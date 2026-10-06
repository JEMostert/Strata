// src/glm/glm_kernels.cu - see include/strata/glm/glm_kernels.hpp. Correctness first: plain kernels, one token.
#include "strata/glm/glm_kernels.hpp"

#include <cuda_runtime.h>

#include <cfloat>
#include <stdexcept>
#include <string>

namespace strata::glm {
namespace {

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("glm kernel ") + what + ": " + cudaGetErrorString(e));
}

__device__ __forceinline__ float warp_sum(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
// Block-wide sum for blockDim.x a multiple of 32 (<= 1024); every thread gets the result.
__device__ float block_sum(float v) {
    __shared__ float part[32];
    __shared__ float total;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_sum(v);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float s = lane < (int) (blockDim.x >> 5) ? part[lane] : 0.0f;
        s = warp_sum(s);
        if (lane == 0) total = s;
    }
    __syncthreads();
    const float r = total;
    __syncthreads();
    return r;
}
__device__ float block_max(float v) {
    __shared__ float part[32];
    __shared__ float total;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    v = warp_max(v);
    if (lane == 0) part[warp] = v;
    __syncthreads();
    if (warp == 0) {
        float s = lane < (int) (blockDim.x >> 5) ? part[lane] : -FLT_MAX;
        s = warp_max(s);
        if (lane == 0) total = s;
    }
    __syncthreads();
    const float r = total;
    __syncthreads();
    return r;
}
__device__ __forceinline__ float sigmoidf_(float x) { return 1.0f / (1.0f + expf(-x)); }

__global__ void rms_norm_rows_kernel(const float* x, const float* w, float* y, int n, float eps) {
    const float* xr = x + (size_t) blockIdx.x * n;
    float* yr = y + (size_t) blockIdx.x * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss);
    const float sc = 1.0f / sqrtf(ss / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) yr[i] = w ? xr[i] * sc * w[i] : xr[i] * sc;
}

__global__ void layer_norm_kernel(const float* x, const float* w, const float* b, float* y, int n, float eps) {
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += x[i];
    const float mean = block_sum(s) / (float) n;
    float v = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { const float d = x[i] - mean; v += d * d; }
    const float sc = 1.0f / sqrtf(block_sum(v) / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = (x[i] - mean) * sc * w[i] + b[i];
}

__global__ void l2_norm_rows_kernel(float* x, int n, float eps) {
    float* xr = x + (size_t) blockIdx.x * n;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) ss += xr[i] * xr[i];
    ss = block_sum(ss);
    // llama.cpp: rms_norm(x, eps / n) / sqrt(n) = x / sqrt(sum x^2 + eps)
    const float sc = 1.0f / sqrtf(ss / (float) n + eps / (float) n) / sqrtf((float) n);
    for (int i = threadIdx.x; i < n; i += blockDim.x) xr[i] *= sc;
}

__global__ void hc_read_kernel(const float* mixes, const float* scale, const float* base, const float* R, float* x_out,
                               float* post_out, float* comb_out, int n_embd, float eps, int iters) {
    __shared__ float pre[kHc];
    if (threadIdx.x == 0) {
        float post[kHc], c[kHc][kHc];   // c[dst][src]
        for (int h = 0; h < kHc; ++h) {
            pre[h] = sigmoidf_(mixes[h] * scale[0] + base[h]) + eps;
            post[h] = 2.0f * sigmoidf_(mixes[kHc + h] * scale[1] + base[kHc + h]);
        }
        for (int s = 0; s < kHc; ++s)
            for (int d = 0; d < kHc; ++d) {
                const int r = d + kHc * s;   // ggml [dst, src]: element (d, s) at d + 4 s
                c[d][s] = mixes[2 * kHc + r] * scale[2] + base[2 * kHc + r];
            }
        // softmax over dst for each src, then + eps
        for (int s = 0; s < kHc; ++s) {
            float m = c[0][s];
            for (int d = 1; d < kHc; ++d) m = fmaxf(m, c[d][s]);
            float sum = 0.0f;
            for (int d = 0; d < kHc; ++d) { c[d][s] = expf(c[d][s] - m); sum += c[d][s]; }
            for (int d = 0; d < kHc; ++d) c[d][s] = c[d][s] / sum + eps;
        }
        auto norm_cols = [&]() {   // over src for each dst
            for (int d = 0; d < kHc; ++d) {
                float sum = 0.0f;
                for (int s = 0; s < kHc; ++s) sum += c[d][s];
                sum += eps;
                for (int s = 0; s < kHc; ++s) c[d][s] /= sum;
            }
        };
        auto norm_rows = [&]() {   // over dst for each src
            for (int s = 0; s < kHc; ++s) {
                float sum = 0.0f;
                for (int d = 0; d < kHc; ++d) sum += c[d][s];
                sum += eps;
                for (int d = 0; d < kHc; ++d) c[d][s] /= sum;
            }
        };
        norm_cols();
        for (int i = 1; i < iters; ++i) { norm_rows(); norm_cols(); }
        for (int h = 0; h < kHc; ++h) post_out[h] = post[h];
        for (int d = 0; d < kHc; ++d)
            for (int s = 0; s < kHc; ++s) comb_out[d + kHc * s] = c[d][s];
    }
    __syncthreads();
    for (int i = threadIdx.x; i < n_embd; i += blockDim.x) {
        float acc = R[i] * pre[0];
        for (int h = 1; h < kHc; ++h) acc += R[(size_t) h * n_embd + i] * pre[h];
        x_out[i] = acc;
    }
}

__global__ void hc_write_kernel(const float* f, const float* R_in, const float* post, const float* comb, float* R_out,
                                int n_embd) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    float r[kHc];
    for (int s = 0; s < kHc; ++s) r[s] = R_in[(size_t) s * n_embd + i];
    const float x = f[i];
    for (int d = 0; d < kHc; ++d) {
        float acc = x * post[d];
        for (int s = 0; s < kHc; ++s) acc += r[s] * comb[d + kHc * s];
        R_out[(size_t) d * n_embd + i] = acc;
    }
}

__global__ void hc_mean_kernel(const float* R, float* y, int n_embd) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    float acc = R[i];
    for (int h = 1; h < kHc; ++h) acc += R[(size_t) h * n_embd + i];
    y[i] = acc * (1.0f / kHc);
}

__global__ void kda_conv_silu_kernel(const float* x, float* state, const float* w, float* y, int channels, int k) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    // window: state[0..k-2] (oldest first), then x; weights w[c*k + j] in the same order
    float* st = state + (size_t) c * (k - 1);
    const float* wc = w + (size_t) c * k;
    float acc = 0.0f;
    for (int j = 0; j < k - 1; ++j) acc += st[j] * wc[j];
    acc += x[c] * wc[k - 1];
    for (int j = 0; j < k - 2; ++j) st[j] = st[j + 1];
    st[k - 2] = x[c];
    y[c] = acc / (1.0f + expf(-acc));
}

__global__ void kda_gate_kernel(const float* gf, const float* dt_bias, const float* A, float* g, int n, int head_dim,
                                float lower) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float t = (gf[i] + dt_bias[i]) * A[i / head_dim];
    g[i] = sigmoidf_(-t) * lower;
}

__global__ void sigmoid_kernel(float* x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = sigmoidf_(x[i]);
}

// One block per head, one thread per row j (= column j of S); S row j holds S[i][j] for i = 0..d-1.
__global__ void kda_step_kernel(float* S, const float* q, const float* k, const float* v, const float* g,
                                const float* beta, float* out, int d) {
    const int h = blockIdx.x, j = threadIdx.x;
    extern __shared__ float sh[];
    float* eg = sh;          // exp(g)
    float* kk = sh + d;
    float* qq = sh + 2 * d;
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        eg[i] = expf(g[(size_t) h * d + i]);
        kk[i] = k[(size_t) h * d + i];
        qq[i] = q[(size_t) h * d + i];
    }
    __syncthreads();
    if (j >= d) return;
    float* row = S + ((size_t) h * d + j) * d;
    float sum = 0.0f;
    for (int i = 0; i < d; ++i) {
        const float s = row[i] * eg[i];
        row[i] = s;
        sum += s * kk[i];
    }
    const float delta = (v[(size_t) h * d + j] - sum) * beta[h];
    float o = 0.0f;
    for (int i = 0; i < d; ++i) {
        const float s = row[i] + kk[i] * delta;
        row[i] = s;
        o += s * qq[i];
    }
    out[(size_t) h * d + j] = o * (1.0f / sqrtf((float) d));
}

__global__ void kda_out_gate_kernel(const float* o, const float* w, const float* g2, float* out, int d, float eps) {
    const float* orow = o + (size_t) blockIdx.x * d;
    float ss = 0.0f;
    for (int i = threadIdx.x; i < d; i += blockDim.x) ss += orow[i] * orow[i];
    ss = block_sum(ss);
    const float sc = 1.0f / sqrtf(ss / (float) d + eps);
    for (int i = threadIdx.x; i < d; i += blockDim.x) {
        const size_t idx = (size_t) blockIdx.x * d + i;
        out[idx] = orow[i] * sc * w[i] * sigmoidf_(g2[idx]);
    }
}

// One block per head: scores into scratch, softmax, weighted sum of the latents.
__global__ void mla_attend_kernel(const float* q_abs, const float* cache, int n_vis, int lat, float scale,
                                  float* out, float* scratch) {
    const int h = blockIdx.x;
    const float* q = q_abs + (size_t) h * lat;
    float* sc = scratch + (size_t) h * n_vis;
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5, warps = blockDim.x >> 5;
    for (int t = warp; t < n_vis; t += warps) {
        const float* c = cache + (size_t) t * lat;
        float s = 0.0f;
        for (int i = lane; i < lat; i += 32) s += q[i] * c[i];
        s = warp_sum(s);
        if (lane == 0) sc[t] = s * scale;
    }
    __syncthreads();
    float m = -FLT_MAX;
    for (int t = threadIdx.x; t < n_vis; t += blockDim.x) m = fmaxf(m, sc[t]);
    m = block_max(m);
    float z = 0.0f;
    for (int t = threadIdx.x; t < n_vis; t += blockDim.x) z += expf(sc[t] - m);
    z = block_sum(z);
    const float inv = n_vis > 0 ? 1.0f / z : 0.0f;
    for (int i = threadIdx.x; i < lat; i += blockDim.x) {
        float acc = 0.0f;
        for (int t = 0; t < n_vis; ++t) acc += expf(sc[t] - m) * cache[(size_t) t * lat + i];
        out[(size_t) h * lat + i] = acc * inv;
    }
}

__global__ void kpool_key_kernel(const float* key, const float* gate, const float* ape, float* pooled, int kpool,
                                 int dim) {
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    float m = -FLT_MAX;
    for (int j = 0; j < kpool; ++j) m = fmaxf(m, gate[(size_t) j * dim + d] + ape[(size_t) j * dim + d]);
    float z = 0.0f;
    for (int j = 0; j < kpool; ++j) z += expf(gate[(size_t) j * dim + d] + ape[(size_t) j * dim + d] - m);
    float acc = 0.0f;
    for (int j = 0; j < kpool; ++j)
        acc += expf(gate[(size_t) j * dim + d] + ape[(size_t) j * dim + d] - m) / z * key[(size_t) j * dim + d];
    pooled[d] = acc;
}

__global__ void gemv_f32_kernel(const float* W, const float* x, float* y, int n_in, int n_out) {
    const int row = blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (row >= n_out) return;
    const float* w = W + (size_t) row * n_in;
    float s = 0.0f;
    for (int i = lane; i < n_in; i += 32) s += w[i] * x[i];
    s = warp_sum(s);
    if (lane == 0) y[row] = s;
}

__global__ void router_topk_kernel(const float* logits, const float* bias, int n_expert, int k, float scale,
                                   int32_t* ids, float* weights) {
    if (threadIdx.x != 0) return;
    float probs[512];
    bool taken[512];
    for (int e = 0; e < n_expert; ++e) { probs[e] = sigmoidf_(logits[e]); taken[e] = false; }
    float sum = 0.0f;
    for (int j = 0; j < k; ++j) {
        int best = -1;
        float bv = -FLT_MAX;
        for (int e = 0; e < n_expert; ++e) {
            const float v = probs[e] + (bias ? bias[e] : 0.0f);
            if (!taken[e] && v > bv) { bv = v; best = e; }
        }
        taken[best] = true;
        ids[j] = best;
        weights[j] = probs[best];
        sum += probs[best];
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (int j = 0; j < k; ++j) weights[j] = weights[j] / sum * scale;
}

__global__ void swiglu_clamp_kernel(const float* g, const float* u, float* out, int n, float limit) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float gg = fminf(g[i], limit);
    const float uu = fminf(fmaxf(u[i], -limit), limit);
    out[i] = gg / (1.0f + expf(-gg)) * uu;
}

__global__ void axpy_dev_kernel(float* y, const float* x, const float* a, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += a[0] * x[i];
}

__global__ void add_kernel(float* y, const float* x, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] += x[i];
}

__global__ void argmax_kernel(const float* x, int n, int32_t* out) {
    __shared__ float bv[1024];
    __shared__ int bi[1024];
    float v = -FLT_MAX;
    int idx = 0;
    for (int i = threadIdx.x; i < n; i += blockDim.x)
        if (x[i] > v) { v = x[i]; idx = i; }
    bv[threadIdx.x] = v;
    bi[threadIdx.x] = idx;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s) {
            const float o = bv[threadIdx.x + s];
            const int oi = bi[threadIdx.x + s];
            if (o > bv[threadIdx.x] || (o == bv[threadIdx.x] && oi < bi[threadIdx.x])) {
                bv[threadIdx.x] = o;
                bi[threadIdx.x] = oi;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x == 0) *out = bi[0];
}

__global__ void iota_kernel(int32_t* d, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) d[i] = i;
}
__global__ void set_pair_kernel(int32_t* d, int32_t a, int32_t b) {
    d[0] = a;
    d[1] = b;
}

inline unsigned blocks(int n, int t) { return (unsigned) ((n + t - 1) / t); }

}  // namespace

void rms_norm_rows(const float* x, const float* w, float* y, int rows, int n, float eps, void* stream) {
    rms_norm_rows_kernel<<<rows, 256, 0, (cudaStream_t) stream>>>(x, w, y, n, eps);
    check("rms_norm_rows");
}
void layer_norm(const float* x, const float* w, const float* b, float* y, int n, float eps, void* stream) {
    layer_norm_kernel<<<1, 256, 0, (cudaStream_t) stream>>>(x, w, b, y, n, eps);
    check("layer_norm");
}
void l2_norm_rows(float* x, int rows, int n, float eps, void* stream) {
    l2_norm_rows_kernel<<<rows, 128, 0, (cudaStream_t) stream>>>(x, n, eps);
    check("l2_norm_rows");
}
void hc_read(const float* mixes, const float* scale, const float* base, const float* R, float* x_out, float* post,
             float* comb, int n_embd, float eps, int iters, void* stream) {
    hc_read_kernel<<<1, 1024, 0, (cudaStream_t) stream>>>(mixes, scale, base, R, x_out, post, comb, n_embd, eps,
                                                          iters);
    check("hc_read");
}
void hc_write(const float* f, const float* R_in, const float* post, const float* comb, float* R_out, int n_embd,
              void* stream) {
    hc_write_kernel<<<blocks(n_embd, 256), 256, 0, (cudaStream_t) stream>>>(f, R_in, post, comb, R_out, n_embd);
    check("hc_write");
}
void hc_mean(const float* R, float* y, int n_embd, void* stream) {
    hc_mean_kernel<<<blocks(n_embd, 256), 256, 0, (cudaStream_t) stream>>>(R, y, n_embd);
    check("hc_mean");
}
void kda_conv_silu(const float* x, float* state, const float* w, float* y, int channels, int k, void* stream) {
    kda_conv_silu_kernel<<<blocks(channels, 256), 256, 0, (cudaStream_t) stream>>>(x, state, w, y, channels, k);
    check("kda_conv_silu");
}
void kda_gate(const float* gf, const float* dt_bias, const float* A, float* g, int n_head, int head_dim, float lower,
              void* stream) {
    const int n = n_head * head_dim;
    kda_gate_kernel<<<blocks(n, 256), 256, 0, (cudaStream_t) stream>>>(gf, dt_bias, A, g, n, head_dim, lower);
    check("kda_gate");
}
void sigmoid_inplace(float* x, int n, void* stream) {
    sigmoid_kernel<<<blocks(n, 256), 256, 0, (cudaStream_t) stream>>>(x, n);
    check("sigmoid");
}
void kda_step(float* S, const float* q, const float* k, const float* v, const float* g, const float* beta,
              float* out, int n_head, int head_dim, void* stream) {
    kda_step_kernel<<<n_head, head_dim, 3 * head_dim * sizeof(float), (cudaStream_t) stream>>>(S, q, k, v, g, beta,
                                                                                             out, head_dim);
    check("kda_step");
}
void kda_out_gate(const float* o, const float* w, const float* g2, float* out, int n_head, int head_dim, float eps,
                  void* stream) {
    kda_out_gate_kernel<<<n_head, 128, 0, (cudaStream_t) stream>>>(o, w, g2, out, head_dim, eps);
    check("kda_out_gate");
}
void mla_attend(const float* q_abs, const float* cache, int n_vis, int n_head, int lat, float scale, float* out,
                float* scratch, void* stream) {
    mla_attend_kernel<<<n_head, 256, 0, (cudaStream_t) stream>>>(q_abs, cache, n_vis, lat, scale, out, scratch);
    check("mla_attend");
}
void kpool_key(const float* key, const float* gate, const float* ape, float* pooled, int kpool, int dim, void* stream) {
    kpool_key_kernel<<<blocks(dim, 128), 128, 0, (cudaStream_t) stream>>>(key, gate, ape, pooled, kpool, dim);
    check("kpool_key");
}
void gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out, void* stream) {
    gemv_f32_kernel<<<blocks(n_out, 8), 256, 0, (cudaStream_t) stream>>>(W, x, y, n_in, n_out);
    check("gemv_f32");
}
void router_topk(const float* logits, const float* bias, int n_expert, int k, float scale, int32_t* ids,
                 float* weights, void* stream) {
    if (n_expert > 512) throw std::runtime_error("router_topk: at most 512 experts");
    router_topk_kernel<<<1, 32, 0, (cudaStream_t) stream>>>(logits, bias, n_expert, k, scale, ids, weights);
    check("router_topk");
}
void swiglu_clamp(const float* g, const float* u, float* out, int n, float limit, void* stream) {
    swiglu_clamp_kernel<<<blocks(n, 256), 256, 0, (cudaStream_t) stream>>>(g, u, out, n, limit);
    check("swiglu_clamp");
}
void axpy_dev(float* y, const float* x, const float* a_dev, int n, void* stream) {
    axpy_dev_kernel<<<blocks(n, 256), 256, 0, (cudaStream_t) stream>>>(y, x, a_dev, n);
    check("axpy_dev");
}
void add_inplace(float* y, const float* x, int n, void* stream) {
    add_kernel<<<blocks(n, 256), 256, 0, (cudaStream_t) stream>>>(y, x, n);
    check("add_inplace");
}
void iota(int32_t* d, int n, void* stream) {
    iota_kernel<<<blocks(n, 256), 256, 0, (cudaStream_t) stream>>>(d, n);
    check("iota");
}
void set_pair(int32_t* d, int32_t a, int32_t b, void* stream) {
    set_pair_kernel<<<1, 1, 0, (cudaStream_t) stream>>>(d, a, b);
    check("set_pair");
}
void argmax(const float* x, int n, int32_t* out, void* stream) {
    argmax_kernel<<<1, 1024, 0, (cudaStream_t) stream>>>(x, n, out);
    check("argmax");
}

}  // namespace strata::glm
