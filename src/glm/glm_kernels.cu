// src/glm/glm_kernels.cu - see include/strata/glm/glm_kernels.hpp. Correctness first: plain kernels, one token.
#include "strata/glm/glm_kernels.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <algorithm>
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

__global__ void scale_kernel(float* x, int n, float a) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] *= a;
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


__global__ void hc_read_rows_kernel(const float* mixes, const float* scale, const float* base, const float* R,
                                    float* x_out, float* post_out, float* comb_out, int n_embd, float eps, int iters) {
    const int t = blockIdx.x;
    mixes += (size_t) t * kHcMix;
    R += (size_t) t * kHc * n_embd;
    x_out += (size_t) t * n_embd;
    post_out += (size_t) t * kHc;
    comb_out += (size_t) t * kHc * kHc;
    __shared__ float pre[kHc];
    if (threadIdx.x == 0) {
        float post[kHc], c[kHc][kHc];
        for (int h = 0; h < kHc; ++h) {
            pre[h] = sigmoidf_(mixes[h] * scale[0] + base[h]) + eps;
            post[h] = 2.0f * sigmoidf_(mixes[kHc + h] * scale[1] + base[kHc + h]);
        }
        for (int s = 0; s < kHc; ++s)
            for (int d = 0; d < kHc; ++d) {
                const int r = d + kHc * s;
                c[d][s] = mixes[2 * kHc + r] * scale[2] + base[2 * kHc + r];
            }
        for (int s = 0; s < kHc; ++s) {
            float m = c[0][s];
            for (int d = 1; d < kHc; ++d) m = fmaxf(m, c[d][s]);
            float sum = 0.0f;
            for (int d = 0; d < kHc; ++d) { c[d][s] = expf(c[d][s] - m); sum += c[d][s]; }
            for (int d = 0; d < kHc; ++d) c[d][s] = c[d][s] / sum + eps;
        }
        for (int it = 0; it < iters; ++it) {
            if (it > 0)
                for (int s = 0; s < kHc; ++s) {
                    float sum = 0.0f;
                    for (int d = 0; d < kHc; ++d) sum += c[d][s];
                    sum += eps;
                    for (int d = 0; d < kHc; ++d) c[d][s] /= sum;
                }
            for (int d = 0; d < kHc; ++d) {
                float sum = 0.0f;
                for (int s = 0; s < kHc; ++s) sum += c[d][s];
                sum += eps;
                for (int s = 0; s < kHc; ++s) c[d][s] /= sum;
            }
        }
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

__global__ void hc_write_rows_kernel(const float* f, const float* R_in, const float* post, const float* comb,
                                     float* R_out, int n_embd) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n_embd) return;
    const float* Ri = R_in + (size_t) t * kHc * n_embd;
    float* Ro = R_out + (size_t) t * kHc * n_embd;
    const float* pt = post + (size_t) t * kHc;
    const float* ct = comb + (size_t) t * kHc * kHc;
    float r[kHc];
    for (int s = 0; s < kHc; ++s) r[s] = Ri[(size_t) s * n_embd + i];
    const float x = f[(size_t) t * n_embd + i];
    for (int d = 0; d < kHc; ++d) {
        float acc = x * pt[d];
        for (int s = 0; s < kHc; ++s) acc += r[s] * ct[d + kHc * s];
        Ro[(size_t) d * n_embd + i] = acc;
    }
}

__global__ void kda_conv_silu_seq_kernel(const float* x, float* state, const float* w, float* y, int channels, int k,
                                         int T) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= channels) return;
    float win[8];
    float* st = state + (size_t) c * (k - 1);
    for (int j = 0; j < k - 1; ++j) win[j] = st[j];
    const float* wc = w + (size_t) c * k;
    for (int t = 0; t < T; ++t) {
        const float xv = x[(size_t) t * channels + c];
        float acc = 0.0f;
        for (int j = 0; j < k - 1; ++j) acc += win[j] * wc[j];
        acc += xv * wc[k - 1];
        for (int j = 0; j < k - 2; ++j) win[j] = win[j + 1];
        win[k - 2] = xv;
        y[(size_t) t * channels + c] = acc / (1.0f + expf(-acc));
    }
    for (int j = 0; j < k - 1; ++j) st[j] = win[j];
}

__global__ void kda_gate_rows_kernel(const float* gf, const float* dt_bias, const float* A, float* g, int64_t n,
                                     int row, int head_dim, float lower) {
    const int64_t i = (int64_t) blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int c = (int) (i % row);
    const float t = (gf[i] + dt_bias[c]) * A[c / head_dim];
    g[i] = sigmoidf_(-t) * lower;
}

__global__ void kda_scan_kernel(float* S, const float* q, const float* k, const float* v, const float* g,
                                const float* beta, float* out, int n_head, int d, int T) {
    const int h = blockIdx.x, j = threadIdx.x;
    extern __shared__ float sh[];
    float* eg = sh;
    float* kk = sh + d;
    float* qq = sh + 2 * d;
    float* row = S + ((size_t) h * d + j) * d;
    const float inv = 1.0f / sqrtf((float) d);
    for (int t = 0; t < T; ++t) {
        const size_t base = ((size_t) t * n_head + h) * d;
        __syncthreads();
        for (int i = threadIdx.x; i < d; i += blockDim.x) {
            eg[i] = expf(g[base + i]);
            kk[i] = k[base + i];
            qq[i] = q[base + i];
        }
        __syncthreads();
        if (j < d) {
            float sum = 0.0f;
            for (int i = 0; i < d; ++i) {
                const float s = row[i] * eg[i];
                row[i] = s;
                sum += s * kk[i];
            }
            const float delta = (v[base + j] - sum) * beta[(size_t) t * n_head + h];
            float o = 0.0f;
            for (int i = 0; i < d; ++i) {
                const float s = row[i] + kk[i] * delta;
                row[i] = s;
                o += s * qq[i];
            }
            out[base + j] = o * inv;
        }
    }
}

// The scan with the state in registers: head = block, row j = tid / 4, its quarter q = tid % 4 holds S[32q..32q+31][j].
// Per token: decay, the row's dot with k (4 partial sums, shuffle-reduced), the delta update, the dot with q.
template <int D>
__global__ void __launch_bounds__(4 * D) kda_scan_reg_kernel(float* S, const float* q, const float* k, const float* v,
                                                             const float* g, const float* beta, float* out, int n_head,
                                                             int T) {
    constexpr int P = D / 4;
    const int h = blockIdx.x, tid = threadIdx.x, j = tid >> 2, qt = tid & 3;
    __shared__ float eg[2][D], kk[2][D], qq[2][D];
    float s[P];
    float* row = S + ((size_t) h * D + j) * D + qt * P;
#pragma unroll
    for (int i = 0; i < P; ++i) s[i] = row[i];
    const float inv = rsqrtf((float) D);
    auto load = [&](int t_, int buf) {
        const size_t base = ((size_t) t_ * n_head + h) * D;
        if (tid < D) eg[buf][tid] = expf(g[base + tid]);
        else if (tid < 2 * D) kk[buf][tid - D] = k[base + tid - D];
        else if (tid < 3 * D) qq[buf][tid - 2 * D] = q[base + tid - 2 * D];
    };
    if (T > 0) load(0, 0);
    __syncthreads();
    for (int t_ = 0; t_ < T; ++t_) {
        const int b = t_ & 1;
        if (t_ + 1 < T) load(t_ + 1, b ^ 1);
        const size_t base = ((size_t) t_ * n_head + h) * D;
        float part = 0.0f;
#pragma unroll
        for (int i = 0; i < P; ++i) {
            s[i] *= eg[b][qt * P + i];
            part += s[i] * kk[b][qt * P + i];
        }
        part += __shfl_xor_sync(0xffffffffu, part, 1);
        part += __shfl_xor_sync(0xffffffffu, part, 2);
        const float delta = (v[base + j] - part) * beta[(size_t) t_ * n_head + h];
        float o = 0.0f;
#pragma unroll
        for (int i = 0; i < P; ++i) {
            s[i] += kk[b][qt * P + i] * delta;
            o += s[i] * qq[b][qt * P + i];
        }
        o += __shfl_xor_sync(0xffffffffu, o, 1);
        o += __shfl_xor_sync(0xffffffffu, o, 2);
        if (qt == 0) out[base + j] = o * inv;
        __syncthreads();
    }
#pragma unroll
    for (int i = 0; i < P; ++i) row[i] = s[i];
}

__global__ void layer_norm_rows_kernel(const float* x, const float* w, const float* b, float* y, int n, float eps) {
    x += (size_t) blockIdx.x * n;
    y += (size_t) blockIdx.x * n;
    float s = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) s += x[i];
    const float mean = block_sum(s) / (float) n;
    float v = 0.0f;
    for (int i = threadIdx.x; i < n; i += blockDim.x) { const float d = x[i] - mean; v += d * d; }
    const float sc = 1.0f / sqrtf(block_sum(v) / (float) n + eps);
    for (int i = threadIdx.x; i < n; i += blockDim.x) y[i] = (x[i] - mean) * sc * w[i] + b[i];
}

// MQA over the latent cache: block = 8 heads (one warp each) of one query token; the latents are read once per
// block in tiles through shared memory, online softmax per head, lane-held accumulators (lat / 32 per lane).
template <int LAT, int QB>
__global__ void mla_attend_rows_kernel(const float* q_abs, const float* cache, int64_t p0, int T, int n_head, int kpool,
                                       float scale, float* out, const uint8_t* sel, int64_t sel_ld) {
    // block = 8 heads (a warp each) of QB consecutive queries; each latent tile is read once for all of them
    constexpr int TILE = 16, PER = LAT / 32;
    __shared__ float tile[TILE][LAT];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const int h = blockIdx.x * 8 + warp;
    const int tq0 = blockIdx.y * QB;
    const int nq = min(QB, T - tq0);
    float q[QB][PER], acc[QB][PER], m[QB], z[QB];
    int64_t n_vis[QB], tail0[QB];
    const uint8_t* srow[QB];
#pragma unroll
    for (int a = 0; a < QB; ++a) {
        const int t = tq0 + (a < nq ? a : 0);
        n_vis[a] = a < nq ? p0 + t + 1 : 0;   // complete pools + the tail up to the query itself
        tail0[a] = ((p0 + t + 1) / kpool) * kpool;
        srow[a] = sel ? sel + (size_t) t * sel_ld : nullptr;
        const float* qh = q_abs + ((size_t) t * n_head + h) * LAT;
        for (int i = 0; i < PER; ++i) { q[a][i] = qh[lane + 32 * i]; acc[a][i] = 0.0f; }
        m[a] = -FLT_MAX;
        z[a] = 0.0f;
    }
    const int64_t n_all = p0 + tq0 + nq;
    for (int64_t t0 = 0; t0 < n_all; t0 += TILE) {
        const int nt = (int) min((int64_t) TILE, n_all - t0);
        __syncthreads();
        for (int i = threadIdx.x; i < nt * LAT; i += blockDim.x)
            tile[i / LAT][i % LAT] = cache[(size_t) (t0 + i / LAT) * LAT + i % LAT];
        __syncthreads();
#pragma unroll
        for (int a = 0; a < QB; ++a) {
            if (t0 >= n_vis[a]) continue;
            float sc[TILE];
            float tmax = -FLT_MAX;
            for (int l = 0; l < nt; ++l) {
                const int64_t ps = t0 + l;
                if (ps >= n_vis[a] || (srow[a] && ps < tail0[a] && !srow[a][ps / kpool])) { sc[l] = -FLT_MAX; continue; }
                float s = 0.0f;
                for (int i = 0; i < PER; ++i) s += q[a][i] * tile[l][lane + 32 * i];
                s = warp_sum(s) * scale;
                sc[l] = s;
                tmax = fmaxf(tmax, s);
            }
            if (tmax == -FLT_MAX) continue;
            const float mn = fmaxf(m[a], tmax);
            const float corr = expf(m[a] - mn);
            z[a] *= corr;
            for (int i = 0; i < PER; ++i) acc[a][i] *= corr;
            for (int l = 0; l < nt; ++l) {
                if (sc[l] == -FLT_MAX) continue;
                const float p = expf(sc[l] - mn);
                z[a] += p;
                for (int i = 0; i < PER; ++i) acc[a][i] += p * tile[l][lane + 32 * i];
            }
            m[a] = mn;
        }
    }
#pragma unroll
    for (int a = 0; a < QB; ++a) {
        if (a >= nq) break;
        float* oh = out + ((size_t) (tq0 + a) * n_head + h) * LAT;
        const float inv = n_vis[a] > 0 ? 1.0f / z[a] : 0.0f;
        for (int i = 0; i < PER; ++i) oh[lane + 32 * i] = acc[a][i] * inv;
    }
}

// one block per query: q (n_head x dim) and the weights in shared memory, a warp per pool
__global__ void ix_scores_kernel(const float* q, const float* w, const float* pooled, int64_t p0, int n_head, int dim,
                                 int kpool, int64_t n_pool, float* scores) {
    extern __shared__ float qs[];   // n_head * dim + n_head
    const int t = blockIdx.y;
    const float* qt = q + (size_t) t * n_head * dim;
    for (int i = threadIdx.x; i < n_head * dim; i += blockDim.x) qs[i] = qt[i];
    for (int i = threadIdx.x; i < n_head; i += blockDim.x) qs[n_head * dim + i] = w[(size_t) t * n_head + i];
    __syncthreads();
    const int64_t vis = (p0 + t + 1) / kpool;
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, warps = blockDim.x >> 5;
    for (int64_t j = (int64_t) blockIdx.x * warps + warp; j < n_pool; j += (int64_t) gridDim.x * warps) {
        float sc = -INFINITY;
        if (j < vis) {
            const float* pk = pooled + (size_t) j * dim;
            float acc = 0.0f;
            for (int hh = 0; hh < n_head; ++hh) {
                float d = 0.0f;
                for (int i = lane; i < dim; i += 32) d += qs[hh * dim + i] * pk[i];
                d = warp_sum(d);
                acc += qs[n_head * dim + hh] * fmaxf(d, 0.0f);
            }
            sc = acc;
        }
        if (lane == 0) scores[(size_t) t * n_pool + j] = sc;
    }
}

__device__ __forceinline__ uint32_t f2key(float f) {   // order-preserving float -> uint32
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

// one block per query: the n_top-th largest key by bisection on the 32 key bits, then mark (ties in index order)
__global__ void ix_select_kernel(const float* scores, int64_t p0, int kpool, int64_t n_pool, int n_top, uint8_t* sel) {
    const int t = blockIdx.x;
    const float* sr = scores + (size_t) t * n_pool;
    uint8_t* so = sel + (size_t) t * n_pool;
    const int64_t vis = min((p0 + t + 1) / kpool, n_pool);
    for (int64_t j = threadIdx.x; j < n_pool; j += blockDim.x) so[j] = 0;
    __syncthreads();
    if (vis <= n_top) {
        for (int64_t j = threadIdx.x; j < vis; j += blockDim.x) so[j] = 1;
        return;
    }
    __shared__ uint32_t s_thr;
    __shared__ unsigned s_cnt;
    uint32_t lo = 0, hi = 0xffffffffu;   // find the largest key thr with count(key >= thr) >= n_top
    while (lo < hi) {
        const uint32_t mid = lo + (uint32_t) (((uint64_t) hi - lo + 1) / 2);
        if (threadIdx.x == 0) s_cnt = 0;
        __syncthreads();
        unsigned c = 0;
        for (int64_t j = threadIdx.x; j < vis; j += blockDim.x) c += f2key(sr[j]) >= mid;
        c = (unsigned) block_sum((float) c);
        if (c >= (unsigned) n_top) lo = mid; else hi = mid - 1;
        __syncthreads();
    }
    if (threadIdx.x == 0) { s_thr = lo; s_cnt = 0; }
    __syncthreads();
    const uint32_t thr = s_thr;
    for (int64_t j = threadIdx.x; j < vis; j += blockDim.x)
        if (f2key(sr[j]) > thr) { so[j] = 1; atomicAdd(&s_cnt, 1u); }
    __syncthreads();
    if (threadIdx.x == 0) {   // ties at the threshold, lowest index first, until n_top
        unsigned have = s_cnt;
        for (int64_t j = 0; j < vis && have < (unsigned) n_top; ++j)
            if (f2key(sr[j]) == thr) { so[j] = 1; ++have; }
    }
}

__global__ void kpool_keys_kernel(const float* key_cache, const float* gate_cache, const float* ape, float* pooled,
                                  int64_t pool0, int kpool, int dim) {
    const int64_t pool = pool0 + blockIdx.y;
    const int d = blockIdx.x * blockDim.x + threadIdx.x;
    if (d >= dim) return;
    const float* key = key_cache + (size_t) pool * kpool * dim;
    const float* gate = gate_cache + (size_t) pool * kpool * dim;
    float m = -FLT_MAX;
    for (int j = 0; j < kpool; ++j) m = fmaxf(m, gate[(size_t) j * dim + d] + ape[(size_t) j * dim + d]);
    float z = 0.0f;
    for (int j = 0; j < kpool; ++j) z += expf(gate[(size_t) j * dim + d] + ape[(size_t) j * dim + d] - m);
    float acc = 0.0f;
    for (int j = 0; j < kpool; ++j)
        acc += expf(gate[(size_t) j * dim + d] + ape[(size_t) j * dim + d] - m) / z * key[(size_t) j * dim + d];
    pooled[(size_t) pool * dim + d] = acc;
}

__global__ void gemm_f32_kernel(const float* W, const float* x, float* y, int n_in, int n_out, int T) {
    const int64_t gw = (int64_t) blockIdx.x * (blockDim.x >> 5) + (threadIdx.x >> 5);
    const int lane = threadIdx.x & 31;
    if (gw >= (int64_t) n_out * T) return;
    const int row = (int) (gw % n_out), t = (int) (gw / n_out);
    const float* w = W + (size_t) row * n_in;
    const float* xt = x + (size_t) t * n_in;
    float s = 0.0f;
    for (int i = lane; i < n_in; i += 32) s += w[i] * xt[i];
    s = warp_sum(s);
    if (lane == 0) y[(size_t) t * n_out + row] = s;
}

__global__ void router_topk_rows_kernel(const float* logits, const float* bias, int n_expert, int k, float scale,
                                        int32_t* ids, float* weights, int T) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= T) return;
    const float* lg = logits + (size_t) t * n_expert;
    int32_t* id = ids + (size_t) t * k;
    float* wt = weights + (size_t) t * k;
    unsigned taken[16] = {0};
    float sum = 0.0f;
    for (int j = 0; j < k; ++j) {
        int best = -1;
        float bv = -FLT_MAX;
        for (int e = 0; e < n_expert; ++e) {
            if (taken[e >> 5] & (1u << (e & 31))) continue;
            const float v = sigmoidf_(lg[e]) + (bias ? bias[e] : 0.0f);
            if (v > bv) { bv = v; best = e; }
        }
        taken[best >> 5] |= 1u << (best & 31);
        id[j] = best;
        wt[j] = sigmoidf_(lg[best]);
        sum += wt[j];
    }
    sum = fmaxf(sum, 6.103515625e-5f);
    for (int j = 0; j < k; ++j) wt[j] = wt[j] / sum * scale;
}

__global__ void scatter_scaled_kernel(const float* y, const int32_t* dst, const int32_t* wi, const float* w,
                                      float* parts, int n) {
    const int r = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    parts[(size_t) dst[r] * n + i] = w[wi[r]] * y[(size_t) r * n + i];
}

__global__ void sum_parts_kernel(const float* parts, const float* extra, float* out, int k, int n) {
    const int t = blockIdx.y;
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float* p = parts + (size_t) t * k * n;
    float acc = p[i];
    for (int j = 1; j < k; ++j) acc += p[(size_t) j * n + i];
    out[(size_t) t * n + i] = acc + extra[(size_t) t * n + i];
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
void scale_inplace(float* x, int n, float a, void* stream) {
    scale_kernel<<<blocks(n, 256), 256, 0, (cudaStream_t) stream>>>(x, n, a);
    check("scale");
}
void sigmoid_inplace(float* x, int n, void* stream) {
    sigmoid_kernel<<<blocks(n, 256), 256, 0, (cudaStream_t) stream>>>(x, n);
    check("sigmoid");
}
void kda_step(float* S, const float* q, const float* k, const float* v, const float* g, const float* beta,
              float* out, int n_head, int head_dim, void* stream) {
    if (head_dim == 128) {   // the scan with one token: the same arithmetic as the prompt path
        kda_scan_reg_kernel<128><<<n_head, 512, 0, (cudaStream_t) stream>>>(S, q, k, v, g, beta, out, n_head, 1);
        check("kda_step (scan)");
        return;
    }
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

void hc_read_rows(const float* mixes, const float* scale, const float* base, const float* R, float* x, float* post,
                  float* comb, int n_embd, float eps, int iters, int T, void* stream) {
    hc_read_rows_kernel<<<T, 512, 0, (cudaStream_t) stream>>>(mixes, scale, base, R, x, post, comb, n_embd, eps, iters);
    check("hc_read_rows");
}
void hc_write_rows(const float* f, const float* R_in, const float* post, const float* comb, float* R_out, int n_embd,
                   int T, void* stream) {
    hc_write_rows_kernel<<<dim3(blocks(n_embd, 256), T), 256, 0, (cudaStream_t) stream>>>(f, R_in, post, comb, R_out,
                                                                                         n_embd);
    check("hc_write_rows");
}
void kda_conv_silu_seq(const float* x, float* state, const float* w, float* y, int channels, int k, int T,
                       void* stream) {
    if (k > 9) throw std::runtime_error("kda_conv_silu_seq: kernel width > 9");
    kda_conv_silu_seq_kernel<<<blocks(channels, 128), 128, 0, (cudaStream_t) stream>>>(x, state, w, y, channels, k, T);
    check("kda_conv_silu_seq");
}
void kda_gate_rows(const float* gf, const float* dt_bias, const float* A, float* g, int n_head, int head_dim, int T,
                   float lower, void* stream) {
    const int64_t n = (int64_t) T * n_head * head_dim;
    kda_gate_rows_kernel<<<(unsigned) ((n + 255) / 256), 256, 0, (cudaStream_t) stream>>>(gf, dt_bias, A, g, n,
                                                                                         n_head * head_dim, head_dim,
                                                                                         lower);
    check("kda_gate_rows");
}
void kda_scan(float* S, const float* q, const float* k, const float* v, const float* g, const float* beta, float* out,
              int n_head, int head_dim, int T, void* stream) {
    if (head_dim == 128) {
        kda_scan_reg_kernel<128><<<n_head, 512, 0, (cudaStream_t) stream>>>(S, q, k, v, g, beta, out, n_head, T);
        check("kda_scan_reg");
        return;
    }
    kda_scan_kernel<<<n_head, head_dim, 3 * head_dim * sizeof(float), (cudaStream_t) stream>>>(S, q, k, v, g, beta, out,
                                                                                             n_head, head_dim, T);
    check("kda_scan");
}
void layer_norm_rows(const float* x, const float* w, const float* b, float* y, int rows, int n, float eps,
                     void* stream) {
    layer_norm_rows_kernel<<<rows, 128, 0, (cudaStream_t) stream>>>(x, w, b, y, n, eps);
    check("layer_norm_rows");
}
void ix_scores(const float* q, const float* w, const float* pooled, int64_t p0, int T, int n_head, int dim, int kpool,
               int64_t n_pool, float* scores, void* stream) {
    const size_t sh = ((size_t) n_head * dim + n_head) * sizeof(float);
    static bool attr = false;
    if (!attr) { cudaFuncSetAttribute(ix_scores_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, 64 * 1024); attr = true; }
    const unsigned gx = (unsigned) std::min<int64_t>((n_pool + 7) / 8, 64);
    ix_scores_kernel<<<dim3(gx, T), 256, sh, (cudaStream_t) stream>>>(q, w, pooled, p0, n_head, dim, kpool, n_pool, scores);
    check("ix_scores");
}
void ix_select(const float* scores, int64_t p0, int T, int kpool, int64_t n_pool, int n_top, uint8_t* sel,
               void* stream) {
    ix_select_kernel<<<T, 256, 0, (cudaStream_t) stream>>>(scores, p0, kpool, n_pool, n_top, sel);
    check("ix_select");
}
void mla_attend_rows(const float* q_abs, const float* cache, int64_t p0, int T, int n_head, int lat, int kpool,
                     float scale, float* out, void* stream, const uint8_t* sel, int64_t sel_ld) {
    if (lat != 512 || n_head % 8) throw std::runtime_error("mla_attend_rows: built for 512-wide latents, 8k heads");
    if (T >= 4)
        mla_attend_rows_kernel<512, 4><<<dim3(n_head / 8, (T + 3) / 4), 256, 0, (cudaStream_t) stream>>>(
            q_abs, cache, p0, T, n_head, kpool, scale, out, sel, sel_ld);
    else
        mla_attend_rows_kernel<512, 1><<<dim3(n_head / 8, T), 256, 0, (cudaStream_t) stream>>>(
            q_abs, cache, p0, T, n_head, kpool, scale, out, sel, sel_ld);
    check("mla_attend_rows");
}
void kpool_keys(const float* key_cache, const float* gate_cache, const float* ape, float* pooled, int64_t pool0, int n,
                int kpool, int dim, void* stream) {
    if (n <= 0) return;
    kpool_keys_kernel<<<dim3(blocks(dim, 128), n), 128, 0, (cudaStream_t) stream>>>(key_cache, gate_cache, ape, pooled,
                                                                                   pool0, kpool, dim);
    check("kpool_keys");
}
void gemm_f32(const float* W, const float* x, float* y, int n_in, int n_out, int T, void* stream) {
    if (T >= 8) {   // cuBLAS SGEMM (plain FP32): y^T (n_out x T) = W (n_out x n_in) . x^T
        static cublasHandle_t hb = nullptr;
        if (!hb && cublasCreate(&hb) != CUBLAS_STATUS_SUCCESS) throw std::runtime_error("cublasCreate");
        cublasSetStream(hb, (cudaStream_t) stream);
        const float one = 1.0f, zero = 0.0f;
        if (cublasSgemm(hb, CUBLAS_OP_T, CUBLAS_OP_N, n_out, T, n_in, &one, W, n_in, x, n_in, &zero, y, n_out) !=
            CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("cublasSgemm");
        return;
    }
    const int64_t warps = (int64_t) n_out * T;
    gemm_f32_kernel<<<(unsigned) ((warps + 7) / 8), 256, 0, (cudaStream_t) stream>>>(W, x, y, n_in, n_out, T);
    check("gemm_f32");
}
void router_topk_rows(const float* logits, const float* bias, int n_expert, int k, float scale, int32_t* ids,
                      float* weights, int T, void* stream) {
    if (n_expert > 512) throw std::runtime_error("router_topk_rows: at most 512 experts");
    router_topk_rows_kernel<<<blocks(T, 64), 64, 0, (cudaStream_t) stream>>>(logits, bias, n_expert, k, scale, ids,
                                                                              weights, T);
    check("router_topk_rows");
}
void scatter_scaled(const float* y, const int32_t* dst, const int32_t* wi, const float* w, float* parts, int rows,
                    int n, void* stream) {
    if (rows <= 0) return;
    scatter_scaled_kernel<<<dim3(blocks(n, 256), rows), 256, 0, (cudaStream_t) stream>>>(y, dst, wi, w, parts, n);
    check("scatter_scaled");
}
void sum_parts(const float* parts, const float* extra, float* out, int T, int k, int n, void* stream) {
    sum_parts_kernel<<<dim3(blocks(n, 256), T), 256, 0, (cudaStream_t) stream>>>(parts, extra, out, k, n);
    check("sum_parts");
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
