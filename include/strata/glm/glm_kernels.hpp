// include/strata/glm/glm_kernels.hpp - the GLM-5.3-Flash ("glm5-next") specific CUDA kernels: the mHC residual
// streams with their Sinkhorn mix, the KDA linear attention, nope-MLA attention over the latent cache, the DSA
// k-pool keys, the sigmoid router and the clamped SwiGLU. The quantized projections use Strata's native_mmvq.
// The maths follows llama.cpp's src/models/glm5-next.cpp (see docs/GLM.md); every kernel here is one token
// unless it says otherwise, activations in f32.
#pragma once

#include <cstdint>

namespace strata::glm {

constexpr int kHc = 4;          // residual streams
constexpr int kHcMix = 24;      // (2 + kHc) * kHc mixes per sublayer: 4 read, 4 write, 16 for the 4 x 4 mix

/// y = x / sqrt(mean(x^2) + eps) * w (w may be null), `rows` rows of n each (one block per row).
void rms_norm_rows(const float* x, const float* w, float* y, int rows, int n, float eps, void* stream);
/// y = (x - mean) / sqrt(var + eps) * w + b, one row of n.
void layer_norm(const float* x, const float* w, const float* b, float* y, int n, float eps, void* stream);
/// The KDA q/k L2 norm (llama.cpp build_gdn_l2_norm): x / sqrt(sum x^2 + eps), `rows` rows of n.
void l2_norm_rows(float* x, int rows, int n, float eps, void* stream);

/// mHC read: from the 24 `mixes` (hc_fn . rmsnorm(flat streams)), `scale` (3) and `base` (24): the read weights
/// pre = sigmoid(m*s0 + b) + eps, the write weights post = 2 sigmoid(m*s1 + b), the 4 x 4 mix by Sinkhorn
/// (softmax over dst, + eps, then column/row normalization `iters` rounds), and the layer input sum_h pre_h R_h.
/// `post` (4) and `comb` (16, element (dst, src) at dst + 4 src) are written for hc_write.
void hc_read(const float* mixes, const float* scale, const float* base, const float* R, float* x_out, float* post,
             float* comb, int n_embd, float eps, int iters, void* stream);
/// mHC write: R_out[dst] = post[dst] * f + sum_src comb(dst, src) R_in[src]. R_out may equal R_in.
void hc_write(const float* f, const float* R_in, const float* post, const float* comb, float* R_out, int n_embd,
              void* stream);
/// The head's input: the mean of the 4 streams.
void hc_mean(const float* R, float* y, int n_embd, void* stream);

/// KDA causal conv (kernel `k`, state = the previous k-1 inputs per channel, oldest first) + SiLU, one token;
/// the state is shifted in place.
void kda_conv_silu(const float* x, float* state, const float* w, float* y, int channels, int k, void* stream);
/// KDA decay: g = lower * sigmoid(-(gf + dt_bias) * A[head]) per channel (A = -exp(A_log) per head).
void kda_gate(const float* gf, const float* dt_bias, const float* A, float* g, int n_head, int head_dim,
              float lower, void* stream);
/// sigmoid in place.
void sigmoid_inplace(float* x, int n, void* stream);
/// The gated delta rule, one token, every head: S (head x [j][i], row j = column j of S) decays by exp(g[i]),
/// delta = beta (v - S^T k), S += k delta^T, out = S^T q / sqrt(head_dim).
void kda_step(float* S, const float* q, const float* k, const float* v, const float* g, const float* beta,
              float* out, int n_head, int head_dim, void* stream);
/// out = rmsnorm(o per head) * w * sigmoid(g2), per head of head_dim.
void kda_out_gate(const float* o, const float* w, const float* g2, float* out, int n_head, int head_dim, float eps,
                  void* stream);

/// Nope-MLA attention for one query over `n_vis` cached latents (rows of `lat` floats): per head,
/// softmax(q_abs . c / sqrt(dk)) weighted sum of c. n_vis == 0 writes zeros. `scratch`: n_head * n_vis floats.
void mla_attend(const float* q_abs, const float* cache, int n_vis, int n_head, int lat, float scale, float* out,
                float* scratch, void* stream);
/// The k-pool key of a completed pool: per dim d, softmax over the pool's members j of (gate[j][d] + ape[j][d]),
/// weighted sum of key[j][d]. key/gate: kpool rows of dim; ape: kpool rows of dim.
void kpool_key(const float* key, const float* gate, const float* ape, float* pooled, int kpool, int dim, void* stream);

/// y = W x for an f32 W of n_out rows of n_in.
void gemv_f32(const float* W, const float* x, float* y, int n_in, int n_out, void* stream);
/// The sigmoid router: probs = sigmoid(logits); top-k of probs + bias; weights = probs of those, normalized
/// (sum clamped to 6.1035e-5) and scaled. ids/weights: k each (device).
void router_topk(const float* logits, const float* bias, int n_expert, int k, float scale, int32_t* ids,
                 float* weights, void* stream);
/// out = silu(min(g, limit)) * clamp(u, -limit, limit).
void swiglu_clamp(const float* g, const float* u, float* out, int n, float limit, void* stream);
/// y += a * x, with a read from device memory a_dev[0] times `mul` (the expert weights live on the device).
void axpy_dev(float* y, const float* x, const float* a_dev, int n, void* stream);
/// y += x.
void add_inplace(float* y, const float* x, int n, void* stream);
// ---- many tokens (the prompt path): T rows, row-major, the same maths as the one-token kernels above

/// hc_read for T tokens: mixes [T][24], R [T][4][n], x [T][n], post [T][4], comb [T][16].
void hc_read_rows(const float* mixes, const float* scale, const float* base, const float* R, float* x, float* post,
                  float* comb, int n_embd, float eps, int iters, int T, void* stream);
/// hc_write for T tokens (f [T][n], R [T][4][n] in place allowed).
void hc_write_rows(const float* f, const float* R_in, const float* post, const float* comb, float* R_out, int n_embd,
                   int T, void* stream);
/// The KDA causal conv + SiLU over T consecutive tokens of one sequence (x, y: [T][channels]); state carried.
void kda_conv_silu_seq(const float* x, float* state, const float* w, float* y, int channels, int k, int T,
                       void* stream);
/// kda_gate over T rows of n_head * head_dim.
void kda_gate_rows(const float* gf, const float* dt_bias, const float* A, float* g, int n_head, int head_dim, int T,
                   float lower, void* stream);
/// The gated delta rule over T consecutive tokens (q, k, v, g, out: [T][heads][d], beta: [T][heads]).
void kda_scan(float* S, const float* q, const float* k, const float* v, const float* g, const float* beta, float* out,
              int n_head, int head_dim, int T, void* stream);
/// layer_norm over `rows` rows of n.
void layer_norm_rows(const float* x, const float* w, const float* b, float* y, int rows, int n, float eps,
                     void* stream);
/// Nope-MLA attention for T consecutive queries at positions p0.. over the latent cache: query t sees the complete
/// pools of `kpool` tokens ending at or before p0 + t and the incomplete tail up to itself (llama.cpp's
/// indexer_kpool_select_tail, true by default), so below the indexer's top-k every position 0..p0+t.
/// q_abs, out: [T][n_head][lat]. n_head % 8 == 0.
void mla_attend_rows(const float* q_abs, const float* cache, int64_t p0, int T, int n_head, int lat, int kpool,
                     float scale, float* out, void* stream);
/// The k-pool keys of pools [pool0, pool0 + n) from the key/gate caches ([pos][dim]) into pooled [pool][dim].
void kpool_keys(const float* key_cache, const float* gate_cache, const float* ape, float* pooled, int64_t pool0, int n,
                int kpool, int dim, void* stream);
/// y [T][n_out] = x [T][n_in] W^T for an f32 W of n_out rows of n_in (the router, the indexer weights).
void gemm_f32(const float* W, const float* x, float* y, int n_in, int n_out, int T, void* stream);
/// router_topk for T rows: logits [T][n_expert], ids/weights [T][k].
void router_topk_rows(const float* logits, const float* bias, int n_expert, int k, float scale, int32_t* ids,
                       float* weights, int T, void* stream);
/// parts[dst[r]] = w[wi[r]] * y[r] for r < rows (y, parts: rows of n).
void scatter_scaled(const float* y, const int32_t* dst, const int32_t* wi, const float* w, float* parts, int rows,
                    int n, void* stream);
/// out[t] = sum_{j<k} parts[t*k + j] + extra[t] (in j order), T rows of n.
void sum_parts(const float* parts, const float* extra, float* out, int T, int k, int n, void* stream);

/// d[i] = i for i < n.
void iota(int32_t* d, int n, void* stream);
/// d[0] = a, d[1] = b.
void set_pair(int32_t* d, int32_t a, int32_t b, void* stream);
/// Index of the largest of n floats into *out (device).
void argmax(const float* x, int n, int32_t* out, void* stream);

}  // namespace strata::glm
