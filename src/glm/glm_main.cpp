// src/glm/glm_main.cpp - strata-glm: GLM-5.3-Flash (glm5-next) bring-up engine. Correctness first: one token at a
// time through every layer, every non-expert weight in VRAM, the 8 routed experts of each layer copied to the GPU
// from the mapped GGUF per token (slow, simple, exact). It prints the top logits of each step so the output can be
// compared with llama.cpp's (docs/GLM.md). The fast paths (expert cache, CPU pool, prompt chunks, MTP) come next.
//
//     strata-glm --model <shard 1 of the GGUF> --tokens 1,2,3 [--max-new N] [--ctx N] [--top K] [--dump]
#include "strata/artifact/dequant.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/glm/glm_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <vector>

namespace K = strata::kernels;
namespace G = strata::glm;

namespace {

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

// A tensor of the model: its GGUF type and shape, the mapped host bytes, and (when uploaded) its device copy.
struct Ten {
    int type = -1;
    std::vector<uint64_t> shape;
    const uint8_t* host = nullptr;
    size_t bytes = 0;
    void* d = nullptr;
    explicit operator bool() const { return host != nullptr; }
    const float* f() const { return (const float*) d; }
};

struct Model {
    std::unique_ptr<strata::GgufModel> gguf;
    // hparams
    int n_layer = 0, n_embd = 0, n_head = 0, n_vocab = 0, n_ff = 0, n_ff_exp = 0, n_expert = 0, n_used = 0;
    int dense_lead = 0, head_dim = 0, d_conv = 0, q_lora = 0, kv_lora = 0, dk_mla = 0, dv_mla = 0;
    int ix_heads = 0, ix_dim = 0, ix_topk = 0, kpool = 0, hc_iters = 0;
    float eps_rms = 1e-5f, eps_ln = 1e-6f, hc_eps = 1e-6f, gate_lower = -5.0f, w_scale = 1.0f;
    bool w_norm = true;
    std::vector<int> head_kv;
    std::vector<float> clamp_exp, clamp_sh;
    // weights
    Ten tok_embd, output, output_norm;
    struct Layer {
        bool mla = false, moe = false;
        Ten attn_norm, ffn_norm, hca_fn, hca_base, hca_scale, hcf_fn, hcf_base, hcf_scale;
        Ten q, k, v, cq, ck, cv, fa, fb, beta, A, dtb, ga, gb, onorm, wo;                    // KDA (+ wo shared)
        Ten qa, qa_norm, qb, kva, kva_norm, kb, vb;                                          // MLA
        Ten ix_k, ix_knw, ix_knb, ix_qb, ix_proj, ix_gate, ix_ape;                           // DSA indexer
        Ten fg, fu, fd;                                                                      // dense FFN
        Ten router, probs_b, sg, su, sd, eg, eu, ed;                                         // MoE
    };
    std::vector<Layer> layers;
    size_t vram_weights = 0;

    const strata::MetaValue* meta(const std::string& k) const { return gguf->meta().get(k); }
    double num(const std::string& k, double def = NAN) const {
        const auto* v = meta(k);
        if (!v) {
            if (std::isnan(def)) throw std::runtime_error("GGUF key missing: " + k);
            return def;
        }
        return v->num();
    }
    Ten get(const std::string& name, bool upload, bool required = true) {
        size_t s = 0;
        const strata::TensorInfo* t = gguf->find(name, &s);
        Ten r;
        if (!t) {
            if (required) throw std::runtime_error("tensor missing: " + name);
            return r;
        }
        r.type = (int) t->type;
        r.shape = t->shape;
        r.host = gguf->shard(s).tensor_data(*t);
        r.bytes = strata::tensor_payload_bytes(*t);
        if (upload) {
            ck(cudaMalloc(&r.d, r.bytes), ("cudaMalloc " + name).c_str());
            ck(cudaMemcpy(r.d, r.host, r.bytes, cudaMemcpyHostToDevice), ("upload " + name).c_str());
            vram_weights += r.bytes;
        }
        return r;
    }
    void load(const std::string& path) {
        gguf = std::make_unique<strata::GgufModel>(strata::GgufModel::open(path));
        const std::string a = "glm5-next.";
        const auto* arch = meta("general.architecture");
        if (!arch || arch->s != "glm5-next") throw std::runtime_error("not a glm5-next GGUF");
        const int n_all = (int) num(a + "block_count");
        const int n_nextn = (int) num(a + "nextn_predict_layers", 0);
        n_layer = n_all - n_nextn;
        n_embd = (int) num(a + "embedding_length");
        n_head = (int) num(a + "attention.head_count");
        n_ff = (int) num(a + "feed_forward_length");
        n_ff_exp = (int) num(a + "expert_feed_forward_length");
        n_expert = (int) num(a + "expert_count");
        n_used = (int) num(a + "expert_used_count");
        dense_lead = (int) num(a + "leading_dense_block_count", 0);
        head_dim = (int) num(a + "kda.head_dim");
        d_conv = (int) num(a + "ssm.conv_kernel");
        q_lora = (int) num(a + "attention.q_lora_rank");
        kv_lora = (int) num(a + "attention.kv_lora_rank");
        dk_mla = (int) num(a + "attention.key_length_mla");
        dv_mla = (int) num(a + "attention.value_length_mla");
        ix_heads = (int) num(a + "attention.indexer.head_count");
        ix_dim = (int) num(a + "attention.indexer.key_length");
        ix_topk = (int) num(a + "attention.indexer.top_k");
        kpool = (int) num(a + "attention.indexer.kpool");
        hc_iters = (int) num(a + "hyper_connection.sinkhorn_iterations");
        hc_eps = (float) num(a + "hyper_connection.epsilon");
        eps_rms = (float) num(a + "attention.layer_norm_rms_epsilon");
        eps_ln = (float) num(a + "attention.layer_norm_epsilon", 1e-6);
        gate_lower = (float) num(a + "kda.gate_lower_bound", -INFINITY);
        w_scale = (float) num(a + "expert_weights_scale", 1.0);
        w_norm = num(a + "expert_weights_norm", 0) != 0;
        if ((int) num(a + "hyper_connection.count") != G::kHc) throw std::runtime_error("hc count != 4");
        if (num(a + "rope.dimension_count", 0) != 0) throw std::runtime_error("this engine assumes nope MLA");
        auto arr = [&](const std::string& k, std::vector<float>& out) {
            const auto* v = meta(a + k);
            if (!v) throw std::runtime_error("GGUF key missing: " + a + k);
            for (const auto& it : v->items) out.push_back((float) it.num());
        };
        std::vector<float> hkv;
        arr("attention.head_count_kv", hkv);
        for (float x : hkv) head_kv.push_back((int) x);
        arr("swiglu_clamp_exp", clamp_exp);
        arr("swiglu_clamp_shexp", clamp_sh);
        tok_embd = get("token_embd.weight", false);
        output = get("output.weight", true);
        output_norm = get("output_norm.weight", true);
        n_vocab = (int) output.shape[1];
        layers.resize(n_layer);
        for (int il = 0; il < n_layer; ++il) {
            Layer& L = layers[il];
            const std::string p = "blk." + std::to_string(il) + ".";
            L.mla = head_kv[il] != 0;
            L.moe = il >= dense_lead;
            L.attn_norm = get(p + "attn_norm.weight", true);
            L.ffn_norm = get(p + "ffn_norm.weight", true);
            L.hca_fn = get(p + "hc_attn_fn.weight", true);
            L.hca_base = get(p + "hc_attn_base.weight", true);
            L.hca_scale = get(p + "hc_attn_scale.weight", true);
            L.hcf_fn = get(p + "hc_ffn_fn.weight", true);
            L.hcf_base = get(p + "hc_ffn_base.weight", true);
            L.hcf_scale = get(p + "hc_ffn_scale.weight", true);
            L.wo = get(p + "attn_output.weight", true);
            if (!L.mla) {
                L.q = get(p + "attn_q.weight", true);
                L.k = get(p + "attn_k.weight", true);
                L.v = get(p + "attn_v.weight", true);
                L.cq = get(p + "ssm_conv1d_q.weight", true);
                L.ck = get(p + "ssm_conv1d_k.weight", true);
                L.cv = get(p + "ssm_conv1d_v.weight", true);
                L.fa = get(p + "ssm_f_a.weight", true);
                L.fb = get(p + "ssm_f_b.weight", true);
                L.beta = get(p + "ssm_beta.weight", true);
                L.A = get(p + "ssm_a", true);
                L.dtb = get(p + "ssm_dt.bias", true);
                L.ga = get(p + "ssm_g_a.weight", true);
                L.gb = get(p + "ssm_g_b.weight", true);
                L.onorm = get(p + "ssm_norm.weight", true);
            } else {
                L.qa = get(p + "attn_q_a.weight", true);
                L.qa_norm = get(p + "attn_q_a_norm.weight", true);
                L.qb = get(p + "attn_q_b.weight", true);
                L.kva = get(p + "attn_kv_a_mqa.weight", true);
                L.kva_norm = get(p + "attn_kv_a_norm.weight", true);
                L.kb = get(p + "attn_k_b.weight", true);
                L.vb = get(p + "attn_v_b.weight", true);
                L.ix_k = get(p + "indexer.attn_k.weight", true);
                L.ix_knw = get(p + "indexer.k_norm.weight", true);
                L.ix_knb = get(p + "indexer.k_norm.bias", true);
                L.ix_qb = get(p + "indexer.attn_q_b.weight", true);
                L.ix_proj = get(p + "indexer.proj.weight", true);
                L.ix_gate = get(p + "indexer_compressor_gate.weight", true);
                L.ix_ape = get(p + "indexer_compressor_ape.weight", true);
            }
            if (!L.moe) {
                L.fg = get(p + "ffn_gate.weight", true);
                L.fu = get(p + "ffn_up.weight", true);
                L.fd = get(p + "ffn_down.weight", true);
            } else {
                L.router = get(p + "ffn_gate_inp.weight", true);
                L.probs_b = get(p + "exp_probs_b.bias", true);
                L.sg = get(p + "ffn_gate_shexp.weight", true);
                L.su = get(p + "ffn_up_shexp.weight", true);
                L.sd = get(p + "ffn_down_shexp.weight", true);
                L.eg = get(p + "ffn_gate_exps.weight", false);
                L.eu = get(p + "ffn_up_exps.weight", false);
                L.ed = get(p + "ffn_down_exps.weight", false);
            }
            for (const Ten* t : {&L.attn_norm, &L.hca_base, &L.hca_scale, &L.A, &L.dtb, &L.onorm, &L.cq, &L.qa_norm,
                                 &L.kva_norm, &L.ix_knw, &L.ix_knb, &L.ix_proj, &L.ix_ape, &L.router, &L.probs_b})
                if (*t && t->type != 0) throw std::runtime_error("expected an f32 tensor in layer " + std::to_string(il));
        }
    }
};

float* dalloc(size_t n) {
    float* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(float)), "cudaMalloc buffer");
    ck(cudaMemset(p, 0, n * sizeof(float)), "cudaMemset");
    return p;
}

struct Engine {
    Model& M;
    cudaStream_t st{};
    int ctx = 0;
    int64_t pos = 0;
    // activations
    float *R, *x, *h, *post, *comb, *mixes, *tmp4k, *out, *acc;
    void* xq = nullptr;
    // KDA
    float *qp, *kp, *vp, *qc, *kc, *vc, *fa, *gf, *g, *beta, *o, *ga, *g2, *gated;
    std::vector<float*> S, conv;   // per KDA layer: state (heads x d x d), conv (3 x (k-1) x channels)
    // MLA
    float *qa, *qr, *q, *lat, *qabs, *olat, *vh, *scores, *ik, *ig, *iqv, *iw, *pooled;
    std::vector<float*> lat_cache, ik_cache, ig_cache, pool_cache;
    // FFN
    float *fg, *fu, *fa2, *fy, *logits_r, *ew;
    int32_t* eids = nullptr;
    uint8_t *stage_g = nullptr, *stage_u = nullptr, *stage_d = nullptr;
    float* logits = nullptr;
    int32_t* d_arg = nullptr;
    bool dump = false;

    explicit Engine(Model& m, int ctx_) : M(m), ctx(ctx_) {
        ck(cudaStreamCreate(&st), "stream");
        const int E = M.n_embd, HD = M.n_head * M.head_dim;
        R = dalloc((size_t) G::kHc * E); x = dalloc(E); h = dalloc(E); post = dalloc(G::kHc); comb = dalloc(16);
        mixes = dalloc(G::kHcMix); tmp4k = dalloc(E); out = dalloc(E); acc = dalloc(E);
        ck(cudaMalloc(&xq, K::native_q8_1_bytes(std::max({G::kHc * E, M.n_ff, HD, M.n_head * M.dv_mla,
                                                          M.n_head * M.kv_lora}), 1)), "xq");
        qp = dalloc(HD); kp = dalloc(HD); vp = dalloc(HD); qc = dalloc(HD); kc = dalloc(HD); vc = dalloc(HD);
        fa = dalloc(M.head_dim); gf = dalloc(HD); g = dalloc(HD); beta = dalloc(M.n_head); o = dalloc(HD);
        ga = dalloc(M.head_dim); g2 = dalloc(HD); gated = dalloc(HD);
        qa = dalloc(M.q_lora); qr = dalloc(M.q_lora); q = dalloc((size_t) M.n_head * M.dk_mla); lat = dalloc(M.kv_lora);
        qabs = dalloc((size_t) M.n_head * M.kv_lora); olat = dalloc((size_t) M.n_head * M.kv_lora);
        vh = dalloc((size_t) M.n_head * M.dv_mla); scores = dalloc((size_t) M.n_head * ctx);
        ik = dalloc(M.ix_dim); ig = dalloc(M.ix_dim); iqv = dalloc((size_t) M.ix_heads * M.ix_dim); iw = dalloc(M.ix_heads);
        pooled = dalloc(M.ix_dim);
        fg = dalloc(M.n_ff); fu = dalloc(M.n_ff); fa2 = dalloc(M.n_ff); fy = dalloc(E);
        logits_r = dalloc(M.n_expert); ew = dalloc(M.n_used);
        ck(cudaMalloc(&eids, M.n_used * sizeof(int32_t)), "eids");
        logits = dalloc(M.n_vocab);
        ck(cudaMalloc(&d_arg, sizeof(int32_t)), "argmax");
        size_t max_g = 0, max_d = 0;
        for (auto& L : M.layers) {
            if (!L.mla) {
                S.push_back(dalloc((size_t) M.n_head * M.head_dim * M.head_dim));
                conv.push_back(dalloc((size_t) 3 * (M.d_conv - 1) * HD));
                lat_cache.push_back(nullptr); ik_cache.push_back(nullptr); ig_cache.push_back(nullptr);
                pool_cache.push_back(nullptr);
            } else {
                S.push_back(nullptr); conv.push_back(nullptr);
                lat_cache.push_back(dalloc((size_t) ctx * M.kv_lora));
                ik_cache.push_back(dalloc((size_t) ctx * M.ix_dim));
                ig_cache.push_back(dalloc((size_t) ctx * M.ix_dim));
                pool_cache.push_back(dalloc((size_t) (ctx / M.kpool + 1) * M.ix_dim));
            }
            if (L.moe) {
                max_g = std::max(max_g, L.eg.bytes / M.n_expert);
                max_d = std::max(max_d, L.ed.bytes / M.n_expert);
            }
        }
        ck(cudaMalloc(&stage_g, max_g), "stage");
        ck(cudaMalloc(&stage_u, max_g), "stage");
        ck(cudaMalloc(&stage_d, max_d), "stage");
    }

    // y = W x, with x quantized to q8_1 first (llama.cpp's MMVQ path for one token).
    void mm(const Ten& w, const void* wd, const float* xin, float* y, int n_in, int n_out) {
        K::native_quantize_q8_1(xin, xq, n_in, 1, st);
        K::native_mmvq(w.type, wd, xq, y, n_in, n_out, 1, st);
    }
    void mm(const Ten& w, const float* xin, float* y) { mm(w, w.d, xin, y, (int) w.shape[0], (int) w.shape[1]); }

    // mHC read: mixes from the RMS-normed flat streams, then pre/post/comb and the layer input into `x`
    void hc_read(const Ten& fn, const Ten& base, const Ten& scale) {
        const int E = M.n_embd;
        G::rms_norm_rows(R, nullptr, flat_norm(), 1, G::kHc * E, M.eps_rms, st);
        mm(fn, flat_norm(), mixes);
        G::hc_read(mixes, scale.f(), base.f(), R, x, post, comb, E, M.hc_eps, M.hc_iters, st);
    }
    float* flat = nullptr;
    float* flat_norm() {
        if (!flat) flat = dalloc((size_t) G::kHc * M.n_embd);
        return flat;
    }

    void swiglu_ffn(const Ten& wg, const void* dg, const Ten& wu, const void* du, const Ten& wd, const void* dd,
                    const float* in, float* y, int n_ff, float limit) {
        const int E = M.n_embd;
        K::native_quantize_q8_1(in, xq, E, 1, st);
        K::native_mmvq(wg.type, dg, xq, fg, E, n_ff, 1, st);
        K::native_mmvq(wu.type, du, xq, fu, E, n_ff, 1, st);
        G::swiglu_clamp(fg, fu, fa2, n_ff, limit, st);
        K::native_quantize_q8_1(fa2, xq, n_ff, 1, st);
        K::native_mmvq(wd.type, dd, xq, y, n_ff, E, 1, st);
    }

    void kda(int il) {
        const auto& L = M.layers[il];
        const int HD = M.n_head * M.head_dim, kc1 = M.d_conv - 1;
        float* cs = conv[il];
        mm(L.q, h, qp);
        mm(L.k, h, kp);
        mm(L.v, h, vp);
        G::kda_conv_silu(qp, cs, L.cq.f(), qc, HD, M.d_conv, st);
        G::kda_conv_silu(kp, cs + (size_t) kc1 * HD, L.ck.f(), kc, HD, M.d_conv, st);
        G::kda_conv_silu(vp, cs + (size_t) 2 * kc1 * HD, L.cv.f(), vc, HD, M.d_conv, st);
        mm(L.fa, h, fa);
        mm(L.fb, fa, gf);
        G::kda_gate(gf, L.dtb.f(), L.A.f(), g, M.n_head, M.head_dim, M.gate_lower, st);
        mm(L.beta, h, beta);
        G::sigmoid_inplace(beta, M.n_head, st);
        G::l2_norm_rows(qc, M.n_head, M.head_dim, 1e-6f, st);
        G::l2_norm_rows(kc, M.n_head, M.head_dim, 1e-6f, st);
        G::kda_step(S[il], qc, kc, vc, g, beta, o, M.n_head, M.head_dim, st);
        mm(L.ga, h, ga);
        mm(L.gb, ga, g2);
        G::kda_out_gate(o, L.onorm.f(), g2, gated, M.n_head, M.head_dim, M.eps_rms, st);
        mm(L.wo, gated, out);
    }

    void mla(int il) {
        const auto& L = M.layers[il];
        const int H = M.n_head;
        mm(L.qa, h, qa);
        G::rms_norm_rows(qa, L.qa_norm.f(), qr, 1, M.q_lora, M.eps_rms, st);
        mm(L.qb, qr, q);
        mm(L.kva, h, lat);
        float* lc = lat_cache[il];
        G::rms_norm_rows(lat, L.kva_norm.f(), lc + (size_t) pos * M.kv_lora, 1, M.kv_lora, M.eps_rms, st);
        // absorbed query: per head, q_abs = W_kb[h] q[h] (kv_lora outputs from dk inputs)
        K::native_quantize_q8_1(q, xq, M.dk_mla, H, st);
        const size_t qrow = K::native_q8_1_bytes(M.dk_mla, 1);
        const size_t kb_head = L.kb.bytes / H;
        for (int hh = 0; hh < H; ++hh)
            K::native_mmvq(L.kb.type, (const uint8_t*) L.kb.d + hh * kb_head, (const uint8_t*) xq + hh * qrow,
                           qabs + (size_t) hh * M.kv_lora, M.dk_mla, M.kv_lora, 1, st);
        // indexer keys for this token (cached; pooled when a pool of kpool completes)
        mm(L.ix_k, h, ik);
        G::layer_norm(ik, L.ix_knw.f(), L.ix_knb.f(), ik_cache[il] + (size_t) pos * M.ix_dim, M.ix_dim, M.eps_ln, st);
        mm(L.ix_gate, h, ig_cache[il] + (size_t) pos * M.ix_dim);
        if ((pos + 1) % M.kpool == 0) {
            const int64_t p0 = pos + 1 - M.kpool;
            G::kpool_key(ik_cache[il] + (size_t) p0 * M.ix_dim, ig_cache[il] + (size_t) p0 * M.ix_dim, L.ix_ape.f(),
                         pool_cache[il] + (size_t) (p0 / M.kpool) * M.ix_dim, M.kpool, M.ix_dim, st);
        }
        // visible: the complete pools ending at or before this token (no tail selection)
        const int64_t n_pool_vis = (pos + 1) / M.kpool;
        if (n_pool_vis > M.ix_topk / M.kpool)
            throw std::runtime_error("contexts past " + std::to_string(M.ix_topk) +
                                     " tokens need the indexer's top-k selection (not implemented yet)");
        const int n_vis = (int) (n_pool_vis * M.kpool);
        G::mla_attend(qabs, lc, n_vis, H, M.kv_lora, 1.0f / std::sqrt((float) M.dk_mla), olat, scores, st);
        // per-head value: W_vb[h] (dv outputs from kv_lora inputs)
        K::native_quantize_q8_1(olat, xq, M.kv_lora, H, st);
        const size_t orow = K::native_q8_1_bytes(M.kv_lora, 1);
        const size_t vb_head = L.vb.bytes / H;
        for (int hh = 0; hh < H; ++hh)
            K::native_mmvq(L.vb.type, (const uint8_t*) L.vb.d + hh * vb_head, (const uint8_t*) xq + hh * orow,
                           vh + (size_t) hh * M.dv_mla, M.kv_lora, M.dv_mla, 1, st);
        mm(L.wo, vh, out);
    }

    void moe(int il) {
        const auto& L = M.layers[il];
        const int E = M.n_embd;
        G::gemv_f32(L.router.f(), h, logits_r, E, M.n_expert, st);
        if (!M.w_norm) throw std::runtime_error("expert_weights_norm = false is not implemented");
        G::router_topk(logits_r, L.probs_b.f(), M.n_expert, M.n_used, M.w_scale, eids, ew, st);
        int32_t ids[64];
        ck(cudaMemcpyAsync(ids, eids, M.n_used * sizeof(int32_t), cudaMemcpyDeviceToHost, st), "ids");
        ck(cudaStreamSynchronize(st), "sync ids");
        const size_t gb = L.eg.bytes / M.n_expert, db = L.ed.bytes / M.n_expert;
        ck(cudaMemsetAsync(acc, 0, E * sizeof(float), st), "acc");
        for (int j = 0; j < M.n_used; ++j) {
            const int e = ids[j];
            ck(cudaMemcpyAsync(stage_g, L.eg.host + (size_t) e * gb, gb, cudaMemcpyHostToDevice, st), "expert g");
            ck(cudaMemcpyAsync(stage_u, L.eu.host + (size_t) e * gb, gb, cudaMemcpyHostToDevice, st), "expert u");
            ck(cudaMemcpyAsync(stage_d, L.ed.host + (size_t) e * db, db, cudaMemcpyHostToDevice, st), "expert d");
            swiglu_ffn(L.eg, stage_g, L.eu, stage_u, L.ed, stage_d, h, fy, M.n_ff_exp, M.clamp_exp[il]);
            G::axpy_dev(acc, fy, ew + j, E, st);
        }
        swiglu_ffn(L.sg, L.sg.d, L.su, L.su.d, L.sd, L.sd.d, h, out, M.n_ff_exp, M.clamp_sh[il]);
        // llama.cpp: moe_out + shexp
        G::add_inplace(acc, out, E, st);
        ck(cudaMemcpyAsync(out, acc, E * sizeof(float), cudaMemcpyDeviceToDevice, st), "moe out");
    }

    // One token through the model; with `want_logits` the head too.
    void forward(int32_t token, bool want_logits) {
        if (pos >= ctx) throw std::runtime_error("context full");
        const int E = M.n_embd;
        // embedding (Q4_K row, dequantized on the host), copied into all four streams
        std::vector<float> emb(E);
        const size_t row = M.tok_embd.bytes / M.n_vocab;
        const uint8_t* r = M.tok_embd.host + (size_t) token * row;
        if (M.tok_embd.type != 12) throw std::runtime_error("token_embd: expected Q4_K");
        for (int b = 0; b < E / 256; ++b) strata::dequantize_q4_K(r + (size_t) b * 144, emb.data() + b * 256);
        for (int s = 0; s < G::kHc; ++s)
            ck(cudaMemcpyAsync(R + (size_t) s * E, emb.data(), E * sizeof(float), cudaMemcpyHostToDevice, st), "emb");
        for (int il = 0; il < M.n_layer; ++il) {
            const auto& L = M.layers[il];
            hc_read(L.hca_fn, L.hca_base, L.hca_scale);
            G::rms_norm_rows(x, L.attn_norm.f(), h, 1, E, M.eps_rms, st);
            if (L.mla) mla(il); else kda(il);
            G::hc_write(out, R, post, comb, R, E, st);
            hc_read(L.hcf_fn, L.hcf_base, L.hcf_scale);
            G::rms_norm_rows(x, L.ffn_norm.f(), h, 1, E, M.eps_rms, st);
            if (L.moe) moe(il);
            else swiglu_ffn(L.fg, L.fg.d, L.fu, L.fu.d, L.fd, L.fd.d, h, out, M.n_ff, M.clamp_sh[il]);
            G::hc_write(out, R, post, comb, R, E, st);
            if (dump) {
                std::vector<float> hr((size_t) G::kHc * E);
                ck(cudaMemcpyAsync(hr.data(), R, hr.size() * sizeof(float), cudaMemcpyDeviceToHost, st), "dump");
                ck(cudaStreamSynchronize(st), "dump sync");
                double ss = 0, sum = 0;
                for (float v : hr) { ss += (double) v * v; sum += v; }
                std::printf("  pos %lld layer %2d %s %s  |R| %.4f  sum %.4f  R0[0..3] %.5f %.5f %.5f %.5f\n",
                            (long long) pos, il, L.mla ? "MLA" : "KDA", L.moe ? "moe  " : "dense", std::sqrt(ss), sum,
                            hr[0], hr[1], hr[2], hr[3]);
            }
        }
        if (want_logits) {
            G::hc_mean(R, x, E, st);
            G::rms_norm_rows(x, M.output_norm.f(), h, 1, E, M.eps_rms, st);
            mm(M.output, h, logits);
        }
        ++pos;
    }
};

std::vector<int32_t> parse_ids(const std::string& s) {
    std::vector<int32_t> v;
    size_t i = 0;
    while (i < s.size()) {
        size_t j = s.find_first_of(", ", i);
        if (j == std::string::npos) j = s.size();
        if (j > i) v.push_back((int32_t) std::stol(s.substr(i, j - i)));
        i = j + 1;
    }
    return v;
}

}  // namespace

int main(int argc, char** argv) {
    std::setvbuf(stdout, nullptr, _IONBF, 0);
    std::string model, tokens;
    int max_new = 8, ctx = 4096, top = 5;
    bool dump = false;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto next = [&]() -> std::string {
            if (i + 1 >= argc) throw std::runtime_error(a + " needs a value");
            return argv[++i];
        };
        if (a == "--model") model = next();
        else if (a == "--tokens") tokens = next();
        else if (a == "--max-new") max_new = std::stoi(next());
        else if (a == "--ctx") ctx = std::stoi(next());
        else if (a == "--top") top = std::stoi(next());
        else if (a == "--dump") dump = true;
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); return 2; }
    }
    if (model.empty() || tokens.empty()) {
        std::fprintf(stderr, "usage: strata-glm --model <GGUF shard 1> --tokens 1,2,3 [--max-new N] [--ctx N] [--top K] [--dump]\n");
        return 2;
    }
    try {
        using clk = std::chrono::steady_clock;
        const auto t0 = clk::now();
        Model M;
        M.load(model);
        std::printf("strata-glm: %d layers (%d MLA), embd %d, %d experts top-%d, vocab %d; %.2f GiB of weights in VRAM, "
                    "loaded in %.1f s\n", M.n_layer,
                    (int) std::count_if(M.layers.begin(), M.layers.end(), [](const Model::Layer& L) { return L.mla; }),
                    M.n_embd, M.n_expert, M.n_used, M.n_vocab, M.vram_weights / 1073741824.0,
                    std::chrono::duration<double>(clk::now() - t0).count());
        Engine E(M, ctx);
        E.dump = dump;
        const std::vector<int32_t> prompt = parse_ids(tokens);
        std::vector<float> lg(M.n_vocab);
        auto report = [&](int step) {
            ck(cudaMemcpy(lg.data(), E.logits, lg.size() * sizeof(float), cudaMemcpyDeviceToHost), "logits");
            std::vector<int> idx(lg.size());
            for (size_t i = 0; i < idx.size(); ++i) idx[i] = (int) i;
            std::partial_sort(idx.begin(), idx.begin() + top, idx.end(), [&](int a, int b) { return lg[a] > lg[b]; });
            double mx = lg[idx[0]], z = 0;
            for (float v : lg) z += std::exp((double) v - mx);
            std::printf("step %d top:", step);
            for (int t = 0; t < top; ++t)
                std::printf(" %d (%.3f, logprob %.3f)", idx[t], lg[idx[t]], lg[idx[t]] - mx - std::log(z));
            std::printf("\n");
            return idx[0];
        };
        const auto tp = clk::now();
        for (size_t i = 0; i < prompt.size(); ++i) E.forward(prompt[i], i + 1 == prompt.size());
        ck(cudaStreamSynchronize(E.st), "prompt");
        const double prompt_s = std::chrono::duration<double>(clk::now() - tp).count();
        std::vector<int32_t> outv;
        int nxt = report(0);
        const auto td = clk::now();
        for (int s = 1; s <= max_new; ++s) {
            outv.push_back(nxt);
            if (s == max_new) break;
            E.forward(nxt, true);
            nxt = report(s);
        }
        const double dec_s = std::chrono::duration<double>(clk::now() - td).count();
        std::printf("output:");
        for (int32_t t : outv) std::printf(" %d", t);
        std::printf("\nprompt %zu tokens in %.2f s (%.2f tok/s); decode %d tokens in %.2f s (%.2f tok/s)\n", prompt.size(),
                    prompt_s, prompt.size() / prompt_s, max_new - 1, dec_s, (max_new - 1) / std::max(dec_s, 1e-9));
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-glm: %s\n", e.what());
        return 1;
    }
    return 0;
}
