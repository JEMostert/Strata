// src/glm/glm_main.cpp - strata-glm: GLM-5.3-Flash (glm5-next) bring-up engine. Correctness first: one token at a
// time through every layer, every non-expert weight in VRAM, the 8 routed experts of each layer copied to the GPU
// from the mapped GGUF per token (slow, simple, exact). It prints the top logits of each step so the output can be
// compared with llama.cpp's (docs/GLM.md). The fast paths (expert cache, CPU pool, prompt chunks, MTP) come next.
//
//     strata-glm --model <shard 1 of the GGUF> --tokens 1,2,3 [--max-new N] [--ctx N] [--top K] [--dump]
#include "strata/artifact/dequant.hpp"
#include "strata/artifact/gguf_reader.hpp"
#include "strata/glm/glm_kernels.hpp"
#include "strata/kernels/cpu/glu.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/prefill/moe_mmq.hpp"

#include <cuda_runtime.h>
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <array>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace K = strata::kernels;
namespace G = strata::glm;
namespace Q = strata::prefill::mmq;
namespace C = strata::kernels::cpu;

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

// y (T rows, ld_y floats apart) = W x for T rows of x (ld_x floats apart): llama.cpp's MMQ (int8 tensor cores) when it
// covers the type and shape, else the one-token MMVQ kernel per group of up to 8 columns.
struct Gemm {
    Q::Context ctx;
    int32_t* ids = nullptr;
    int32_t* bounds = nullptr;
    void* xq = nullptr;
    int64_t rows_cap = 0, cols_cap = 0;
    float* pack = nullptr;   // contiguous copy for the fallback when ld_x != n_in

    Gemm(int64_t rows, int64_t cols, cudaStream_t st) : rows_cap(rows), cols_cap(cols) {
        ck(cudaMalloc(&ids, rows * sizeof(int32_t)), "gemm ids");
        ck(cudaMalloc(&bounds, 2 * sizeof(int32_t)), "gemm bounds");
        ck(cudaMalloc(&xq, std::max(Q::q8_bytes(rows, cols), K::native_q8_1_bytes((int) cols, 8))), "gemm xq");
        ck(cudaMalloc(&pack, (size_t) 8 * cols * sizeof(float)), "gemm pack");
        G::iota(ids, (int) rows, st);
    }
    static bool mmq_ok(int type, int64_t n_out) { return Q::built() && Q::supported(type) && Q::fits(type, n_out); }
    void run(int type, const void* w, int64_t n_in, int64_t n_out, size_t wbytes, const float* x, int64_t ld_x,
             int64_t T, float* y, int64_t ld_y, cudaStream_t st, bool force_mmvq = false) {
        if (T > rows_cap || n_in > cols_cap) throw std::runtime_error("gemm: buffers too small");
        if (!force_mmvq && T > 1 && mmq_ok(type, n_out)) {
            G::set_pair(bounds, 0, (int32_t) T, st);
            Q::quantize(x, nullptr, xq, type, n_in, ld_x, T, st);
            Q::Product p;
            p.w = w; p.type = type; p.w_rows = n_out; p.w_cols = n_in; p.expert_bytes = wbytes; p.n = 1;
            p.xq = xq; p.bounds = bounds; p.ids = ids; p.total_rows = T; p.max_rows = T; p.dst = y; p.ld_dst = ld_y;
            ctx.run(p, st);
            return;
        }
        for (int64_t c0 = 0; c0 < T; c0 += 8) {
            const int nc = (int) std::min<int64_t>(8, T - c0);
            const float* xs = x + c0 * ld_x;
            if (ld_x != n_in) {
                ck(cudaMemcpy2DAsync(pack, n_in * sizeof(float), xs, ld_x * sizeof(float), n_in * sizeof(float), nc,
                                     cudaMemcpyDeviceToDevice, st), "gemm pack");
                xs = pack;
            }
            K::native_quantize_q8_1(xs, xq, (int) n_in, nc, st);
            if (ld_y == n_out) {
                K::native_mmvq(type, w, xq, y + c0 * ld_y, (int) n_in, (int) n_out, nc, st);
            } else {
                for (int c = 0; c < nc; ++c)
                    K::native_mmvq(type, w, (const uint8_t*) xq + c * K::native_q8_1_bytes((int) n_in, 1),
                                   y + (c0 + c) * ld_y, (int) n_in, (int) n_out, 1, st);
            }
        }
    }
};

// ---- the GLM expert pack: <dir>/experts.bin, every MoE layer's experts as contiguous [gate | up | down] blobs (the
// GGUF bytes of the three slices, back to back: the layout of NativeExpertLayout / NativeFmt), layer by layer, each
// layer 4 KiB aligned; <dir>/glm-pack.txt indexes it ("glm-pack 1 <gguf bytes> <file bytes>", then one line per
// layer: "<layer> <offset> <blob bytes> <gate/up type> <down type>").
struct PackLayer { int64_t off = -1; size_t blob = 0; int gu = -1, d = -1; };

uint64_t gguf_bytes(const strata::GgufModel& g) {
    uint64_t n = 0;
    for (size_t i = 0; i < g.size(); ++i) n += g.shard(i).file_size();
    return n;
}

int write_pack(Model& M, const std::string& dir) {
    ::mkdir(dir.c_str(), 0755);
    const std::string bin = dir + "/experts.bin", idx = dir + "/glm-pack.txt";
    const int fd = ::open(bin.c_str(), O_WRONLY | O_CREAT | O_TRUNC, 0644);
    if (fd < 0) throw std::runtime_error("cannot create " + bin);
    std::vector<PackLayer> lay(M.n_layer);
    int64_t off = 0;
    std::vector<uint8_t> buf;
    const auto t0 = std::chrono::steady_clock::now();
    for (int il = 0; il < M.n_layer; ++il) {
        const auto& L = M.layers[il];
        if (!L.moe) continue;
        if (L.eg.type != L.eu.type) throw std::runtime_error("pack: gate and up types differ");
        const size_t gb = L.eg.bytes / M.n_expert, db = L.ed.bytes / M.n_expert, blob = 2 * gb + db;
        off = (off + 4095) & ~int64_t(4095);
        lay[il] = {off, blob, L.eg.type, L.ed.type};
        const int per = 32;
        buf.resize((size_t) per * blob);
        for (int e0 = 0; e0 < M.n_expert; e0 += per) {
            const int n = std::min(per, M.n_expert - e0);
            for (int j = 0; j < n; ++j) {
                uint8_t* b = buf.data() + (size_t) j * blob;
                std::memcpy(b, L.eg.host + (size_t) (e0 + j) * gb, gb);
                std::memcpy(b + gb, L.eu.host + (size_t) (e0 + j) * gb, gb);
                std::memcpy(b + 2 * gb, L.ed.host + (size_t) (e0 + j) * db, db);
            }
            const size_t want = (size_t) n * blob;
            if (::pwrite(fd, buf.data(), want, off + (int64_t) e0 * (int64_t) blob) != (ssize_t) want)
                throw std::runtime_error("pack: write failed");
        }
        off += (int64_t) M.n_expert * (int64_t) blob;
        ::fdatasync(fd);
        ::posix_fadvise(fd, 0, 0, POSIX_FADV_DONTNEED);   // keep the page cache for the GGUF reads
        for (const Ten* t : {&L.eg, &L.eu, &L.ed})        // and drop the source pages already copied
            ::madvise((void*) ((uintptr_t) t->host & ~uintptr_t(4095)), t->bytes, MADV_DONTNEED);
        std::printf("  layer %d: %d experts of %.2f MB (%s/%s), %.1f GB written, %.0f s\n", il, M.n_expert, blob / 1e6,
                    strata::ggml_type_name(L.eg.type), strata::ggml_type_name(L.ed.type), off / 1e9,
                    std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count());
    }
    ::close(fd);
    std::FILE* f = std::fopen(idx.c_str(), "w");
    if (!f) throw std::runtime_error("cannot write " + idx);
    std::fprintf(f, "glm-pack 1 %llu %llu\n", (unsigned long long) gguf_bytes(*M.gguf), (unsigned long long) off);
    for (int il = 0; il < M.n_layer; ++il)
        if (lay[il].off >= 0)
            std::fprintf(f, "%d %lld %zu %d %d\n", il, (long long) lay[il].off, lay[il].blob, lay[il].gu, lay[il].d);
    std::fclose(f);
    std::printf("pack: %s, %.2f GB\n", bin.c_str(), off / 1e9);
    return 0;
}

// The pack at run time: experts.bin mapped; with a RAM budget the first experts (in the order given) are copied
// into pinned memory, which the copy engine reads at the link's full speed; the rest stay in the mapping.
struct Pack {
    std::vector<PackLayer> lay;
    const uint8_t* map = nullptr;
    size_t map_bytes = 0;
    uint8_t* pinned = nullptr;
    size_t pinned_bytes = 0;
    std::vector<const uint8_t*> where;   // [layer * n_expert + e]: the blob's address (pinned or mapped)
    std::vector<uint8_t> is_pinned;
    int n_expert = 0;

    void open(const Model& M, const std::string& dir) {
        std::FILE* f = std::fopen((dir + "/glm-pack.txt").c_str(), "r");
        if (!f) throw std::runtime_error("no GLM pack in " + dir + " (strata-glm --pack-out " + dir + " writes it)");
        unsigned long long gb = 0, fb = 0;
        int ver = 0;
        if (std::fscanf(f, "glm-pack %d %llu %llu", &ver, &gb, &fb) != 3 || ver != 1) throw std::runtime_error("bad glm-pack.txt");
        if (gb != gguf_bytes(*M.gguf)) throw std::runtime_error("the GLM pack was written from another GGUF: write it again");
        lay.assign(M.n_layer, PackLayer{});
        int il;
        long long off;
        size_t blob;
        int gu, d;
        while (std::fscanf(f, "%d %lld %zu %d %d", &il, &off, &blob, &gu, &d) == 5)
            if (il >= 0 && il < M.n_layer) lay[il] = {off, blob, gu, d};
        std::fclose(f);
        const int fd = ::open((dir + "/experts.bin").c_str(), O_RDONLY);
        if (fd < 0) throw std::runtime_error("cannot open experts.bin");
        struct stat sb;
        ::fstat(fd, &sb);
        if ((unsigned long long) sb.st_size != fb) throw std::runtime_error("experts.bin has the wrong size");
        map_bytes = (size_t) sb.st_size;
        map = (const uint8_t*) ::mmap(nullptr, map_bytes, PROT_READ, MAP_SHARED, fd, 0);
        ::close(fd);
        if (map == MAP_FAILED) throw std::runtime_error("cannot map experts.bin");
        n_expert = M.n_expert;
        where.assign((size_t) M.n_layer * n_expert, nullptr);
        is_pinned.assign(where.size(), 0);
        for (int l = 0; l < M.n_layer; ++l)
            if (lay[l].off >= 0)
                for (int e = 0; e < n_expert; ++e) where[(size_t) l * n_expert + e] = map + lay[l].off + (size_t) e * lay[l].blob;
    }
    // Copy experts into pinned RAM in `order` (layer * n_expert + e) until `budget` bytes are used.
    void pin(const std::vector<int>& order, size_t budget) {
        size_t need = 0;
        for (int k : order) {
            const size_t b = lay[k / n_expert].blob;
            if (need + b > budget) break;
            need += b;
        }
        if (need == 0) return;
        if (cudaHostAlloc((void**) &pinned, need, cudaHostAllocPortable) != cudaSuccess) {
            cudaGetLastError();
            throw std::runtime_error("cannot pin " + std::to_string(need >> 20) + " MiB of RAM for experts");
        }
        pinned_bytes = need;
        size_t at = 0;
        for (int k : order) {
            const size_t b = lay[k / n_expert].blob;
            if (at + b > need) break;
            std::memcpy(pinned + at, where[k], b);
            where[k] = pinned + at;
            is_pinned[k] = 1;
            at += b;
        }
        ::madvise((void*) map, map_bytes, MADV_DONTNEED);   // the copied pages are not needed from the file
    }
    const uint8_t* blob(int l, int e) const { return where[(size_t) l * n_expert + e]; }
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

    // ---- the prompt path: T tokens per chunk, every layer for the whole chunk (allocated on first use)
    int Tmax = 0;
    std::unique_ptr<Gemm> gemm;
    float *bR = nullptr, *bflat, *bmix, *bx, *bh, *bpost, *bcomb, *bout;
    float *bqp, *bkp, *bvp, *bfa, *bgf, *bg, *bbeta, *bo, *bga, *bg2, *bgated;
    float *bqa, *bqr, *bq, *blat, *bqabs, *bolat, *bvh, *bik;
    float *blog, *bw, *bsg, *bsu, *bshared, *bparts, *bge, *bue, *bae, *bye, *bfg, *bfu;
    int32_t *bids = nullptr, *rtok = nullptr, *rdst = nullptr, *rwi = nullptr, *e_ids = nullptr, *e_bounds = nullptr;
    void *exq = nullptr, *exq2 = nullptr;
    double ms_copy = 0, ms_layers = 0;
    size_t bytes_streamed = 0;

    // ---- the pack and the CPU's share of the experts
    Pack* pk = nullptr;
    std::unique_ptr<C::ExpertPool> pool;
    std::vector<C::NativeFmt> fmt;           // per MoE layer
    float* h_host = nullptr;                 // pinned: the layer inputs, T rows
    float* y_host = nullptr;                 // pinned: the CPU experts' outputs, T*k rows
    int have_t = 0, have_k = 0;
    std::vector<uint8_t> nact;               // per-token quantized activations for the CPU
    uint8_t* stage_blob = nullptr;           // one expert blob on the device (the pack layout)
    double cpu_gbps = 16.0;                  // the CPU's measured rate, blob bytes per pass per second (EMA)
    double cpu_bias = 1.0;                   // feedback on the split: >1 when the CPU side took longer than the model said
    double gpu_gbps_pinned = 7.0, gpu_gbps_mapped = 4.5;
    bool cpu_share = true;
    int64_t n_cpu_experts = 0, n_gpu_experts = 0;
    double ms_cpu = 0;

    void use_pack(Pack* p, int workers) {
        pk = p;
        fmt.assign(M.n_layer, C::NativeFmt{});
        size_t max_blob = 0;
        for (int il = 0; il < M.n_layer; ++il) {
            const auto& pl = pk->lay[il];
            if (pl.off < 0) continue;
            std::string err;
            if (!C::native_fmt(pl.gu, pl.d, M.n_embd, M.n_ff_exp, fmt[il], err)) throw std::runtime_error(err);
            if (fmt[il].bytes != pl.blob) throw std::runtime_error("pack blob size does not match the layer format");
            max_blob = std::max(max_blob, pl.blob);
        }
        float lim = M.clamp_exp.empty() ? 0.0f : M.clamp_exp[0];
        for (float v : M.clamp_exp) if (v != lim) throw std::runtime_error("per-layer SwiGLU limits are not supported by the CPU path");
        C::g_glu_limit = lim;
        pool = std::make_unique<C::ExpertPool>(workers);
        ck(cudaMalloc(&stage_blob, max_blob), "stage blob");
        std::printf("strata-glm: CPU expert pool, %d workers + the host thread; %.1f GiB of experts pinned in RAM\n",
                    pool->workers(), pk->pinned_bytes / 1073741824.0);
    }
    void ensure_host(int rows_t, int rows_k) {
        if (rows_t > have_t) {
            if (h_host) cudaFreeHost(h_host);
            ck(cudaHostAlloc((void**) &h_host, (size_t) rows_t * M.n_embd * sizeof(float), 0), "h_host");
            nact.resize((size_t) rows_t * C::kNativeActBytes);
            have_t = rows_t;
        }
        if (rows_k > have_k) {
            if (y_host) cudaFreeHost(y_host);
            ck(cudaHostAlloc((void**) &y_host, (size_t) rows_k * M.n_embd * sizeof(float), 0), "y_host");
            have_k = rows_k;
        }
    }
    // The CPU's experts of one layer: `cpu` holds (expert, first row, rows) of the grouped rows (htok: token of a row);
    // outputs into y_host rows (the grouped row index).
    void cpu_experts(int il, const std::vector<std::array<int, 3>>& cpu, const std::vector<int32_t>& htok) {
        if (cpu.empty()) return;
        const auto t0 = std::chrono::steady_clock::now();
        const C::NativeFmt& f = fmt[il];
        std::vector<C::ExpertJobMulti> jobs;
        size_t passes_bytes = 0;
        for (const auto& c : cpu) {
            const uint8_t* b = pk->blob(il, c[0]);
            for (int r = c[1]; r < c[1] + c[2]; r += C::MAXT) {
                C::ExpertJobMulti j;
                j.blob = b;
                j.nt = std::min(C::MAXT, c[1] + c[2] - r);
                for (int t = 0; t < j.nt; ++t) {
                    j.nact[t] = nact.data() + (size_t) htok[r + t] * C::kNativeActBytes;
                    j.out[t] = y_host + (size_t) (r + t) * M.n_embd;
                }
                jobs.push_back(j);
                passes_bytes += f.bytes;
            }
        }
        pool->run_split_multi_native(f, jobs.data(), (int) jobs.size());
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        ms_cpu += ms;
        if (ms > 0.5) cpu_gbps = 0.8 * cpu_gbps + 0.2 * (passes_bytes / 1e6 / ms);
        n_cpu_experts += (int64_t) cpu.size();
    }

    void alloc_batch(int T) {
        Tmax = T;
        const int E = M.n_embd, HD = M.n_head * M.head_dim, H = M.n_head, k = M.n_used;
        gemm = std::make_unique<Gemm>((int64_t) T * k, (int64_t) G::kHc * E, st);
        bR = dalloc((size_t) T * G::kHc * E); bflat = dalloc((size_t) T * G::kHc * E); bmix = dalloc((size_t) T * G::kHcMix);
        bx = dalloc((size_t) T * E); bh = dalloc((size_t) T * E); bpost = dalloc((size_t) T * G::kHc);
        bcomb = dalloc((size_t) T * 16); bout = dalloc((size_t) T * E);
        bqp = dalloc((size_t) T * HD); bkp = dalloc((size_t) T * HD); bvp = dalloc((size_t) T * HD);
        bfa = dalloc((size_t) T * M.head_dim); bgf = dalloc((size_t) T * HD); bg = dalloc((size_t) T * HD);
        bbeta = dalloc((size_t) T * H); bo = dalloc((size_t) T * HD); bga = dalloc((size_t) T * M.head_dim);
        bg2 = dalloc((size_t) T * HD); bgated = dalloc((size_t) T * HD);
        bqa = dalloc((size_t) T * M.q_lora); bqr = dalloc((size_t) T * M.q_lora); bq = dalloc((size_t) T * H * M.dk_mla);
        blat = dalloc((size_t) T * M.kv_lora); bqabs = dalloc((size_t) T * H * M.kv_lora);
        bolat = dalloc((size_t) T * H * M.kv_lora); bvh = dalloc((size_t) T * H * M.dv_mla); bik = dalloc((size_t) T * M.ix_dim);
        blog = dalloc((size_t) T * M.n_expert); bw = dalloc((size_t) T * k);
        bsg = dalloc((size_t) T * M.n_ff_exp); bsu = dalloc((size_t) T * M.n_ff_exp); bshared = dalloc((size_t) T * E);
        bparts = dalloc((size_t) T * k * E);
        bge = dalloc((size_t) T * k * M.n_ff_exp); bue = dalloc((size_t) T * k * M.n_ff_exp);
        bae = dalloc((size_t) T * k * M.n_ff_exp); bye = dalloc((size_t) T * k * E);
        bfg = dalloc((size_t) T * M.n_ff); bfu = dalloc((size_t) T * M.n_ff);
        ck(cudaMalloc(&bids, (size_t) T * k * sizeof(int32_t)), "bids");
        ck(cudaMalloc(&rtok, (size_t) T * k * sizeof(int32_t)), "rtok");
        ck(cudaMalloc(&rdst, (size_t) T * k * sizeof(int32_t)), "rdst");
        ck(cudaMalloc(&rwi, (size_t) T * k * sizeof(int32_t)), "rwi");
        ck(cudaMalloc(&e_ids, (size_t) T * k * sizeof(int32_t)), "e_ids");
        ck(cudaMalloc(&e_bounds, 2 * sizeof(int32_t)), "e_bounds");
        G::iota(e_ids, T * k, st);
        ck(cudaMalloc(&exq, Q::q8_bytes((int64_t) T * k, E)), "exq");
        ck(cudaMalloc(&exq2, Q::q8_bytes((int64_t) T * k, M.n_ff_exp)), "exq2");
    }
    void gm(const Ten& w, const void* wd, size_t wbytes, int64_t n_in, int64_t n_out, const float* xin, int64_t ld_x,
           int T, float* y, int64_t ld_y) {
        gemm->run(w.type, wd, n_in, n_out, wbytes, xin, ld_x, T, y, ld_y, st);
    }
    void gm(const Ten& w, const float* xin, int T, float* y) {
        gm(w, w.d, w.bytes, (int64_t) w.shape[0], (int64_t) w.shape[1], xin, (int64_t) w.shape[0], T, y,
          (int64_t) w.shape[1]);
    }
    void hc_read_b(const Ten& fn, const Ten& base, const Ten& scale, int T) {
        const int E = M.n_embd;
        G::rms_norm_rows(bR, nullptr, bflat, T, G::kHc * E, M.eps_rms, st);
        gm(fn, bflat, T, bmix);
        G::hc_read_rows(bmix, scale.f(), base.f(), bR, bx, bpost, bcomb, E, M.hc_eps, M.hc_iters, T, st);
    }
    void kda_b(int il, int T) {
        const auto& L = M.layers[il];
        const int HD = M.n_head * M.head_dim, kc1 = M.d_conv - 1;
        float* cs = conv[il];
        gm(L.q, bh, T, bqp);
        gm(L.k, bh, T, bkp);
        gm(L.v, bh, T, bvp);
        G::kda_conv_silu_seq(bqp, cs, L.cq.f(), bqp, HD, M.d_conv, T, st);   // in place: each value read once first
        G::kda_conv_silu_seq(bkp, cs + (size_t) kc1 * HD, L.ck.f(), bkp, HD, M.d_conv, T, st);
        G::kda_conv_silu_seq(bvp, cs + (size_t) 2 * kc1 * HD, L.cv.f(), bvp, HD, M.d_conv, T, st);
        gm(L.fa, bh, T, bfa);
        gm(L.fb, bfa, T, bgf);
        G::kda_gate_rows(bgf, L.dtb.f(), L.A.f(), bg, M.n_head, M.head_dim, T, M.gate_lower, st);
        gm(L.beta, bh, T, bbeta);
        G::sigmoid_inplace(bbeta, T * M.n_head, st);
        G::l2_norm_rows(bqp, T * M.n_head, M.head_dim, 1e-6f, st);
        G::l2_norm_rows(bkp, T * M.n_head, M.head_dim, 1e-6f, st);
        G::kda_scan(S[il], bqp, bkp, bvp, bg, bbeta, bo, M.n_head, M.head_dim, T, st);
        gm(L.ga, bh, T, bga);
        gm(L.gb, bga, T, bg2);
        G::kda_out_gate(bo, L.onorm.f(), bg2, bgated, T * M.n_head, M.head_dim, M.eps_rms, st);
        gm(L.wo, bgated, T, bout);
    }
    void mla_b(int il, int T) {
        const auto& L = M.layers[il];
        const int H = M.n_head;
        const int64_t p0 = pos;
        gm(L.qa, bh, T, bqa);
        G::rms_norm_rows(bqa, L.qa_norm.f(), bqr, T, M.q_lora, M.eps_rms, st);
        gm(L.qb, bqr, T, bq);
        gm(L.kva, bh, T, blat);
        G::rms_norm_rows(blat, L.kva_norm.f(), lat_cache[il] + (size_t) p0 * M.kv_lora, T, M.kv_lora, M.eps_rms, st);
        const size_t kb_head = L.kb.bytes / H, vb_head = L.vb.bytes / H;
        for (int hh = 0; hh < H; ++hh)
            gm(L.kb, (const uint8_t*) L.kb.d + hh * kb_head, kb_head, M.dk_mla, M.kv_lora, bq + (size_t) hh * M.dk_mla,
              (int64_t) H * M.dk_mla, T, bqabs + (size_t) hh * M.kv_lora, (int64_t) H * M.kv_lora);
        gm(L.ix_k, bh, T, bik);
        G::layer_norm_rows(bik, L.ix_knw.f(), L.ix_knb.f(), ik_cache[il] + (size_t) p0 * M.ix_dim, T, M.ix_dim, M.eps_ln, st);
        gm(L.ix_gate, bh, T, ig_cache[il] + (size_t) p0 * M.ix_dim);
        G::kpool_keys(ik_cache[il], ig_cache[il], L.ix_ape.f(), pool_cache[il], p0 / M.kpool,
                      (int) ((p0 + T) / M.kpool - p0 / M.kpool), M.kpool, M.ix_dim, st);
        const uint8_t* sel = indexer(L, il, bh, bqr, p0, T);
        G::mla_attend_rows(bqabs, lat_cache[il], p0, T, H, M.kv_lora, M.kpool, 1.0f / std::sqrt((float) M.dk_mla), bolat, st,
                           sel, sel ? n_pools() : 0);
        for (int hh = 0; hh < H; ++hh)
            gm(L.vb, (const uint8_t*) L.vb.d + hh * vb_head, vb_head, M.kv_lora, M.dv_mla, bolat + (size_t) hh * M.kv_lora,
              (int64_t) H * M.kv_lora, T, bvh + (size_t) hh * M.dv_mla, (int64_t) H * M.dv_mla);
        gm(L.wo, bvh, T, bout);
    }
    // ---- the DSA indexer: scores of the complete pools, top-k of them per query (null while all fit)
    float *ix_q = nullptr, *ix_w = nullptr, *ix_sc = nullptr;
    uint8_t* ix_sel = nullptr;
    int ix_rows = 0;
    int64_t n_pools() const { return ctx / M.kpool + 1; }
    const uint8_t* indexer(const Model::Layer& L, int il, const float* hin, const float* qrin, int64_t p0, int T) {
        const int64_t last_vis = (p0 + T) / M.kpool;   // pools visible to the chunk's last query
        if (last_vis <= M.ix_topk / M.kpool) return nullptr;
        if (T > ix_rows) {
            for (void* p : {(void*) ix_q, (void*) ix_w, (void*) ix_sc, (void*) ix_sel}) if (p) cudaFree(p);
            ix_q = dalloc((size_t) T * M.ix_heads * M.ix_dim);
            ix_w = dalloc((size_t) T * M.ix_heads);
            ix_sc = dalloc((size_t) T * n_pools());
            ck(cudaMalloc(&ix_sel, (size_t) T * n_pools()), "ix_sel");
            ix_rows = T;
        }
        if (T == 1) {
            mm(L.ix_qb, qrin, ix_q);
            G::gemv_f32(L.ix_proj.f(), hin, ix_w, M.n_embd, M.ix_heads, st);
        } else {
            gm(L.ix_qb, qrin, T, ix_q);
            G::gemm_f32(L.ix_proj.f(), hin, ix_w, M.n_embd, M.ix_heads, T, st);
        }
        // llama.cpp: weights * 1/sqrt(dim * heads)
        G::scale_inplace(ix_w, T * M.ix_heads, 1.0f / std::sqrt((float) (M.ix_dim * M.ix_heads)), st);
        G::ix_scores(ix_q, ix_w, pool_cache[il], p0, T, M.ix_heads, M.ix_dim, M.kpool, n_pools(), ix_sc, st);
        G::ix_select(ix_sc, p0, T, M.kpool, n_pools(), M.ix_topk / M.kpool, ix_sel, st);
        return ix_sel;
    }
    void expert_product(int type, const void* w, int64_t n_in, int64_t n_out, size_t wbytes, const void* xq_, int rows,
                        float* dst) {
        Q::Product p;
        p.w = w; p.type = type; p.w_rows = n_out; p.w_cols = n_in; p.expert_bytes = wbytes; p.n = 1;
        p.xq = xq_; p.bounds = e_bounds; p.ids = e_ids; p.total_rows = rows; p.max_rows = rows;
        p.dst = dst; p.ld_dst = n_out;
        gemm->ctx.run(p, st);
    }
    void moe_b(int il, int T) {
        const auto& L = M.layers[il];
        const int E = M.n_embd, k = M.n_used, F = M.n_ff_exp;
        G::gemm_f32(L.router.f(), bh, blog, E, M.n_expert, T, st);
        G::router_topk_rows(blog, L.probs_b.f(), M.n_expert, k, M.w_scale, bids, bw, T, st);
        // shared expert for every token
        gm(L.sg, bh, T, bsg);
        gm(L.su, bh, T, bsu);
        G::swiglu_clamp(bsg, bsu, bsg, T * F, M.clamp_sh[il], st);
        gm(L.sd, bsg, T, bshared);
        // routed: rows grouped by expert, each expert's weights copied to the GPU once for all its rows
        std::vector<int32_t> ids((size_t) T * k);
        ck(cudaMemcpyAsync(ids.data(), bids, ids.size() * sizeof(int32_t), cudaMemcpyDeviceToHost, st), "ids");
        ck(cudaStreamSynchronize(st), "ids sync");
        std::vector<int32_t> count(M.n_expert + 1, 0), htok(ids.size()), hdst(ids.size());
        for (int32_t e : ids) count[e + 1]++;
        for (int e = 0; e < M.n_expert; ++e) count[e + 1] += count[e];
        std::vector<int32_t> fill(count.begin(), count.end() - 1);
        for (int t = 0; t < T; ++t)
            for (int j = 0; j < k; ++j) {
                const int e = ids[(size_t) t * k + j];
                const int r = fill[e]++;
                htok[r] = t;
                hdst[r] = t * k + j;
            }
        ck(cudaMemcpyAsync(rtok, htok.data(), htok.size() * sizeof(int32_t), cudaMemcpyHostToDevice, st), "rtok");
        ck(cudaMemcpyAsync(rdst, hdst.data(), hdst.size() * sizeof(int32_t), cudaMemcpyHostToDevice, st), "rdst");
        const size_t gbytes = L.eg.bytes / M.n_expert, dbytes = L.ed.bytes / M.n_expert;
        // which experts the CPU takes: the ones with the fewest rows, while that shortens the slower side
        std::vector<int> order;
        for (int e = 0; e < M.n_expert; ++e) if (count[e + 1] > count[e]) order.push_back(e);
        std::vector<uint8_t> on_cpu(M.n_expert, 0);
        if (pk && cpu_share) {
            std::sort(order.begin(), order.end(), [&](int a, int b) { return count[a + 1] - count[a] < count[b + 1] - count[b]; });
            double gpu_ms = 0, cpu_ms = 0;
            auto gcost = [&](int e) { return pk->lay[il].blob / 1e6 / (pk->is_pinned[(size_t) il * M.n_expert + e] ? gpu_gbps_pinned : gpu_gbps_mapped); };
            auto ccost = [&](int e) { return cpu_bias * (double) ((count[e + 1] - count[e] + C::MAXT - 1) / C::MAXT) * pk->lay[il].blob / 1e6 / cpu_gbps; };
            for (int e : order) gpu_ms += gcost(e);
            for (int e : order) {
                const double g2 = gpu_ms - gcost(e), c2 = cpu_ms + ccost(e);
                if (std::max(g2, c2) >= std::max(gpu_ms, cpu_ms)) break;
                gpu_ms = g2; cpu_ms = c2; on_cpu[e] = 1;
            }
        }
        std::vector<std::array<int, 3>> cpu_list;
        for (int e = 0; e < M.n_expert; ++e)
            if (on_cpu[e]) cpu_list.push_back({e, count[e], count[e + 1] - count[e]});
        if (!cpu_list.empty()) {   // the CPU needs the layer inputs on the host
            ensure_host(T, T * k);
            ck(cudaMemcpyAsync(h_host, bh, (size_t) T * E * sizeof(float), cudaMemcpyDeviceToHost, st), "h to host");
            ck(cudaStreamSynchronize(st), "h sync");
            for (int t = 0; t < T; ++t)
                C::native_quant_act(fmt[il], h_host + (size_t) t * E, nact.data() + (size_t) t * C::kNativeActBytes);
        }
        // the GPU's experts, queued from their own thread while this one runs the CPU's
        auto gpu_side = [&]() {
            for (int e = 0; e < M.n_expert; ++e) {
                const int r0 = count[e], n = count[e + 1] - count[e];
                if (n == 0 || on_cpu[e]) continue;
                const void *wg, *wu, *wd;
                if (pk) {
                    ck(cudaMemcpyAsync(stage_blob, pk->blob(il, e), pk->lay[il].blob, cudaMemcpyHostToDevice, st), "blob");
                    wg = stage_blob; wu = stage_blob + gbytes; wd = stage_blob + 2 * gbytes;
                    bytes_streamed += pk->lay[il].blob;
                } else {
                    ck(cudaMemcpyAsync(stage_g, L.eg.host + (size_t) e * gbytes, gbytes, cudaMemcpyHostToDevice, st), "eg");
                    ck(cudaMemcpyAsync(stage_u, L.eu.host + (size_t) e * gbytes, gbytes, cudaMemcpyHostToDevice, st), "eu");
                    ck(cudaMemcpyAsync(stage_d, L.ed.host + (size_t) e * dbytes, dbytes, cudaMemcpyHostToDevice, st), "ed");
                    wg = stage_g; wu = stage_u; wd = stage_d;
                    bytes_streamed += 2 * gbytes + dbytes;
                }
                ++n_gpu_experts;
                G::set_pair(e_bounds, 0, n, st);
                Q::quantize(bh, rtok + r0, exq, L.eg.type, E, E, n, st);
                expert_product(L.eg.type, wg, E, F, gbytes, exq, n, bge);
                expert_product(L.eu.type, wu, E, F, gbytes, exq, n, bue);
                G::swiglu_clamp(bge, bue, bae, n * F, M.clamp_exp[il], st);
                Q::quantize(bae, nullptr, exq2, L.ed.type, F, F, n, st);
                expert_product(L.ed.type, wd, F, E, dbytes, exq2, n, bye);
                G::scatter_scaled(bye, rdst + r0, rdst + r0, bw, bparts, n, E, st);
            }
            ck(cudaStreamSynchronize(st), "gpu experts");
        };
        const auto c0 = std::chrono::steady_clock::now();
        double gpu_side_ms = 0;
        std::thread gpu_thread([&]() {
            gpu_side();
            gpu_side_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
        });
        cpu_experts(il, cpu_list, htok);
        const double cpu_side_ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
        gpu_thread.join();
        // feedback: the side that finished last gets less next layer (the cost model's error, smoothed)
        if (!cpu_list.empty() && gpu_side_ms > 1.0 && cpu_side_ms > 1.0)
            cpu_bias = std::min(4.0, std::max(0.25, cpu_bias * std::sqrt(cpu_side_ms / gpu_side_ms)));
        ms_copy += std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - c0).count();
        // the CPU's rows to the device, weighted into their slots
        for (const auto& c : cpu_list) {
            ck(cudaMemcpyAsync(bye + (size_t) c[1] * E, y_host + (size_t) c[1] * E, (size_t) c[2] * E * sizeof(float),
                               cudaMemcpyHostToDevice, st), "cpu rows");
            G::scatter_scaled(bye + (size_t) c[1] * E, rdst + c[1], rdst + c[1], bw, bparts, c[2], E, st);
        }
        G::sum_parts(bparts, bshared, bout, T, k, E, st);
    }
    // T prompt tokens at positions pos.. through every layer; the logits of the last one into `logits`.
    void prefill(const int32_t* toks, int T) {
        if (T > Tmax) throw std::runtime_error("prefill: chunk larger than the buffers");
        if (pos + T > ctx) throw std::runtime_error("context full");
        const int E = M.n_embd;
        std::vector<float> emb((size_t) T * G::kHc * E);
        const size_t row = M.tok_embd.bytes / M.n_vocab;
        for (int t = 0; t < T; ++t) {
            const uint8_t* r = M.tok_embd.host + (size_t) toks[t] * row;
            float* e0 = emb.data() + (size_t) t * G::kHc * E;
            for (int b = 0; b < E / 256; ++b) strata::dequantize_q4_K(r + (size_t) b * 144, e0 + b * 256);
            for (int s = 1; s < G::kHc; ++s) std::memcpy(e0 + (size_t) s * E, e0, E * sizeof(float));
        }
        ck(cudaMemcpyAsync(bR, emb.data(), emb.size() * sizeof(float), cudaMemcpyHostToDevice, st), "emb");
        for (int il = 0; il < M.n_layer; ++il) {
            const auto& L = M.layers[il];
            hc_read_b(L.hca_fn, L.hca_base, L.hca_scale, T);
            G::rms_norm_rows(bx, L.attn_norm.f(), bh, T, E, M.eps_rms, st);
            if (L.mla) mla_b(il, T); else kda_b(il, T);
            G::hc_write_rows(bout, bR, bpost, bcomb, bR, E, T, st);
            hc_read_b(L.hcf_fn, L.hcf_base, L.hcf_scale, T);
            G::rms_norm_rows(bx, L.ffn_norm.f(), bh, T, E, M.eps_rms, st);
            if (L.moe) {
                moe_b(il, T);
            } else {
                gm(L.fg, bh, T, bfg);
                gm(L.fu, bh, T, bfu);
                G::swiglu_clamp(bfg, bfu, bfg, T * M.n_ff, M.clamp_sh[il], st);
                gm(L.fd, bfg, T, bout);
            }
            G::hc_write_rows(bout, bR, bpost, bcomb, bR, E, T, st);
        }
        // the head for the last token
        G::hc_mean(bR + (size_t) (T - 1) * G::kHc * E, x, E, st);
        G::rms_norm_rows(x, M.output_norm.f(), h, 1, E, M.eps_rms, st);
        mm(M.output, h, logits);
        pos += T;
    }

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
        fg = dalloc(M.n_ff); fu = dalloc(M.n_ff); fa2 = dalloc(M.n_ff); fy = dalloc((size_t) M.n_used * E);
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
        K::native_quantize_q8_1(q, xq, M.dk_mla * H, 1, st);   // one vector: no block crosses a head
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
        // visible: the complete pools ending at or before this token (the indexer's top-k of them once there are
        // more) plus the incomplete tail up to the token itself (llama.cpp's indexer_kpool_select_tail defaults to
        // true and the GGUF does not set it)
        const uint8_t* sel = indexer(L, il, h, qr, pos, 1);
        G::mla_attend_rows(qabs, lc, pos, 1, H, M.kv_lora, M.kpool, 1.0f / std::sqrt((float) M.dk_mla), olat, st, sel,
                           sel ? n_pools() : 0);
        // per-head value: W_vb[h] (dv outputs from kv_lora inputs)
        K::native_quantize_q8_1(olat, xq, M.kv_lora * H, 1, st);
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
        if (pk) {   // the CPU computes the 8 experts from RAM while the GPU runs the shared expert
            ensure_host(1, M.n_used);
            ck(cudaMemcpyAsync(h_host, h, E * sizeof(float), cudaMemcpyDeviceToHost, st), "h to host");
            ck(cudaStreamSynchronize(st), "h sync");
            swiglu_ffn(L.sg, L.sg.d, L.su, L.su.d, L.sd, L.sd.d, h, out, M.n_ff_exp, M.clamp_sh[il]);
            C::native_quant_act(fmt[il], h_host, nact.data());
            std::vector<std::array<int, 3>> cl;
            std::vector<int32_t> htok(M.n_used, 0);
            for (int j = 0; j < M.n_used; ++j) cl.push_back({ids[j], j, 1});
            cpu_experts(il, cl, htok);
            ck(cudaMemcpyAsync(fy, y_host, (size_t) M.n_used * E * sizeof(float), cudaMemcpyHostToDevice, st), "cpu rows");
            for (int j = 0; j < M.n_used; ++j) G::axpy_dev(acc, fy + (size_t) j * E, ew + j, E, st);
            G::add_inplace(acc, out, E, st);
            ck(cudaMemcpyAsync(out, acc, E * sizeof(float), cudaMemcpyDeviceToDevice, st), "moe out");
            return;
        }
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

// --selftest: every kind of weight GLM multiplies, MMQ (Gemm) against the one-token MMVQ path on random rows.
int selftest(Model& M) {
    cudaStream_t st;
    ck(cudaStreamCreate(&st), "stream");
    const int64_t T = 37;
    Gemm gm(T, 16384, st);
    struct Case { const char* name; const Ten* t; size_t expert; };
    const auto& L0 = M.layers[0];
    const auto& L3 = M.layers[3];
    const auto& L11 = M.layers[11];
    std::vector<Case> cases = {{"kda q Q5_K", &L0.q, 0}, {"kda wo Q5_K", &L0.wo, 0}, {"f_a Q8_0", &L0.fa, 0},
                               {"f_b Q8_0", &L0.fb, 0}, {"beta Q8_0", &L0.beta, 0}, {"hc_fn Q8_0", &L0.hca_fn, 0},
                               {"ffn_down Q6_K", &L0.fd, 0}, {"ffn_gate Q5_K", &L0.fg, 0}, {"q_a Q5_K", &L3.qa, 0},
                               {"q_b Q8_0", &L3.qb, 0}, {"kv_a Q8_0", &L3.kva, 0}, {"ix_q_b Q8_0", &L3.ix_qb, 0},
                               {"mla wo Q5_K", &L3.wo, 0}, {"shexp down Q6_K", &L3.sd, 0}, {"output Q4_K", &M.output, 0},
                               {"exp gate IQ2_XXS", &L3.eg, 1}, {"exp down IQ3_XXS", &L3.ed, 1},
                               {"exp gate IQ2_S", &L11.eg, 1}, {"exp down IQ4_XS", &L11.ed, 1}};
    int bad = 0;
    for (const auto& c : cases) {
        const Ten& t = *c.t;
        const int64_t n_in = (int64_t) t.shape[0], n_out = (int64_t) t.shape[1];
        const void* w = t.d;
        void* tmp = nullptr;
        size_t wb = t.bytes;
        if (c.expert) {   // one expert's slice copied to the device
            wb = t.bytes / M.n_expert;
            ck(cudaMalloc(&tmp, wb), "expert");
            ck(cudaMemcpy(tmp, t.host + 5 * wb, wb, cudaMemcpyHostToDevice), "expert copy");
            w = tmp;
        }
        std::vector<float> hx((size_t) T * n_in);
        uint32_t seed = 12345;
        for (auto& v : hx) { seed = seed * 1664525u + 1013904223u; v = ((seed >> 8) / 16777216.0f - 0.5f) * 2.0f; }
        float *x = dalloc(hx.size()), *y1 = dalloc((size_t) T * n_out), *y2 = dalloc((size_t) T * n_out);
        ck(cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice), "x");
        gm.run(t.type, w, n_in, n_out, wb, x, n_in, T, y1, n_out, st, false);
        gm.run(t.type, w, n_in, n_out, wb, x, n_in, T, y2, n_out, st, true);
        ck(cudaStreamSynchronize(st), "selftest");
        std::vector<float> a((size_t) T * n_out), b(a.size());
        ck(cudaMemcpy(a.data(), y1, a.size() * 4, cudaMemcpyDeviceToHost), "y1");
        ck(cudaMemcpy(b.data(), y2, b.size() * 4, cudaMemcpyDeviceToHost), "y2");
        double md = 0, mx = 0;
        for (size_t i = 0; i < a.size(); ++i) { md = std::max(md, (double) std::fabs(a[i] - b[i])); mx = std::max(mx, (double) std::fabs(b[i])); }
        const bool mmq = Gemm::mmq_ok(t.type, n_out);
        const bool ok = std::isfinite(md) && md <= 2e-2 * std::max(mx, 1e-3);
        std::printf("  %-18s %6lld x %6lld  %s  max |diff| %.3g of max |y| %.3g  %s\n", c.name, (long long) n_out,
                    (long long) n_in, mmq ? "MMQ " : "MMVQ", md, mx, ok ? "ok" : "MISMATCH");
        bad += !ok;
        cudaFree(x); cudaFree(y1); cudaFree(y2);
        if (tmp) cudaFree(tmp);
    }
    std::printf("selftest: %d mismatches\n", bad);
    return bad ? 1 : 0;
}

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
    int max_new = 8, ctx = 4096, top = 5, chunk = 0, workers = 0;
    std::string pack_out, pack_dir;
    double ram_gib = 0;
    bool no_cpu = false;
    bool dump = false, self = false;
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
        else if (a == "--chunk") chunk = std::stoi(next());
        else if (a == "--pack-out") pack_out = next();
        else if (a == "--pack") pack_dir = next();
        else if (a == "--ram-gib") ram_gib = std::stod(next());
        else if (a == "--pool-workers") workers = std::stoi(next());
        else if (a == "--no-cpu-share") no_cpu = true;
        else if (a == "--dump") dump = true;
        else if (a == "--selftest") self = true;
        else { std::fprintf(stderr, "unknown argument: %s\n", a.c_str()); return 2; }
    }
    if (model.empty() || (tokens.empty() && !self && pack_out.empty())) {
        std::fprintf(stderr, "usage: strata-glm --model <GGUF shard 1> --tokens 1,2,3 [--max-new N] [--ctx N] [--top K] [--chunk T] [--dump]\n"
                     "       [--pack DIR [--ram-gib N] [--pool-workers N] [--no-cpu-share]] | --pack-out DIR | --selftest\n");
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
        if (self) return selftest(M);
        if (!pack_out.empty()) return write_pack(M, pack_out);
        Pack P;
        if (!pack_dir.empty()) {
            P.open(M, pack_dir);
            std::vector<int> order;   // pinned first: every layer's experts in order (a routing profile comes later)
            for (int e = 0; e < M.n_expert; ++e)
                for (int l = 0; l < M.n_layer; ++l)
                    if (P.lay[l].off >= 0) order.push_back(l * M.n_expert + e);
            const auto tp0 = clk::now();
            P.pin(order, (size_t) (ram_gib * 1073741824.0));
            if (P.pinned_bytes)
                std::printf("strata-glm: %.1f GiB of experts copied into pinned RAM in %.1f s\n", P.pinned_bytes / 1073741824.0,
                            std::chrono::duration<double>(clk::now() - tp0).count());
        }
        Engine E(M, ctx);
        if (!pack_dir.empty()) E.use_pack(&P, workers);
        E.cpu_share = !no_cpu;
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
        if (chunk > 0) {
            E.alloc_batch(chunk);
            for (size_t i = 0; i < prompt.size(); i += chunk)
                E.prefill(prompt.data() + i, (int) std::min<size_t>(chunk, prompt.size() - i));
        } else {
            for (size_t i = 0; i < prompt.size(); ++i) E.forward(prompt[i], i + 1 == prompt.size());
        }
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
        if (chunk > 0)
            std::printf("prompt path: %.2f GB of experts copied to the GPU; experts on the GPU %lld, on the CPU %lld "
                        "(CPU %.0f ms, %.1f GB/s per pass); MoE wall %.0f ms\n", E.bytes_streamed / 1e9,
                        (long long) E.n_gpu_experts, (long long) E.n_cpu_experts, E.ms_cpu, E.cpu_gbps, E.ms_copy);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-glm: %s\n", e.what());
        return 1;
    }
    return 0;
}
