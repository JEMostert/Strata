# GLM-5.3-Flash in Strata (work in progress)

A second model family for Strata: GLM-5.3-Flash ("glm5-next" in GGUF), 320B parameters with 18B active per token,
on one consumer GPU plus system RAM. This file is the plan and the progress log; the numbers in it are measured
unless they are marked as an estimate.

## The model (from the GGUF and llama.cpp's `src/models/glm5-next.cpp`)

- 45 layers plus one MTP (NextN) layer for drafting. hidden 4096, vocabulary 154,880, context 1M.
- Layer types: KDA (Kimi Delta Attention, linear) where `head_count_kv` is 0, otherwise MLA attention without RoPE
  plus a DSA indexer (34 KDA layers, 11 MLA layers in the 45; the NextN layer is MLA).
- Residual: 4 streams (mHC). Per sublayer: RMS norm of the flat 4 x 4096 streams, one projection to 24 mixes:
  4 read weights (sigmoid + eps), 4 write weights (2 sigmoid), a 4 x 4 mixing matrix (Sinkhorn, 20 rounds).
  Output = mean of the streams, RMS norm, head.
- KDA: q/k/v projections (4096 -> 64 heads x 128), causal conv (kernel 4) + SiLU each, L2-normed q/k,
  per-channel decay g = -5 * sigmoid(-(A * (f_b(f_a(x)) + dt_bias))), beta = sigmoid(W_beta x), the gated delta
  rule (S *= exp(g) per key channel; delta = beta (v - S^T k); S += k delta^T; o = S^T q / sqrt(128)), RMS norm per
  head times sigmoid(g_b(g_a(x))), output projection 8192 -> 4096.
- MLA (nope): q = W_qb(rmsnorm(W_qa x)) 64 x 256; latent c = rmsnorm(W_kva x) (512, the only thing cached);
  scores = (W_kb^T q) . c / sqrt(256); per-head value W_vb (512 -> 256); output 16384 -> 4096. KV cache: 512
  values per token per MLA layer.
- DSA indexer: per token key (LayerNorm with bias, 128) and gate (128); pools of 4 tokens, pooled key = per-dim
  softmax(gate + ape) weighted sum; per query 32 heads from the q LoRA, score(pool) = sum_h w_h relu(q_h . k_pool);
  top 512 pools (2048 tokens) attended. Below 2048 tokens of context it is plain causal attention.
- FFN: layers 0-2 dense (12288). Layers 3-44 MoE: 288 experts (4096 x 2048), top 8 by sigmoid score + bias,
  weights normalized and scaled by 2.5, plus one shared expert (2048).

## The file (unsloth UD-IQ2_XXS, 101.8 GB)

| Part | Formats | Size |
| --- | --- | ---: |
| Expert gate/up | IQ2_XXS (IQ2_S in layer 11) | 52.6 GB |
| Expert down | IQ3_XXS (IQ4_XS in layers 11, 12, 44) | 39.9 GB |
| NextN experts | Q2_K / Q3_K | 2.6 GB |
| Everything else (attention, KDA, shared experts, router, head, embedding) | Q5_K / Q6_K / Q8_0 / F32 | ~6.5 GB |

One expert is 7.5 MB (gate/up 4.3 MB + down 3.2 MB); 12,096 of them. Strata already has GPU and CPU kernels for
IQ2_XXS, IQ3_XXS, IQ4_XS and IQ2_S.

## Target PC

Ryzen AI 9 HX 370 (12 cores, AVX-512 VNNI), 96 GB DDR5-5600, RTX 5060 Ti 16 GB on PCIe x4, Fedora 43, CUDA 13.2
(driver 580 / 13.0, minor-version compatibility). Memory plan (estimate): ~6.5 GB of dense weights + buffers on the
GPU, ~7.5 GB of experts cached in VRAM, the other ~84 GB of experts in RAM (each expert in exactly one tier), so
nothing is read from the SSD while answering.

## Plan

1. **Reference.** Mainline llama.cpp (glm5-next merged in #27773) on the same PC: baseline speed and per-layer
   outputs to compare against.
2. **Bring-up (correctness first).** A `strata-glm` engine: GGUF loading, the new kernels (mHC, KDA, MLA + indexer,
   router), dense weights on the GPU, experts computed by Strata's CPU pool and VRAM cache. Checked layer by layer
   against llama.cpp.
3. **Speed.** Captured decode graphs, the VRAM expert cache with adaptive swaps, the balanced PCIe share, MTP
   speculation (llama.cpp has no NextN graph yet: checked by acceptance and against plain greedy output).
4. **Prompt processing on a narrow link.** Each prompt chunk streams every expert it routes to over PCIe, so with
   most of the model outside VRAM the link sets the speed and a bigger chunk is the lever. Plan: decouple the
   streaming granularity from the compute granularity (one pass of expert streaming over a large super-chunk, the
   residual streams kept in VRAM, compute in sub-chunks), keep the link busy across layer boundaries, and give the
   CPU the experts with few tokens (both sides balanced, as in decode).
5. **Serving.** The server's chat template and tokenizer for GLM; Unsloth Desktop as a client of the
   OpenAI-compatible API.

## Progress

- 2026-10-07: model downloaded on the target PC; CUDA 13.2 installed; Strata (upstream main + the balanced PCIe
  share) builds and passes its tests there except three that need a Qwen pack or a higher `ulimit -l`.
