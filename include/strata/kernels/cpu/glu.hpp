// include/strata/kernels/cpu/glu.hpp - the SwiGLU of the CPU expert kernels.
#pragma once

#include <algorithm>
#include <cmath>

namespace strata::kernels::cpu {

/// GLM-5's clamp (llama.cpp's ggml_swiglu_clamp): with a limit L > 0 the gate is capped at L and the up projection
/// clamped to [-L, L] before silu(gate) * up. 0 (the default, every Qwen pack) leaves the SwiGLU exactly as it was.
/// One model per process, so one process-wide value, set before the first expert runs.
inline float g_glu_limit = 0.0f;

inline float swiglu(float g, float u) {
    const float L = g_glu_limit;
    if (L > 0.0f) {
        g = std::min(g, L);
        u = std::min(std::max(u, -L), L);
    }
    return (g / (1.f + std::exp(-g))) * u;
}

}  // namespace strata::kernels::cpu
