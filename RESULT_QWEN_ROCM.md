# Qwen3.6-35B-A3B on ROCm (Radeon 780M) — result

Date: 2026-07-16. Hardware: Minisforum UM790 Pro, Ryzen 9 7940HS, **iGPU Radeon
780M (gfx1103, run as gfx1100 via HSA_OVERRIDE_GFX_VERSION=11.0.0)**, 64 GB DDR5,
Pop!_OS, ROCm 6.3.1. Branch `rocm-qwen` (fork of andreaborio/ds4).

## Status: **Qwen3.6-35B-A3B runs end-to-end on the 780M GPU.**

Resident mode (19.37 GiB payload wired in GTT, no SSD streaming). Coherent
generation; numerically matches the scalar CPU reference.

### Measured (greedy, temp 0)

| path | prefill t/s | generation t/s | notes |
|---|---|---|---|
| ds4 **ROCm GPU** (this port) | 11.8–14.3 | **13.4–14.9** | resident, 780M |
| ds4 CPU reference | ~10 | ~10–11.5 | scalar oracle |
| llama.cpp CPU (same artifact) | ~100 (pp64) | 16.1 (tg64) | 8 threads |

The GPU path already beats the ds4 CPU reference. It does not yet beat
llama.cpp — expected for a v1 correctness-first port (no HIP-graph batching, F32
throughout, top-8 MoE run as two top-4 halves). Andrea's Metal numbers (58–66
t/s on M5 Pro) are on ~2× the memory bandwidth and a mature kernel set.

### Correctness (M2 parity gate)

CPU-vs-GPU top-logprobs, prompt "The sea is", 5 steps:
- **top-1 agreement: 5/5** (GPU selects the same token as the CPU reference at
  every step)
- **cosine 0.9994–0.9999** on the shared top-k logit vectors (Andrea's own
  Metal-vs-llama.cpp gate is 0.99994)
- max logit diff 0.5–1.7 out of ~26 (sub-1%) — pure f32 accumulation-order
  difference between the scalar CPU path and the GPU kernels.

Greedy 96-token text is token-identical to the CPU golden for the first ~40
tokens, then diverges at a near-tie — the documented upstream behavior, not a
bug.

Per-op kernel parity (`tests/test_qwen35_rocm.c`, all 22 checks pass on the
780M vs `ds4_qwen_ref.c`): split_q_gate, sigmoid_mul (elements/rows), rope,
router (exact top-8 + weights), gqa decode/prefill, gated_delta_step (kd=128
fast path + kd=64 generic), gated_delta_sequence_128, causal_conv step/sequence,
rmsnorm_gated, gated_delta_controls, dequant_embedding_q8_0. The GDN recurrence
(the novel, highest-risk kernel) matches to ~1e-8 relative.

## What was built

- **`rocm/ds4_rocm_qwen.cuh`**: 19 HIP kernels (1:1 port of `metal/qwen35.metal`;
  Metal simdgroup → RDNA3 wave32, simd_sum/max/min → `__shfl_xor` butterfly) +
  the 20 `ds4_gpu_qwen35_*` entry points. Byte-stride args mirror `ds4_metal.m`.
- **Top-8 MoE**: the shared ROCm Q4_K MoE core is specialized for DeepSeek's
  top-6 (fused `sum6` kernels). Qwen's top-8 runs as **two top-4 halves + add**
  (`moe_split_selected_kernel` + `routed_moe_batch_two_half`), leaving the
  DeepSeek path untouched.
- **Host wiring (`ds4.c`)**: Qwen GPU forward/session/dispatch guards widened
  from `__APPLE__` to `DS4_QWEN_GPU_BUILD`. No `DS4_BACKEND_ROCM` enum exists —
  ROCm runs under `DS4_BACKEND_CUDA` — so `ds4_backend_is_qwen_gpu()` replaces
  the `backend==METAL` checks. Env gate `DS4_QWEN_EXPERIMENTAL_ROCM=1`; resident
  forced (SSD streaming for Qwen not ported).
- **Artifact**: `tests/qwen/normalize_qwen36_gguf.py` reproduces the blessed
  `Qwen3.6-35B-A3B-ds4-Q4_K_S.gguf` from the Unsloth UD-Q4_K_S source (Q6_K
  ffn_down_exps → Q4_K, output → Q8_0, pad-id + chat template). ds4 accepts it;
  inventory **f32=361 / q8_0=252 / q4_k=120 == the prereg histogram exactly**.
- **Forward-ported La Bestia ROCm hardening** (multi-model range cache, readback
  event ring + host-side waits, adaptive free-floor) onto the fork; build-sha
  bug fixed.

## Run

```sh
DS4_QWEN_EXPERIMENTAL_ROCM=1 HSA_OVERRIDE_GFX_VERSION=11.0.0 \
  ./ds4 -m gguf/Qwen3.6-35B-A3B-ds4-Q4_K_S.gguf -c 4096 -p "…"
```

## Not done / next

- Soak at 8k and 32k context; formal `ds4-bench` CSV.
- Throughput: HIP-graph batching, fused gate/up+SwiGLU for the two-half MoE,
  parallel GQA decode variant (only the serial GQA kernel is wired).
- Server path on ROCm (chat.html) — engine wiring is done; needs a server smoke.
- **Caveat**: the Qwen server is text-only (no tool-call), so it cannot replace
  Ollama for JARVIS function-calling until upstream adds tool parsing.
