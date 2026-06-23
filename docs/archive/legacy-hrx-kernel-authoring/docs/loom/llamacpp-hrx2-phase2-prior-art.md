# HRX2 Phase 2 Fusion Prior Art

Date: 2026-06-15

This note seeds HRX2 Phase 2 fusion search. These are not accepted HRX2 routes;
they are motifs that other llama.cpp backends already considered valuable
enough to encode. HRX2 still requires same-target, same-shape fused-vs-unfused
evidence before routing a fusion.

## Backend Motifs

| Motif | Backend evidence | Why it matters for HRX2 |
| --- | --- | --- |
| `RMS_NORM -> MUL` | Vulkan admits `RMS_NORM_MUL` in `ggml-vulkan.cpp`; CUDA has `ggml_cuda_op_rms_norm_fused` in `norm.cu`. | Common norm-weight application. It saves an F32 vector write/read and one dispatch across decode and prompt regimes. |
| `RMS_NORM -> MUL -> ROPE` | Vulkan generates `rms_norm_mul_rope_*` shaders and routes `RMS_NORM_MUL_ROPE` when layout/mode constraints pass. | Q/K projection setup often materializes normalized/scaled vectors before ROPE. This is bandwidth-relevant, not just launch reduction. Defer enablement until attention-adjacent Loom stability is settled. |
| `MUL_MAT -> ADD` and `MUL_MAT -> ADD -> ADD` | Vulkan checks and routes `MUL_MAT_ADD` and `MUL_MAT_ADD_ADD` for mat-vec epilogues. | Bias/add epilogues should be evaluated per quant/layout/shape bucket. Prior HRX work also called out Q8_0 `MUL_MAT + ADD`. |
| `MUL_MAT_ID -> ADD_ID`, `MUL_MAT_ID -> ADD_ID -> MUL`, `MUL_MAT_ID -> MUL` | Vulkan routes MoE matmul-id epilogues; CUDA `mmvq.cu` has fused quant matvec support for bias, gate, gate bias, and GLU. | MoE routed matvecs are high-frequency on the basket. These fusions should specialize by route-density and decode/prompt shape. |
| Multi-`ADD` chains | CUDA has fused binary broadcast add for 2-8 adds; Vulkan routes `MULTI_ADD`. | Useful when graph dataflow shows same-shape F32 add chains. Accept only if memory traffic dominates, not for launch count alone. |
| Top-k MoE selection pipeline | CUDA `topk-moe.cu` explicitly fuses softmax/top-k/get-rows style MoE selection; Vulkan has top-k MoE fusion modes. | Potentially important for MoE models, but graph variants and masks make this a separate evidence bucket. |
| `ROPE -> VIEW -> SET_ROWS` | Vulkan has `ROPE_VIEW_SET_ROWS` and a larger `RMS_NORM_MUL_ROPE_VIEW_SET_ROWS` path. | Targets KV-cache update traffic and the Phase 1 `SET_ROWS` infrastructure blocker. This should be tracked separately from ordinary compute fusions. |

## Search Guidance

- Mine candidates from the HRX2 basket traces first, then cross-check this
  prior-art list. Do not route a fusion only because another backend has it.
- Score memory bandwidth savings separately from dispatch-count savings. Hero
  ops such as matmul epilogues and attention/KV-cache updates are worthwhile
  even when llama.cpp remains dispatch-heavy overall.
- Keep decode-like, grouped/narrow prompt, and prefill-like regimes separate.
  The same logical fusion can have different winners or lose entirely across
  those regimes.
- Favor small, stable 2-3 op fusions for tranche 1. Larger clusters from other
  systems are useful directionally, but llama.cpp graph and allocation
  structure makes a dozen-cluster target unrealistic for HRX2 v1.

## Source Anchors

- `sources/llama.cpp/ggml/src/ggml-vulkan/ggml-vulkan.cpp`: fusion admission
  and graph compute routing around `ggml_vk_can_fuse`, `MUL_MAT_ADD`,
  `MUL_MAT_ID_ADD_ID_MUL`, `RMS_NORM_MUL`, `RMS_NORM_MUL_ROPE`, top-k MoE, and
  `ROPE_VIEW_SET_ROWS`.
- `sources/llama.cpp/ggml/src/ggml-vulkan/vulkan-shaders/vulkan-shaders-gen.cpp`:
  generated RMS norm/ROPE fusion shader variants.
- `sources/llama.cpp/ggml/src/ggml-cuda/norm.cu`: fused RMS norm plus MUL and
  RMS norm plus MUL plus ADD entry points.
- `sources/llama.cpp/ggml/src/ggml-cuda/mmvq.cu`: quant matvec fusion support
  for bias, gate, gate bias, and GLU epilogues.
- `sources/llama.cpp/ggml/src/ggml-cuda/binbcast.cu`: fused broadcast ADD
  support.
- `sources/llama.cpp/ggml/src/ggml-cuda/topk-moe.cu`: MoE softmax/top-k/get
  rows fusion intent.
