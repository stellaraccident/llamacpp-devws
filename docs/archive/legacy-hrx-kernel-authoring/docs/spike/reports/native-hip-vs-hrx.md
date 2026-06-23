# Native HIP vs HRX Falsification Report

Date: 2026-04-21

## Summary

I replaced the HRX runtime dependency in `sources/llama.cpp-hip` with a native HIP compatibility shim that drives the same generated HSACO kernel catalog directly through HIP. The shim supports two modes:

- `GGML_HIP_COMPARE_SUBMISSION=stream`: direct `hipModuleLaunchKernel` on a HIP stream.
- `GGML_HIP_COMPARE_SUBMISSION=graph`: HIP stream capture, instantiate, and launch on flush/sync boundaries.

The result falsifies the simple claim that HRX performance comes from being slower than native HIP submission. After rerunning the current HRX build at `build/llama-hrx-integration`, HRX is faster on decode, while native HIP stream and HRX are effectively comparable on the generalized prefill sweep. Native HIP graph mode is substantially slower.

## Headline Results

All numbers are local W7900 / Qwen3.5-35B-A3B Q4_K_L, serial runs, `-dev HRX0 -ngl 99 --no-host 1 -fa 1 -b 512 -ub 512` unless noted. Current HRX was measured from existing build dir `build/llama-hrx-integration` on 2026-04-21; `llama-bench` reports build `cbc385495 (8746)`.

| Case | Native HIP stream | Native HIP graph | Current HRX | Readout |
| --- | ---: | ---: | ---: | --- |
| Decode `p0 n64 r7` | `112.26 +/- 0.28 tok/s` | `52.99 +/- 0.65 tok/s` | `115.55 +/- 0.33 tok/s` | HRX is ~2.9% faster than stream; graph is much worse |
| Prefill `p512 n0 r5` | `2362.56 +/- 7.46 tok/s` | `2208.33 +/- 12.26 tok/s` | `2347.61 +/- 26.40 tok/s` | Stream and HRX are effectively tied; graph is slower |

The earlier report draft used stale documented HRX numbers. This revision uses the measured current HRX build requested by the user. HRX logs are under `build/llama-hip-compare/logs/hrx-current/`.

## Implementation

Changed `sources/llama.cpp-hip` only:

- Added `ggml/src/ggml-hrx/hrx_hip_compat.h`.
- Swapped `ggml-hrx` from `find_package(hrx)` / `hrx::hrx` to `find_package(hip)` / `hip::host`.
- Implemented enough of the HRX API surface on HIP: device enumeration, buffer allocation/mapping, stream sync, async copy/fill, executable load, export lookup, raw kernarg packing, and kernel launch.
- Added graph mode via HIP stream capture, graph instantiate, and graph launch on stream flush.
- Flushed graph capture before allocation/staging boundaries that HIP cannot capture.
- Treated zero-work dispatches as no-ops, matching HRX behavior and avoiding HIP `grid=(0,...)` launch failures.
- Bounded native HIP `GET_ROWS` support to `ggml_nelements(src1) <= 65535` because the existing kernel maps selected rows to HIP grid Y, and native HIP rejects larger Y dimensions.

## Correctness

Final validation logs are under `build/llama-hip-compare/logs/`.

| Check | Result |
| --- | --- |
| Rebuild `test-backend-hrx test-backend-ops llama-cli llama-completion llama-bench` | passed |
| `test-backend-hrx`, stream mode | passed |
| `test-backend-hrx`, graph mode | passed |
| `test-backend-ops test -b HRX0` targeted 23-op set, stream mode | `1060/1060 tests passed` |
| Same targeted op set, graph mode | `1060/1060 tests passed` |
| `llama-cli --simple-io -st` stream smoke | passed, generation reported `112.0 t/s` |
| `llama-cli --simple-io -st` graph smoke | passed, generation reported `55.4 t/s` |
| `git diff --check` in `sources/llama.cpp-hip` | passed |
| Current HRX decode benchmark from `build/llama-hrx-integration` | passed |
| Current HRX prefill sweep from `build/llama-hrx-integration` | passed |

The op gate used:

`RMS_NORM,ADD,MUL,DIV,SCALE,CPY,CONT,SET_ROWS,GET_ROWS,MUL_MAT,FLASH_ATTN_EXT,CONCAT,SOFT_MAX,ARGSORT,ROPE,UNARY,GLU,SUM_ROWS,L2_NORM,CLAMP,SSM_CONV,GATED_DELTA_NET,MUL_MAT_ID`

## Prefill Sweep

`llama-bench -p 2,3,31,32,33,127,128,129,255,256,257,512,1023,1024,1025 -n 0 -r 5`

| Prompt tokens | Native HIP stream tok/s | Native HIP graph tok/s | Current HRX tok/s |
| ---: | ---: | ---: | ---: |
| 2 | `159.47 +/- 1.75` | `90.93 +/- 1.11` | `164.61 +/- 1.12` |
| 3 | `218.59 +/- 2.54` | `128.37 +/- 4.71` | `224.21 +/- 1.61` |
| 31 | `504.21 +/- 15.03` | `431.49 +/- 4.89` | `509.43 +/- 18.02` |
| 32 | `489.55 +/- 8.56` | `414.54 +/- 6.00` | `498.78 +/- 12.39` |
| 33 | `486.91 +/- 4.58` | `403.91 +/- 10.51` | `488.12 +/- 8.73` |
| 127 | `1231.13 +/- 28.94` | `1116.39 +/- 19.16` | `1257.17 +/- 32.69` |
| 128 | `1339.97 +/- 43.71` | `1196.51 +/- 19.75` | `1355.98 +/- 28.42` |
| 129 | `1254.90 +/- 22.73` | `1117.52 +/- 19.65` | `1266.49 +/- 34.68` |
| 255 | `1888.18 +/- 18.53` | `1716.03 +/- 20.56` | `1895.85 +/- 12.99` |
| 256 | `1900.23 +/- 38.41` | `1730.11 +/- 30.65` | `1931.22 +/- 26.09` |
| 257 | `1803.78 +/- 29.63` | `1654.14 +/- 19.85` | `1826.32 +/- 23.04` |
| 512 | `2362.56 +/- 7.46` | `2208.33 +/- 12.26` | `2347.61 +/- 26.40` |
| 1023 | `2331.68 +/- 13.19` | `2188.18 +/- 16.77` | `2297.33 +/- 15.57` |
| 1024 | `2335.19 +/- 24.69` | `2181.60 +/- 18.18` | `2314.24 +/- 14.82` |
| 1025 | `2271.34 +/- 18.77` | `2086.13 +/- 5.93` | `2256.14 +/- 7.09` |

Graph mode is consistently slower in this shim. At `p512`, graph is about 6.5% below stream. At decode, graph is about 52.8% below stream. Current HRX and native HIP stream are close across prefill; HRX is slightly ahead on most small/mid prompt points, while native HIP stream is slightly ahead at `p512` and around `p1024`.

## Interpretation

Native HIP stream mode can reproduce the same decode regime, but current HRX is faster in the direct rerun: `115.55 tok/s` versus native HIP stream `112.26 tok/s`. That is enough to reject the claim that direct native HIP submission is inherently faster than HRX for this decode shape.

For prefill, the current HRX build is not the stale `29xx tok/s` documented outlier. It measures `2347.61 tok/s` at `p512`, while native HIP stream measures `2362.56 tok/s`. That is a tie for practical purposes. Across the sweep, HRX is slightly ahead on many small/mid prompt sizes and native stream is slightly ahead on some larger prompt sizes. The direct HIP stream path does not show a broad win.

The naive HIP graph-capture path is not a win. It is worse for decode and modestly worse for prefill. This does not prove a highly engineered reusable HIP graph implementation could not help, but it does rule out "plain HIP graph capture explains the HRX result."

## Caveats

- This is a falsification harness over the existing HRX generated kernel catalog, not a separate hand-optimized HIP backend.
- Graph mode currently captures, instantiates, launches, and retires graphs at flush/sync boundaries. It does not implement graph exec update/reuse.
- Current HRX was rerun from the existing `build/llama-hrx-integration` build, not rebuilt.
- Native HIP now declines oversized `GET_ROWS` cases that exceed the HIP grid-Y limit instead of claiming support and failing at launch.
