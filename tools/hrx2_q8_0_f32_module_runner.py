#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from pathlib import Path

from hrx2_pipeline_lib import DEFAULT_ROCM, WORKSPACE


DEFAULT_OUT_ROOT = WORKSPACE / "cache" / "hrx2" / "q8_0_f32_common_runner"
DEFAULT_HIP_SOURCE = (
    WORKSPACE
    / "cache"
    / "hrx2"
    / "q8_0_f32_refute"
    / "gfx1100-q8-hip-rerun-20260612"
    / "q8_0_f32_refute.hip.cpp"
)
DEFAULT_LOOM_ARTIFACT = (
    WORKSPACE
    / "cache"
    / "hrx2"
    / "q8_0_f32_tune"
    / "q8-stella-focused-k512-r64-c8-20260612"
    / "bundles"
    / "q8_0_f32_word4_bitunpack_rhsvec_dotf_k512_r64_c8_rpg1_cpg1_wg128_rep2"
    / "target_artifacts"
    / "r00052f870d700453_c0_per_sample_sample0_target.elf"
)
DEFAULT_LOOM_ARTIFACT_WG64 = (
    WORKSPACE
    / "cache"
    / "hrx2"
    / "q8_0_f32_tune"
    / "q8-stella-focused-k512-r64-c8-20260612"
    / "bundles"
    / "q8_0_f32_word4_bitunpack_rhsvec_dotf_k512_r64_c8_rpg1_cpg1_wg64_rep2"
    / "target_artifacts"
    / "r00052f84794d3ae8_c0_per_sample_sample0_target.elf"
)


RUNNER_SOURCE = r'''
#include <hip/hip_runtime.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <string>
#include <vector>

#define HIP_CHECK(expr) do { \
  hipError_t _err = (expr); \
  if (_err != hipSuccess) { \
    std::fprintf(stderr, "%s:%d: HIP error: %s\n", __FILE__, __LINE__, hipGetErrorString(_err)); \
    std::exit(2); \
  } \
} while (0)

struct block_q8_0 {
  unsigned short d;
  signed char qs[32];
};

static_assert(sizeof(block_q8_0) == 34, "unexpected Q8_0 block size");

struct options {
  std::string kind = "loom";
  std::string module_path;
  std::string function_name;
  int k = 512;
  int rows = 64;
  int cols = 8;
  int rows_per_workgroup = 1;
  int workgroup_size = 128;
  int iters = 1000;
  int warmup = 100;
  int repeats = 7;
  bool check = true;
};

static int parse_int_arg(const char *arg, const char *prefix) {
  const size_t n = std::strlen(prefix);
  if (std::strncmp(arg, prefix, n) != 0) return -1;
  return std::atoi(arg + n);
}

static options parse_options(int argc, char **argv) {
  options opts;
  for (int i = 1; i < argc; ++i) {
    if (std::strncmp(argv[i], "--kind=", 7) == 0) {
      opts.kind = argv[i] + 7;
    } else if (std::strncmp(argv[i], "--module=", 9) == 0) {
      opts.module_path = argv[i] + 9;
    } else if (std::strncmp(argv[i], "--function=", 11) == 0) {
      opts.function_name = argv[i] + 11;
    } else if (std::strncmp(argv[i], "--k=", 4) == 0) {
      opts.k = std::max(32, parse_int_arg(argv[i], "--k="));
    } else if (std::strncmp(argv[i], "--rows=", 7) == 0) {
      opts.rows = std::max(1, parse_int_arg(argv[i], "--rows="));
    } else if (std::strncmp(argv[i], "--cols=", 7) == 0) {
      opts.cols = std::max(1, parse_int_arg(argv[i], "--cols="));
    } else if (std::strncmp(argv[i], "--rows-per-workgroup=", 21) == 0) {
      opts.rows_per_workgroup = std::max(1, parse_int_arg(argv[i], "--rows-per-workgroup="));
    } else if (std::strncmp(argv[i], "--workgroup-size=", 17) == 0) {
      opts.workgroup_size = std::max(1, parse_int_arg(argv[i], "--workgroup-size="));
    } else if (std::strncmp(argv[i], "--iters=", 8) == 0) {
      opts.iters = std::max(1, parse_int_arg(argv[i], "--iters="));
    } else if (std::strncmp(argv[i], "--warmup=", 9) == 0) {
      opts.warmup = std::max(0, parse_int_arg(argv[i], "--warmup="));
    } else if (std::strncmp(argv[i], "--repeats=", 10) == 0) {
      opts.repeats = std::max(1, parse_int_arg(argv[i], "--repeats="));
    } else if (std::strcmp(argv[i], "--no-check") == 0) {
      opts.check = false;
    } else {
      std::fprintf(stderr,
          "usage: %s --kind=loom|hip --module=PATH --function=NAME [--k=N] [--rows=N] [--cols=N] "
          "[--rows-per-workgroup=N] [--workgroup-size=N] [--iters=N] [--warmup=N] [--repeats=N] [--no-check]\n",
          argv[0]);
      std::exit(2);
    }
  }
  opts.k = (opts.k / 32) * 32;
  if (opts.module_path.empty() || opts.function_name.empty()) {
    std::fprintf(stderr, "--module and --function are required\n");
    std::exit(2);
  }
  if (opts.kind != "loom" && opts.kind != "hip") {
    std::fprintf(stderr, "--kind must be loom or hip\n");
    std::exit(2);
  }
  return opts;
}

template <typename T>
struct device_buffer {
  T *ptr = nullptr;
  size_t count = 0;

  explicit device_buffer(size_t count) : count(count) {
    HIP_CHECK(hipMalloc(&ptr, count * sizeof(T)));
  }
  ~device_buffer() {
    if (ptr) (void)hipFree(ptr);
  }
  device_buffer(const device_buffer &) = delete;
  device_buffer &operator=(const device_buffer &) = delete;
};

static float make_value(int index, int seed) {
  const int raw = (index * 17 + seed * 31) % 257;
  const float wave = std::sin(static_cast<float>(index + seed) * 0.013f) * 0.25f;
  return (static_cast<float>(raw) - 128.0f) * 0.00390625f + wave;
}

static unsigned short scale_bits(int row, int block_idx) {
  static const unsigned short bits[] = {0x3400, 0x3800, 0x3a00, 0x3c00};
  return bits[(row * 3 + block_idx * 5) & 3];
}

static float scale_value(unsigned short bits) {
  switch (bits) {
    case 0x3400: return 0.25f;
    case 0x3800: return 0.5f;
    case 0x3a00: return 0.75f;
    case 0x3c00: return 1.0f;
    default: return 0.0f;
  }
}

static void fill_q8(std::vector<block_q8_0> &blocks, int rows, int blocks_per_row) {
  for (int row = 0; row < rows; ++row) {
    for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
      block_q8_0 &block = blocks[static_cast<size_t>(row) * blocks_per_row + block_idx];
      block.d = scale_bits(row, block_idx);
      for (int i = 0; i < 32; ++i) {
        block.qs[i] = static_cast<signed char>(((row * 11 + block_idx * 13 + i * 7) & 63) - 32);
      }
    }
  }
}

static void fill_rhs(std::vector<float> &src1, int k, int cols) {
  for (int col = 0; col < cols; ++col) {
    for (int i = 0; i < k; ++i) {
      src1[static_cast<size_t>(col) * k + i] = make_value(i + col * k, 23);
    }
  }
}

static void reference_q8(
    const std::vector<block_q8_0> &src0,
    const std::vector<float> &src1,
    std::vector<float> &dst,
    int k,
    int rows,
    int cols) {
  const int blocks_per_row = k / 32;
  for (int col = 0; col < cols; ++col) {
    for (int row = 0; row < rows; ++row) {
      float acc = 0.0f;
      for (int block_idx = 0; block_idx < blocks_per_row; ++block_idx) {
        const block_q8_0 &block = src0[static_cast<size_t>(row) * blocks_per_row + block_idx];
        const float d = scale_value(block.d);
        for (int i = 0; i < 32; ++i) {
          acc += d * static_cast<float>(block.qs[i]) * src1[static_cast<size_t>(col) * k + block_idx * 32 + i];
        }
      }
      dst[static_cast<size_t>(col) * rows + row] = acc;
    }
  }
}

static void check_close(const std::vector<float> &actual, const std::vector<float> &expected) {
  double max_abs = 0.0;
  double max_rel = 0.0;
  size_t bad_idx = 0;
  for (size_t i = 0; i < actual.size(); ++i) {
    const double diff = std::abs(static_cast<double>(actual[i]) - static_cast<double>(expected[i]));
    const double denom = std::max(1.0, std::abs(static_cast<double>(expected[i])));
    const double rel = diff / denom;
    if (diff > max_abs) {
      max_abs = diff;
      max_rel = rel;
      bad_idx = i;
    }
  }
  if (max_abs > 5.0e-3 && max_rel > 5.0e-4) {
    std::fprintf(stderr,
        "correctness failed: max_abs=%g max_rel=%g idx=%zu actual=%g expected=%g\n",
        max_abs, max_rel, bad_idx, actual[bad_idx], expected[bad_idx]);
    std::exit(1);
  }
}

static void launch_kernel(
    const options &opts,
    hipFunction_t function,
    block_q8_0 *src0,
    float *src1,
    float *dst) {
  const unsigned int grid_x = static_cast<unsigned int>((opts.rows + opts.rows_per_workgroup - 1) / opts.rows_per_workgroup);
  const unsigned int grid_y = static_cast<unsigned int>(opts.cols);
  const unsigned int block_x = static_cast<unsigned int>(opts.workgroup_size);
  if (opts.kind == "loom") {
    void *args[] = {&src0, &src1, &dst};
    HIP_CHECK(hipModuleLaunchKernel(function, grid_x, grid_y, 1, block_x, 1, 1, 0, nullptr, args, nullptr));
  } else {
    long long k = opts.k;
    long long rows = opts.rows;
    long long cols = opts.cols;
    void *args[] = {&src0, &src1, &dst, &k, &rows, &cols};
    HIP_CHECK(hipModuleLaunchKernel(function, grid_x, grid_y, 1, block_x, 1, 1, 0, nullptr, args, nullptr));
  }
}

int main(int argc, char **argv) {
  const options opts = parse_options(argc, argv);
  HIP_CHECK(hipSetDevice(0));

  hipModule_t module;
  HIP_CHECK(hipModuleLoad(&module, opts.module_path.c_str()));
  hipFunction_t function;
  HIP_CHECK(hipModuleGetFunction(&function, module, opts.function_name.c_str()));

  const int blocks_per_row = opts.k / 32;
  const size_t block_count = static_cast<size_t>(opts.rows) * blocks_per_row;
  const size_t src1_count = static_cast<size_t>(opts.cols) * opts.k;
  const size_t dst_count = static_cast<size_t>(opts.cols) * opts.rows;

  std::vector<block_q8_0> h_src0(block_count);
  std::vector<float> h_src1(src1_count);
  std::vector<float> h_ref(dst_count, 0.0f);
  std::vector<float> h_dst(dst_count, 0.0f);
  fill_q8(h_src0, opts.rows, blocks_per_row);
  fill_rhs(h_src1, opts.k, opts.cols);
  reference_q8(h_src0, h_src1, h_ref, opts.k, opts.rows, opts.cols);

  device_buffer<block_q8_0> d_src0(block_count);
  device_buffer<float> d_src1(src1_count);
  device_buffer<float> d_dst(dst_count);
  HIP_CHECK(hipMemcpy(d_src0.ptr, h_src0.data(), block_count * sizeof(block_q8_0), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemcpy(d_src1.ptr, h_src1.data(), src1_count * sizeof(float), hipMemcpyHostToDevice));
  HIP_CHECK(hipMemset(d_dst.ptr, 0, dst_count * sizeof(float)));

  if (opts.check) {
    launch_kernel(opts, function, d_src0.ptr, d_src1.ptr, d_dst.ptr);
    HIP_CHECK(hipDeviceSynchronize());
    HIP_CHECK(hipMemcpy(h_dst.data(), d_dst.ptr, dst_count * sizeof(float), hipMemcpyDeviceToHost));
    check_close(h_dst, h_ref);
  }

  for (int i = 0; i < opts.warmup; ++i) {
    launch_kernel(opts, function, d_src0.ptr, d_src1.ptr, d_dst.ptr);
  }
  HIP_CHECK(hipDeviceSynchronize());

  std::vector<float> samples;
  samples.reserve(opts.repeats);
  for (int repeat = 0; repeat < opts.repeats; ++repeat) {
    hipEvent_t start;
    hipEvent_t stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));
    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < opts.iters; ++i) {
      launch_kernel(opts, function, d_src0.ptr, d_src1.ptr, d_dst.ptr);
    }
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));
    float ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&ms, start, stop));
    samples.push_back(ms * 1000.0f / static_cast<float>(opts.iters));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));
  }

  std::vector<float> sorted = samples;
  std::sort(sorted.begin(), sorted.end());
  double mean = 0.0;
  for (float sample : samples) mean += sample;
  mean /= samples.size();
  double var = 0.0;
  for (float sample : samples) {
    const double diff = sample - mean;
    var += diff * diff;
  }
  var /= samples.size();

  const float p50 = sorted[sorted.size() / 2];
  const float p90 = sorted[std::min(sorted.size() - 1, static_cast<size_t>(std::ceil(sorted.size() * 0.9) - 1))];
  const unsigned int grid_x = static_cast<unsigned int>((opts.rows + opts.rows_per_workgroup - 1) / opts.rows_per_workgroup);
  std::printf(
      "{\"kind\":\"%s\",\"module\":\"%s\",\"function\":\"%s\",\"k\":%d,\"rows\":%d,\"cols\":%d,"
      "\"rows_per_workgroup\":%d,\"workgroup_size\":%d,\"grid_x\":%u,\"grid_y\":%d,"
      "\"mean_us\":%.6f,\"stdev_us\":%.6f,\"p50_us\":%.6f,\"p90_us\":%.6f,\"samples_us\":[",
      opts.kind.c_str(), opts.module_path.c_str(), opts.function_name.c_str(),
      opts.k, opts.rows, opts.cols, opts.rows_per_workgroup, opts.workgroup_size,
      grid_x, opts.cols, mean, std::sqrt(var), p50, p90);
  for (size_t i = 0; i < samples.size(); ++i) {
    std::printf("%s%.6f", i == 0 ? "" : ",", samples[i]);
  }
  std::printf("]}\n");
  HIP_CHECK(hipModuleUnload(module));
  return 0;
}
'''


def parse_args():
    parser = argparse.ArgumentParser(description="Build/run a common HIP module runner for Q8_0/F32 Loom and HIP code objects.")
    parser.add_argument("--out-root", default=str(DEFAULT_OUT_ROOT))
    parser.add_argument("--run-id", default="gfx1100-q8-common-runner")
    parser.add_argument("--rocm-path", default=str(DEFAULT_ROCM))
    parser.add_argument("--offload-arch", default=os.environ.get("HRX2_TARGET", "gfx1100"))
    parser.add_argument("--hip-source", default=str(DEFAULT_HIP_SOURCE))
    parser.add_argument("--hip-module", default="")
    parser.add_argument("--loom-module", default=str(DEFAULT_LOOM_ARTIFACT))
    parser.add_argument("--loom-module-wg64", default=str(DEFAULT_LOOM_ARTIFACT_WG64))
    parser.add_argument(
        "--cases",
        default="loom128,hip128,loom64,hip64",
        help="Comma-separated cases to run: loom128, hip128, loom64, hip64.",
    )
    parser.add_argument("--build-only", action="store_true")
    parser.add_argument("--k", type=int, default=512)
    parser.add_argument("--rows", type=int, default=64)
    parser.add_argument("--cols", type=int, default=8)
    parser.add_argument("--iters", type=int, default=1000)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--no-check", action="store_true")
    return parser.parse_args()


def run(cmd, cwd=WORKSPACE):
    subprocess.run(cmd, cwd=cwd, check=True)


def build(args, run_dir):
    rocm = Path(args.rocm_path)
    hipcc = rocm / "bin" / "hipcc"
    if not hipcc.exists():
        raise FileNotFoundError(f"hipcc not found at {hipcc}")
    runner_src = run_dir / "q8_common_module_runner.cpp"
    runner_bin = run_dir / "q8_common_module_runner"
    runner_src.write_text(RUNNER_SOURCE, encoding="utf-8")
    run([
        str(hipcc),
        f"--offload-arch={args.offload_arch}",
        "-O3",
        "-std=c++17",
        str(runner_src),
        "-o",
        str(runner_bin),
    ])
    hip_module = Path(args.hip_module) if args.hip_module else run_dir / "q8_hip_refute.hsaco"
    if not args.hip_module:
        run([
            str(hipcc),
            f"--offload-arch={args.offload_arch}",
            "-O3",
            "-std=c++17",
            "--genco",
            str(args.hip_source),
            "-o",
            str(hip_module),
        ])
    return runner_bin, hip_module


def run_one(runner_bin, kind, module, function, rows_per_workgroup, workgroup_size, args):
    cmd = [
        str(runner_bin),
        f"--kind={kind}",
        f"--module={module}",
        f"--function={function}",
        f"--k={args.k}",
        f"--rows={args.rows}",
        f"--cols={args.cols}",
        f"--rows-per-workgroup={rows_per_workgroup}",
        f"--workgroup-size={workgroup_size}",
        f"--iters={args.iters}",
        f"--warmup={args.warmup}",
        f"--repeats={args.repeats}",
    ]
    if args.no_check:
        cmd.append("--no-check")
    proc = subprocess.run(cmd, cwd=WORKSPACE, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    row = json.loads(proc.stdout.strip().splitlines()[-1])
    row["cmd"] = cmd
    return row


def selected_cases(args, hip_module):
    cases = {
        "loom128": ("loom", Path(args.loom_module), "q8_0_f32_candidate", 1, 128),
        "hip128": ("hip", hip_module, "refute_q8_0_rows1_wg128_f32", 1, 128),
        "loom64": ("loom", Path(args.loom_module_wg64), "q8_0_f32_candidate", 1, 64),
        "hip64": ("hip", hip_module, "refute_q8_0_rows1_wg64_f32", 1, 64),
    }
    selected = []
    for name in [part.strip() for part in args.cases.split(",") if part.strip()]:
        if name not in cases:
            raise ValueError(f"unknown case {name!r}; expected one of {', '.join(sorted(cases))}")
        selected.append((name, *cases[name]))
    return selected


def main():
    args = parse_args()
    run_dir = Path(args.out_root) / args.run_id
    run_dir.mkdir(parents=True, exist_ok=True)
    runner_bin, hip_module = build(args, run_dir)
    manifest = {
        "run_id": args.run_id,
        "runner": str(runner_bin),
        "hip_module": str(hip_module),
        "loom_module": str(args.loom_module),
        "loom_module_wg64": str(args.loom_module_wg64),
        "offload_arch": args.offload_arch,
        "cases": args.cases,
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    if args.build_only:
        print(json.dumps(manifest, indent=2))
        return 0

    results = []
    for case_name, kind, module, function, rows_per_workgroup, workgroup_size in selected_cases(args, hip_module):
        row = run_one(runner_bin, kind, module, function, rows_per_workgroup, workgroup_size, args)
        row["case"] = case_name
        results.append(row)
        print(json.dumps(row, sort_keys=True))
    (run_dir / "results.jsonl").write_text("\n".join(json.dumps(row, sort_keys=True) for row in results) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
