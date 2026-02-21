# onnxruntime-builder

A Nix flake that produces ONNX Runtime v1.23.2 libraries with platform-specific accelerator support.

## Quick Reference

```bash
# Pre-flight (CUDA builds saturate CPU for 30-60 min)
pgrep -f "onnxruntime\|nvcc" && echo "⚠️  BUILD RUNNING" || echo "✓ Clear"

# Run (prefer nix run — auto-configures ONNX_TEST_MODEL, LD_LIBRARY_PATH, etc.)
nix run                            # Default: CPU-only (fast iteration)
nix run .#ort-wrapper-cuda-dlopen  # CUDA hybrid (recommended for CUDA)
nix run .#ort-wrapper-cpu          # CPU-only explicit

# Build libraries only
nix build .#onnxruntime-cpu          # CPU-only static (~5-10 min)
nix build .#onnxruntime-cuda-hybrid  # Hybrid: static core + CUDA provider .so (~20-40 min)

# Dev shells
nix develop                        # Default: CPU-only (fast iteration)
nix develop .#cpu                  # CPU-only (explicit)
nix develop .#cuda-dlopen          # CUDA hybrid
```

## Status

- [x] CPU-only static build — works end-to-end
- [x] Hybrid CUDA build — static ort core + CUDA providers via dlopen, no SIOF
- [x] CUDA provider loads, detects GPU, runs inference on Blackwell (sm_120)
- [x] `ort-wrapper` re-exports `ort` + `ndarray`, auto-initializes
- [x] `SessionBuilder` (file/memory) + one-shot `infer()` API
- [x] Test suite: `usage.rs` (API docs), `smoke.rs` (linking), `validation.rs` (quality/stability)
- [x] Default dev shell is CPU-only (no misleading CUDA hints)
- [x] All build paths use `ORT_LIB_LOCATION`
- [ ] Self-contained cuda-dlopen package for downstream consumption
- [ ] macOS/aarch64-darwin build verified

## Architecture

| Platform | Backends | Accelerator |
|----------|----------|-------------|
| x86_64-linux | CPU + CUDA | NVIDIA GPU (Hopper/Blackwell) |
| aarch64-darwin | CPU + CoreML | Apple Neural Engine / Metal |

### Key Design Decisions

- **Hybrid CUDA is upstream's default** — ONNX Runtime cmake builds `providers_cuda.so` (MODULE) and `providers_shared.so` (SHARED) unconditionally on Linux, regardless of `BUILD_SHARED_LIB`. Our static CUDA patches override this; hybrid mode omits them.
- **`ort-wrapper` re-exports `ort` and `ndarray`** — downstream crates depend only on `ort-wrapper`. ndarray version matches ort's (0.17), so `Tensor::from_array(ndarray)` works across the crate boundary. `SessionBuilder` is the primary API; `infer()` provides one-shot convenience.
- **`build.rs` discovers abseil libraries** from `ORT_LIB_LOCATION` via filesystem scan — no hardcoded list to maintain across abseil versions.
- **`alternative-backend`** feature in `ort` crate disables its linking; `build.rs` takes full control.
- **`envDefault = envCpu`** — the default dev environment is CPU-only for fast iteration. CUDA development uses `nix develop .#cuda-dlopen`.
- **Test files** separated by purpose: `usage.rs` (literate examples, no macros), `smoke.rs` (build/link), `validation.rs` (providers, quality, determinism, stability).

## Reference

- [Build reference](docs/build-reference.md) — outputs, build times, cargo features, env vars, patches, GPU architectures
- [Nix build discipline](docs/nix-build-discipline.md) — pre-flight checks, stderr capture, build progress monitoring
- [SIOF patches](docs/siof-patches.md) — Static Initialization Order Fiasco fixes
- [Static CUDA hang](docs/static-cuda-hang.md) — root cause, debug tools, alternative approaches
- [nixpkgs onnxruntime](https://github.com/NixOS/nixpkgs/blob/nixos-unstable/pkgs/by-name/on/onnxruntime/package.nix) — upstream shared library definition used as starting point

## Current Focus: Downstream Integration

The hybrid build is validated end-to-end: static `libonnxruntime.a` (77 MB) linked at compile time + CUDA providers (341 MB `.so`) loaded via dlopen at runtime. CPU inference 2ms, CUDA inference 546ms (first run, includes GPU warmup).

### `lib` Output

The flake exports `mkOnnxruntime` and `mkOrtWrapperEnv` via `lib`. Downstream usage:
```nix
ort = onnxruntime-builder.lib;
onnxruntime = ort.mkOnnxruntime { inherit pkgs system; hybridCuda = true; };
env = ort.mkOrtWrapperEnv { inherit pkgs system onnxruntime; };
# env.buildInputs, env.envVars.ORT_LIB_LOCATION, env.cargoFeatures
```

### Remaining Work

1. **Self-contained package** — bundle CUDA provider `.so` files and cuDNN libs so the output is deployable without Nix store references at runtime
2. **macOS verification** — validate aarch64-darwin CPU + CoreML path
