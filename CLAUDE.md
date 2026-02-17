# onnxruntime-builder

A Nix flake that produces ONNX Runtime v1.23.2 libraries with platform-specific accelerator support.

## Current Focus: ort-wrapper as Self-Contained Library

**Goal**: `ort-wrapper` is the single dependency for downstream Rust applications needing ONNX Runtime. It re-exports `ort` and `ndarray`, handles FFI initialization, and will carry all runtime dependencies (shared libs for cuda-dyn).

**Status**: Public API surface is production-shaped: `SessionBuilder` (file/memory, provider preference), one-shot `infer()`, and `create_session` (thin delegate). Next: verify cuda-dyn is truly self-contained when bundled (shared libs packaged alongside binary).

**Long-term**: Fully static CUDA linking (currently hangs — see [static-cuda-hang.md](docs/static-cuda-hang.md)).

## Quick Reference

```bash
# Pre-flight (CUDA builds saturate CPU for 30-60 min)
pgrep -f "onnxruntime\|nvcc" && echo "⚠️  BUILD RUNNING" || echo "✓ Clear"

# Run (prefer nix run — auto-configures ONNX_TEST_MODEL, LD_LIBRARY_PATH, etc.)
nix run                           # Default: CPU-only (fast iteration)
nix run .#ort-wrapper-cuda-dyn    # Dynamic CUDA (recommended for CUDA)
nix run .#ort-wrapper-cuda        # Static CUDA (hangs — use cuda-dyn)

# Build libraries only
nix build .#onnxruntime-cpu       # CPU-only static (~5-10 min)
nix build .#onnxruntime-cuda-dyn  # CUDA shared (~20-40 min)
nix build .#                      # CUDA static (~30-60 min, hangs at runtime)

# Dev shells
nix develop                       # Default (CPU-only on Linux, CoreML on macOS)
nix develop .#cpu                 # CPU-only
nix develop .#cuda-dyn            # Dynamic CUDA
```

## Status

- [x] CPU-only static build — works end-to-end
- [x] Dynamic CUDA build (`cuda-dyn`) — bypasses static init hang
- [x] CUDA provider loads and detects GPU
- [x] Blackwell GPU support (sm_120)
- [x] `ort-wrapper` re-exports `ort` + `ndarray`, auto-initializes
- [x] `SessionBuilder` (file/memory) + one-shot `infer()` API
- [x] Test suite organized: `usage.rs` (API docs), `smoke.rs` (linking), `validation.rs` (quality/stability)
- [ ] Self-contained cuda-dyn packaging (bundled shared libs)
- [ ] Static CUDA build — hangs during static initialization
- [ ] macOS/aarch64-darwin build verified

## Architecture

| Platform | Backends | Accelerator |
|----------|----------|-------------|
| x86_64-linux | CPU + CUDA | NVIDIA GPU (Hopper/Blackwell) |
| aarch64-darwin | CPU + CoreML | Apple Neural Engine / Metal |

### Key Design Decisions

- **`ort-wrapper` re-exports `ort` and `ndarray`** — downstream crates depend only on `ort-wrapper`. `init()` uses `OnceLock` for idempotent, retryable initialization. `SessionBuilder` is the primary session API; `create_session` delegates to it. `infer()` provides one-shot convenience.
- **`build.rs` discovers abseil libraries** from `ORT_LIB_LOCATION` via filesystem scan — no hardcoded list to maintain across abseil versions.
- **MRI merge uses glob** (`libonnxruntime_*.a`) — single script works for both CUDA and CPU static builds.
- **`alternative-backend`** feature in `ort` crate disables its linking; `build.rs` takes full control.
- **`ORT_STATIC_PROVIDERS` define** gates provider bridge code that creates infinite loops in static builds.
- **Test files** are separated by purpose: `usage.rs` (literate API examples — 4 tests, no macros), `smoke.rs` (build/link verification), `validation.rs` (provider behavior, output quality, determinism, stability).

## Reference

- [Build reference](docs/build-reference.md) — outputs, build times, cargo features, env vars, patches, GPU architectures
- [Nix build discipline](docs/nix-build-discipline.md) — pre-flight checks, stderr capture, build progress monitoring
- [SIOF patches](docs/siof-patches.md) — Static Initialization Order Fiasco fixes
- [Static CUDA hang](docs/static-cuda-hang.md) — root cause, debug tools, fallback approaches
- [nixpkgs onnxruntime](https://github.com/NixOS/nixpkgs/blob/nixos-unstable/pkgs/by-name/on/onnxruntime/package.nix) — upstream shared library definition used as starting point
