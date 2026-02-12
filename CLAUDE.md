# onnxruntime-builder

A Nix flake that produces ONNX Runtime v1.23.2 libraries with platform-specific accelerator support.

## Current Task: Static CUDA Build (Primary Goal)

**Goal**: Fully static CUDA linking. All other derivations (`cpu`, `cuda-dyn`) are stepping stones.

**Status**: Two infinite loop patterns fixed, rebuild needed to test.

**Root cause**: Provider bridge (`provider_bridge_provider.cc`) creates infinite loops when statically linked. Methods call through vtable indirection back to themselves.

**Progress**:
- `ORT_STATIC_PROVIDERS` guard added for `GetType<T>` specializations (`static-datatype-loop.patch`)
- `ORT_STATIC_PROVIDERS` guard added for `TensorShape` methods (`static-tensorshape-loop.patch`)
- `cuda-dyn` confirmed working (validates CUDA integration; not the end goal)

**Next step**: Rebuild and test. If still hangs, capture another backtrace with `nix run .#debug-bt -- 3`.

**Identifying infinite loops** in the backtrace:
- `ProviderHostImpl::SomeMethod__Thing()` calling `Thing::SomeMethod()`
- Both resolve to the same address (vtable indirection back to itself)

Then add a guard in `provider_bridge_provider.cc` similar to the `GetType<T>` fix:
```cpp
#ifndef ORT_STATIC_PROVIDERS
// bridge implementation that calls through g_host
#endif
```

**Approaches if patching becomes untenable**:
1. **Continue patching** - Guard remaining methods with `ORT_STATIC_PROVIDERS` (current path)
2. **Disable provider bridge entirely** - Investigate `onnxruntime_USE_FULL_PROTOBUF` or similar flags that bypass the bridge for static builds
3. **Hybrid linking** - Keep provider bridge as shared lib, link everything else static

## Supported Platforms

| Platform | Backends | Accelerator |
|----------|----------|-------------|
| x86_64-linux | CPU + CUDA | NVIDIA GPU (Ampere/Ada/Hopper/Blackwell) |
| aarch64-darwin | CPU + CoreML | Apple Neural Engine / Metal |

## Running & Building

**BEFORE ANY BUILD**: Check for active builds. CUDA builds saturate CPU for 30-60 min.
```bash
pgrep -f "onnxruntime\|nvcc" && echo "⚠️  BUILD RUNNING" || echo "✓ Clear"
```

**PREFER `nix run` OVER `nix build`** - apps automatically configure the runtime environment (`ONNX_TEST_MODEL`, `LD_LIBRARY_PATH` for cuDNN, etc.).

```bash
# Run the wrapper (builds if needed, sets up env automatically)
nix run                       # Default: CPU-only (fast iteration)
nix run .#ort-wrapper-cpu     # Explicit CPU-only
nix run .#ort-wrapper-cuda-dyn # Dynamic CUDA linking (recommended for CUDA)
nix run .#ort-wrapper-cuda    # Static CUDA (hangs - use cuda-dyn instead)

# Build libraries only
nix build .#onnxruntime-cpu        # CPU-only static library (fast)
nix build .#onnxruntime-cuda-dyn # CUDA shared library (for cuda-dyn)
nix build .#                        # CUDA static library (slow, has runtime issues)
```

### Build Times

Approximate build times on a modern workstation (first build, no cache):

| Build | Time | Notes |
|-------|------|-------|
| `onnxruntime-cpu` | ~5-10 min | CPU-only, fastest iteration |
| `onnxruntime-cuda-dyn` | ~20-40 min | Dynamic CUDA libs, scales with `cudaArchitectures` |
| `onnxruntime` (static CUDA) | ~30-60 min | Full static build with archive merging |
| `ort-wrapper-*` | ~1-2 min | Rust wrapper (after ONNX Runtime is cached) |

**Factors affecting build time:**
- **CUDA architectures**: Each sm_XX target adds compilation time. `"90;100"` is ~2x slower than `"90"` alone.
- **Nix cache**: Subsequent builds reuse cached artifacts. Only changed components rebuild.
- **NVCC threads**: Set to 1 (`onnxruntime_NVCC_THREADS`) to avoid OOM on systems with limited RAM.

### Available Apps & Packages (x86_64-linux)

| Command | Description | Status |
|---------|-------------|--------|
| **`nix run`** | **ORT wrapper CPU-only (recommended for dev)** | ✅ Works |
| `nix run .#ort-wrapper-cpu` | ORT wrapper CPU-only (explicit) | ✅ Works |
| `nix run .#ort-wrapper-cuda-dyn` | ORT wrapper with dynamic CUDA | ✅ Works |
| `nix run .#ort-wrapper-cuda` | ORT wrapper with static CUDA | ❌ Hangs |
| `nix build .#onnxruntime-cpu` | ONNX Runtime CPU-only library | ✅ Works |
| `nix build .#onnxruntime-cuda-dyn` | ONNX Runtime dynamic CUDA library | ✅ Works |
| `nix build .#` | ONNX Runtime static CUDA library | ⚠️ Builds but hangs |
| `nix run .#build-cuda` | Build static CUDA with visible logs | 🔧 Debug tool |

### Dev Shells

```bash
nix develop           # Default (static CUDA on Linux, CoreML on macOS)
nix develop .#cpu     # CPU-only
nix develop .#cuda-dyn # Dynamic CUDA linking
```

## Build Outputs

### Static Build (default)
```
result/lib/
├── libonnxruntime.a              # Unified static library
├── libonnxruntime_providers_cuda.a    # CUDA provider (static)
└── cudnn/libcudnn*.so*           # cuDNN shared libs (for bundling)
```

### Shared Build (cuda-dyn)
```
result/lib/
├── libonnxruntime.so             # Main shared library
├── libonnxruntime_providers_cuda.so   # CUDA provider (dlopened at runtime)
├── libonnxruntime_providers_shared.so # Provider bridge
└── cudnn/libcudnn*.so*           # cuDNN shared libs
```

## Status

- [x] CPU-only static build - works end-to-end
- [x] Dynamic CUDA build (`cuda-dyn`) - **bypasses static init hang**
- [x] CUDA provider loads and detects GPU
- [x] GPU architecture support for Blackwell (RTX 50xx) - sm_120 for compute 12.0
- [ ] Static CUDA build - hangs during static initialization (unsolved)
- [ ] macOS/aarch64-darwin build verified

## Known Issues

### Static CUDA Hang (Provider Bridge Infinite Loops)

The provider bridge (`provider_bridge_provider.cc`) is designed for shared libraries where provider and host code live in separate binaries. When statically linked, methods call through vtable indirection back to themselves:

1. `DataTypeImpl::GetType<T>()` → `Provider_GetHost()->DataTypeImpl__GetType_T()`
2. `ProviderHostImpl::DataTypeImpl__GetType_T()` → `DataTypeImpl::GetType<T>()`
3. Infinite loop (same function via vtable indirection)

See **Current Task** for debugging workflow and patch progress.

**Workaround**: Use `cuda-dyn` (dynamic linking avoids the bridge entirely).

### GPU Architecture

The `cudaArchitectures` variable in `flake.nix` controls GPU targets:
- Current: `"90;120"` (Hopper + Blackwell consumer)
- sm_90 = Hopper (H100, compute 9.0)
- sm_100 = Blackwell datacenter (B100/B200, compute 10.0)
- sm_120 = Blackwell consumer (RTX 50xx, compute 12.0)

If you see `cudaErrorSymbolNotFound` during CUDA inference, the GPU architecture isn't supported. Update `cudaArchitectures` in `flake.nix`.

## Implementation Notes

### Build Variants

The flake supports two CUDA build strategies via `mkOnnxruntime`:

1. **Static (`buildShared=false`)**: Produces `.a` files, requires complex linking, patches for static CUDA provider. Currently hangs at runtime.

2. **Shared (`buildShared=true`)**: Produces `.so` files, standard dynamic linking. Used by `cuda-dyn`. Works correctly.

### Cargo Features

| Feature | Linking | Library |
|---------|---------|---------|
| `cuda` | Static | `libonnxruntime.a` + complex deps |
| `cuda-dyn` | Dynamic | `libonnxruntime.so` |
| `coreml` | Static | `libonnxruntime.a` + frameworks |

The `cuda-dyn` feature enables `cuda` (for runtime detection) but uses a different `build.rs` path that links against the shared library.

### Environment Variables

| Variable | Feature | Purpose |
|----------|---------|---------|
| `ORT_LIB_LOCATION` | cuda, coreml | Path to static `.a` files (ort crate convention) |
| `ORT_DYLIB_PATH` | cuda-dyn | Path to shared `.so` files |
| `ONNX_TEST_MODEL` | all | Path to test ONNX model |
| `LD_LIBRARY_PATH` | cuda* | Must include cuDNN path at runtime |

### Patches

All patches live in `patches/`. Grouped by purpose:

**SIOF fixes** (Static Initialization Order Fiasco) — see `docs/siof-patches.md` for details:
- `static-init-bridge.patch` — Meyers singleton for `ProviderHostImpl`
- `static-init-cpu.patch` — Meyers singleton for `ProviderHostCPUImpl`
- `static-init-provider.patch` — Lazy init for `g_host`/`g_host_cpu`
- `static-init-common.patch` — Fallback to static host when `g_host` is NULL

**Provider bridge loop fixes** (static CUDA hang):
- `static-datatype-loop.patch` — `ORT_STATIC_PROVIDERS` guard for `GetType<T>`
- `static-tensorshape-loop.patch` — `ORT_STATIC_PROVIDERS` guard for `TensorShape`

**Build system**:
- `cuda-static-provider.patch` — Enable static CUDA provider compilation
- `providers-shared-static.patch` — Build `providers_shared` as static lib

**Portability**:
- `musl-execinfo.patch` — Stub `execinfo.h` for musl libc
- `musl-cstdint.patch` — Add missing `<cstdint>` include

## Debug Tools

**Always use `nix run`** — apps configure `ONNX_TEST_MODEL`, `LD_LIBRARY_PATH`, etc.

```bash
nix run .#debug-bt          # Auto-interrupt GDB backtrace (default 2s)
nix run .#debug-bt -- 5     # Custom delay (5 seconds)
nix run .#debug-gdb         # Interactive GDB session (Ctrl+C, then bt)
nix run .#build-cuda        # Build with visible logs
```

See **Current Task** for the debugging workflow.

## Nix Build Discipline

**Step 1 - ALWAYS check for active builds first:**
```bash
pgrep -f "onnxruntime\|nvcc" && echo "⚠️  BUILD RUNNING" || echo "✓ Clear"
```

**Step 2 - If clear, run synchronously (foreground):**
- Use `nix run .#build-cuda` for visible progress on long builds
- Use `nix run` (CPU-only default) for fast iteration (~1-2 min)
- Use `cuda-dyn` to validate CUDA integration while static build is in progress

**Why this matters**: CUDA builds take 30-60 minutes and saturate CPU. The Nix daemon runs builds independently - `nix build` can return while compilation continues. Starting a second build creates resource contention and confusion.

**Anti-pattern:**
```
❌ Skip precondition check → Start build → Discover conflict → Waste time
```

## Nix Command Output

**Always capture stderr**: Many nix commands output progress/errors to stderr only. Use `2>&1` to capture both streams:

```bash
# BAD: may show empty output
nix path-info .#foo

# GOOD: captures all output
nix path-info .#foo 2>&1
nix build .#foo --print-build-logs 2>&1
```

**Check build progress**: Nix builds run in the daemon. The `nix build` command may return but the daemon continues working.

```bash
# Reliable status check
nix path-info .#ort-wrapper-cuda 2>/dev/null && echo "DONE" || echo "NOT DONE"

# Check if actively compiling
pgrep -f "onnxruntime" && echo "Build running" || echo "Not running"

# Count active compile jobs
ps aux | grep -E "nvcc|g\+\+.*onnxruntime" | grep -v grep | wc -l
```

## Reference

The nixpkgs shared library definition (v1.22.2) was used as a starting point:
https://github.com/NixOS/nixpkgs/blob/nixos-unstable/pkgs/by-name/on/onnxruntime/package.nix
