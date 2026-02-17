# Build Reference

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

## Build Times

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

## Build Variants

The flake supports two CUDA build strategies via `mkOnnxruntime`:

1. **Static (`buildShared=false`)**: Produces `.a` files, requires complex linking, patches for static CUDA provider. Currently hangs at runtime.

2. **Shared (`buildShared=true`)**: Produces `.so` files, standard dynamic linking. Used by `cuda-dyn`. Works correctly.

## Cargo Features

| Feature | Linking | Library |
|---------|---------|---------|
| `cuda` | Static | `libonnxruntime.a` + complex deps |
| `cuda-dyn` | Dynamic | `libonnxruntime.so` |
| `coreml` | Static | `libonnxruntime.a` + frameworks |

The `cuda-dyn` feature enables `cuda` (for runtime detection) but uses a different `build.rs` path that links against the shared library.

## Environment Variables

| Variable | Feature | Purpose |
|----------|---------|---------|
| `ORT_LIB_LOCATION` | cuda, coreml | Path to static `.a` files (ort crate convention) |
| `ORT_DYLIB_PATH` | cuda-dyn | Path to shared `.so` files |
| `ONNX_TEST_MODEL` | all | Path to test ONNX model |
| `LD_LIBRARY_PATH` | cuda* | Must include cuDNN path at runtime |

## Patches

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

## GPU Architecture

The `cudaArchitectures` variable in `flake.nix` controls GPU targets:
- Current: `"90;120"` (Hopper + Blackwell consumer)
- sm_90 = Hopper (H100, compute 9.0)
- sm_100 = Blackwell datacenter (B100/B200, compute 10.0)
- sm_120 = Blackwell consumer (RTX 50xx, compute 12.0)

If you see `cudaErrorSymbolNotFound` during CUDA inference, the GPU architecture isn't supported. Update `cudaArchitectures` in `flake.nix`.
