# Build Reference

## Build Outputs

### Hybrid Build (active — static core + CUDA provider dlopen)
```
result/lib/
├── libonnxruntime.a                    # Static core library (linked at compile time)
├── libonnxruntime_providers_shared.so  # Provider bridge (dlopen'd at runtime)
├── libonnxruntime_providers_cuda.so    # CUDA provider (dlopen'd at runtime)
└── cudnn/libcudnn*.so*                 # cuDNN shared libs
```

### Static CUDA Build (hangs — see docs/static-cuda-hang.md)
```
result/lib/
├── libonnxruntime.a              # Unified static library (includes CUDA providers)
├── libonnxruntime_providers_cuda.a    # CUDA provider (static)
└── cudnn/libcudnn*.so*           # cuDNN shared libs (for bundling)
```

### Shared Build (removed — superseded by hybrid)
```
result/lib/
├── libonnxruntime.so             # Shared library
├── libonnxruntime_providers_cuda.so   # CUDA provider (dlopen'd at runtime)
├── libonnxruntime_providers_shared.so # Provider bridge
└── cudnn/libcudnn*.so*           # cuDNN shared libs
```
Available via `mkOnnxruntime { buildShared = true; }` but no longer exposed as a package.

## Build Times

Approximate build times on a modern workstation (first build, no cache):

| Build | Time | Notes |
|-------|------|-------|
| `onnxruntime-cpu` | ~5-10 min | CPU-only, fastest iteration |
| `onnxruntime-cuda-hybrid` | ~20-40 min | Hybrid CUDA, scales with `cudaArchitectures` |
| `onnxruntime` (static CUDA) | ~30-60 min | Full static build with archive merging |
| `ort-wrapper-*` | ~1-2 min | Rust wrapper (after ONNX Runtime is cached) |

**Factors affecting build time:**
- **CUDA architectures**: Each sm_XX target adds compilation time. `"90;120"` is ~2x slower than `"90"` alone.
- **Nix cache**: Subsequent builds reuse cached artifacts. Only changed components rebuild.
- **NVCC threads**: Set to 1 (`onnxruntime_NVCC_THREADS`) to avoid OOM on systems with limited RAM.

## Build Variants

The flake supports two CUDA strategies via `mkOnnxruntime`:

1. **Hybrid (`hybridCuda=true`)**: `libonnxruntime.a` (static core) + CUDA providers as separate `.so` files loaded at runtime via dlopen. **This is the active architecture.** Uses upstream cmake's default behavior — CUDA provider and provider bridge are unconditionally shared modules on Linux.

2. **Static CUDA (default, static CUDA patches applied)**: `libonnxruntime.a` with CUDA providers compiled in. Hangs at runtime due to SIOF. Our patches (`cuda-static-provider.patch`, `providers-shared-static.patch`) force providers into static archives, overriding upstream defaults.

## Cargo Features

| Feature | Architecture | Status |
|---------|-------------|--------|
| `cuda` | Static ort, static CUDA | Hangs at runtime (SIOF) |
| `cuda-dlopen` | Static ort, CUDA via dlopen | Works (hybrid: static core + provider `.so` files) |
| `coreml` | Static ort, CoreML | Works (macOS) |

The `cuda-dlopen` feature enables `cuda` (for runtime provider detection) and uses the hybrid `build.rs` path — static `libonnxruntime.a` linked at compile time, CUDA providers loaded at runtime from `.so` files via ORT's `GetRuntimePath()`.

## Environment Variables

| Variable | Used By | Purpose |
|----------|---------|---------|
| `ORT_LIB_LOCATION` | all | Path to directory containing `libonnxruntime.a` (ort crate convention) |
| `ONNX_TEST_MODEL` | all | Path to test ONNX model |
| `LD_LIBRARY_PATH` | cuda* | Must include cuDNN path at runtime |

## Patches

All patches live in `patches/`. Grouped by purpose:

**Portability** (applied to all builds):
- `musl-execinfo.patch` — Stub `execinfo.h` for musl libc
- `musl-cstdint.patch` — Add missing `<cstdint>` include

**Static-only CUDA** (applied only when `useCuda && !buildShared && !hybridCuda`):
- `cuda-static-provider.patch` — Force CUDA provider from shared module to static archive
- `providers-shared-static.patch` — Force provider bridge from shared lib to static archive
- `static-init-bridge.patch` — Meyers singleton for `ProviderHostImpl`
- `static-init-cpu.patch` — Meyers singleton for `ProviderHostCPUImpl`
- `static-init-provider.patch` — Lazy init for `g_host`/`g_host_cpu`
- `static-init-common.patch` — Fallback to static host when `g_host` is NULL
- `static-datatype-loop.patch` — `ORT_STATIC_PROVIDERS` guard for `GetType<T>`
- `static-tensorshape-loop.patch` — `ORT_STATIC_PROVIDERS` guard for `TensorShape`

See [siof-patches.md](siof-patches.md) and [static-cuda-hang.md](static-cuda-hang.md) for details on the static-only patches.

## GPU Architecture

The `cudaArchitectures` variable in `flake.nix` controls GPU targets:
- Current: `"90;120"` (Hopper + Blackwell consumer)
- sm_90 = Hopper (H100, compute 9.0)
- sm_100 = Blackwell datacenter (B100/B200, compute 10.0)
- sm_120 = Blackwell consumer (RTX 50xx, compute 12.0)

If you see `cudaErrorSymbolNotFound` during CUDA inference, the GPU architecture isn't supported. Update `cudaArchitectures` in `flake.nix`.
