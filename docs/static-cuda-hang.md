# Static CUDA Hang

The static CUDA build hangs during static initialization. The static CUDA `onnxruntime` package builds, but any binary linking it statically with CUDA providers hangs at startup.

## Root Cause

Provider bridge (`provider_bridge_provider.cc`) creates infinite loops when statically linked — methods call through vtable indirection back to themselves. Guarded with `#ifndef ORT_STATIC_PROVIDERS`.

## Identifying Infinite Loops

In backtrace output: `ProviderHostImpl::SomeMethod__Thing()` calling `Thing::SomeMethod()` where both resolve to the same address.

Two patterns patched so far:
1. `GetType<T>()` → `Provider_GetHost()->DataTypeImpl__GetType_T()` → back to `GetType<T>()`
2. `TensorShape::Allocate()` → `g_host->TensorShape__Allocate()` → back to `TensorShape::Allocate()`

## Status

Two infinite loop patterns patched (`GetType<T>`, `TensorShape::Allocate`). The static CUDA build compiles but hangs at runtime. The hybrid build (`hybridCuda=true`) avoids this entirely and is the active architecture.

## Alternative Approaches

1. **Hybrid linking (validated, active path)** — static `libonnxruntime.a` (CPU core) + CUDA providers as separate `.so` files loaded via dlopen. Avoids SIOF entirely by keeping CUDA out of the static archive. This is upstream cmake's default behavior — `cuda-static-provider.patch` and `providers-shared-static.patch` are what override it. See CLAUDE.md implementation plan.
2. **Disable provider bridge** — investigate `onnxruntime_USE_FULL_PROTOBUF` or similar cmake flags to eliminate the bridge code that causes infinite loops.
