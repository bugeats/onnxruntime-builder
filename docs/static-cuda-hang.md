# Static CUDA Hang

The static CUDA build (`nix build .#` / `nix run .#ort-wrapper-cuda`) hangs during static initialization.

## Root Cause

Provider bridge (`provider_bridge_provider.cc`) creates infinite loops when statically linked — methods call through vtable indirection back to themselves. Guarded with `#ifndef ORT_STATIC_PROVIDERS`.

## Identifying Infinite Loops

In backtrace output: `ProviderHostImpl::SomeMethod__Thing()` calling `Thing::SomeMethod()` where both resolve to the same address.

Two patterns patched so far:
1. `GetType<T>()` → `Provider_GetHost()->DataTypeImpl__GetType_T()` → back to `GetType<T>()`
2. `TensorShape::Allocate()` → `g_host->TensorShape__Allocate()` → back to `TensorShape::Allocate()`

## Debug Tools

```bash
nix run .#debug-bt                # Auto-interrupt GDB backtrace (default 2s)
nix run .#debug-bt -- 5           # Custom delay
nix run .#debug-gdb               # Interactive GDB session
nix run .#build-cuda              # Build with visible logs
```

## Status

Two infinite loop patterns patched, rebuild needed to test. If still hangs, capture backtrace with `nix run .#debug-bt -- 3`.

## Fallback Approaches

If patching becomes untenable:
1. Disable provider bridge entirely (investigate `onnxruntime_USE_FULL_PROTOBUF` or similar flags)
2. Hybrid linking — keep provider bridge as shared lib, link everything else static
