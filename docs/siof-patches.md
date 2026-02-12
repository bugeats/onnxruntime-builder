# Static Initialization Order Fiasco (SIOF) Patches

This document explains the SIOF problem in ONNX Runtime static linking and how our patches solve it.

## Background

When statically linking ONNX Runtime, the C++ static initialization order between translation units is undefined. Multiple files have global static objects that depend on each other, causing crashes or hangs when initialization happens in the wrong order.

## Problem Files (4 total)

| File | Global Static | Problem |
|------|---------------|---------|
| `provider_bridge_ort.cc` | `ProviderHostImpl provider_host_` | Main ProviderHost implementation |
| `cpu_provider_shared.cc` | `ProviderHostCPUImpl provider_host_cpu_` | CPU-specific provider host |
| `provider_bridge_provider.cc` | `g_host`, `g_host_cpu` | Call `Provider_GetHost()` at init time |
| `common.cc` | `g_host` (atomic) | Provider_GetHost/SetHost implementations |

## Root Cause

```cpp
// In provider_bridge_provider.cc - runs at static init time
ProviderHost* g_host = Provider_GetHost();  // Returns NULL in static builds!
ProviderHostCPU& g_host_cpu = g_host->GetProviderHostCPU();  // CRASH or hang
```

In shared library builds, `Provider_SetHost()` is called via dlopen before any provider code runs. In static builds, no one calls `Provider_SetHost()`, so `Provider_GetHost()` returns NULL.

## Solution: Meyers Singletons

All patches convert global statics to **Meyers Singletons** - function-local statics that are:
- Thread-safe in C++11+ (compiler generates locks)
- Initialized on first use, not at static init time
- Break circular dependencies by lazy initialization

### Patch 1: static-init-bridge.patch

**File:** `onnxruntime/core/session/provider_bridge_ort.cc`

```cpp
// Before: } provider_host_;  // Global static!
// After:
static ProviderHostImpl& GetProviderHostImpl() {
  static ProviderHostImpl instance;  // Meyers Singleton
  return instance;
}
ProviderHost* GetStaticProviderHost() {
  return &GetProviderHostImpl();
}
```

### Patch 2: static-init-cpu.patch

**File:** `onnxruntime/core/providers/cpu/cpu_provider_shared.cc`

```cpp
// Before: ProviderHostCPUImpl provider_host_cpu_;
// After:
ProviderHostCPU& GetProviderHostCPU() {
  static ProviderHostCPUImpl provider_host_cpu_;  // Meyers Singleton
  return provider_host_cpu_;
}
```

### Patch 3: static-init-provider.patch

**File:** `onnxruntime/core/providers/shared_library/provider_bridge_provider.cc`

```cpp
// Before:
// ProviderHost* g_host = Provider_GetHost();
// ProviderHostCPU& g_host_cpu = g_host->GetProviderHostCPU();
// After:
static ProviderHost*& GetGHost() {
  static ProviderHost* host = Provider_GetHost();
  return host;
}
static ProviderHostCPU& GetGHostCPU() {
  static ProviderHostCPU& host_cpu = GetGHost()->GetProviderHostCPU();
  return host_cpu;
}
#define g_host (GetGHost())      // Macro for compatibility
#define g_host_cpu (GetGHostCPU())
```

### Patch 4: static-init-common.patch

**File:** `onnxruntime/core/providers/shared/common.cc`

```cpp
// Adds fallback to GetStaticProviderHost() when g_host is NULL
// Uses thread_local guard to prevent recursive initialization
onnxruntime::ProviderHost* Provider_GetHost() {
  auto* host = g_host.load(std::memory_order_acquire);
  if (host != nullptr) return host;
  if (g_in_init) return onnxruntime::GetStaticProviderHost();  // Recursion guard
  g_in_init = true;
  host = onnxruntime::GetStaticProviderHost();
  g_in_init = false;
  return host;
}
```

## Initialization Flow (Static Linking)

1. Some code calls `Provider_GetHost()` during static init
2. `g_host` is NULL (no one called `Provider_SetHost`)
3. Falls through to `GetStaticProviderHost()`
4. This calls `GetProviderHostImpl()` which constructs the singleton
5. `ProviderHostImpl` constructor may call `GetProviderHostCPU()`
6. This constructs its singleton, returns reference
7. All singletons now initialized, program continues

**Key Insight:** The `thread_local g_in_init` flag prevents infinite recursion if `ProviderHostImpl`'s constructor calls `Provider_GetHost()` again.

## Status

These patches are verified working for CPU-only builds. The CUDA build has a separate hang issue unrelated to these SIOF fixes (see main CLAUDE.md).
