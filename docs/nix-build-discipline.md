# Nix Build Discipline

## Pre-flight Check

**ALWAYS check for active builds first:**
```bash
pgrep -f "onnxruntime\|nvcc" && echo "⚠️  BUILD RUNNING" || echo "✓ Clear"
```

## Running Builds

- Use `nix run .#build-cuda` for visible progress on long builds
- Use `nix run` (CPU-only default) for fast iteration (~1-2 min)
- Use `cuda-dyn` to validate CUDA integration while static build is in progress

**Why this matters**: CUDA builds take 30-60 minutes and saturate CPU. The Nix daemon runs builds independently — `nix build` can return while compilation continues. Starting a second build creates resource contention and confusion.

## Nix Command Output

**Always capture stderr**: Many nix commands output progress/errors to stderr only. Use `2>&1` to capture both streams:

```bash
# BAD: may show empty output
nix path-info .#foo

# GOOD: captures all output
nix path-info .#foo 2>&1
nix build .#foo --print-build-logs 2>&1
```

**Check build progress**:

```bash
# Reliable status check
nix path-info .#ort-wrapper-cuda 2>/dev/null && echo "DONE" || echo "NOT DONE"

# Check if actively compiling
pgrep -f "onnxruntime" && echo "Build running" || echo "Not running"

# Count active compile jobs
ps aux | grep -E "nvcc|g\+\+.*onnxruntime" | grep -v grep | wc -l
```
