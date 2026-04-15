# Rocprofiler On-Demand Profiling Repro

Demonstrates that unpatched Tracy crashes when a profiler connects to a
HIP application built with `TRACY_ON_DEMAND` and `TRACY_ROCPROF`.

## Root cause

Two bugs in `TracyRocprof.cpp` break on-demand profiling:

1. **GpuNewContext not deferred.** `gpu_context_allocate()` writes a
   `GpuNewContext` queue item but does not call `DeferItem()`. When a
   Tracy client connects late, the context creation message is never
   replayed. The server then receives `GpuZoneBegin` events for a
   context it has never seen, triggering:

       Assertion `ctx' failed in ProcessGpuZoneBeginImplCommon

2. **Kernel symbols dropped before init.** The `data->init` guard at the
   top of `tool_callback_tracing_callback()` blocks all callbacks before
   the GPU context is allocated. Kernel symbol registrations
   (`CODE_OBJECT_DEVICE_KERNEL_SYMBOL_REGISTER`) happen at HIP init
   time — before `data->init` is true — so they are silently dropped.
   Even if the crash is worked around, kernel names would be missing.

## Prerequisites

- AMD GPU with working ROCm driver
- `librocprofiler-sdk.so` available (typically at `/opt/rocm/lib/`)
- `/opt/rocm/bin/hipcc`

## Build and run

```bash
make
./repro &
tracy-capture -o repro.tracy -s 5
```

## What to expect

| Tracy version | Result |
|---|---|
| Unpatched | `tracy-capture` crashes: `Assertion 'ctx' failed` |
| Patched   | Capture succeeds with ~50 GPU zones (`vectorAdd`) with kernel names |
