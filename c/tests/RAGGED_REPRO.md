# Ragged attention — production-scale repro for PR #365

These tests reproduce the non-deterministic corruption found in the ragged
attention path added by PR #365 (`attention_absorb_ragged_kernel` +
`coli_cuda_attention_project_ragged`). They accompany the review comment at
<https://github.com/JustVugg/colibri/pull/365>.

## TL;DR

The shipped `tests/test_ragged_attention.cu` passes (`ragged_relative_rms=0`),
but it runs at toy dimensions (K=3, T=3, H=2, S=3) where the bug does not
trigger. At GLM-5.2's real MLA dimensions the ragged path intermittently
corrupts its output. The batch path (the existing, shipping kernel) is stable
under the identical harness, so the bug is specific to the ragged path.

## Build

Same toolchain as `make cuda-test` (nvcc + MSVC `cl.exe` on Windows; nvcc +
gcc/clang on Linux). From `c/`:

```bat
:: Windows (run from a shell with vcvars64 active, or call it first)
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
nvcc -O3 -std=c++17 -arch=native -Xcompiler=-W3 -DCOLI_CUDA_BUILDING_DLL ^
    -L"%CUDA_PATH%\lib\x64" -lcudart ^
    backend_cuda.cu tests\test_ragged_repro.cu -o test_ragged_repro.exe
```

```sh
# Linux
nvcc -O3 -std=c++17 -arch=native -Xcompiler=-Wall,-Wextra \
    backend_cuda.cu tests/test_ragged_repro.cu -o test_ragged_repro
```

Run a test several times in a row — the bug is intermittent (~1 in 5 runs).

## The tests

### `test_ragged_repro.cu` — the core repro

Runs a ragged S=4 call (cold `DeviceContext`), then pollutes the shared
scratch with S=1 batch calls, then runs the *identical* ragged S=4 call again.
Reports three rms values vs the per-sequence batch reference:

- `cold-vs-batchref` — ragged on a fresh `DeviceContext`
- `warm-vs-batchref` — ragged after the scratch was reused
- `cold-vs-warm` — the two ragged calls vs each other

On a healthy build all three are `0`. On this PR they intermittently read
`rms ≈ 0.01–0.03`. Run it 5–10×; a single run can be `0` by chance.

### `test_batch_stability.cu` — the control

Runs the existing **batch** kernel 8× with identical inputs and compares each
run to the first. This is the critical control: it confirms the non-determinism
is specific to the ragged path, not a harness/driver/scratch-reuse artifact.
On this PR the batch path is **0/8 divergent** (rock-stable), while the ragged
path intermittently corrupts.

### `test_ragged_attention_real.cu` — production-dimension A/B

The same A/B the shipped test does, but at real MLA dimensions
(K=512, Q=192, R=64, V=256, H=64, S=4, ragged lengths {8,17,31,40}), with a
third arm (ragged S=1 vs ragged S=4) to isolate multi-row batching.

## What the evidence points to

The ragged kernel itself is textually identical to the working batch kernel
and matches the batch reference bit-for-bit when it isn't corrupted. The
signature — intermittent, non-deterministic, no crash — points to a
**race / uninitialized-device-memory read** in the ragged path's use of the
shared `DeviceContext` scratch (`dc->ac` / `dc->al` / `dc->y`). `cudaMalloc`
scratch is uninitialized and `reserve()` is grow-only, so a buffer retained
from a prior (larger) geometry can be read before the current call overwrites
it. Suggested fix in `coli_cuda_attention_project_ragged`
(`backend_cuda.cu`): `cudaMemsetAsync` the reused device scratch to zero
before the kernel, or ensure the `[S,T,K]` latent/rope device uploads cover
the full padded region.
