# CUDA molecular dynamics from scratch

A one-dimensional Lennard-Jones simulator written from scratch in C++ and CUDA, benchmarked on a Kaggle T4. No library does any of the computation; OpenMM is used only as an external check.

| Notebook | Content |
|---|---|
| `cuda-md-5000x-speedup-with-cell-lists.ipynb` | Part 1: CPU reference, a first CUDA kernel, fewer divisions, shared-memory tiling, cell lists (about 5,000x faster at N = 65,536) and the energy drift caused by a hard cutoff |
| `cuda-md-fp32-vs-fp64-precision.ipynb` | Part 2: why full FP32 fails, mixed precision, a GPU-resident time loop, validation against OpenMM |

`validation/` holds the scripts and results of the OpenMM comparison.

## Running

Open a notebook in a Kaggle session with a GPU and run the cells in order. Each code cell writes a source file with `%%writefile`, then compiles it with `g++` or `nvcc` and runs it. The outputs stored in the notebooks are from a Kaggle T4 session.

## Main results (N = 65,536, 1D chain, T4)

- Naive FP64 force kernel: 1514 ms. One reciprocal instead of three divisions: 1.7x faster. Shared-memory tiling: about 1% faster. Cell list with a 2.5 sigma cutoff: 0.18 ms.
- A hard energy cutoff makes the total energy drift by -173.6 over t = 1. The drift does not depend on the time step; a shifted-force potential reduces it to -6.5e-4 (relative error about 3e-8).
- Storing absolute positions in `float` freezes the particles (float spacing near 1e5 is 0.0078, steps are about 1e-6). Keeping state in FP64 and doing only the pair-force arithmetic in FP32 gives an energy error 10 to 15% above the FP64 one and a 4x faster force kernel.
- With the cell list on the host, the force kernel is 4% of a step and the mixed-precision loop is 1.18x faster than FP64. With everything on the GPU it is 2.06x faster, and the whole loop is 13x to 23x faster than the host-driven one.
- The FP64 code agrees with OpenMM to about 1e-14 in forces and 5e-13 in energy over 30,000 steps (N = 4096).

## Limits

One dimension with about two neighbours per particle, a single GPU, no profiler data (Nsight Compute counters are blocked on Kaggle), and the GPU-resident cell list relies on particles never overtaking each other. See the end of each notebook for details.
