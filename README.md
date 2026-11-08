# CEPARCO Integrating Project: Register-Width For K-Means Clustering With SIMD

**1st Term AY 2026-2027 | S01 Group 2**

> **Status:** Milestone #1 (project proposal). This README describes the plan. Sections marked `TODO` are filled in as the code and results come in.

---

# Project Proposal Google Drive Link
- **Video Pitch (5-8 min):** [Watch here](https://drive.google.com/file/d/17PanjDcpvq3jb04gxR7asMjCW3-3cH5n/view?usp=sharing)

---

## Abstract
This project investigates how SIMD register-width optimization techniques affect the performance of K-means clustering on an x86-64 CPU. It compares four implementations: scalar C as the baseline, compiler auto-vectorized C, an SSE XMM-register kernel (128-bit, four single-precision values per instruction), and an AVX2 YMM-register kernel (256-bit, eight values per instruction).

All versions use identical datasets, initial centroids, parameters, and empty-cluster handling rules to ensure a controlled comparison. Experiments vary the number of data points (`N`), the number of clusters (`K`), and the implementation to investigate how workload size and computational demand affect performance. The results will show the benefits and limitations of wider SIMD registers and loop optimizations for K-means clustering on the CPU.

## Introduction

K-means clustering sorts a set of points into `K` groups. Each group has a center called a **centroid**, and every point belongs to the group of its nearest centroid. The algorithm repeats two steps until the centroids stop moving:

1. **Assignment:** give every point the label of its nearest centroid.
2. **Update:** move each centroid to the average position of the points with its label.

The assignment step needs the distance from every point to every centroid (`N x K` distances per iteration). Each point's distance calculation is independent of the others, which makes it a good fit for **SIMD** (Single Instruction, Multiple Data), where one instruction works on several numbers at once.

The distance used is the squared Euclidean distance between a point `(x, y)` and a centroid `(cx, cy)`:

```text
d2 = (x - cx)^2 + (y - cy)^2
```

The square root is skipped because it never changes which centroid is the nearest. The project uses Lloyd's algorithm without algorithmic distance-pruning methods, so all versions perform the same `N x K` distance comparisons per iteration.

---

## Task

Implement k-means clustering on 2D points in three versions and compare them:

| # | Version | File | Points per step |
|---|---|---|---:|
| 1 | Sequential scalar C (baseline; compiler vectorization disabled) | `main.c` (`kmeansC`) | 1 |
| 2 | Compiler auto-vectorized C (AVX2 build) | C built with `/arch:AVX2` | Compiler-dependent |
| 3 | x86-64 SIMD using XMM registers (128-bit, AVX/VEX encoding) | `xmm.asm` (`assignXMM`) | 4 |
| 4 | x86-64 SIMD using YMM registers (256-bit, AVX2) | `ymm.asm` (`assignYMM`) | 8 |

The goal is to find out how register width and compiler vectorization affect run time, and how the speedup depends on the number of points (`N`) and clusters (`K`). The scalar C implementation is the baseline; the auto-vectorized C build provides a comparison against compiler-generated SIMD code.

Optional, only if time allows after the CUDA lecture: a CUDA (SIMT) version of the same step.

---

## Requirements

### C Program

The C program is responsible for:
- Collecting the inputs (or generating the dataset)
- Allocating memory
- Choosing the initial centroids
- Running the scalar C baseline and the auto-vectorized C version
- Calling the x86-64 assembly kernels
- Implementing centroid updates, convergence checks, and empty-cluster handling
- Calling the x86-64 assembly kernels
- Checking correctness and timing each stage
- Printing outputs and benchmark statistics
  
### x86-64 Assembly Kernels

The assembly kernels are responsible for the **assignment step**:
- Loading several points into a register
- Computing the squared distance to each centroid
- Keeping the smallest distance and its centroid number
- Storing the labels

### SIMD Requirement
- Packed SIMD registers: XMM (128-bit) and YMM (256-bit)
- Packed SIMD floating-point instructions (`vsubps`, `vmulps`, `vaddps`, `vcmpltps`, `vblendvps`)
- Instruction details are checked against the Intel Intrinsics Guide [4] and the Intel Software Developer's Manual [5]

### Processes in the Project

| Process | Language | Parallelized? |
| :--- | :--- | :-: |
| Data generation and input checking | C | No |
| Initial centroid selection | C | No |
| **Assignment step** (distance + nearest centroid) | C / ASM | **Yes** |
| Cluster counting (members per cluster) | C | No; performed as part of the update workflow |
| Centroid recomputation (divide sums by counts) | C | No |
| Convergence check | C | No |
| Empty-cluster detection and relocation | C | No; sequential and separately counted/timed |
| Timing and accuracy checking | C | No |

---

## Input Format

The first line has the number of points (`N`) and the number of clusters (`K`). The next `N` lines each have the `x` and `y` of one point.

```text
N, K
x, y
x, y
...
```

For benchmarking, the program generates synthetic Gaussian blobs using a fixed seed (`SEED`) so every implementation receives identical data. The initial centroids are the first `K` points in the array.

**Input rules:**
- `X[]` and `Y[]` must have the same length.
- `N >= K >= 1`.
- Every coordinate must be a finite numerical value.
- The same seed, data, starting centroids, and parameters must be used for all implementations.
  
**Internal inputs:**

| Name | Meaning |
| :--- | :--- |
| `X[]`, `Y[]` | float32 coordinates, stored in **two separate arrays** so a register can load several x values (or y values) in one go |
| `MAX_ITER` | Most iterations allowed |
| `TOL` | Convergence threshold on centroid movement |
| `SEED` | Fixed random seed so every version receives identical data and initial centroids |
| `EMPTY_POLICY` | Empty-cluster handling: Policy A (keep the previous centroid) or Policy B (relocate) |
| `FIXED_ITERS` | Number of iterations in timing mode; timing mode uses `TOL = 0` so every version performs the same work |

For correctness tests, the algorithm can run until convergence or `MAX_ITER`. For timing tests, use a fixed iteration count with `TOL = 0` so differences in convergence do not change the amount of work.

**Planned benchmark sizes:** `N = 2^16`, `2^20`, and `2^24`, with `K = 4, 8, 16, 32`. Add smaller sizes down to `N = 2^10` to exercise lower cache levels, and boundary sizes `N = 7, 1003, 1005` with representative `K` values to test leftover-point handling.

### Example Input

```text
6, 2
1.0, 1.0
1.5, 2.0
1.2, 0.8
8.0, 8.0
9.0, 9.5
8.5, 9.0
```

---

## Output Format

The program outputs:
- The label of every point, the number of members in each cluster, and the final centroids.
- Execution time per stage for each implementation: initial centroid selection, assignment with member counting, centroid recomputation, and full runtime.
- Similarity score against scalar C, the largest centroid difference, and inertia.
- Number of empty-cluster events and the extra time by Policy B.
- Benchmark statistics, speedup, efficiency, points processed per second, and measured memory bandwidth.


### Example Output

```text
Labels: 0 0 0 1 1 1
Cluster 0: 3 points, centroid (1.23, 1.27)
Cluster 1: 3 points, centroid (8.50, 8.83)
Similarity Score: 100%
Inertia: 
```

_(This example was worked out by hand and will be used as the first test case.)_

---

## Runtime Assessment

The run is timed in four parts:

| Part | What it covers | Where it runs |
| :--- | :--- | :--- |
| Initial centroid assessment | Picking the starting centroids | C |
| Clustering and member count | The assignment step, then counting the members of each cluster | ASM kernel (assignment) |
| Centroid recomputation | Summing the points of each cluster and dividing to get the new centroids | C (ASM if time permits) |
| Full runtime and accuracy check | Whole run, plus comparing the results with the C version | C |

---

## How to Compile

### Option A: Visual Studio (main)

1. Open `KMeans.sln`.
2. Choose **x64**, then **Debug** or **Release**.
3. Make sure the NASM custom build step (`C:\nasm\nasm.exe -f win64 <file>.asm`) is set for **both** Debug|x64 and Release|x64.
4. Build.

### Option B: Windows command line (NASM + GCC)

```bash
nasm -f win64 -o xmm.obj xmm.asm
nasm -f win64 -o ymm.obj ymm.asm
gcc -O2 -o kmeans.exe main.c xmm.obj ymm.obj
```

_TODO: confirm the exact commands once the code is in the repo._

**Requirements:** NASM, Visual Studio (or GCC), and a CPU that supports AVX2.

---

## How to Run

```bash
.\kmeans.exe
```

The program will:
1. Read the input (or generate the dataset)
2. Run the C, XMM and YMM versions on the same data
3. Print the labels, cluster counts and centroids
4. Print the correctness check against the C version
5. Run the timing benchmark for the chosen sizes of `N` and `K`

---

## Correctness Check

The C version is the answer key. The XMM and YMM results are compared against it:

- **Similarity Score** is the percentage of points that got the same label as in the C version. It should be **100%**.
- **Centroid difference** is the largest difference between any centroid and the C centroid. Small differences are allowed (relative `1e-4`), because adding numbers in a different order can change the last digits.
- **Inertia** is the sum of the squared distances of all points to their centroids. It is reported for each version.
- **Empty-cluster checks:** all implementations must apply the same policy and produce matching labels, counts, and centroids for the forced empty-cluster tests.

Labels should match exactly because every version computes the distance in the same order, uses separate multiply and add instructions (no FMA), and sends ties to the lower centroid number.

**Empty-cluster handling**

A cluster is empty when no point is assigned to it, making its mean undefined. Two policies are planned:

- **Policy A (default, keep):** leave the empty cluster's centroid at its previous position and count the event.
- **Policy B (relocate):** move an empty centroid to the point with the largest squared distance from its assigned centroid; ties go to the lowest point index. If multiple clusters are empty, use the next farthest points in order.

The assignment kernel stores each point's smallest squared distance, so Policy B can scan those distances without doing extra distance calculations. Empty-cluster detection checks the cluster member counts after each update. The scan runs only when an empty cluster exists.

Two datasets will force empty-cluster cases: (1) the five-point example described by Nielsen and Nock [9], using `N = 5`, `K = 3`, and initial centroids `(0.03, 0.92)`, `(0.66, 0.79)`, and `(0.6, 0.85)`; and (2) a duplicate-point case with `N = 8`, the first three points identical, and `K = 3`, so clusters 1 and 2 are empty because ties go to centroid 0.


**Result**

_TODO: example output, for example `6/6 labels match` and `Similarity Score: 100%`._

---

## Performance Results

Each size is run **30 times** (after one warm-up run) and the average execution time is recorded, for both **Debug** and **Release**. Only the clustering is timed. Timing uses `QueryPerformanceCounter`.

**Controlled benchmark settings:**

- Timing mode uses `FIXED_ITERS` with `TOL = 0`, ensuring every implementation performs identical work.
- Use identical data, seed, starting centroids, and parameters.
- Run a memory-copy test to estimate the practical memory-bandwidth ceiling.
- Sweep one factor at a time. Planned main sizes are `N = 2^16, 2^20, 2^24` and `K = 4, 8, 16, 32`; also test smaller `N` values down to `2^10` and boundary sizes `N = 7, 1003, 1005`.
  
### Release, whole run

| **N** | **K** | **Scalar C Avg Time** | **Auto-vectorized C Avg Time** | **XMM Avg Time** | **YMM Avg Time** | **Speedup Auto-C** | **Speedup XMM** | **Speedup YMM** |
|---|---:|---|---|---|---|---|---|---|
| 2^16 | 8 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 2^20 | 8 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |
| 2^24 | 8 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

### Release, assignment step only

| **N** | **K** | **Scalar C Avg Time** | **Auto-vectorized C Avg Time** | **XMM Avg Time** | **YMM Avg Time** | **Speedup Auto-C** | **Speedup XMM** | **Speedup YMM** |
|---|---:|---|---|---|---|---|---|---|
| 2^20 | 8 | TODO | TODO | TODO | TODO | TODO | TODO | TODO |

### Geometric Mean of Speedup

The **geometric mean** is a fair way to average speedups across different sizes.

| **Metric** | **Auto-vectorized C** | **XMM** | **YMM** |
|---|---|---|---|
| Assignment step geometric mean speedup | TODO | TODO | TODO |
| Whole run geometric mean speedup | TODO | TODO | TODO |
| Assignment step efficiency | N/A | TODO | TODO |
| Whole run efficiency | N/A | TODO | TODO |

_TODO: repeat the tables for Debug._

---

## Performance Analysis

_TODO. Questions to answer once the results are in:_
- Which version is fastest, and why?
- Is the assignment step **memory-bound or compute-bound**? (Memory-bound means the CPU waits for data from RAM, compute-bound means it is busy doing math. For large `N`, memory speed may become the limit and XMM and YMM may end up close together.)
- Why isn't the speedup the ideal 4x (XMM) and 8x (YMM)? (Memory speed, the leftover loop, the sequential parts, cache effects.)
- How does the speedup change with `K`? A bigger `K` means more math per point, so the SIMD advantage should grow.
- How much of the total time is the assignment step, and what does that say about the best possible total speedup (Amdahl's law)?
- Debug vs Release.

---

## Parallelization Plan

1. **Data layout:** keep the data in separate `X[]` and `Y[]` arrays so one load fills a register with neighboring values.
2. **Remove the square root:** compare squared distances only.
3. **Broadcast:** for each centroid, copy `cx` and `cy` into every slot of a register.
4. **Distance:** compute `dx = X - cx`, `dy = Y - cy`, then `d2 = dx*dx + dy*dy` for all points in the register at once, using separate multiply and add instructions (no fused multiply-add).
5. **Branch-free minimum:** keep the smallest `d2` so far and the centroid index that produced it, using a compare and blend instead of a conditional jump. Ties go to the lowest centroid index. Store each point's minimum squared distance for inertia and empty-cluster relocation.
6. **Centroid loop:** iterate over the `K` centroids sequentially, then store the labels for the group of points.
7. **Leftover points:** when `N` is not a multiple of 4 (XMM) or 8 (YMM), finish the remaining points with a scalar loop. For `N = 1003`, YMM processes 125 groups of 8 (1000 points), then handles the final 3 points in the scalar loop.
8. **Width variants:** XMM processes 4 points per step and YMM processes 8, using the same kernel design with a different register width, loop stride, and leftover size.
9. **Compare implementations:** build scalar C with compiler vectorization disabled, a separate auto-vectorized C version, and the hand-written XMM and YMM kernels.

**Why the update step is harder.** Many points add into the same cluster total, so they are not independent. One possible optimization is to process one cluster at a time: compare labels against that cluster's number, mask out other points, and sum the rest. This requires `K` passes and may only be worthwhile for small `K`; centroid accumulation is a secondary target if time permits.

**Empty-cluster policies.** Policy A keeps the previous centroid when a cluster is empty. Policy B relocates empty centroids to the farthest points from their assigned centroids, selecting points deterministically and using the lowest index to break ties. The same policy must be used for every implementation in a given comparison.

---

## Related Works

- **Lloyd's algorithm** is the standard K-means formulation [1] and is the algorithm implemented in this project.
- **Algorithmic accelerations:** Hamerly [6] uses distance bounds and the triangle inequality to skip distance calculations while preserving Lloyd's answer. scikit-learn [8] uses Lloyd's algorithm by default and also offers an Elkan variant, which reduces distance calculations at the cost of an additional `n_samples x n_clusters` array. This project retains full Lloyd's algorithm so every version computes the same `N x K` distances; only the cost of each distance calculation changes with the implementation.
- **SIMD implementations:** Al Hasib et al. [7] present a vectorized K-means algorithm using block data layout, an in-register argmin, SIMD compression, and OpenMP. Their study evaluates 128-bit SSE and 512-bit AVX-512, reporting a 4.5x improvement for their compressed SSE version on Haswell with one thread. It also reports that compiler auto-vectorization can remain below ideal speedups. The paper does not show its 256-bit AVX2 results, making direct measurement of XMM and YMM widths a useful focus for this project.
- **GPU implementations:** FAISS [2] and NVIDIA cuVS [3] provide GPU-based similarity search and clustering. A CUDA (SIMT) version is optional because this project targets CPU SIMD registers.
- **How ours differs:** this project does not claim a new K-means algorithm. Its contribution is a controlled measurement of register width for exact Lloyd's K-means using float32 data and a single thread. It compares scalar C, compiler auto-vectorized C, hand-written 128-bit XMM, and hand-written 256-bit YMM implementations with the same data layout and empty-cluster policy. It investigates where speedup comes from and what limits it, including memory bandwidth and sequential stages.


---

## Milestones

| Milestone | Week | What we hand in |
| :--- | :-: | :--- |
| #1 Project proposal | 6 | Proposal document and 5-8 minute video pitch, in this public repo |
| #2 Mid-project update demo | 10 | See the list below |
| #3 Complete submission and demo | 13 | Source code, discussion of the parallel algorithms, sequential vs parallel timing, optional paper |

**Mid-project (Week 10) target**
- [ ] Data generator and timing code completed, including fixed-iteration mode, aligned memory, and pinned-core timing.
- [ ] Sequential scalar C K-means completed with compiler vectorization disabled and verified on a small hand-worked example.
- [ ] Empty-cluster Policies A and B implemented in C and verified on two forced-empty test datasets.
- [ ] Memory-bandwidth copy test completed.
- [ ] XMM assignment kernel working and producing the same labels as the C version.
- [ ] YMM kernel started, with preliminary C versus XMM timing results.

**Final (Week 13) target**
- [ ] YMM assignment kernel completed and verified.
- [ ] Auto-vectorized C build compared against the hand-written kernels.
- [ ] Boundary sizes tested.
- [ ] Full benchmarks across planned `N` and `K` under the controlled protocol, with Release primary and Debug secondary.
- [ ] Empty-cluster results documented: event counts, cost of Policy B, and effect on inertia.
- [ ] Optional vectorized centroid update and optional CUDA version.
- [ ] GitHub repository with source code, discussion of the parallel algorithms, and sequential-versus-parallel execution-time comparison.
- [ ] Optional conference paper.

---

## Program Output Screenshot

_TODO: add after the program runs._

---

## Project Files

| File | Description |
| :--- | :--- |
| `CEPARCO_Group2_ProjectProposalRevised.pdf` | Revised project proposal document |
| `CEPARCO_Group2_ProjectProposal.pdf` | Project proposal document |
| `CEPARCO_Group2_ProposalSlides.pdf` | Slide deck for the video presentation |
| `README.md` | This file |

_TODO: update this table to match the real files._

---

## References

1. S. P. Lloyd, "Least squares quantization in PCM," *IEEE Trans. Information Theory*, vol. 28, no. 2, pp. 129-137, 1982. [PDF](https://www.stat.cmu.edu/~rnugent/PCMI2016/papers/LloydKMeans.pdf)
2. J. Johnson, M. Douze, and H. Jégou, "Billion-scale similarity search with GPUs," *IEEE Trans. Big Data*, vol. 7, no. 3, pp. 535-547, 2021. [arXiv](https://arxiv.org/abs/1702.08734)
3. NVIDIA cuVS, "K-Means" clustering guide. [Link](https://docs.nvidia.com/cuvs/user-guide/api-guides/clustering-guide/k-means)
4. Intel, *Intel Intrinsics Guide*. [Link](https://www.intel.com/content/www/us/en/docs/intrinsics-guide/index.html)
5. Intel, *Intel 64 and IA-32 Architectures Software Developer's Manual*. [Link](https://www.intel.com/content/www/us/en/content-details/825743/intel-64-and-ia-32-architectures-software-developer-s-manual-combined-volumes-1-2a-2b-2c-2d-3a-3b-3c-3d-and-4.html)
6. G. Hamerly, "Making k-means even faster," *Proc. 2010 SIAM International Conference on Data Mining (SDM)*, pp. 130-140, 2010. [PDF](https://cs.baylor.edu/~hamerly/papers/sdm_2010.pdf)
7. A. Al Hasib, J. M. Cebrian, and L. Natvig, "A vectorized k-means algorithm for compressed datasets: design and experimental analysis," *J. Supercomputing*, vol. 74, pp. 2705-2728, 2018. [DOI](https://doi.org/10.1007/s11227-018-2310-0)
8. scikit-learn developers, "sklearn.cluster.KMeans" documentation. [Link](https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html)
9. F. Nielsen and R. Nock, "Further heuristics for k-means: The merge-and-split heuristic and the (k,l)-means," arXiv:1406.6314, 2014. [PDF](https://arxiv.org/pdf/1406.6314)

---

## Submitted By

- [Daniel Jedrick C. Dabuit](https://github.com/danueli)
- [Ethan Daniel Magbatoc](https://github.com/ethanndaniel)
- [Jose Bryan Perez](https://github.com/JoseBryanPerez)
- [Tara Ysabel P. Uy](https://github.com/tartar121)
