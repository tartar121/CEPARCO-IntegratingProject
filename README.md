# CEPARCO Integrating Project: K-Means Clustering with SIMD

**1st Term AY 2026-2027 | S01 Group 2**

> **Status:** Milestone #1 (project proposal). This README describes the plan. Sections marked `TODO` are filled in as the code and results come in.

---

# Project Proposal Google Drive Link
- **Video Pitch (5-8 min):** [Watch here](https://drive.google.com/file/d/17PanjDcpvq3jb04gxR7asMjCW3-3cH5n/view?usp=sharing)

---


## Introduction

K-means clustering sorts a set of points into `K` groups. Each group has a center called a **centroid**, and every point belongs to the group of its nearest centroid. The algorithm repeats two steps until the centroids stop moving:

1. **Assignment:** give every point the label of its nearest centroid.
2. **Update:** move each centroid to the average position of the points with its label.

The assignment step needs the distance from every point to every centroid (`N x K` distances per iteration). Each point's distance calculation is independent of the others, which makes it a good fit for **SIMD** (Single Instruction, Multiple Data), where one instruction works on several numbers at once.

The distance used is the squared Euclidean distance between a point `(x, y)` and a centroid `(cx, cy)`:

```text
d2 = (x - cx)^2 + (y - cy)^2
```

The square root is skipped because it never changes which centroid is the nearest.

---

## Task

Implement k-means clustering on 2D points in three versions and compare them:

| # | Version | File | Points per step |
| :-: | :--- | :--- | :-: |
| 1 | Sequential C (baseline) | `main.c` (`kmeansC`) | 1 |
| 2 | x86-64 SIMD AVX2 using **XMM** registers (128-bit) | `xmm.asm` (`assignXMM`) | 4 |
| 3 | x86-64 SIMD AVX2 using **YMM** registers (256-bit) | `ymm.asm` (`assignYMM`) | 8 |

The goal is to find out which register width gives the best run time, and how much the speedup depends on the number of points (`N`) and clusters (`K`).

Optional, only if time allows after the CUDA lecture: a CUDA (SIMT) version of the same step.

---

## Requirements

### C Program

The C program is responsible for:
- Collecting the inputs (or generating the dataset)
- Allocating memory
- Choosing the initial centroids
- Running the sequential baseline
- Calling the x86-64 assembly kernels
- Checking correctness and timing each stage
- Printing the outputs

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
| Cluster counting (members per cluster) | C | No (done during the update step) |
| Centroid recomputation (sums, then division) | C | Only if time permits |
| Convergence check | C | No |
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

For the benchmark, the program can instead **generate** `N` points as synthetic Gaussian blobs, seeded with `srand(67)` so every run and every version gets the same data.

**Input rules:**
- `X[]` and `Y[]` must have the same length.
- `N >= K >= 1`.
- Every coordinate must be a valid number.

**Internal inputs:**

| Name | Meaning |
| :--- | :--- |
| `X[]`, `Y[]` | float32 coordinates, stored in **two separate arrays** so a register can load several x values (or y values) in one go |
| `MAX_ITER` | Most iterations allowed |
| `TOL` | Stop when no centroid moves more than this |

The starting centroids are the first `K` points, so every version starts from the same place.

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

The program outputs the label of each point, the number of points per cluster, and the final centroids.

### Example Output

```text
Labels: 0 0 0 1 1 1
Cluster 0: 3 points, centroid (1.23, 1.27)
Cluster 1: 3 points, centroid (8.50, 8.83)
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

Labels should match exactly because every version computes the distance in the same order, uses separate multiply and add instructions (no FMA), and sends ties to the lower centroid number.

**Result**

_TODO: example output, for example `6/6 labels match` and `Similarity Score: 100%`._

---

## Performance Results

Each size is run **30 times** (after one warm-up run) and the average execution time is recorded, for both **Debug** and **Release**. Only the clustering is timed. Timing uses `QueryPerformanceCounter`.

**Sizes to test** _(TODO: finalize after the first timing runs)_: `N` = 2^16, 2^20, 2^24 and `K` = 4, 8, 16, 32. Boundary sizes: `N` = 7, 1003, 1005 with `K = 4`.

### Release, whole run

| N | K | C Avg Time | XMM Avg Time | YMM Avg Time | Speedup XMM | Speedup YMM |
| :-: | :-: | -------: | -------: | -------: | -------: | -------: |
| 2^16 | 8 | TODO | TODO | TODO | TODO | TODO |
| 2^20 | 8 | TODO | TODO | TODO | TODO | TODO |
| 2^24 | 8 | TODO | TODO | TODO | TODO | TODO |

### Release, assignment step only

| N | K | C Avg Time | XMM Avg Time | YMM Avg Time | Speedup XMM | Speedup YMM |
| :-: | :-: | -------: | -------: | -------: | -------: | -------: |
| 2^20 | 8 | TODO | TODO | TODO | TODO | TODO |

### Geometric Mean of Speedup

The **geometric mean** is a fair way to average speedups across different sizes.

| | XMM | YMM |
| :--- | :-: | :-: |
| Assignment step | TODO | TODO |
| Whole run | TODO | TODO |

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

1. Keep the data in separate `X[]` and `Y[]` arrays, so one load fills a register with neighboring values.
2. For each centroid, copy `cx` and `cy` into every slot of a register (**broadcast**).
3. Compute `dx = X - cx`, `dy = Y - cy`, then `d2 = dx*dx + dy*dy` for all points in the register at once.
4. For every slot, keep the smallest `d2` so far and the centroid that gave it, using a compare and a **blend** (pick one value or the other per slot), so there is no jump.
5. Loop over the `K` centroids one after another, then store the labels.
6. If `N` is not a multiple of 4 or 8, finish the last points with a one-at-a-time loop. Example: for `N = 1003`, YMM does 125 steps of 8 = 1000 points, then the leftover loop does the last 3.

**Why the update step is harder.** Many points add into the same cluster total, so they are not independent. One idea is to go cluster by cluster: compare the labels against that cluster's number, zero out the other points, and add up the rest. This takes `K` passes, so it only pays off for small `K`.

---

## Related Works

- **Lloyd's algorithm** is the standard k-means procedure (Lloyd, 1982) [1]: assign each point to its nearest centroid, then move each centroid to the average of its points.
- **FAISS** (Johnson et al.) is a library for similarity search and clustering of dense vectors. Its paper notes that GPUs are well suited to data-parallel tasks such as distance computation [2].
- **NVIDIA cuVS** has a GPU-accelerated k-means that follows the same two steps and reports centroids, labels, inertia and the iteration count. Its guide says the expensive part is repeatedly comparing many rows against many centroids, which is the step we target [3].
- **How ours differs:** those use GPUs or tuned libraries. We write the SIMD kernels **by hand in x86-64 assembly** and compare 128-bit and 256-bit registers against a plain C baseline on the same CPU, to show where the speedup comes from and where it stops.

---

## Milestones

| Milestone | Week | What we hand in |
| :--- | :-: | :--- |
| #1 Project proposal | 6 | Proposal document and 5-8 minute video pitch, in this public repo |
| #2 Mid-project update demo | 10 | See the list below |
| #3 Complete submission and demo | 13 | Source code, discussion of the parallel algorithms, sequential vs parallel timing, optional paper |

**Mid-project (Week 10) target**
- [ ] Data generator and timing code
- [ ] Sequential C k-means, checked against the hand-worked example
- [ ] XMM assignment kernel working and matching the C labels
- [ ] YMM kernel started, with a first timing of C vs XMM

**Final (Week 13) target**
- [ ] YMM assignment kernel matching the C labels
- [ ] Boundary sizes tested
- [ ] Timing across all `N` and `K`, in Debug and Release
- [ ] Optional: vectorized centroid update, CUDA version, conference paper

---

## Program Output Screenshot

_TODO: add after the program runs._

---

## Project Files

| File | Description |
| :--- | :--- |
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

---

## Submitted By

- [Daniel Jedrick C. Dabuit](https://github.com/danueli)
- [Ethan Daniel Magbatoc](https://github.com/ethanndaniel)
- [Jose Bryan Perez](https://github.com/JoseBryanPerez)
- [Tara Ysabel P. Uy](https://github.com/tartar121)
