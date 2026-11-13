#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <stdint.h>
#include <windows.h>
#include <string.h>
#include "kmeans.h"

// gaussian generator parameters
#define PI 3.14159265358979323846
#define BLOB_RADIUS 100.0
#define BLOB_STDDEV 10.0


// asm test cases
void xmmkernel(void);
void ymmkernel(void);


int validate_data(const KMeansData* data); //validate input data
int allocate_arrays(KMeansData* data); //validate X[] and Y[]
void free_arrays(KMeansData* data); //free X[] and Y[]

double random_float(void); //uniform (0,1) from xorshift32
void rng_seed(unsigned int seed); //seeds our own PRNG (pseudo-random number generator)
void random_pair(double* num1, double* num2); //box-muller transform generates the gaussian pair
void generate_data(KMeansData* data); //guassian blob coords

typedef struct {
    double mean;
    double median;
    double stddev;
    double min;
} TimingStats;

void initialize_centroids(const KMeansData* data, float* centroid_X, float* centroid_Y); //centroids for X and Y
void timer_init(void); //caches the QPC (QueryPerformanceCounter) frequency once
int benchmark_kmeans(const KMeansData* data, AssignFn assign,
    float* centroid_X, float* centroid_Y, int* labels, float* min_d2,
    ClusterSums* sums, int runs, TimingStats* stats); //1 warm-up + `runs` timed runs
int measure_bandwidth(size_t bytes, int runs, double* best_gbps, double* mean_gbps); //copy test (GB/s)

DWORD_PTR pin_core(DWORD core); //pinned core

//FOR TESTING
//known-answer tests for the C baseline
static void run_tests(void) {
    printf("--------------------------------------\n");
    printf("TESTS\n");

    //Test 1: README hand-worked example with the real rule (first K points = starting centroids)
    {
        float X[] = { 1.0f, 1.5f, 1.2f, 8.0f, 9.0f, 8.5f };
        float Y[] = { 1.0f, 2.0f, 0.8f, 8.0f, 9.5f, 9.0f };
        float cx[] = { X[0], X[1] };
        float cy[] = { Y[0], Y[1] };
        int labels[6];
        float d2[6];
        KMeansData d = { 0 };
        d.X = X; d.Y = Y; d.N = 6; d.K = 2;
        d.MAX_ITER = 100; d.TOL = 0.001f; d.FIXED_ITERS = 0;

        ClusterSums sums;
        KMeansResult r;
        cluster_sums_init(d.K, &sums);
        int run_ok = kmeans(&d, assign_c, cx, cy, labels, d2, &sums, &r, d.EMPTY_POLICY);
        compute_inertia(&d, assign_c, cx, cy, labels, d2, &r.inertia);
        cluster_sums_free(&sums);

        int ok = run_ok && r.converged && r.iterations == 3 && r.empty_events == 0 &&
            labels[0] == 0 && labels[1] == 0 && labels[2] == 0 &&
            labels[3] == 1 && labels[4] == 1 && labels[5] == 1 &&
            fabsf(cx[0] - 1.2333f) < 1e-3f && fabsf(cy[0] - 1.2667f) < 1e-3f &&
            fabsf(cx[1] - 8.5f) < 1e-3f && fabsf(cy[1] - 8.8333f) < 1e-3f &&
            fabs(r.inertia - 2.62) < 1e-3;
        printf("Test 1 (hand-worked example, converges): %s\n", ok ? "PASS" : "FAIL");
    }

    //Test 2: first 3 points identical, K = 3, one iteration -> clusters 1 and 2 stay empty
    {
        float X[] = { 1, 1, 1, 5, 6, 7, 8, 9 };
        float Y[] = { 1, 1, 1, 5, 6, 7, 8, 9 };
        float cx[] = { 1, 1, 1 };
        float cy[] = { 1, 1, 1 };
        int labels[8];
        float d2[8];
        KMeansData d = { 0 };
        d.X = X; d.Y = Y; d.N = 8; d.K = 3;
        d.MAX_ITER = 100; d.TOL = 0.0f; d.FIXED_ITERS = 1;

        ClusterSums sums;
        KMeansResult r;
        cluster_sums_init(d.K, &sums);
        int run_ok = kmeans(&d, assign_c, cx, cy, labels, d2, &sums, &r, d.EMPTY_POLICY);
        cluster_sums_free(&sums);

        int all_zero = 1;
        for (int i = 0; i < 8; i++) if (labels[i] != 0) all_zero = 0;

        //clusters 1 and 2 are empty and keep their old position (Policy A)
        int ok = run_ok && all_zero && r.iterations == 1 && r.empty_events == 2 &&
            cx[1] == 1.0f && cx[2] == 1.0f;
        printf("Test 2 (duplicate points / empty clusters): %s\n", ok ? "PASS" : "FAIL");
    }

    //Test 3: fixed mode runs exactly FIXED_ITERS iterations and never reports converged
    {
        float X[] = { 1.0f, 1.5f, 1.2f, 8.0f, 9.0f, 8.5f };
        float Y[] = { 1.0f, 2.0f, 0.8f, 8.0f, 9.5f, 9.0f };
        float cx[] = { X[0], X[1] };
        float cy[] = { Y[0], Y[1] };
        int labels[6];
        float d2[6];
        KMeansData d = { 0 };
        d.X = X; d.Y = Y; d.N = 6; d.K = 2;
        d.MAX_ITER = 100; d.TOL = 0.0f; d.FIXED_ITERS = 10;

        ClusterSums sums;
        KMeansResult r;
        cluster_sums_init(d.K, &sums);
        int run_ok = kmeans(&d, assign_c, cx, cy, labels, d2, &sums, &r, d.EMPTY_POLICY);
        cluster_sums_free(&sums);

        int ok = run_ok && r.iterations == 10 && r.converged == 0;
        printf("Test 3 (fixed iterations): %s\n", ok ? "PASS" : "FAIL");
    }
    printf("--------------------------------------\n\n");
}

int main(void) {
    run_tests(); //run small tests to verify correctness

    //fill with temp values for now
    KMeansData data = { 0 };

    data.X = NULL;
    data.Y = NULL;

    data.N = 1024;
    data.K = 4;

    data.MAX_ITER = 100;
    data.TOL = 0.001f;

    data.SEED = 67;
    rng_seed(data.SEED);
    timer_init();

    data.EMPTY_POLICY = EMPTY_POLICY_A;
    data.FIXED_ITERS = 0;


    //validate data
    printf("--------------------------------------\n");
    printf("1. Input Validation\n");
    if (!validate_data(&data)) {
        return 1;
    }
    printf("Input is valid.\n");
    printf("--------------------------------------\n\n");


    //allocate arrays
    printf("--------------------------------------\n");
    printf("2. X[] and Y[]\n");

    if (!allocate_arrays(&data)) {
        return 1;
    }

    printf("Points: %d\n", data.N);
    printf("X Address: %p\n", (void*)data.X);
    printf("Y Address: %p\n", (void*)data.Y);
    printf("--------------------------------------\n\n");


    //generate gaussian blobs and centroids
    printf("--------------------------------------\n");
    printf("3. Data Generation and Centroids\n");

    generate_data(&data);
    float* centroid_X = (float*)malloc(data.K * sizeof(float));
    float* centroid_Y = (float*)malloc(data.K * sizeof(float));

    if (centroid_X == NULL || centroid_Y == NULL) {
        printf("Centroid allocation failed.\n");

        free(centroid_X);
        free(centroid_Y);
        free_arrays(&data);

        return 1;
    }

    initialize_centroids(&data, centroid_X, centroid_Y);

    //test 5 points
    printf("\nGenerated Points:\n");

    for (int i = 0; i < 5; i++) {
        printf("Point %d: X = %.2f, Y = %.2f\n",
            i, data.X[i], data.Y[i]);
    }

    //initial centroids
    printf("\nInitial Centroids:\n");

    for (int i = 0; i < data.K; i++) {
        printf("Centroid %d: X = %.2f, Y = %.2f\n",
            i, centroid_X[i], centroid_Y[i]);
    }
    printf("--------------------------------------\n\n");


    printf("--------------------------------------\n");

    //pin thread to core 0
    printf("5. Pinning thread to CPU Core 0\n");
    DWORD_PTR old_affinity = pin_core(0);

    if (old_affinity == 0) {
        printf("Failed to pin.\n");
        return 1;
    }
    printf("Successfully pinned.\n");
    printf("--------------------------------------\n\n");


    //allocate labels and smallest distances (32-byte aligned for the kernels)
    int* labels = (int*)_aligned_malloc(data.N * sizeof(int), 32);
    float* min_d2 = (float*)_aligned_malloc(data.N * sizeof(float), 32);
    ClusterSums sums = { 0 };
    KMeansResult result = { 0 };
    int exit_code = 1;

    if (labels == NULL || min_d2 == NULL) {
        printf("Label allocation failed.\n");
        goto cleanup;
    }

    if (!cluster_sums_init(data.K, &sums)) {
        printf("Cluster sums allocation failed.\n");
        goto cleanup;
    }

    //k-means clustering
    printf("--------------------------------------\n");
    printf("6. Sequential C K-Means\n");

    if (!kmeans(&data, assign_c, centroid_X, centroid_Y, labels, min_d2, &sums, &result, data.EMPTY_POLICY)) {
        printf("K-means failed.\n");
        goto cleanup;
    }

    //inertia is computed outside the timed region
    if (!compute_inertia(&data, assign_c, centroid_X, centroid_Y, labels, min_d2, &result.inertia)) {
        printf("Inertia failed.\n");
        goto cleanup;
    }

    printf("Iterations: %d\n", result.iterations);
    printf("Converged: %d\n", result.converged);
    printf("Empty-cluster events: %d\n", result.empty_events);
    printf("Inertia: %.4f\n", result.inertia);

    //print first 10 labels
    for (int i = 0; i < 10 && i < data.N; i++) {
        printf("Point %d : Cluster %d\n", i, labels[i]);
    }
    printf("--------------------------------------\n\n");

    //final centroids
    printf("--------------------------------------\n");
    printf("7. Final Centroids\n");

    for (int i = 0; i < data.K; i++) {
        printf("Centroid %d: X = %.2f, Y = %.2f\n",
            i, centroid_X[i], centroid_Y[i]);
    }
    printf("--------------------------------------\n\n");


    //timing is fixed-iteration mode (TOL = 0) so every version does identical work
    {
        KMeansData timing_data = data;
        timing_data.FIXED_ITERS = 100;
        timing_data.TOL = 0.0f;

        TimingStats stats;

        printf("--------------------------------------\n");
        printf("8. Timing (Scalar C, %d iterations per run, 30 runs)\n", timing_data.FIXED_ITERS);

        if (!benchmark_kmeans(&timing_data, assign_c, centroid_X, centroid_Y,
            labels, min_d2, &sums, 30, &stats)) {
            printf("Benchmark failed.\n");
            goto cleanup;
        }

        printf("Mean:   %.3f ms\n", stats.mean);
        printf("Median: %.3f ms\n", stats.median);
        printf("StdDev: %.3f ms\n", stats.stddev);
        printf("Min:    %.3f ms\n", stats.min);
        printf("Per iteration (mean): %.4f ms\n", stats.mean / timing_data.FIXED_ITERS);
        printf("--------------------------------------\n\n");
    }

    //memory bandwidth ceiling (single thread, copy)
    {
        double best_gbps, mean_gbps;

        printf("--------------------------------------\n");
        printf("9. Memory Bandwidth (copy, 256 MB, 10 runs)\n");

        if (!measure_bandwidth((size_t)256 << 20, 10, &best_gbps, &mean_gbps)) {
            printf("Bandwidth test failed.\n");
            goto cleanup;
        }

        printf("Best: %.2f GB/s\n", best_gbps);
        printf("Mean: %.2f GB/s\n", mean_gbps);
        printf("--------------------------------------\n\n");
    }

    // Test asm 

    // XMM vs C check (assign_xmm / xmm_vs_c_check live in kmeans.c)
    xmm_vs_c_check(&data, labels, centroid_X, centroid_Y, &result, &sums);

    // YMM vs C check (assign_ymm / ymm_vs_c_check live in kmeans.c)
    ymm_vs_c_check(&data, labels, centroid_X, centroid_Y, &result, &sums);

    // xmmkernel();
    // ymmkernel();

    exit_code = 0;

cleanup:
    cluster_sums_free(&sums);
    _aligned_free(labels);
    _aligned_free(min_d2);
    free(centroid_X);
    free(centroid_Y);
    free_arrays(&data);

    SetThreadAffinityMask(GetCurrentThread(), old_affinity); //restore old cpu affinity

    return exit_code;
}

// validate internal inputs
int validate_data(const KMeansData* data)
{
    //check if data exists
    if (data == NULL) {
        printf("Invalid data structure.\n");
        return 0;
    }

    //verify N and K
    if (data->N < 1 || data->K < 1) {
        printf("N and K must be positive.\n");
        return 0;
    }

    if (data->N < data->K) {
        printf("N must be greater than or equal to K.\n");
        return 0;
    }

    //verify iterations
    if (data->MAX_ITER < 1) {
        printf("MAX_ITER must be positive.\n");
        return 0;
    }

    if (data->FIXED_ITERS < 0) {
        printf("FIXED_ITERS cannot be negative.\n");
        return 0;
    }

    //verify tolerance
    if (!isfinite(data->TOL) || data->TOL < 0.0f) {
        printf("TOL must be a finite non-negative value.\n");
        return 0;
    }

    //check policy
    if (data->EMPTY_POLICY != EMPTY_POLICY_A && data->EMPTY_POLICY != EMPTY_POLICY_B) {
        printf("Invalid empty cluster policy.\n");
        return 0;
    }

    return 1;
}


//allocate X[] and Y[]
int allocate_arrays(KMeansData* data) {
    size_t bytes;

    if (data == NULL || data->N < 1) {
        return 0;
    }

    //get and verify allocation size
    bytes = (size_t)data->N * sizeof(float);

    if (bytes / sizeof(float) != (size_t)data->N) {
        printf("Array size is too large.\n");
        return 0;
    }

    //32 byte aligned
    data->X = (float*)_aligned_malloc(bytes, 32);
    data->Y = (float*)_aligned_malloc(bytes, 32);

    //verify allocation
    if (data->X == NULL || data->Y == NULL) {
        printf("Memory allocation failed.\n");

        _aligned_free(data->X);
        _aligned_free(data->Y);

        data->X = NULL;
        data->Y = NULL;

        return 0;
    }

    return 1;
}

//free X[] and Y[]
void free_arrays(KMeansData* data) {

    _aligned_free(data->X);
    _aligned_free(data->Y);

    data->X = NULL;
    data->Y = NULL;
}

//xorshift32 PRNG (Marsaglia, 2003)
//same seed gives the same data on any compiler
static unsigned int rng_state = 1;

//uniform in (0, 1), never exactly 0 or 1 (safe for log in box-muller)
double random_float(void) {
    unsigned int x = rng_state;
    x ^= x << 13;
    x ^= x >> 17;
    x ^= x << 5;
    rng_state = x;
    return ((double)x + 0.5) / 4294967296.0;
}

void rng_seed(unsigned int seed) {
    rng_state = seed * 2654435761u; //spread small seeds across the 32 bits
    if (rng_state == 0) rng_state = 1;//state must never be 0
    for (int i = 0; i < 16; i++) random_float(); //discard the first few outputs
}

//normally distributed random pairs using box-muller transform
//REF: https://en.wikipedia.org/wiki/Box%E2%80%93Muller_transform
void random_pair(double* num1, double* num2) {

    double u1 = random_float();
    double u2 = random_float();

    double radius = sqrt(-2.0 * log(u1));
    double angle = 2.0 * PI * u2;

    *num1 = radius * cos(angle);
    *num2 = radius * sin(angle);
}

//generate gaussian blob coords
void generate_data(KMeansData* data) {

    for (int i = 0; i < data->N; i++) {

        int cluster = i % data->K;

        //calculate cluster center
        double angle = 2.0 * PI * cluster / data->K;

        double center_x = BLOB_RADIUS * cos(angle);
        double center_y = BLOB_RADIUS * sin(angle);

        //box-muller
        double random_x, random_y;
        random_pair(&random_x, &random_y);

        //final coords
        data->X[i] = (float)(center_x + random_x * BLOB_STDDEV);
        data->Y[i] = (float)(center_y + random_y * BLOB_STDDEV);
    }
}

//initialize X and Y centroids
void initialize_centroids(const KMeansData* data, float* centroid_X, float* centroid_Y) {
    for (int i = 0; i < data->K; i++) {

        centroid_X[i] = data->X[i];
        centroid_Y[i] = data->Y[i];
    }
}

static double g_ms_per_tick = 0.0;

//call once at startup (the frequency never changes while the program runs)
void timer_init(void) {
    LARGE_INTEGER frequency;
    QueryPerformanceFrequency(&frequency);
    g_ms_per_tick = 1000.0 / (double)frequency.QuadPart;
}

static int compare_double(const void* a, const void* b) {
    double x = *(const double*)a;
    double y = *(const double*)b;
    return (x > y) - (x < y);
}

//times kmeans() for one assignment version
//1 warm-up run, then `runs` timed runs; Centroids are reset to the first K
//points before every run, and the reset is outside the timer
//only timing mode is allowed (FIXED_ITERS > 0, TOL = 0) so every version does identical work
int benchmark_kmeans(const KMeansData* data, AssignFn assign,
    float* centroid_X, float* centroid_Y, int* labels, float* min_d2,
    ClusterSums* sums, int runs, TimingStats* stats) {

    if (data == NULL || assign == NULL || stats == NULL || runs < 1) {
        return 0;
    }

    if (data->FIXED_ITERS < 1 || data->TOL != 0.0f) {
        printf("Timing needs FIXED_ITERS > 0 and TOL = 0.\n");
        return 0;
    }

    double* times = (double*)malloc(runs * sizeof(double));
    if (times == NULL) {
        return 0;
    }

    KMeansResult result;

    //warm-up run (not timed)
    //loads caches and touches every page
    initialize_centroids(data, centroid_X, centroid_Y);
    if (!kmeans(data, assign, centroid_X, centroid_Y, labels, min_d2, sums, &result, data->EMPTY_POLICY)) {
        free(times);
        return 0;
    }

    for (int r = 0; r < runs; r++) {

        initialize_centroids(data, centroid_X, centroid_Y); //reset, not timed

        LARGE_INTEGER start, end;
        QueryPerformanceCounter(&start);
        int ok = kmeans(data, assign, centroid_X, centroid_Y, labels, min_d2, sums, &result, data->EMPTY_POLICY);
        QueryPerformanceCounter(&end);

        if (!ok) {
            free(times);
            return 0;
        }

        times[r] = (double)(end.QuadPart - start.QuadPart) * g_ms_per_tick;
    }

    //statistics
    double sum = 0.0;
    double min = times[0];
    for (int r = 0; r < runs; r++) {
        sum += times[r];
        if (times[r] < min) min = times[r];
    }
    double mean = sum / runs;

    double var = 0.0;
    for (int r = 0; r < runs; r++) {
        var += (times[r] - mean) * (times[r] - mean);
    }

    qsort(times, runs, sizeof(double), compare_double);

    stats->mean = mean;
    stats->min = min;
    stats->stddev = (runs > 1) ? sqrt(var / (runs - 1)) : 0.0;
    stats->median = (runs % 2 == 1) ? times[runs / 2]
        : (times[runs / 2 - 1] + times[runs / 2]) / 2.0;

    free(times);
    return 1;
}

//single-thread copy bandwidth in GB/s (read + write bytes, STREAM-style)
//bytes must be much larger than the last-level cache (e.g. 256 MB)
int measure_bandwidth(size_t bytes, int runs, double* best_gbps, double* mean_gbps) {

    if (best_gbps == NULL || mean_gbps == NULL || runs < 1 || bytes == 0) {
        return 0;
    }

    char* src = (char*)_aligned_malloc(bytes, 32);
    char* dst = (char*)_aligned_malloc(bytes, 32);

    if (src == NULL || dst == NULL) {
        _aligned_free(src);
        _aligned_free(dst);
        return 0;
    }

    //touch both buffers once so page faults are not measured
    memset(src, 1, bytes);
    memset(dst, 0, bytes);

    //warm-up copy (not timed)
    memcpy(dst, src, bytes);

    double best = 0.0;
    double total = 0.0;

    for (int r = 0; r < runs; r++) {

        LARGE_INTEGER start, end;
        QueryPerformanceCounter(&start);
        memcpy(dst, src, bytes);
        QueryPerformanceCounter(&end);

        double seconds = (double)(end.QuadPart - start.QuadPart) * g_ms_per_tick / 1000.0;
        double gbps = (2.0 * (double)bytes) / seconds / 1e9; //read + write bytes

        total += gbps;
        if (gbps > best) best = gbps;
    }

    //read a byte back so the copies cannot be optimized away
    volatile char sink = dst[bytes - 1];
    (void)sink;

    *best_gbps = best;
    *mean_gbps = total / runs;

    _aligned_free(src);
    _aligned_free(dst);
    return 1;
}

//pin thread to specific core
DWORD_PTR pin_core(DWORD core) {

    SYSTEM_INFO system_info;
    GetSystemInfo(&system_info);

    if (core >= system_info.dwNumberOfProcessors) {
        printf("Invalid core number.\n");
        return 0;
    }

    DWORD_PTR mask = ((DWORD_PTR)1 << core);
    return SetThreadAffinityMask(GetCurrentThread(), mask);
}