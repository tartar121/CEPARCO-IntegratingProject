
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <stdint.h>
#include <windows.h>
#include "kmeans.h"

// gaussian generator parameters
#define PI 3.14159265358979323846
#define BLOB_RADIUS 100.0
#define BLOB_STDDEV 10.0

//define policies
typedef enum {
    EMPTY_POLICY_A = 0,
    EMPTY_POLICY_B = 1
} EmptyPolicy;


// asm test cases
void xmmkernel(void);
void ymmkernel(void);


int validate_data(const KMeansData* data); //validate input data
int allocate_arrays(KMeansData* data); //validate X[] and Y[]
void free_arrays(KMeansData* data); //free X[] and Y[]

double random_float(void); //conver srand numbers to floats
void random_pair(double* num1, double* num2); //box-muller transform generates the gaussian pair
void generate_data(KMeansData* data); //guassian blob coords

void initialize_centroids(KMeansData* data, float* centroid_X, float* centroid_Y); //centroids for X and Y
double get_time(void); //used QueryPerformanceFrequency
double benchmark(void (*function)(void), int runs); //benchmark N runs

DWORD_PTR pin_core(DWORD core); //pinned core

//FOR TESTING
//small known-answer tests for the C baseline
static void run_tests(void) {
    printf("--------------------------------------\n");
    printf("TESTS\n");

    //Test 1: README hand-worked example, starting centroids set by hand
    {
        float X[] = { 1.0f, 1.5f, 1.2f, 8.0f, 9.0f, 8.5f };
        float Y[] = { 1.0f, 2.0f, 0.8f, 8.0f, 9.5f, 9.0f };
        float cx[] = { 1.0f, 8.0f };
        float cy[] = { 1.0f, 8.0f };
        int labels[6];
        KMeansData d = { 0 };
        d.X = X; d.Y = Y; d.N = 6; d.K = 2;

        ClusterSums sums;
        cluster_sums_init(d.K, &sums);
        kmeans(&d, cx, cy, labels);
        int empty = update_centroids(&d, labels, cx, cy, &sums);
        cluster_sums_free(&sums);

        int ok = labels[0] == 0 && labels[1] == 0 && labels[2] == 0 &&
            labels[3] == 1 && labels[4] == 1 && labels[5] == 1 &&
            fabsf(cx[0] - 1.2333f) < 1e-3f && fabsf(cy[0] - 1.2667f) < 1e-3f &&
            fabsf(cx[1] - 8.5f) < 1e-3f && fabsf(cy[1] - 8.8333f) < 1e-3f &&
            empty == 0;
        printf("Test 1 (hand-worked example): %s\n", ok ? "PASS" : "FAIL");
    }

    //Test 2: duplicate first 3 points, K = 3 -> every point ties to centroid 0
    {
        float X[] = { 1, 1, 1, 5, 6, 7, 8, 9 };
        float Y[] = { 1, 1, 1, 5, 6, 7, 8, 9 };
        float cx[] = { 1, 1, 1 };
        float cy[] = { 1, 1, 1 };
        int labels[8];
        KMeansData d = { 0 };
        d.X = X; d.Y = Y; d.N = 8; d.K = 3;

        ClusterSums sums;
        cluster_sums_init(d.K, &sums);
        kmeans(&d, cx, cy, labels);
        int empty = update_centroids(&d, labels, cx, cy, &sums);
        cluster_sums_free(&sums);

        int all_zero = 1;
        for (int i = 0; i < 8; i++) if (labels[i] != 0) all_zero = 0;

        //clusters 1 and 2 must be empty and keep their old position (Policy A)
        int ok = all_zero && empty == 2 && cx[1] == 1.0f && cx[2] == 1.0f;
        printf("Test 2 (duplicate points / empty clusters): %s\n", ok ? "PASS" : "FAIL");
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
    srand(data.SEED);

    data.EMPTY_POLICY = EMPTY_POLICY_A;
    data.FIXED_ITERS = 100;


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


    // generate gaussian blobs and centroids
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


	//allocate labels for clusters
    int* labels = (int*)malloc(data.N * sizeof(int)); 
    if (labels == NULL) {
        printf("Label allocation failed.\n");

        free(centroid_X);
        free(centroid_Y);
        free_arrays(&data);

        return 1;
    }

    //k means clustering
    printf("--------------------------------------\n");
    printf("6. Sequential C K-Means\n");

    int result = kmeans(&data, centroid_X, centroid_Y, labels);

    if (!result) {
        printf("K-means failed.\n");

        free(labels);
        free(centroid_X);
        free(centroid_Y);
        free_arrays(&data);

        return 1;
    }

    //print first 10 labels
    for (int i = 0; i < 10 && i < data.N; i++) {
        printf("Point %d : Cluster %d\n", i, labels[i]);
    }
    printf("--------------------------------------\n\n");


    // buffers for the centroid update
    ClusterSums sums;

    if (!cluster_sums_init(data.K, &sums)) {
        printf("Cluster sums allocation failed.\n");

        free(labels);
        free(centroid_X);
        free(centroid_Y);
        free_arrays(&data);

        return 1;
    }

    // update centroid positions
    int empty = update_centroids(&data, labels, centroid_X, centroid_Y, &sums);

    if (empty < 0) {
        printf("Centroid update failed.\n");

        cluster_sums_free(&sums);
        free(labels);
        free(centroid_X);
        free(centroid_Y);
        free_arrays(&data);

        return 1;
    }

    //updated centroids
    printf("--------------------------------------\n");
    printf("7.Updated Centroids\n");
    printf("Empty clusters this update: %d\n", empty);

    for (int i = 0; i < data.K; i++) {
        printf("Centroid %d: X = %.2f, Y = %.2f\n",
            i, centroid_X[i], centroid_Y[i]);
    }
    printf("--------------------------------------\n\n");


    //test asm
    xmmkernel();
    ymmkernel();


    //free memory
    cluster_sums_free(&sums);
	free(labels);
    free(centroid_X);
    free(centroid_Y);
    free_arrays(&data);

	SetThreadAffinityMask(GetCurrentThread(), old_affinity); //restore old cpu affinity

    return 0;
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

//convert rand to float between 0.0 to 1.0
double random_float(void) {
    return ((double)rand() + 1.0) / ((double)RAND_MAX + 2.0);
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
void initialize_centroids(KMeansData* data, float* centroid_X, float* centroid_Y) {
    for (int i = 0; i < data->K; i++) {

        centroid_X[i] = data->X[i];
        centroid_Y[i] = data->Y[i];
    }
}

//timing function (in ms)
double get_time(void)
{
    LARGE_INTEGER frequency;
    LARGE_INTEGER counter;

    QueryPerformanceFrequency(&frequency);
    QueryPerformanceCounter(&counter);

    return (double)counter.QuadPart * 1000.0 / (double)frequency.QuadPart;
}


//benchmark function
double benchmark(void (*function)(void), int runs) {
    double start_time, end_time;
    double total_time = 0.0;

    // warm-up run
    //TODO
    function();

    for (int i = 0; i < runs; i++) {

        start_time = get_time();
        function();
        end_time = get_time();

        total_time += end_time - start_time;
    }
    return total_time / runs;
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




