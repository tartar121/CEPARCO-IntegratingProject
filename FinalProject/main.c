
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include <malloc.h>
#include <stdint.h>


//define policies
typedef enum {
    EMPTY_POLICY_A = 0,
    EMPTY_POLICY_B = 1
} EmptyPolicy;

// internal inputs
typedef struct {
    //arrays
    float *X;
    float *Y;

    //points and clusters
    int N;
    int K;

    //parameters
    int MAX_ITER;
    float TOL;
    unsigned int SEED;
    EmptyPolicy EMPTY_POLICY;
    int FIXED_ITERS;

} KMeansData;


// asm test cases
void xmmkernel(void);
void ymmkernel(void);


int validate_data(const KMeansData* data); //validate input data
int allocate_arrays(KMeansData* data); //validate X[] and Y[]
void free_arrays(KMeansData* data); //free X[] and Y[]


int main(void)
{
    //fill with temp values for now
    KMeansData data = { 0 };

    data.X = NULL;
    data.Y = NULL;

    data.N = 1024;
    data.K = 4;

    data.MAX_ITER = 100;
    data.TOL = 0.001f;
    data.SEED = 67;
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

    printf("Number of Points: %d\n", data.N);
    printf("X Address: %p\n", (void*)data.X);
    printf("Y Address: %p\n", (void*)data.Y);
    printf("--------------------------------------\n\n");


	//TODO: guassian blob generator 
    printf("--------------------------------------\n");
    printf("3. Gaussian Blob Generator\n");

    printf("--------------------------------------\n\n");


	//test asm 
    xmmkernel();
    ymmkernel();


    free_arrays(&data);
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

