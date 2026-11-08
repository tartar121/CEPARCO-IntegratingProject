
#include <stdio.h>
#include <stdlib.h>

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

	//test asm 
    xmmkernel();
    ymmkernel();

    return 0;
}
