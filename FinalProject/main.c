// main.c
#include <stdio.h>
#include <stdlib.h>
#include <windows.h>
#include <math.h>

#define NUM_RUNS 30

// four kernels to benchmark: C, x86-64, XMM, and YMM
extern double asmkernel(const double* A, int n);
extern double xmmkernel(const double* A, int n);
extern double ymmkernel(const double* A, int n);
double ckernel(const double* A, int n);

typedef double (*kernel_func)(const double*, int);

// function declarations
void generate_values(double* A, int n);
double benchmark_kernel(kernel_func kernel, const double* A, int n);
int check_result(double reference, double result);
void run_test(int n);

int main(void)
{
    // large test sizes
	// increase these if memory allows, 
    // but keep in mind that the ASUM kernel is O(n) and will take longer to run for larger sizes
    int testSizes[] = {
        1 << 20,
        1 << 26,
        1 << 27
    };

    int numTests = sizeof(testSizes) / sizeof(testSizes[0]);

    // use the same random sequence for all tests
    srand(67);

    printf("============================================\n");
    printf("          ASUM KERNEL BENCHMARK\n");
    printf("============================================\n");
    printf("Input values: random doubles in [-50, 50]\n");
    printf("Benchmark runs: %d\n", NUM_RUNS);

    for (int i = 0; i < numTests; i++) {
        run_test(testSizes[i]);
    }

    // test sizes that are not divisible by 2 or 4
    int boundarySizes[] = { 1, 3, 7, 1003, 1005 };

    int numBoundary =
        sizeof(boundarySizes) / sizeof(boundarySizes[0]);

    printf("\n============================================\n");
    printf("              BOUNDARY TESTS\n");
    printf("============================================\n");

    for (int i = 0; i < numBoundary; i++) {
        run_test(boundarySizes[i]);
    }

    return 0;
}

void run_test(int n)
{
    if (n <= 0) {
        printf("Invalid vector size: %d\n", n);
        return;
    }

    double* A = (double*)malloc((size_t)n * sizeof(double));

    if (A == NULL) {
        printf("Memory allocation failed for n = %d\n", n);
        return;
    }

    generate_values(A, n);

    printf("\n--------------------------------------------\n");
    printf("Vector size: n = %d\n", n);
    printf("--------------------------------------------\n");

    // calculate the results.
    double resultC = ckernel(A, n);
    double resultASM = asmkernel(A, n);
    double resultXMM = xmmkernel(A, n);
    double resultYMM = ymmkernel(A, n);

    // measure each kernel separately
    double timeC = benchmark_kernel(ckernel, A, n);
    double timeASM = benchmark_kernel(asmkernel, A, n);
    double timeXMM = benchmark_kernel(xmmkernel, A, n);
    double timeYMM = benchmark_kernel(ymmkernel, A, n);

    // display the results and check correctness against C
    printf("\nResults:\n");
    printf("C kernel:       %.12f [REFERENCE]\n", resultC);

    printf("x86-64 kernel:  %.12f [%s]\n",
        resultASM,
        check_result(resultC, resultASM) ? "CORRECT" : "WRONG");

    printf("XMM kernel:     %.12f [%s]\n",
        resultXMM,
        check_result(resultC, resultXMM) ? "CORRECT" : "WRONG");

    printf("YMM kernel:     %.12f [%s]\n",
        resultYMM,
        check_result(resultC, resultYMM) ? "CORRECT" : "WRONG");

    // display the average execution times
    printf("\nAverage time (%d runs):\n", NUM_RUNS);
    printf("C:       %.9f seconds\n", timeC);

    printf("x86-64:  %.9f seconds\n", timeASM);

    printf("XMM:     %.9f seconds\n", timeXMM);

    printf("YMM:     %.9f seconds\n", timeYMM);

    free(A);
}

// generate random values between -50 and 50
void generate_values(double* A, int n)
{
    for (int i = 0; i < n; i++) {
        A[i] = ((double)rand() / RAND_MAX) * 100.0 - 50.0;
    }
}

// C reference implementation
double ckernel(const double* A, int n)
{
    double sum = 0.0;

    for (int i = 0; i < n; i++) {
        sum += fabs(A[i]);
    }

    return sum;
}

// return 1 if the results are close
int check_result(double reference, double result)
{
    double difference = fabs(reference - result);
    double tolerance = 1e-8 * fmax(1.0, fabs(reference));

    return difference <= tolerance;
}

// measure the kernel execution time over 30 runs
double benchmark_kernel(
    kernel_func kernel,
    const double* A,
    int n)
{
    LARGE_INTEGER frequency;
    LARGE_INTEGER start;
    LARGE_INTEGER end;

    QueryPerformanceFrequency(&frequency);

    double warmup = kernel(A, n);
    (void)warmup;

    double totalTime = 0.0;
    double checksum = 0.0;

    for (int i = 0; i < NUM_RUNS; i++) {
        QueryPerformanceCounter(&start);

        double result = kernel(A, n);

        QueryPerformanceCounter(&end);

        // use the returned result outside the timed interval
        checksum += result;

        totalTime +=
            (double)(end.QuadPart - start.QuadPart)
            / (double)frequency.QuadPart;
    }

    // keep the checksum observable without printing during timing.
    if (checksum < 0.0) {
        printf("Unexpected negative ASUM result.\n");
    }

    return totalTime / NUM_RUNS;
}