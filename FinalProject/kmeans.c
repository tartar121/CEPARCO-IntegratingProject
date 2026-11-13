#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kmeans.h"


//put point to nearest centroid (scalar C version of the assignment step)
int assign_c(const KMeansData* data, const float* centroid_X,
    const float* centroid_Y, int* labels, float* min_d2) {
#pragma loop(no_vector)
    for (int i = 0; i < data->N; i++) {

        float dx = data->X[i] - centroid_X[0];
        float dy = data->Y[i] - centroid_Y[0];

        float min_distance = (dx * dx) + (dy * dy);
        int closest_cluster = 0;
#pragma loop(no_vector)
        for (int j = 1; j < data->K; j++) {

            dx = data->X[i] - centroid_X[j];
            dy = data->Y[i] - centroid_Y[j];

            float distance = (dx * dx) + (dy * dy);

            if (distance < min_distance) {
                min_distance = distance;
                closest_cluster = j;
            }
        }

        labels[i] = closest_cluster;
        min_d2[i] = min_distance;
    }

    return 1;
}


//K-means loop. The assignment step is passed in, everything else is shared.
//FIXED_ITERS > 0: timing mode, runs exactly FIXED_ITERS iterations, no convergence check
//FIXED_ITERS == 0: correctness mode, stops when max centroid movement <= TOL or at MAX_ITER
int kmeans(const KMeansData* data, AssignFn assign, float* centroid_X, float* centroid_Y,
    int* labels, float* min_d2, ClusterSums* sums, KMeansResult* result) {

    //validate all inputs
    if (data == NULL || data->X == NULL || data->Y == NULL || assign == NULL ||
        centroid_X == NULL || centroid_Y == NULL || labels == NULL ||
        min_d2 == NULL || sums == NULL || result == NULL ||
        data->N < 1 || data->K < 1 || data->N < data->K) {
        return 0;
    }

    int fixed_mode = (data->FIXED_ITERS > 0);
    int max_iters = fixed_mode ? data->FIXED_ITERS : data->MAX_ITER;

    result->iterations = 0;
    result->converged = 0;
    result->empty_events = 0;
    result->inertia = 0.0;

    for (int iter = 0; iter < max_iters; iter++) {

        //assignment step
        if (!assign(data, centroid_X, centroid_Y, labels, min_d2)) {
            return 0;
        }

        //update step (movement is only measured when it will be checked)
        double max_move = 0.0;
        int empty = update_centroids(data, labels, min_d2, centroid_X, centroid_Y,
            sums, fixed_mode ? NULL : &max_move, data->EMPTY_POLICY);
        if (empty < 0) {
            return 0;
        }

        result->empty_events += empty;
        result->iterations++;

        if (!fixed_mode && max_move <= (double)data->TOL) {
            result->converged = 1;
            break;
        }
    }

    return 1;
}


//final inertia: one extra assignment against the FINAL centroids, then a
//sequential double-precision sum of the smallest squared distances.
//leaves labels[] consistent with the final centroids. Run this OUTSIDE the timer.
int compute_inertia(const KMeansData* data, AssignFn assign, const float* centroid_X,
    const float* centroid_Y, int* labels, float* min_d2, double* inertia) {

    if (data == NULL || assign == NULL || centroid_X == NULL || centroid_Y == NULL ||
        labels == NULL || min_d2 == NULL || inertia == NULL) {
        return 0;
    }

    if (!assign(data, centroid_X, centroid_Y, labels, min_d2)) {
        return 0;
    }

    double sum = 0.0;
    for (int i = 0; i < data->N; i++) {
        sum += (double)min_d2[i];
    }

    *inertia = sum;
    return 1;
}


int cluster_sums_init(int K, ClusterSums* sums) {
    sums->sum_X = (double*)malloc(K * sizeof(double));
    sums->sum_Y = (double*)malloc(K * sizeof(double));
    sums->count = (int*)malloc(K * sizeof(int));

    if (sums->sum_X == NULL || sums->sum_Y == NULL || sums->count == NULL) {
        free(sums->sum_X);
        free(sums->sum_Y);
        free(sums->count);
        sums->sum_X = NULL;
        sums->sum_Y = NULL;
        sums->count = NULL;
        return 0;
    }
    return 1;
}

void cluster_sums_free(ClusterSums* sums) {
    free(sums->sum_X);
    free(sums->sum_Y);
    free(sums->count);
    sums->sum_X = NULL;
    sums->sum_Y = NULL;
    sums->count = NULL;
}

//updates centroids based on current labels
//returns the number of empty clusters (>= 0) or -1 on error
int update_centroids(const KMeansData* data, const int* labels, float* min_d2, float* centroid_X,
    float* centroid_Y, ClusterSums* sums, double* max_move, int policy) {

    if (data == NULL || labels == NULL || centroid_X == NULL ||
        centroid_Y == NULL || sums == NULL) {
        return -1;
    }

    //reset sums and counts
    for (int j = 0; j < data->K; j++) {
        sums->sum_X[j] = 0.0;
        sums->sum_Y[j] = 0.0;
        sums->count[j] = 0;
    }

    //sum up points in each cluster (double, in point order)
    for (int i = 0; i < data->N; i++) {

        int cluster = labels[i];

        if (cluster < 0 || cluster >= data->K) {
            return -1;
        }

        sums->sum_X[cluster] += (double)data->X[i];
        sums->sum_Y[cluster] += (double)data->Y[i];
        sums->count[cluster]++;
    }

    //new centroid positions
    int empty = 0;
    double move = 0.0;

    for (int j = 0; j < data->K; j++) {

        if (sums->count[j] > 0) {
            float new_x = (float)(sums->sum_X[j] / sums->count[j]);
            float new_y = (float)(sums->sum_Y[j] / sums->count[j]);

            if (max_move != NULL) {
                double dx = (double)new_x - (double)centroid_X[j];
                double dy = (double)new_y - (double)centroid_Y[j];
                double d = sqrt(dx * dx + dy * dy);
                if (d > move) move = d;
            }

            centroid_X[j] = new_x;
            centroid_Y[j] = new_y;
        }
        else {
            empty++; //Policy A: keep old centroid (Policy B goes here later)
        }
    }
    // Policy B
    if (policy == EMPTY_POLICY_B && empty > 0) {
        
        int prev_index = -1;
        float prev_d2 = 0.0f;

        for (int j = 0; j < data->K; j++) {
            
            // Verification for an empty cluster
            if (sums->count[j] > 0) {
                continue;
            }

            int max_idx = -1;
            float max_d2 = 0.0f;
                for (int i = 0; i < data->N; i++) {
                    float d2 = min_d2[i];
                    
                    // Skip used points
                    if (prev_index >= 0 && !(d2 < prev_d2 || (d2 == prev_d2 && i > prev_index)))
                        continue;
                    // Checks for the max and assign index
                    if (max_idx < 0 || d2 > max_d2) {
                        max_idx = i;
                        max_d2 = d2;
                    }
                }

            if (max_move != NULL) {
               double dx = (double)data->X[max_idx] - (double)centroid_X[j];
               double dy = (double)data->Y[max_idx] - (double)centroid_Y[j];
               double d = sqrt(dx * dx + dy * dy);
               if (d > move) move = d;
            }
            // Assign centroid value
            centroid_X[j] = data->X[max_idx];
            centroid_Y[j] = data->Y[max_idx];

            prev_index = max_idx;
            prev_d2 = max_d2;
        }
  
    }

    if (max_move != NULL) *max_move = move;

    return empty;
}


//=====================================================================
//[ADDED] XMM (AVX-128) assignment step
//Everything above this line is unchanged. Everything below is new.
//The asm kernel lives in xmm.asm (below the printf stub, NASM, win64).
//Call direction: C (assign_xmm) calls asm (xmm_assign_kernel), and the asm
//calls back into C (assign_tail_c) for the N % 4 leftover points.
//=====================================================================

#include <malloc.h> //for _aligned_malloc / _aligned_free

//asm kernel in xmm.asm: processes floor(N/4)*4 points, 4 per step,
//then calls assign_tail_c (below) for the leftover points
extern void xmm_assign_kernel(const float* X, const float* Y,
    const float* cx, const float* cy,
    int* labels, float* min_d2, int N, int K);


//C CALLEE, called FROM xmm.asm (xmm_assign_kernel) for the leftover points
//(N % 4 for XMM; the YMM kernel can call the same function for N % 8).
//X, Y, labels and min_d2 already point at the first leftover point; count = how many.
//No distance math is repeated here: it wraps the leftover points in a small
//KMeansData view and lets assign_c compute them, so the XMM/YMM tail and the
//scalar baseline are guaranteed to use the same code (same math, same tie-break).
void assign_tail_c(const float* X, const float* Y, const float* cx, const float* cy,
    int* labels, float* min_d2, int count, int K) {

    KMeansData tail = { 0 };
    tail.X = (float*)X; //view only: assign_c never writes to X or Y
    tail.Y = (float*)Y;
    tail.N = count;
    tail.K = K;

    assign_c(&tail, cx, cy, labels, min_d2);
}


//XMM assignment step (same AssignFn signature as assign_c).
//C calls the asm; the asm does the vector part and calls assign_tail_c for the leftover.
int assign_xmm(const KMeansData* data, const float* centroid_X,
    const float* centroid_Y, int* labels, float* min_d2) {

    if (data == NULL || data->K < 1) {
        return 0;
    }

    xmm_assign_kernel(data->X, data->Y, centroid_X, centroid_Y, labels, min_d2, data->N, data->K);

    return 1;
}


//XMM vs C correctness check (uses its own buffers, leaves the C results untouched)
//c_labels / c_centroid_X / c_centroid_Y / c_result = results of the finished C run
//returns 1 if the labels match exactly, 0 otherwise (or on error)
int xmm_vs_c_check(const KMeansData* data, const int* c_labels,
    const float* c_centroid_X, const float* c_centroid_Y,
    const KMeansResult* c_result, ClusterSums* sums) {

    int ok = 0;
    float* cx = (float*)malloc(data->K * sizeof(float));
    float* cy = (float*)malloc(data->K * sizeof(float));
    int* labels = (int*)_aligned_malloc(data->N * sizeof(int), 32);
    float* d2 = (float*)_aligned_malloc(data->N * sizeof(float), 32);
    KMeansResult r = { 0 };

    printf("--------------------------------------\n");
    printf("XMM K-Means vs C\n");

    if (cx == NULL || cy == NULL || labels == NULL || d2 == NULL) {
        printf("XMM buffer allocation failed.\n");
        goto done;
    }

    //same starting centroids as the C run: first K points
    for (int j = 0; j < data->K; j++) {
        cx[j] = data->X[j];
        cy[j] = data->Y[j];
    }

    if (!kmeans(data, assign_xmm, cx, cy, labels, d2, sums, &r)) {
        printf("XMM k-means failed.\n");
        goto done;
    }

    if (!compute_inertia(data, assign_xmm, cx, cy, labels, d2, &r.inertia)) {
        printf("XMM inertia failed.\n");
        goto done;
    }

    //label match
    int mismatches = 0;
    for (int i = 0; i < data->N; i++) {
        if (labels[i] != c_labels[i]) mismatches++;
    }

    //largest centroid difference
    double max_diff = 0.0;
    for (int j = 0; j < data->K; j++) {
        double ddx = fabs((double)cx[j] - (double)c_centroid_X[j]);
        double ddy = fabs((double)cy[j] - (double)c_centroid_Y[j]);
        if (ddx > max_diff) max_diff = ddx;
        if (ddy > max_diff) max_diff = ddy;
    }

    printf("Iterations: C = %d, XMM = %d\n", c_result->iterations, r.iterations);
    printf("Label mismatches vs C: %d of %d (match %.4f%%)\n",
        mismatches, data->N, 100.0 * (data->N - mismatches) / data->N);
    printf("Max centroid difference: %g\n", max_diff);
    printf("Inertia: C = %.4f, XMM = %.4f\n", c_result->inertia, r.inertia);
    printf("XMM validity: %s\n", (mismatches == 0) ? "PASS" : "FAIL");

    ok = (mismatches == 0);

done:
    printf("--------------------------------------\n\n");
    free(cx);
    free(cy);
    _aligned_free(labels);
    _aligned_free(d2);
    return ok;
}