#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kmeans.h"


//put point to nearest centroid (scalar C version of the assignment step)
int assign_c(const KMeansData* data, const float* centroid_X,
    const float* centroid_Y, int* labels, float* min_d2) {

    for (int i = 0; i < data->N; i++) {

        float dx = data->X[i] - centroid_X[0];
        float dy = data->Y[i] - centroid_Y[0];

        float min_distance = (dx * dx) + (dy * dy);
        int closest_cluster = 0;

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
        int empty = update_centroids(data, labels, centroid_X, centroid_Y,
            sums, fixed_mode ? NULL : &max_move);
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
    if (policy == 1 && empty > 0) {
        
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