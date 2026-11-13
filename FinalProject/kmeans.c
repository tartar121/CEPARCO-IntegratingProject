
#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "kmeans.h"


//put point to nearest centroid
static int assign_points(const KMeansData* data, const float* centroid_X, const float* centroid_Y, int* labels) {
    
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
    }

    return 1;
}

//sequential C K-means
int kmeans(const KMeansData* data, float* centroid_X, float* centroid_Y, int* labels) {
    
    //va,idate all inputs
    if (data == NULL || data->X == NULL || data->Y == NULL ||
        centroid_X == NULL || centroid_Y == NULL || labels == NULL ||
        data->N < 1 || data->K < 1 || data->N < data->K) {
        return 0;
    }

    //assignment step
    if (!assign_points(data, centroid_X, centroid_Y, labels)) {
        return 0;
    }

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
//returns the number of empty clusters (>= 0), or -1 on error
int update_centroids(const KMeansData* data, const int* labels,
    float* centroid_X, float* centroid_Y, ClusterSums* sums) {

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

    for (int j = 0; j < data->K; j++) {

        if (sums->count[j] > 0) {
            centroid_X[j] = (float)(sums->sum_X[j] / sums->count[j]);
            centroid_Y[j] = (float)(sums->sum_Y[j] / sums->count[j]);
        }
        else {
            empty++; //Policy A: keep old centroid (Policy B goes here later)
        }
    }

    return empty;
}
