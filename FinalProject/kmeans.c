
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


//updates centroids based on current labels
int update_centroids(const KMeansData* data, const int* labels, float* centroid_X, float* centroid_Y) {
    
    if (data == NULL || labels == NULL || centroid_X == NULL || centroid_Y == NULL) {
        return 0;
    }

    //for X[] and Y[] sums and point count
    float* sum_X = (float*)calloc(data->K, sizeof(float));
    float* sum_Y = (float*)calloc(data->K, sizeof(float));
    int* count = (int*)calloc(data->K, sizeof(int));

    if (sum_X == NULL || sum_Y == NULL || count == NULL) {
        free(sum_X);
        free(sum_Y);
        free(count);
        return 0;
    }

	//sum up points in each cluster
    for (int i = 0; i < data->N; i++) {

        int cluster = labels[i];

        if (cluster < 0 || cluster >= data->K) {
            free(sum_X);
            free(sum_Y);
            free(count);
            return 0;
        }

        sum_X[cluster] += data->X[i];
        sum_Y[cluster] += data->Y[i];

        count[cluster]++;
    }

    //new centroid positions
    for (int j = 0; j < data->K; j++) {

        if (count[j] > 0) {
            centroid_X[j] = sum_X[j] / count[j];
            centroid_Y[j] = sum_Y[j] / count[j];
        }
    }

    //free memory
    free(sum_X);
    free(sum_Y);
    free(count);

    return 1;
}
