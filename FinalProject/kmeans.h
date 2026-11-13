#ifndef KMEANS_H
#define KMEANS_H

//internal inputs
typedef struct {
    float* X;
    float* Y;

    int N;
    int K;

    int MAX_ITER;
    float TOL;
    unsigned int SEED;

    int EMPTY_POLICY;
    int FIXED_ITERS;
} KMeansData;

// reusable buffer for the centroid update (allocated once, not per iteration)
typedef struct {
    double* sum_X;
    double* sum_Y;
    int* count;
} ClusterSums;

int kmeans(const KMeansData* data, float* centroid_X, float* centroid_Y, int* labels);

int cluster_sums_init(int K, ClusterSums* sums);
void cluster_sums_free(ClusterSums* sums);

int update_centroids(const KMeansData* data, const int* labels, float* centroid_X, float* centroid_Y, ClusterSums* sums);

#endif