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


int kmeans(const KMeansData* data, float* centroid_X, float* centroid_Y, int* labels);
int update_centroids(const KMeansData* data, const int* labels, float* centroid_X, float* centroid_Y);

#endif