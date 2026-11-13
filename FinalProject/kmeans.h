#ifndef KMEANS_H
#define KMEANS_H

//define policies
typedef enum {
    EMPTY_POLICY_A = 0,
    EMPTY_POLICY_B = 1
} EmptyPolicy;

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

//reusable buffer for the centroid update (allocated once, not per iteration)
typedef struct {
    double* sum_X;
    double* sum_Y;
    int* count;
} ClusterSums;

//what a run reports back
typedef struct {
    int iterations; //iterations actually run
    int converged; //1 if stopped because movement <= TOL (always 0 in fixed mode)
    int empty_events; //empty clusters, summed over all update steps
    double inertia; //filled in by compute_inertia(), not by kmeans()
} KMeansResult;

//assignment step: C, XMM and YMM versions all use this same signature
//writes labels[i] and min_d2[i] (the smallest squared distance) for every point
typedef int (*AssignFn)(const KMeansData* data, const float* centroid_X,
    const float* centroid_Y, int* labels, float* min_d2);

int assign_c(const KMeansData* data, const float* centroid_X,
    const float* centroid_Y, int* labels, float* min_d2);

int kmeans(const KMeansData* data, AssignFn assign, float* centroid_X, float* centroid_Y,
    int* labels, float* min_d2, ClusterSums* sums, KMeansResult* result);

int compute_inertia(const KMeansData* data, AssignFn assign, const float* centroid_X,
    const float* centroid_Y, int* labels, float* min_d2, double* inertia);

int cluster_sums_init(int K, ClusterSums* sums);
void cluster_sums_free(ClusterSums* sums);

//returns the number of empty clusters (>= 0) or -1 on error
//max_move (optional, may be NULL) receives the largest distance any centroid moved
int update_centroids(const KMeansData* data, const int* labels, float* centroid_X,
    float* centroid_Y, ClusterSums* sums, double* max_move);

#endif