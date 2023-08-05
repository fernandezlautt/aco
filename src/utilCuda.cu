#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "utilCuda.cuh"

__global__ void probabilities_calculation(double *pheromone, double *probabilities, double sum, bool *visited, double *distances, int alpha, int beta)
{
    int i = threadIdx.x;

    if (visited[i] || distances[i] == 0)
        probabilities[i] = 0;
    else
        probabilities[i] = pow(pheromone[i], alpha) * pow(1 / distances[i], beta) / sum;
}

__global__ void copy_vector(int *vector1, int *vector2)
{
    int i = threadIdx.x;
    vector1[i] = vector2[i];
}
