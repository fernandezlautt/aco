#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include "utilCuda.cuh"

__global__ void probabilities_calculation(double *pheromone, double *probabilities, double sum, bool *visited)
{
    int i = threadIdx.x;
    if (visited[i])
        probabilities[i] = 0;
    else
        probabilities[i] = pheromone[i] / sum;
}
