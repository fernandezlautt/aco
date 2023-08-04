#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <stdio.h>

__global__ void probabilities_calculation(double *pheromone, double *probabilities, double sum, bool *visited);