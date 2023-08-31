#include "graph.cuh"
#include "aco.cuh"
#include "system.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <time.h>

// Parameters
#define N_CITIES 100
#define N_ANTS 50
#define ALPHA 1
#define BETA 1
#define N_ITERATIONS 50
#define EVAPORATION_RATE 0.5
#define THREADS 5
#define REINFORCEMENT 1

int main(int argc, char *argv[])
{
    char *p;
    int n_cities = strtol(argv[1], &p, 10);
    int n_ants = strtol(argv[2], &p, 10);
    float alpha = atof(argv[3]);
    float beta = atof(argv[4]);
    float evaporation_rate = atof(argv[5]);
    int cycles = strtol(argv[6], &p, 10);

    clock_t start, end;
    RESULT *result = nullptr;

    // Initialize the system
    SYSTEM *system = initialize_system(n_cities, n_ants, alpha, beta, evaporation_rate, REINFORCEMENT);

    // Reset gpu
    cudaDeviceReset();

    start = clock();

    // Run the algorithm
    result = aco(system, cycles, THREADS);

    end = clock();

    result->time = ((double)(end - start)) / CLOCKS_PER_SEC;

    printf("%ld\n", end - start);

    // // Data for python
    // printf("%f\n", result->time);
    // printf("%i\n", N_CITIES);
    // printf("%i\n", N_ANTS);
    // printf("%i\n", N_ITERATIONS);

    // Print the results
    // print_vector(result->path, N_CITIES);

    // print_vector_double(result->costs, N_ITERATIONS);

    // Free memory
    free_system(system);

    return 0;
}
