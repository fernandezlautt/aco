#include "graph.cuh"
#include "aco.cuh"
#include "system.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// Parameters
#define N_CITIES 500
#define N_ANTS 50
#define ALPHA 1
#define BETA 1
#define N_ITERATIONS 100
#define EVAPORATION_RATE 0.75
#define THREADS 5
#define REINFORCEMENT 0.5

int main(int argc, char *argv[])
{
    clock_t start, end;
    RESULT *result = nullptr;

    // Initialize the system
    SYSTEM *system = initialize_system(N_CITIES, N_ANTS, ALPHA, BETA, EVAPORATION_RATE, REINFORCEMENT);

    start = clock();
    // Run the algorithm
    result = aco(system, N_ITERATIONS, THREADS);
    end = clock();
    result->time = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Print the results
    print_vector(result->path, N_CITIES);
    print_vector_double(result->costs, N_ITERATIONS);
    printf("Time taken: %f\n", result->time);

    // Free memory
    free_system(system);

    return 0;
}
