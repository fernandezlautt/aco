#include "graph.cuh"
#include "aco.cuh"
#include "system.cuh"
#include <stdlib.h>
#include <stdio.h>
#include <time.h>

// Parameters
#define N_CITIES 100
#define N_ANTS 50
#define ALPHA 0.5
#define BETA 0.5
#define N_ITERATIONS 100
#define EVAPORATION_RATE 0.8
#define THREADS 4

int main(int argc, char *argv[])
{
    clock_t start, end;
    double time_taken;

    // Initialize the system
    SYSTEM *system = initialize_system(N_CITIES, N_ANTS, ALPHA, BETA, EVAPORATION_RATE);

    start = clock();
    // Run the algorithm
    int *path = aco(system, N_ITERATIONS, THREADS);
    end = clock();
    time_taken = ((double)(end - start)) / CLOCKS_PER_SEC;

    // Print the results
    print_vector(path, N_CITIES);
    // print_matrix(system->pheromone_matrix);
    printf("Distance: %f\n", system->best_cost);
    printf("Time taken: %f\n", time_taken);

    // Free memory
    free_system(system);

    return 0;
}