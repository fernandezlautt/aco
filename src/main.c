#include "graph.h"
#include "aco.h"
#include "system.h"
#include <stdlib.h>
#include <stdio.h>

// Parameters
#define N_CITIES 100
#define N_ANTS 50
#define ALPHA 0.5
#define BETA 0.5
#define N_ITERATIONS 100
#define EVAPORATION_RATE 0.75

int main(int argc, char *argv[])
{
    // Initialize the system
    SYSTEM *system = initialize_system(N_CITIES, N_ANTS, ALPHA, BETA, EVAPORATION_RATE);

    // Run the algorithm
    int *path = aco(system, N_ITERATIONS);

    // Print the results
    print_vector(path, N_CITIES);
    print_matrix(system->pheromone_matrix);

    // Free memory
    free_system(system);

    return 0;
}