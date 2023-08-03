#include "graph.h"
#include "aco.h"
#include "system.h"
#include <stdlib.h>
#include <stdio.h>

#define N_CITIES 10
#define N_ANTS 10
#define ALPHA 0.5
#define BETA 0.5
#define N_ITERATIONS 100
#define EVAPORATION_RATE 0.75

int main(int argc, char *argv[])
{
    SYSTEM *system = initialize_system(N_CITIES, N_ANTS, ALPHA, BETA, EVAPORATION_RATE);

    printf("here we are\n");

    int *path = aco(system, N_ITERATIONS);

    printf("here we are 2\n");

    print_vector(path, N_CITIES);
    print_matrix(system->pheromone_matrix);

    free_system(system);

    return 0;
}