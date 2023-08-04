#include "graph.cuh"
#include "system.cuh"
#include <stdlib.h>
#include <stdio.h>

/*
    Initialize the system with the given parameters.
    The distance matrix is initialized with random values.
    The pheromone matrix is initialized with 1.
*/
SYSTEM *initialize_system(int n_cities, int n_ants, double alpha, double beta, double evaporation_rate)
{
    SYSTEM *system = (SYSTEM *)malloc(sizeof(SYSTEM));

    system->distance_matrix = (MATRIX *)malloc(sizeof(MATRIX));
    system->pheromone_matrix = (MATRIX *)malloc(sizeof(MATRIX));
    system->best_path = (int *)malloc(sizeof(int) * n_cities);
    system->best_cost = 0;
    system->alpha = alpha;
    system->beta = beta;
    system->n_ants = n_ants;
    system->evaporation_rate = evaporation_rate;

    initialize_matrix(system->distance_matrix, n_cities);
    initialize_matrix(system->pheromone_matrix, n_cities);
    initialize_distances(system->distance_matrix);
    initialize_pheromones(system->pheromone_matrix);
    // ToDo -> delete this
    // int i;
    /*     print_matrix(system->distance_matrix);
        print_matrix(system->pheromone_matrix);
        for (i = 0; i < n_ants; i++)
            print_vector(system->ants[i].path, n_cities); */

    return system;
}

/*
    Initialize the ants with the given parameters.
    The path is initialized with -1.
    The current city is initialized with a random value.
*/
ANT *initialize_ants(int n_ants, int n_cities)
{
    ANT *ants = (ANT *)malloc(sizeof(ANT) * n_ants);
    int i;
    int j;

    for (i = 0; i < n_ants; i++)
    {
        ants[i].path = (int *)malloc(sizeof(int) * n_cities);
        ants[i].visited = (bool *)malloc(sizeof(bool) * n_cities);

        for (j = 0; j < n_cities; j++)
            ants[i]
                .path[j] = -1;

        for (j = 0; j < n_cities; j++)
            ants[i].visited[j] = false;

        ants[i].cost = 0;

        ants[i].current_city = rand() % n_cities;

        ants[i].path[0] = ants[i].current_city;
        ants[i].visited[ants[i].current_city] = true;
        ants[i].step = 0;
    }
    return ants;
}

void free_ants(ANT *ants, int n_ants)
{
    int i;
    for (i = 0; i < n_ants; i++)
    {
        free(ants[i].path);
        free(ants[i].visited);
    }
    free(ants);
}

// Free memory allocated for the system
void free_system(SYSTEM *system)
{
    free_matrix(system->distance_matrix);
    free_matrix(system->pheromone_matrix);
    free_ants(system->ants, system->n_ants);
    free(system->best_path);
    free(system);
}