#include "graph.h"
#include "system.h"
#include <stdlib.h>
#include <stdio.h>

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
    /*     print_matrix(system->distance_matrix);
        print_matrix(system->pheromone_matrix);
        for (int i = 0; i < n_ants; i++)
            print_vector(system->ants[i].path, n_cities); */

    return system;
}

ANT *initialize_ants(int n_ants, int n_cities)
{
    ANT *ants = (ANT *)malloc(sizeof(ANT) * n_ants);
    for (int i = 0; i < n_ants; i++)
    {
        ants[i].path = (int *)malloc(sizeof(int) * n_cities);
        for (int j = 0; j < n_cities; j++)
            ants[i].path[j] = -1;
        ants[i].cost = 0;
        ants[i].current_city = rand() % n_cities;
        ants[i].path[0] = ants[i].current_city;
    }
    return ants;
}

void free_system(SYSTEM *system)
{
    free_matrix(system->distance_matrix);
    free_matrix(system->pheromone_matrix);
    free(system->ants);
    free(system->best_path);
    free(system);
}