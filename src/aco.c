#include <stdlib.h>
#include <stdio.h>
#include "aco.h"
#include "graph.h"
#include "system.h"
#include "util.h"

void pheromone_update(SYSTEM *system)
{
    for (int i = 0; i < system->distance_matrix->n; i++)
        for (int j = 0; j < system->distance_matrix->n; j++)
            system->pheromone_matrix->adj[i][j] *= system->evaporation_rate;
    for (int i = 0; i < system->distance_matrix->n - 1; i++)
    {
        system->pheromone_matrix->adj[system->best_path[i]][system->best_path[i + 1]] += 1.0;
    }
    system->pheromone_matrix->adj[system->best_path[system->distance_matrix->n - 1]][system->best_path[0]] += 1.0;
}

void calculate_best_path(SYSTEM *s)
{
    int best_ant = 0;
    for (int i = 0; i < s->distance_matrix->n; i++)
        if (s->ants[i].cost < s->best_cost)
        {
            s->best_cost = s->ants[i].cost;
            best_ant = i;
        }
    for (int j = 0; j < s->distance_matrix->n; j++)
        s->best_path[j] = s->ants[best_ant].path[j];
}

double *calculate_probabilities(SYSTEM *system, ANT *ant)
{
    double *probabilities = malloc(sizeof(double) * system->distance_matrix->n);
    double *pheromones = malloc(sizeof(double) * system->distance_matrix->n);
    double sum = 0;

    for (int i = 0; i < system->distance_matrix->n; i++)
    {
        pheromones[i] = system->pheromone_matrix->adj[ant->current_city][i];
        for (int j = 0; j < system->distance_matrix->n; j++)
        {
            if (ant->path[j] == i)
            {
                pheromones[i] = 0;
                break;
            }
        }
        sum += pheromones[i];
    }

    for (int i = 0; i < system->distance_matrix->n; i++)
    {
        probabilities[i] = pheromones[i] / sum;
    }
    return probabilities;
}

void ant_movement(SYSTEM *system, int n_ant)
{

    double *probabilities = calculate_probabilities(system, &system->ants[n_ant]);

    double r = random_zero_one();
    double sum = 0;
    int next_city = 0;

    for (int i = 0; i < system->distance_matrix->n; i++)
    {

        sum += probabilities[i];
        if (r <= sum)
        {
            next_city = i;
            break;
        }
    }

    //   free(probabilities);

    system->ants[n_ant].step++;
    system->ants[n_ant].path[system->ants[n_ant].step] = next_city;
    system->ants[n_ant].current_city = next_city;
    system->ants[n_ant].cost += system->distance_matrix->adj[system->ants[n_ant].path[system->ants[n_ant].step - 1]][next_city];
}

int *aco(SYSTEM *system, int n_iterations)
{
    int n = 0;
    while (n < n_iterations)
    {
        system->ants = initialize_ants(system->n_ants, system->distance_matrix->n);
        for (int i = 0; i < system->distance_matrix->n; i++)
        {
            for (int j = 0; j < system->n_ants; j++)
            {
                ant_movement(system, j);
                printf("Ant %d path: ", j);
                print_vector(system->ants[j].path, system->distance_matrix->n);
            }
        }
        calculate_best_path(system);
        pheromone_update(system);
        n++;
    }
    return system->best_path;
}
