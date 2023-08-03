#include <stdlib.h>
#include <stdio.h>
#include "aco.h"
#include "graph.h"
#include "system.h"
#include "util.h"

/*
Update pheromone matrix
    - Only the best path is updated
    - The rest of the matrix is updated with the evaporation rate
*/
void pheromone_update(SYSTEM *system)
{
    int i;
    int j;

    for (i = 0; i < system->distance_matrix->n; i++)
        for (j = 0; j < system->distance_matrix->n; j++)
            system->pheromone_matrix->adj[i][j] *= system->evaporation_rate;

    for (i = 0; i < system->distance_matrix->n - 1; i++)
    {
        system->pheromone_matrix->adj[system->best_path[i]][system->best_path[i + 1]] += 1.0;
    }
    // Last city to first city
    system->pheromone_matrix->adj[system->best_path[system->distance_matrix->n - 1]][system->best_path[0]] += 1.0;
}

/*
Verify who ant has the best path and update the best path
*/
void calculate_best_path(SYSTEM *s)
{
    int i;
    int j;

    int best_ant = 0;
    for (i = 0; i < s->distance_matrix->n; i++)
        if (s->ants[i].cost < s->best_cost)
        {
            s->best_cost = s->ants[i].cost;
            best_ant = i;
        }
    for (j = 0; j < s->distance_matrix->n; j++)
        s->best_path[j] = s->ants[best_ant].path[j];
}

/*
Calculate probabilities of visit each city
    - The probability of visit a city is proportional to the pheromone level
    - P[i]=pheromone[i]/sum(pheromone)
    - The probability of visit a city is zero if the ant has already visited that city
*/
double *calculate_probabilities(SYSTEM *system, ANT *ant)
{
    double *probabilities = malloc(sizeof(double) * system->distance_matrix->n);
    double *pheromones = malloc(sizeof(double) * system->distance_matrix->n);
    double sum = 0;
    int i;
    int j;

    for (i = 0; i < system->distance_matrix->n; i++)
    {
        pheromones[i] = system->pheromone_matrix->adj[ant->current_city][i];
        for (j = 0; j < system->distance_matrix->n; j++)
        {
            if (ant->path[j] == i)
            {
                pheromones[i] = 0;
                break;
            }
        }
        sum += pheromones[i];
    }

    for (i = 0; i < system->distance_matrix->n; i++)
    {
        probabilities[i] = pheromones[i] / sum;
    }
    return probabilities;
}

/*
Move the ant to the next city
    - The next city is chosen based on the probabilities of visit each city
*/
void ant_movement(SYSTEM *system, int n_ant)
{

    double *probabilities = calculate_probabilities(system, &system->ants[n_ant]);

    double r = random_zero_one();
    double sum = 0;
    int next_city = 0;
    int i;

    for (i = 0; i < system->distance_matrix->n; i++)
    {

        sum += probabilities[i];
        if (r <= sum)
        {
            next_city = i;
            break;
        }
    }

    free(probabilities);

    system->ants[n_ant].step++;
    system->ants[n_ant].path[system->ants[n_ant].step] = next_city;
    system->ants[n_ant].current_city = next_city;
    system->ants[n_ant].cost += system->distance_matrix->adj[system->ants[n_ant].path[system->ants[n_ant].step - 1]][next_city];
}

/*
Initialize the ants
-   The cycle is repeated until reach the desired number of iterations
    1- Each ant starts in a random city
    2- Each ant visits each city only once
    3- The best path is calculated
    4- The pheromone matrix is updated
*/
int *aco(SYSTEM *system, int n_iterations)
{
    int n = 0;
    int i;
    int j;

    while (n < n_iterations)
    {
        system->ants = initialize_ants(system->n_ants, system->distance_matrix->n);
        for (i = 0; i < system->distance_matrix->n; i++)
        {
            for (j = 0; j < system->n_ants; j++)
            {
                ant_movement(system, j);
            }
        }
        calculate_best_path(system);
        pheromone_update(system);
        n++;
    }
    return system->best_path;
}
