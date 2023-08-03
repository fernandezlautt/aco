#ifndef SYSTEM_H
#define SYSTEM_H

#include "graph.h"

typedef struct ANT
{
    int *path;
    int cost;
    int current_city;
    int step;
} ANT;

typedef struct SYSTEM
{
    ANT *ants;
    int *best_path;
    MATRIX *distance_matrix;
    MATRIX *pheromone_matrix;
    int best_cost;
    int n_ants;
    double evaporation_rate;
    double alpha;
    double beta;
} SYSTEM;

SYSTEM *initialize_system(int n_cities, int n_ants, double alpha, double beta, double evaporation_rate);
ANT *initialize_ants(int n_ants, int n_cities);
void free_system(SYSTEM *system);

#endif