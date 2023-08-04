#ifndef SYSTEM_H
#define SYSTEM_H

#include "graph.cuh"

typedef struct ANT
{
    int *path;
    double cost;
    int current_city;
    int step;
    bool *visited;
} ANT;

typedef struct RESULT
{
    int *path;
    double time;
    double *costs;
} RESULT;

typedef struct SYSTEM
{
    ANT *ants;
    int *best_path;
    MATRIX *distance_matrix;
    MATRIX *pheromone_matrix;
    double best_cost;
    int n_ants;
    double evaporation_rate;
    double reinforcement_rate;
    double alpha;
    double beta;
} SYSTEM;

SYSTEM *initialize_system(int n_cities, int n_ants, double alpha, double beta, double evaporation_rate, double reinforcement_rate);
ANT *initialize_ants(int n_ants, int n_cities);
RESULT *initialize_result(int n_cities, int n_iterations);
void free_system(SYSTEM *system);
void free_ants(ANT *ants, int n_ants);

#endif