#include "graph.h"
#include <stdlib.h>
#include <stdio.h>

// Initialize a matrix with n rows and n columns
void initialize_matrix(MATRIX *g, int n)
{
    g->n = n;
    g->adj = (double **)malloc(n * sizeof(double *));
    for (int i = 0; i < n; i++)
        g->adj[i] = (double *)malloc(n * sizeof(double));
}

// Initialize the distance matrix with random values
void initialize_distances(MATRIX *g)
{
    // symetric matrix with diagonal 0
    for (int i = 0; i < g->n; i++)
        for (int j = 0; j <= i; j++)
            // adj[i][j] has the same value as adj[j][i] and adj[i][i] = 0
            g->adj[i][j] = g->adj[j][i] = i == j ? 0 : rand() % 100;
}

// Initialize the pheromone matrix with 1
void initialize_pheromones(MATRIX *g)
{
    for (int i = 0; i < g->n; i++)
        for (int j = 0; j < g->n; j++)
            g->adj[i][j] = 1;
}

// Print the given matrix
void print_matrix(MATRIX *g)
{
    for (int i = 0; i < g->n; i++)
    {
        for (int j = 0; j < g->n; j++)
            printf("%lf ", g->adj[i][j]);
        printf("\n");
    }
}

// Print the given vector
void print_vector(int *v, int n)
{
    for (int i = 0; i < n; i++)
        printf("%d ", v[i]);
    printf("\n");
}

// Free memory allocated for the matrix
void free_matrix(MATRIX *g)
{
    for (int i = 0; i < g->n; i++)
        free(g->adj[i]);
    free(g->adj);
}