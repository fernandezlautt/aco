#include <stdlib.h>

#ifndef MATRIX_H
#define MATRIX_H
typedef struct MATRIX
{
    double **adj;
    int n;
} MATRIX;

void initialize_distances(MATRIX *g);
void initialize_pheromones(MATRIX *g);
void print_matrix(MATRIX *g);
void initialize_matrix(MATRIX *g, int n);
void print_vector(int *v, int n);
void print_vector_double(double *v, int n);
void print_vector_bool(bool *v, int n);
void free_matrix(MATRIX *g);
#endif