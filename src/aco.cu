#include <stdlib.h>
#include <stdio.h>
#include "aco.cuh"
#include "graph.cuh"
#include "system.cuh"
#include "util.cuh"
#include "utilCuda.cuh"
#include <cuda_runtime.h>
#include <cuda_profiler_api.h>
#include <pthread.h>
#include <math.h>

/*
Update pheromone matrix
    - Only the best path is updated
    - The rest of the matrix is updated with the evaporation rate
*/
void pheromone_update(SYSTEM *system)
{
    int i;

    // Device variables
    double *d_pheromone_matrix = nullptr;

    // Allocate memory in device
    cudaMalloc(&d_pheromone_matrix, sizeof(double) * system->distance_matrix->n * system->distance_matrix->n);

    for (i = 0; i < system->distance_matrix->n; i++)
    {
        cudaMemcpy(&d_pheromone_matrix[i * system->distance_matrix->n], system->pheromone_matrix->adj[i], sizeof(double) * system->distance_matrix->n, cudaMemcpyHostToDevice);
    }

    // Update pheromone matrix
    multiply_matrix_escalar<<<system->distance_matrix->n, system->distance_matrix->n>>>(d_pheromone_matrix, system->evaporation_rate, system->distance_matrix->n);

    // Copy data to host
    for (i = 0; i < system->distance_matrix->n; i++)
    {
        cudaMemcpy(system->pheromone_matrix->adj[i], &d_pheromone_matrix[i * system->distance_matrix->n], sizeof(double) * system->distance_matrix->n, cudaMemcpyDeviceToHost);
    }

    // Free memory in device
    cudaFree(d_pheromone_matrix);

    // Reinforce the best path
    for (i = 0; i < system->distance_matrix->n - 1; i++)
    {
        system->pheromone_matrix->adj[system->best_path[i]][system->best_path[i + 1]] += system->reinforcement_rate;
    }

    // Last city to first city
    system->pheromone_matrix->adj[system->best_path[system->distance_matrix->n - 1]][system->best_path[0]] += system->reinforcement_rate;
}

/*
Verify who ant has the best path and update the best path
*/
void calculate_best_path(SYSTEM *s)
{
    int i;
    int best_ant = 0;
    // Device variables
    int *d_best_path = nullptr;
    int *d_best_ant_path = nullptr;

    // Allocate memory in device
    cudaMalloc(&d_best_path, sizeof(int) * s->distance_matrix->n);
    cudaMalloc(&d_best_ant_path, sizeof(int) * s->distance_matrix->n);

    for (i = 0; i < s->n_ants; i++)
        if (s->ants[i].cost < s->best_cost)
        {
            s->best_cost = s->ants[i].cost;
            best_ant = i;
        }

    // Copy the best path to device
    cudaMemcpy(d_best_ant_path, s->ants[best_ant].path, sizeof(int) * s->distance_matrix->n, cudaMemcpyHostToDevice);

    // Copy path of the best ant to best path
    copy_vector<<<1, s->distance_matrix->n>>>(d_best_path, d_best_ant_path);

    // Copy the best path to host
    cudaMemcpy(s->best_path, d_best_path, sizeof(int) * s->distance_matrix->n, cudaMemcpyDeviceToHost);

    // Free memory in device
    cudaFree(d_best_path);
    cudaFree(d_best_ant_path);
}

/*
Calculate the probabilities of visit each city
    - The probabilities are calculated based on the pheromone matrix and the distance matrix
    - The probabilities are calculated in parallel
*/
double *calculate_probabilities_parallel(SYSTEM *system, ANT *ant)
{
    double *probabilities = (double *)malloc(sizeof(double) * system->distance_matrix->n);
    double sum = 0;
    int i;
    // Device variables
    double *d_pheromones = nullptr;
    double *d_probabilities = nullptr;
    double *d_distances = nullptr;
    bool *d_ant_visited = nullptr;

    // Calculate the sum of the probabilities
    for (i = 0; i < system->distance_matrix->n; i++)
    {
        if (!ant->visited[i])
            sum += pow(system->pheromone_matrix->adj[ant->current_city][i], system->alpha) * pow(1.0 / system->distance_matrix->adj[ant->current_city][i], system->beta);
    }
    // Allocate memory in device
    cudaMalloc(&d_pheromones, sizeof(double) * system->distance_matrix->n);
    cudaMalloc(&d_probabilities, sizeof(double) * system->distance_matrix->n);
    cudaMalloc(&d_distances, sizeof(double) * system->distance_matrix->n);
    cudaMalloc(&d_ant_visited, sizeof(bool) * system->distance_matrix->n);
    // Copy data to device
    cudaMemcpy(d_pheromones, system->pheromone_matrix->adj[ant->current_city], sizeof(double) * system->distance_matrix->n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ant_visited, ant->visited, sizeof(bool) * system->distance_matrix->n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_distances, system->distance_matrix->adj[ant->current_city], sizeof(double) * system->distance_matrix->n, cudaMemcpyHostToDevice);
    // Calculate the probabilities
    probabilities_calculation<<<1, (system->distance_matrix->n)>>>(d_pheromones, d_probabilities, sum, d_ant_visited, d_distances, system->alpha, system->beta);
    // Copy the probabilities to host
    cudaMemcpy(probabilities, d_probabilities, sizeof(double) * system->distance_matrix->n, cudaMemcpyDeviceToHost);

    // Free memory in device
    cudaFree(d_pheromones);
    cudaFree(d_probabilities);
    cudaFree(d_ant_visited);
    cudaFree(d_distances);
    return probabilities;
}

/*
Move the ant to the next city
    - The next city is chosen based on the probabilities of visit each city
*/
void ant_movement(SYSTEM *system, int n_ant)
{
    double *probabilities = calculate_probabilities_parallel(system, &system->ants[n_ant]);
    double r = random_zero_one();
    double sum = 0;
    int next_city = 0;
    int i;

    // Choose the next city
    for (i = 0; i < system->distance_matrix->n; i++)
    {
        sum += probabilities[i];
        if (r <= sum)
        {
            next_city = i;
            break;
        }
    }

    // Free memory
    free(probabilities);

    // Update the ant
    system->ants[n_ant].step++;
    system->ants[n_ant].path[system->ants[n_ant].step] = next_city;
    system->ants[n_ant].current_city = next_city;
    system->ants[n_ant].visited[next_city] = true;
    system->ants[n_ant].cost += system->distance_matrix->adj[system->ants[n_ant].path[system->ants[n_ant].step - 1]][next_city];
}

/*
    Run the thread
    - Each thread is responsible for a part of the ants
*/
void *run_thread(void *arg)
{
    int i;
    int j;
    THREAD *thread = (THREAD *)arg;
    SYSTEM *system = thread->system;

    for (i = 0; i < system->distance_matrix->n - 1; i++)
    {
        for (j = thread->thread_id; j < system->n_ants; j += thread->num_threads)
        {
            ant_movement(system, j);
        }
    }
}

/*
Initialize the ants
-   The cycle is repeated until reach the desired number of iterations
    1- Each ant starts in a random city
    2- Each ant visits each city only once
    3- The best path is calculated
    4- The pheromone matrix is updated
*/
RESULT *aco(SYSTEM *system, int n_iterations, int n_threads)
{
    int n = 0;
    int i;
    // Initialize the result struct
    RESULT *result = initialize_result(system->distance_matrix->n, n_iterations);
    // Allocate threads and thread data
    pthread_t *threads = (pthread_t *)malloc(sizeof(pthread_t) * n_threads);
    THREAD *thread_data = initialize_thread_data(n_threads, system);
    THREAD **thread_data_ptr = (THREAD **)malloc(sizeof(THREAD *) * n_threads);
    for (i = 0; i < n_threads; i++)
    {
        thread_data_ptr[i] = &thread_data[i];
    }

    while (n < n_iterations)
    {
        system->ants = initialize_ants(system->n_ants, system->distance_matrix->n);

        for (i = 0; i < n_threads; i++)
        {
            pthread_create(&threads[i], NULL, run_thread, (void *)thread_data_ptr[i]);
        }

        for (i = 0; i < n_threads; i++)
        {
            pthread_join(threads[i], NULL);
        }

        calculate_best_path(system);
        result->costs[n] = system->best_cost;
        pheromone_update(system);
        free_ants(system->ants, system->n_ants);

        n++;
    }

    free(threads);
    free(thread_data);
    free(thread_data_ptr);
    result->path = system->best_path;

    return result;
}
