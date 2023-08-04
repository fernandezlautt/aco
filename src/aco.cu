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

    int *d_best_path = nullptr;
    int *d_best_ant_path = nullptr;

    cudaMalloc(&d_best_path, sizeof(int) * s->distance_matrix->n);
    cudaMalloc(&d_best_ant_path, sizeof(int) * s->distance_matrix->n);

    for (i = 0; i < s->n_ants; i++)
        if (s->ants[i].cost < s->best_cost)
        {
            s->best_cost = s->ants[i].cost;
            best_ant = i;
        }

    cudaMemcpy(d_best_ant_path, s->ants[best_ant].path, sizeof(int) * s->distance_matrix->n, cudaMemcpyHostToDevice);

    copy_vector<<<1, s->distance_matrix->n>>>(d_best_path, d_best_ant_path);

    cudaMemcpy(s->best_path, d_best_path, sizeof(int) * s->distance_matrix->n, cudaMemcpyDeviceToHost);

    cudaFree(d_best_path);
    cudaFree(d_best_ant_path);
}

/*
Calculate probabilities of visit each city (SEQUENTIAL VERSION)
    - The probability of visit a city is proportional to the pheromone level
    - P[i]=pheromone[i]/sum(pheromone)
    - The probability of visit a city is zero if the ant has already visited that city
*/
double *calculate_probabilities(SYSTEM *system, ANT *ant)
{
    double *probabilities = (double *)malloc(sizeof(double) * system->distance_matrix->n);
    double *pheromones = (double *)malloc(sizeof(double) * system->distance_matrix->n);
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

double *calculate_probabilities_parallel(SYSTEM *system, ANT *ant)
{
    double *probabilities = (double *)malloc(sizeof(double) * system->distance_matrix->n);
    double *pheromones = (double *)malloc(sizeof(double) * system->distance_matrix->n);
    double sum = 0;
    int i;

    double *d_pheromones = nullptr;
    double *d_probabilities = nullptr;
    bool *d_ant_visited = nullptr;

    for (i = 0; i < system->distance_matrix->n; i++)
    {
        if (ant->visited[i])
        {
        }
        else
        {
            pheromones[i] = system->pheromone_matrix->adj[ant->current_city][i];
            sum += pheromones[i];
        }
    }

    cudaMalloc(&d_pheromones, sizeof(double) * system->distance_matrix->n);
    cudaMalloc(&d_probabilities, sizeof(double) * system->distance_matrix->n);
    cudaMalloc(&d_ant_visited, sizeof(bool) * system->distance_matrix->n);

    cudaMemcpy(d_pheromones, pheromones, sizeof(double) * system->distance_matrix->n, cudaMemcpyHostToDevice);
    cudaMemcpy(d_ant_visited, ant->visited, sizeof(bool) * system->distance_matrix->n, cudaMemcpyHostToDevice);

    probabilities_calculation<<<1, (system->distance_matrix->n)>>>(d_pheromones, d_probabilities, sum, d_ant_visited);

    cudaMemcpy(probabilities, d_probabilities, sizeof(double) * system->distance_matrix->n, cudaMemcpyDeviceToHost);

    cudaFree(d_pheromones);
    cudaFree(d_probabilities);
    cudaFree(d_ant_visited);

    return probabilities;
}

/*
Move the ant to the next city
    - The next city is chosen based on the probabilities of visit each city
*/
void ant_movement(SYSTEM *system, int n_ant)
{
    // sequential version
    // double *probabilities = calculate_probabilities(system, &system->ants[n_ant]);
    double *probabilities = calculate_probabilities_parallel(system, &system->ants[n_ant]);
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
    system->ants[n_ant].visited[next_city] = true;
    system->ants[n_ant].cost += system->distance_matrix->adj[system->ants[n_ant].path[system->ants[n_ant].step - 1]][next_city];
}

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
    RESULT *result = initialize_result(system->distance_matrix->n, n_iterations);
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
