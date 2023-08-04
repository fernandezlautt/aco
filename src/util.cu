#include "util.cuh"
#include "system.cuh"
#include <stdlib.h>
#include <stdio.h>

// Random number between 0 and 1
double random_zero_one()
{
    return (double)rand() / (double)RAND_MAX;
}

THREAD *initialize_thread_data(int num_threads, SYSTEM *system)
{
    THREAD *thread_data = (THREAD *)malloc(num_threads * sizeof(THREAD));

    for (int i = 0; i < num_threads; i++)
    {
        thread_data[i].thread_id = i;
        thread_data[i].system = system;
        thread_data[i].num_threads = num_threads;
    }
    return thread_data;
}