

#ifndef UTIL_H
#define UTIL_h

#include "system.cuh"

typedef struct THREAD
{
    SYSTEM *system;
    int thread_id;
    int num_threads;
} THREAD;

double random_zero_one();
THREAD *initialize_thread_data(int num_threads, SYSTEM *system);
#endif