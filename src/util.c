#include "util.h"
#include <stdlib.h>
#include <stdio.h>

// Random number between 0 and 1
double random_zero_one()
{
    return (double)rand() / (double)RAND_MAX;
}