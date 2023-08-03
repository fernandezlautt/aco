#include "util.h"
#include <stdlib.h>
#include <stdio.h>

double random_zero_one()
{
    return (double)rand() / (double)RAND_MAX;
}