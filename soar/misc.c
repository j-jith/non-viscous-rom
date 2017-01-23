#include "globals.h"

PetscReal* linspace(PetscReal start, PetscReal stop, unsigned int len)
{
    PetscReal *array, h;
    unsigned int i;

    array = (PetscReal*)malloc(sizeof(PetscReal)*len);

    h = (stop-start)/(len-1);

    for(i = 0; i < len; i++)
    {
        array[i] = start + i*h;
    }

    return array;
}
