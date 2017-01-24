#include "globals.h"

PetscReal* linspace(PetscReal start, PetscReal stop, PetscInt len)
{
    PetscReal *array, h;
    PetscInt i;

    PetscMalloc(sizeof(PetscReal)*len, &array);

    h = (stop-start)/(len-1);

    for(i = 0; i < len; i++)
    {
        array[i] = start + i*h;
    }

    return array;
}

