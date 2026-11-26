#ifndef BH_COMMON_H
#define BH_COMMON_H

#include <stdint.h>

/* Tres reales contiguos por partícula: equivale a pos[i, :] en NumPy. */
typedef struct {
    double x;
    double y;
    double z;
} Vec3;

#define BH_MAX_DEPTH 21   /* PUNTO DE MODIFICACIÓN: profundidad máxima (raíz = 0) */
#define BH_LEAF_SIZE 1    /* PUNTO DE MODIFICACIÓN: una hoja tiene como máximo esta cantidad de partículas */
#define BH_BITS_PER_AXIS 21
#define BH_CUBE_MARGIN 1e-6

#endif
