#ifndef BH_IO_H
#define BH_IO_H

#include "common.h"

typedef struct {
    int64_t step;
    double k;
    double u;
    double e;
} EnergyRow;

int bh_read_input(const char *path, Vec3 **pos, Vec3 **vel, double **mass, int64_t *n);
void bh_write_forces(const char *path, const Vec3 *acc, int64_t n);
void bh_write_energy(const char *path, const EnergyRow *rows, int64_t n);

#endif
