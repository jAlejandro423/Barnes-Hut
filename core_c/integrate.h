#ifndef BH_INTEGRATE_H
#define BH_INTEGRATE_H

#include "common.h"

void bh_kick(Vec3 *vel, const Vec3 *acc, int64_t n, double dt);
void bh_drift(Vec3 *pos, const Vec3 *vel, int64_t n, double dt);
double bh_kinetic_energy(const Vec3 *vel, const double *mass, int64_t n);
double bh_potential_energy(const Vec3 *pos, const double *mass, int64_t n, double eps);

#endif
