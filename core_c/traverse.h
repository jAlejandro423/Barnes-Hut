#ifndef BH_TRAVERSE_H
#define BH_TRAVERSE_H

#include "common.h"
#include "tree.h"

void bh_accel_one(const Tree *tree, const Vec3 *src_pos, const double *src_mass,
                  Vec3 target, double theta, double eps, Vec3 *acc, int64_t *n_inter);
void bh_compute_accelerations(const Tree *tree, const Vec3 *src_pos, const double *src_mass,
                              const Vec3 *tgt, int64_t m, double theta, double eps,
                              Vec3 *acc, int64_t *cost);
void bh_compute_accelerations_omp(const Tree *tree, const Vec3 *src_pos, const double *src_mass,
                                  const Vec3 *tgt, int64_t m, double theta, double eps,
                                  const int64_t *prev_cost, Vec3 *acc, int64_t *cost);

#endif
