#ifndef BH_MORTON_H
#define BH_MORTON_H

#include "common.h"

void bh_local_bounds(const Vec3 *pos, int64_t n, Vec3 *lo, Vec3 *hi);
void bh_cube_from_bounds(const Vec3 *lo, const Vec3 *hi, Vec3 *origin, double *size);
void bh_morton_codes(const Vec3 *pos, int64_t n, Vec3 origin, double size, uint64_t *codes);
void bh_morton_order(const uint64_t *codes, int64_t n, int64_t *perm);

#endif
