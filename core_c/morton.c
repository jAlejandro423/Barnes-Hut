#include "morton.h"

#include <math.h>
#include <stdint.h>
#include <stdlib.h>

void bh_local_bounds(const Vec3 *pos, int64_t n, Vec3 *lo, Vec3 *hi)
{
    lo->x = lo->y = lo->z = INFINITY;
    hi->x = hi->y = hi->z = -INFINITY;
    for (int64_t i = 0; i < n; i++) {
        if (pos[i].x < lo->x) lo->x = pos[i].x;
        if (pos[i].y < lo->y) lo->y = pos[i].y;
        if (pos[i].z < lo->z) lo->z = pos[i].z;
        if (pos[i].x > hi->x) hi->x = pos[i].x;
        if (pos[i].y > hi->y) hi->y = pos[i].y;
        if (pos[i].z > hi->z) hi->z = pos[i].z;
    }
}

void bh_cube_from_bounds(const Vec3 *lo, const Vec3 *hi, Vec3 *origin, double *size)
{
    double side = fmax(hi->x - lo->x, fmax(hi->y - lo->y, hi->z - lo->z));
    if (side == 0.0) side = 1.0;
    side *= 1.0 + BH_CUBE_MARGIN;
    origin->x = 0.5 * (lo->x + hi->x) - 0.5 * side;
    origin->y = 0.5 * (lo->y + hi->y) - 0.5 * side;
    origin->z = 0.5 * (lo->z + hi->z) - 0.5 * side;
    *size = side;
}

static int64_t to_grid(double value, double origin, double size)
{
    const double n_cells = (double)(1LL << BH_BITS_PER_AXIS);
    int64_t i = (int64_t)((value - origin) / size * n_cells);
    if (i < 0) i = 0;
    if (i > n_cells - 1.0) i = (int64_t)n_cells - 1;
    return i;
}

/* Intercala los 21 bits de cada eje: octante = (bit_x << 2) | (bit_y << 1) | bit_z. */
static uint64_t interleave_bits(int64_t ix, int64_t iy, int64_t iz)
{
    uint64_t code = 0;
    for (int level = 0; level < BH_BITS_PER_AXIS; level++) {
        int bit = BH_BITS_PER_AXIS - 1 - level;
        int octant = (int)(((ix >> bit) & 1) << 2) | (int)(((iy >> bit) & 1) << 1) | (int)((iz >> bit) & 1);
        code = (code << 3) | (uint64_t)octant;
    }
    return code;
}

void bh_morton_codes(const Vec3 *pos, int64_t n, Vec3 origin, double size, uint64_t *codes)
{
    for (int64_t i = 0; i < n; i++) {
        int64_t ix = to_grid(pos[i].x, origin.x, size);
        int64_t iy = to_grid(pos[i].y, origin.y, size);
        int64_t iz = to_grid(pos[i].z, origin.z, size);
        codes[i] = interleave_bits(ix, iy, iz);
    }
}

typedef struct {
    uint64_t code;
    int64_t idx;
} CodeItem;

/* PUNTO DE MODIFICACIÓN: orden de Morton. El desempate por índice hace estable el qsort,
   igual que np.argsort(kind="stable") en core_py/morton.py. */
static int cmp_code_item(const void *a, const void *b)
{
    const CodeItem *x = (const CodeItem *)a;
    const CodeItem *y = (const CodeItem *)b;
    if (x->code < y->code) return -1;
    if (x->code > y->code) return 1;
    if (x->idx < y->idx) return -1;
    if (x->idx > y->idx) return 1;
    return 0;
}

void bh_morton_order(const uint64_t *codes, int64_t n, int64_t *perm)
{
    CodeItem *items = (CodeItem *)malloc((size_t)n * sizeof(CodeItem));
    if (items == NULL) abort();
    for (int64_t i = 0; i < n; i++) {
        items[i].code = codes[i];
        items[i].idx = i;
    }
    qsort(items, (size_t)n, sizeof(CodeItem), cmp_code_item);
    for (int64_t i = 0; i < n; i++) perm[i] = items[i].idx;
    free(items);
}
