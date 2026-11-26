#include "traverse.h"

#include <math.h>
#include <omp.h>
#include <stdlib.h>

static void pair_accel(double dx, double dy, double dz, double m, double eps2, Vec3 *a)
{
    double r2 = dx * dx + dy * dy + dz * dz + eps2;
    double r3 = r2 * sqrt(r2);
    if (r3 == 0.0) {           /* r = 0 con eps = 0: se salta el par, igual que en core_py */
        a->x = a->y = a->z = 0.0;
        return;
    }
    double factor = m / r3;
    a->x = factor * dx;
    a->y = factor * dy;
    a->z = factor * dz;
}

static int accept_cell(double half, double d2, double theta)
{
    /* PUNTO DE MODIFICACIÓN: criterio de apertura θ.
       Alternativas: distancia al centro geométrico o exigir además d > 2·half. */
    double size = 2.0 * half;
    return size * size < theta * theta * d2;
}

void bh_accel_one(const Tree *tree, const Vec3 *src_pos, const double *src_mass,
                  Vec3 target, double theta, double eps, Vec3 *acc, int64_t *n_inter)
{
    double eps2 = eps * eps;
    double ax = 0.0, ay = 0.0, az = 0.0;
    int64_t interactions = 0;
    int64_t node = 0;
    Vec3 part;

    while (node != -1) {
        if (tree->node_n_children[node] == 0) {
            int64_t count = tree->node_body_count[node];
            if (count > 0) {                       /* hoja: suma directa */
                int64_t start = tree->node_body_start[node];
                for (int64_t b = start; b < start + count; b++) {
                    pair_accel(src_pos[b].x - target.x, src_pos[b].y - target.y,
                               src_pos[b].z - target.z, src_mass[b], eps2, &part);
                    ax += part.x;
                    ay += part.y;
                    az += part.z;
                    interactions++;
                }
            } else {                               /* pseudo-hoja del LET: masa puntual */
                pair_accel(tree->node_com[node].x - target.x, tree->node_com[node].y - target.y,
                           tree->node_com[node].z - target.z, tree->node_mass[node], eps2, &part);
                ax += part.x;
                ay += part.y;
                az += part.z;
                interactions++;
            }
            node = tree->node_next[node];
            continue;
        }

        double dx = tree->node_com[node].x - target.x;
        double dy = tree->node_com[node].y - target.y;
        double dz = tree->node_com[node].z - target.z;
        double d2 = dx * dx + dy * dy + dz * dz;
        if (accept_cell(tree->node_half[node], d2, theta)) {
            pair_accel(dx, dy, dz, tree->node_mass[node], eps2, &part);
            ax += part.x;
            ay += part.y;
            az += part.z;
            interactions++;
            node = tree->node_next[node];          /* se salta el subárbol */
        } else {
            node = tree->node_first_child[node];   /* se abre la celda */
        }
    }

    acc->x = ax;
    acc->y = ay;
    acc->z = az;
    *n_inter = interactions;
}

void bh_compute_accelerations(const Tree *tree, const Vec3 *src_pos, const double *src_mass,
                              const Vec3 *tgt, int64_t m, double theta, double eps,
                              Vec3 *acc, int64_t *cost)
{
    for (int64_t i = 0; i < m; i++) {
        bh_accel_one(tree, src_pos, src_mass, tgt[i], theta, eps, &acc[i], &cost[i]);
    }
}

static double weight_at(const int64_t *prev_cost, int64_t i)
{
    if (prev_cost == NULL || prev_cost[i] < 1) return 1.0;
    return (double)prev_cost[i];
}

/* PUNTO DE MODIFICACIÓN: reparto de carga entre hilos.
   Se parte el arreglo ordenado por Morton en P tramos de igual costo acumulado (interacciones del paso
   anterior). Alternativas: schedule(dynamic, chunk), o partir por número de partículas. */
void bh_compute_accelerations_omp(const Tree *tree, const Vec3 *src_pos, const double *src_mass,
                                  const Vec3 *tgt, int64_t m, double theta, double eps,
                                  const int64_t *prev_cost, Vec3 *acc, int64_t *cost)
{
    if (m == 0) return;
    int n_threads = omp_get_max_threads();
    if (n_threads > m) n_threads = (int)m;

    int64_t *bounds = (int64_t *)malloc((size_t)(n_threads + 1) * sizeof(int64_t));
    if (bounds == NULL) abort();
    bounds[0] = 0;
    bounds[n_threads] = m;

    double total = 0.0;
    for (int64_t i = 0; i < m; i++) total += weight_at(prev_cost, i);

    int64_t i = 0;
    double prefix = 0.0;
    for (int p = 1; p < n_threads; p++) {
        double goal = total * (double)p / (double)n_threads;
        while (i < m && prefix < goal) {
            prefix += weight_at(prev_cost, i);
            i++;
        }
        bounds[p] = i;
    }

    #pragma omp parallel for schedule(static)
    for (int p = 0; p < n_threads; p++) {
        for (int64_t k = bounds[p]; k < bounds[p + 1]; k++) {
            bh_accel_one(tree, src_pos, src_mass, tgt[k], theta, eps, &acc[k], &cost[k]);
        }
    }
    free(bounds);
}
