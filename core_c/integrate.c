#include "integrate.h"

#include <math.h>
#include <omp.h>

void bh_kick(Vec3 *vel, const Vec3 *acc, int64_t n, double dt)
{
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++) {
        vel[i].x += acc[i].x * dt;
        vel[i].y += acc[i].y * dt;
        vel[i].z += acc[i].z * dt;
    }
}

void bh_drift(Vec3 *pos, const Vec3 *vel, int64_t n, double dt)
{
    #pragma omp parallel for schedule(static)
    for (int64_t i = 0; i < n; i++) {
        pos[i].x += vel[i].x * dt;
        pos[i].y += vel[i].y * dt;
        pos[i].z += vel[i].z * dt;
    }
}

double bh_kinetic_energy(const Vec3 *vel, const double *mass, int64_t n)
{
    double k = 0.0;
    #pragma omp parallel for reduction(+:k) schedule(static)
    for (int64_t i = 0; i < n; i++) {
        k += 0.5 * mass[i] * (vel[i].x * vel[i].x + vel[i].y * vel[i].y + vel[i].z * vel[i].z);
    }
    return k;
}

/* O(N²): solo se llama cuando el usuario pide --energy-log y N es pequeño. */
double bh_potential_energy(const Vec3 *pos, const double *mass, int64_t n, double eps)
{
    double eps2 = eps * eps;
    double u = 0.0;
    #pragma omp parallel for reduction(+:u) schedule(dynamic, 64)
    for (int64_t i = 0; i < n; i++) {
        double row = 0.0;
        for (int64_t j = i + 1; j < n; j++) {
            double dx = pos[j].x - pos[i].x;
            double dy = pos[j].y - pos[i].y;
            double dz = pos[j].z - pos[i].z;
            row -= mass[i] * mass[j] / sqrt(dx * dx + dy * dy + dz * dz + eps2);
        }
        u += row;
    }
    return u;
}
