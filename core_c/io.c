#include "io.h"

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

int bh_read_input(const char *path, Vec3 **pos, Vec3 **vel, double **mass, int64_t *n)
{
    FILE *file = fopen(path, "r");
    if (file == NULL) {
        fprintf(stderr, "No se puede abrir %s\n", path);
        return -1;
    }

    size_t capacity = 1024;
    size_t count = 0;
    double *buffer = (double *)malloc(capacity * 7 * sizeof(double));
    if (buffer == NULL) abort();

    double values[7];
    while (fscanf(file, "%lf %lf %lf %lf %lf %lf %lf",
                  &values[0], &values[1], &values[2], &values[3],
                  &values[4], &values[5], &values[6]) == 7) {
        if (count == capacity) {
            capacity *= 2;
            double *bigger = (double *)realloc(buffer, capacity * 7 * sizeof(double));
            if (bigger == NULL) abort();
            buffer = bigger;
        }
        for (int d = 0; d < 7; d++) buffer[count * 7 + d] = values[d];
        count++;
    }
    fclose(file);
    if (count == 0) {
        fprintf(stderr, "El archivo %s no tiene partículas\n", path);
        free(buffer);
        return -1;
    }

    *pos = (Vec3 *)malloc(count * sizeof(Vec3));
    *vel = (Vec3 *)malloc(count * sizeof(Vec3));
    *mass = (double *)malloc(count * sizeof(double));
    if (*pos == NULL || *vel == NULL || *mass == NULL) abort();

    for (size_t i = 0; i < count; i++) {
        (*pos)[i].x = buffer[i * 7 + 0];
        (*pos)[i].y = buffer[i * 7 + 1];
        (*pos)[i].z = buffer[i * 7 + 2];
        (*vel)[i].x = buffer[i * 7 + 3];
        (*vel)[i].y = buffer[i * 7 + 4];
        (*vel)[i].z = buffer[i * 7 + 5];
        (*mass)[i] = buffer[i * 7 + 6];
    }
    free(buffer);
    *n = (int64_t)count;
    return 0;
}

void bh_write_forces(const char *path, const Vec3 *acc, int64_t n)
{
    FILE *file = fopen(path, "w");
    if (file == NULL) {
        fprintf(stderr, "No se puede escribir %s\n", path);
        exit(1);
    }
    for (int64_t i = 0; i < n; i++) {
        fprintf(file, "%.17g %.17g %.17g\n", acc[i].x, acc[i].y, acc[i].z);
    }
    fclose(file);
}

void bh_write_energy(const char *path, const EnergyRow *rows, int64_t n)
{
    FILE *file = fopen(path, "w");
    if (file == NULL) {
        fprintf(stderr, "No se puede escribir %s\n", path);
        exit(1);
    }
    fprintf(file, "step,K,U,E\n");
    for (int64_t i = 0; i < n; i++) {
        fprintf(file, "%lld,%.17g,%.17g,%.17g\n",
                (long long)rows[i].step, rows[i].k, rows[i].u, rows[i].e);
    }
    fclose(file);
}
