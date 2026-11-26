#include <getopt.h>
#include <math.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../core_c/common.h"
#include "../core_c/integrate.h"
#include "../core_c/io.h"
#include "../core_c/morton.h"
#include "../core_c/traverse.h"
#include "../core_c/tree.h"

static const char *PHASES[4] = {"ordenamiento", "arbol", "recorrido", "integracion"};
#define F_ORDER 0
#define F_TREE 1
#define F_TRAVERSE 2
#define F_INTEGRATE 3

typedef struct {
    const char *input;
    int steps;
    double dt;
    double theta;
    double eps;
    int workers;
    const char *partition;
    const char *dump_forces;
    const char *energy_log;
    const char *timing;
} Args;

static Args parse_args(int argc, char **argv)
{
    Args args = {"", 100, 1e-3, 0.5, 1e-3, 1, "morton", NULL, NULL, NULL};
    static struct option options[] = {
        {"input", required_argument, NULL, 'i'},
        {"steps", required_argument, NULL, 's'},
        {"dt", required_argument, NULL, 'T'},
        {"theta", required_argument, NULL, 'h'},
        {"eps", required_argument, NULL, 'p'},
        {"workers", required_argument, NULL, 'w'},
        {"partition", required_argument, NULL, 'P'},
        {"dump-forces", required_argument, NULL, 'f'},
        {"energy-log", required_argument, NULL, 'n'},
        {"timing", required_argument, NULL, 'm'},
        {NULL, 0, NULL, 0}
    };

    opterr = 0;
    int option;
    while ((option = getopt_long(argc, argv, "", options, NULL)) != -1) {
        switch (option) {
            case 'i': args.input = optarg; break;
            case 's': args.steps = atoi(optarg); break;
            case 'T': args.dt = atof(optarg); break;
            case 'h': args.theta = atof(optarg); break;
            case 'p': args.eps = atof(optarg); break;
            case 'w': args.workers = atoi(optarg); break;
            case 'P': args.partition = optarg; break;
            case 'f': args.dump_forces = optarg; break;
            case 'n': args.energy_log = optarg; break;
            case 'm': args.timing = optarg; break;
            default:
                fprintf(stderr, "Opción desconocida: %s\n", argv[optind - 1]);
                exit(1);
        }
    }
    if (args.input[0] == '\0' || args.workers < 1 || args.steps < 0) {
        fprintf(stderr, "Uso: bh --input ARCHIVO [--steps N] [--dt X] [--theta X] [--eps X] "
                        "[--workers P] [--partition morton|orb] [--dump-forces F] "
                        "[--energy-log F] [--timing F]\n");
        exit(1);
    }
    return args;
}

/* Todas las partículas se reordenan con la misma permutación; orig guarda el índice del archivo. */
typedef struct {
    int64_t n;
    Vec3 *pos;
    Vec3 *vel;
    double *mass;
    int64_t *orig;
    Vec3 *pos_tmp;
    Vec3 *vel_tmp;
    double *mass_tmp;
    int64_t *orig_tmp;
    int64_t *perm;
    uint64_t *codes;
    uint64_t *codes_tmp;
    Tree tree;
    Vec3 *acc;
    int64_t *cost;
    int64_t *cost_prev;
} Sim;

static void sim_alloc(Sim *sim, int64_t n)
{
    memset(sim, 0, sizeof(*sim));
    sim->n = n;
    sim->pos = (Vec3 *)malloc((size_t)n * sizeof(Vec3));
    sim->vel = (Vec3 *)malloc((size_t)n * sizeof(Vec3));
    sim->mass = (double *)malloc((size_t)n * sizeof(double));
    sim->orig = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    sim->pos_tmp = (Vec3 *)malloc((size_t)n * sizeof(Vec3));
    sim->vel_tmp = (Vec3 *)malloc((size_t)n * sizeof(Vec3));
    sim->mass_tmp = (double *)malloc((size_t)n * sizeof(double));
    sim->orig_tmp = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    sim->perm = (int64_t *)malloc((size_t)n * sizeof(int64_t));
    sim->codes = (uint64_t *)malloc((size_t)n * sizeof(uint64_t));
    sim->codes_tmp = (uint64_t *)malloc((size_t)n * sizeof(uint64_t));
    sim->acc = (Vec3 *)malloc((size_t)n * sizeof(Vec3));
    sim->cost = (int64_t *)calloc((size_t)n, sizeof(int64_t));
    sim->cost_prev = (int64_t *)calloc((size_t)n, sizeof(int64_t));
    if (!sim->pos || !sim->vel || !sim->mass || !sim->orig || !sim->pos_tmp || !sim->vel_tmp ||
        !sim->mass_tmp || !sim->orig_tmp || !sim->perm || !sim->codes || !sim->codes_tmp ||
        !sim->acc || !sim->cost || !sim->cost_prev) abort();
    for (int64_t i = 0; i < n; i++) sim->orig[i] = i;
}

static void sim_free(Sim *sim)
{
    bh_tree_free(&sim->tree);
    free(sim->pos);
    free(sim->vel);
    free(sim->mass);
    free(sim->orig);
    free(sim->pos_tmp);
    free(sim->vel_tmp);
    free(sim->mass_tmp);
    free(sim->orig_tmp);
    free(sim->perm);
    free(sim->codes);
    free(sim->codes_tmp);
    free(sim->acc);
    free(sim->cost);
    free(sim->cost_prev);
    memset(sim, 0, sizeof(*sim));
}

static double add_time(double times[4], int phase, double start)
{
    double now = omp_get_wtime();
    times[phase] += now - start;
    return now;
}

static void sim_forces(Sim *sim, double theta, double eps, double times[4])
{
    double t = omp_get_wtime();

    Vec3 lo, hi, origin;
    double size;
    bh_local_bounds(sim->pos, sim->n, &lo, &hi);
    bh_cube_from_bounds(&lo, &hi, &origin, &size);
    bh_morton_codes(sim->pos, sim->n, origin, size, sim->codes);
    bh_morton_order(sim->codes, sim->n, sim->perm);
    for (int64_t i = 0; i < sim->n; i++) {
        int64_t j = sim->perm[i];
        sim->pos_tmp[i] = sim->pos[j];
        sim->vel_tmp[i] = sim->vel[j];
        sim->mass_tmp[i] = sim->mass[j];
        sim->orig_tmp[i] = sim->orig[j];
        sim->codes_tmp[i] = sim->codes[j];
    }
    Vec3 *p = sim->pos; sim->pos = sim->pos_tmp; sim->pos_tmp = p;
    Vec3 *v = sim->vel; sim->vel = sim->vel_tmp; sim->vel_tmp = v;
    double *m = sim->mass; sim->mass = sim->mass_tmp; sim->mass_tmp = m;
    int64_t *o = sim->orig; sim->orig = sim->orig_tmp; sim->orig_tmp = o;
    uint64_t *c = sim->codes; sim->codes = sim->codes_tmp; sim->codes_tmp = c;
    t = add_time(times, F_ORDER, t);

    bh_tree_free(&sim->tree);
    if (bh_build_tree(sim->codes, sim->n, origin, size, &sim->tree) != 0) {
        fprintf(stderr, "No se pudo construir el árbol\n");
        exit(1);
    }
    bh_compute_mass_com(&sim->tree, sim->pos, sim->mass);
    t = add_time(times, F_TREE, t);

    bh_compute_accelerations_omp(&sim->tree, sim->pos, sim->mass, sim->pos, sim->n,
                                 theta, eps, sim->cost_prev, sim->acc, sim->cost);
    t = add_time(times, F_TRAVERSE, t);

    int64_t *swap = sim->cost_prev;
    sim->cost_prev = sim->cost;
    sim->cost = swap;
}

static void warm_up(const Sim *full, double theta, double eps)
{
    int64_t k = full->n < 64 ? full->n : 64;
    Sim small;
    sim_alloc(&small, k);
    for (int64_t i = 0; i < k; i++) {
        small.pos[i] = full->pos[i];
        small.vel[i] = full->vel[i];
        small.mass[i] = full->mass[i];
        small.orig[i] = full->orig[i];
    }
    double times[4] = {0.0, 0.0, 0.0, 0.0};
    sim_forces(&small, theta, eps, times);
    sim_free(&small);
}

static EnergyRow energy_row(int64_t step, const Sim *sim, double eps)
{
    EnergyRow row;
    row.step = step;
    row.k = bh_kinetic_energy(sim->vel, sim->mass, sim->n);
    row.u = bh_potential_energy(sim->pos, sim->mass, sim->n, eps);
    row.e = row.k + row.u;
    return row;
}

static void write_timing(const char *path, int64_t n, int workers, const double times[4])
{
    FILE *file = fopen(path, "w");
    if (file == NULL) {
        fprintf(stderr, "No se puede escribir %s\n", path);
        exit(1);
    }
    fprintf(file, "version,N,P,fase,segundos\n");
    for (int phase = 0; phase < 4; phase++) {
        fprintf(file, "omp,%lld,%d,%s,%.6f\n",
                (long long)n, workers, PHASES[phase], times[phase]);
    }
    fclose(file);
}

int main(int argc, char **argv)
{
    Args args = parse_args(argc, argv);

    Vec3 *pos = NULL;
    Vec3 *vel = NULL;
    double *mass = NULL;
    int64_t n = 0;
    if (bh_read_input(args.input, &pos, &vel, &mass, &n) != 0) return 1;

    /* PUNTO DE MODIFICACIÓN: con N grande no se reporta energía (U es O(N²)). */
    if (args.energy_log && n > BH_MAX_N_ENERGY) {
        fprintf(stderr, "Aviso: N = %lld > %d; no se calcula U ni se escribe %s\n",
                (long long)n, BH_MAX_N_ENERGY, args.energy_log);
        args.energy_log = NULL;
    }

    omp_set_num_threads(args.workers);
    printf("N = %lld, pasos = %d, dt = %g, theta = %g, eps = %g\n",
           (long long)n, args.steps, args.dt, args.theta, args.eps);

    Sim sim;
    sim_alloc(&sim, n);
    for (int64_t i = 0; i < n; i++) {
        sim.pos[i] = pos[i];
        sim.vel[i] = vel[i];
        sim.mass[i] = mass[i];
    }
    free(pos);
    free(vel);
    free(mass);

    warm_up(&sim, args.theta, args.eps);

    double times[4] = {0.0, 0.0, 0.0, 0.0};
    double start_total = omp_get_wtime();

    sim_forces(&sim, args.theta, args.eps, times);

    Vec3 *acc_orig = (Vec3 *)malloc((size_t)n * sizeof(Vec3));
    if (acc_orig == NULL) abort();
    if (args.dump_forces) {
        for (int64_t i = 0; i < n; i++) acc_orig[sim.orig[i]] = sim.acc[i];
        bh_write_forces(args.dump_forces, acc_orig, n);
    }
    free(acc_orig);

    EnergyRow *rows = NULL;
    int64_t n_rows = 0;
    if (args.energy_log) {
        rows = (EnergyRow *)malloc((size_t)(args.steps + 1) * sizeof(EnergyRow));
        if (rows == NULL) abort();
        rows[n_rows++] = energy_row(0, &sim, args.eps);
    }

    double half_dt = 0.5 * args.dt;
    for (int step = 1; step <= args.steps; step++) {
        double t = omp_get_wtime();
        bh_kick(sim.vel, sim.acc, n, half_dt);
        bh_drift(sim.pos, sim.vel, n, args.dt);
        add_time(times, F_INTEGRATE, t);

        sim_forces(&sim, args.theta, args.eps, times);

        t = omp_get_wtime();
        bh_kick(sim.vel, sim.acc, n, half_dt);
        add_time(times, F_INTEGRATE, t);

        if (args.energy_log) rows[n_rows++] = energy_row(step, &sim, args.eps);
    }

    if (args.energy_log) {
        bh_write_energy(args.energy_log, rows, n_rows);
        free(rows);
    }
    if (args.timing) write_timing(args.timing, n, args.workers, times);

    double mean_cost = 0.0;
    int64_t max_cost = 0;
    for (int64_t i = 0; i < n; i++) {
        mean_cost += (double)sim.cost_prev[i] / (double)n;
        if (sim.cost_prev[i] > max_cost) max_cost = sim.cost_prev[i];
    }
    double measured = times[0] + times[1] + times[2] + times[3];
    printf("Terminado: %d pasos en %.2f s (medido en fases: %.2f s). "
           "ordenamiento %.2f s, arbol %.2f s, recorrido %.2f s, integracion %.2f s\n",
           args.steps, omp_get_wtime() - start_total, measured,
           times[0], times[1], times[2], times[3]);
    printf("Interacciones por partícula en el último paso: media %.0f, máx %lld\n",
           mean_cost, (long long)max_cost);

    sim_free(&sim);
    return 0;
}
