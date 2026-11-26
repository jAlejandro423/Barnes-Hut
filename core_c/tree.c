#include "tree.h"

#include <stdint.h>
#include <stdlib.h>

static int tree_reserve(Tree *tree, int64_t capacity)
{
    if (capacity <= tree->capacity) return 0;

    Vec3 *center = (Vec3 *)realloc(tree->node_center, (size_t)capacity * sizeof(Vec3));
    double *half = (double *)realloc(tree->node_half, (size_t)capacity * sizeof(double));
    double *mass = (double *)realloc(tree->node_mass, (size_t)capacity * sizeof(double));
    Vec3 *com = (Vec3 *)realloc(tree->node_com, (size_t)capacity * sizeof(Vec3));
    int64_t *first = (int64_t *)realloc(tree->node_first_child, (size_t)capacity * sizeof(int64_t));
    int64_t *nchild = (int64_t *)realloc(tree->node_n_children, (size_t)capacity * sizeof(int64_t));
    int64_t *start = (int64_t *)realloc(tree->node_body_start, (size_t)capacity * sizeof(int64_t));
    int64_t *count = (int64_t *)realloc(tree->node_body_count, (size_t)capacity * sizeof(int64_t));
    int64_t *next = (int64_t *)realloc(tree->node_next, (size_t)capacity * sizeof(int64_t));
    int64_t *level = (int64_t *)realloc(tree->node_level, (size_t)capacity * sizeof(int64_t));

    if (!center || !half || !mass || !com || !first || !nchild || !start || !count || !next || !level) abort();

    tree->node_center = center;
    tree->node_half = half;
    tree->node_mass = mass;
    tree->node_com = com;
    tree->node_first_child = first;
    tree->node_n_children = nchild;
    tree->node_body_start = start;
    tree->node_body_count = count;
    tree->node_next = next;
    tree->node_level = level;
    tree->capacity = capacity;
    return 0;
}

int bh_tree_init(Tree *tree, int64_t capacity)
{
    tree->node_center = NULL;
    tree->node_half = NULL;
    tree->node_mass = NULL;
    tree->node_com = NULL;
    tree->node_first_child = NULL;
    tree->node_n_children = NULL;
    tree->node_body_start = NULL;
    tree->node_body_count = NULL;
    tree->node_next = NULL;
    tree->node_level = NULL;
    tree->n = 0;
    tree->capacity = 0;
    return tree_reserve(tree, capacity);
}

void bh_tree_free(Tree *tree)
{
    free(tree->node_center);
    free(tree->node_half);
    free(tree->node_mass);
    free(tree->node_com);
    free(tree->node_first_child);
    free(tree->node_n_children);
    free(tree->node_body_start);
    free(tree->node_body_count);
    free(tree->node_next);
    free(tree->node_level);
    tree->node_center = NULL;
    tree->node_half = NULL;
    tree->node_mass = NULL;
    tree->node_com = NULL;
    tree->node_first_child = NULL;
    tree->node_n_children = NULL;
    tree->node_body_start = NULL;
    tree->node_body_count = NULL;
    tree->node_next = NULL;
    tree->node_level = NULL;
    tree->n = 0;
    tree->capacity = 0;
}

/* Octante del nivel `level`; el nivel 0 lee los bits más significativos. */
static int octant_at_level(uint64_t code, int level)
{
    return (int)((code >> (3 * (20 - level))) & 7ULL);
}

static int is_leaf_candidate(int64_t count, int level)
{
    return count <= BH_LEAF_SIZE || level >= BH_MAX_DEPTH || level >= BH_BITS_PER_AXIS;
}

/* El árbol se construye por niveles: todos los nodos de un nivel tienen índices contiguos,
   así que se pueden contar y crear los hijos de cada padre en paralelo. El orden de los hijos
   sigue siendo el de los octantes (el arreglo de códigos está ordenado). */
int bh_build_tree(const uint64_t *codes, int64_t n, Vec3 origin, double size, Tree *tree)
{
    if (bh_tree_init(tree, 2 * n + 64) != 0) return -1;

    tree->node_center[0].x = origin.x + 0.5 * size;
    tree->node_center[0].y = origin.y + 0.5 * size;
    tree->node_center[0].z = origin.z + 0.5 * size;
    tree->node_half[0] = 0.5 * size;
    tree->node_mass[0] = 0.0;
    tree->node_com[0].x = tree->node_com[0].y = tree->node_com[0].z = 0.0;
    tree->node_first_child[0] = -1;
    tree->node_n_children[0] = 0;
    tree->node_body_start[0] = 0;
    tree->node_body_count[0] = n;
    tree->node_level[0] = 0;
    tree->n = 1;

    int64_t level_start = 0;
    int64_t level_end = 1;
    while (level_start < level_end) {
        /* Paso 1: contar los tramos de octantes de cada padre del nivel. */
        #pragma omp parallel for schedule(static)
        for (int64_t p = level_start; p < level_end; p++) {
            int64_t count = tree->node_body_count[p];
            int level = (int)tree->node_level[p];
            int64_t n_children = 0;
            if (!is_leaf_candidate(count, level)) {
                int64_t k = tree->node_body_start[p];
                int64_t end = k + count;
                while (k < end) {
                    int octant = octant_at_level(codes[k], level);
                    n_children++;
                    while (k < end && octant_at_level(codes[k], level) == octant) k++;
                }
            }
            tree->node_n_children[p] = n_children;
            tree->node_first_child[p] = -1;
        }

        /* Paso 2: asignar rangos contiguos de hijos, en el orden de los padres. */
        int64_t next = tree->n;
        for (int64_t p = level_start; p < level_end; p++) {
            int64_t n_children = tree->node_n_children[p];
            if (n_children > 0) {
                tree->node_first_child[p] = next;
                next += n_children;
            }
        }
        if (next > tree->capacity) {
            int64_t new_capacity = 2 * tree->capacity > next ? 2 * tree->capacity : next;
            if (tree_reserve(tree, new_capacity) != 0) return -1;
        }
        tree->n = next;

        /* Paso 3: crear los hijos. Cada padre escribe en un rango distinto del arreglo. */
        #pragma omp parallel for schedule(static)
        for (int64_t p = level_start; p < level_end; p++) {
            int64_t n_children = tree->node_n_children[p];
            if (n_children == 0) continue;
            int64_t child = tree->node_first_child[p];
            int level = (int)tree->node_level[p];
            double child_half = 0.5 * tree->node_half[p];
            int64_t k = tree->node_body_start[p];
            int64_t end = k + tree->node_body_count[p];
            while (k < end) {
                int octant = octant_at_level(codes[k], level);
                int64_t k_end = k + 1;
                while (k_end < end && octant_at_level(codes[k_end], level) == octant) k_end++;

                int bit_x = (octant >> 2) & 1;
                int bit_y = (octant >> 1) & 1;
                int bit_z = octant & 1;
                tree->node_center[child].x = tree->node_center[p].x + (2 * bit_x - 1) * child_half;
                tree->node_center[child].y = tree->node_center[p].y + (2 * bit_y - 1) * child_half;
                tree->node_center[child].z = tree->node_center[p].z + (2 * bit_z - 1) * child_half;
                tree->node_half[child] = child_half;
                tree->node_mass[child] = 0.0;
                tree->node_com[child].x = tree->node_com[child].y = tree->node_com[child].z = 0.0;
                tree->node_first_child[child] = -1;
                tree->node_n_children[child] = 0;
                tree->node_body_start[child] = k;
                tree->node_body_count[child] = k_end - k;
                tree->node_next[child] = -1;
                tree->node_level[child] = level + 1;

                child++;
                k = k_end;
            }
        }
        level_start = level_end;
        level_end = tree->n;
    }

    bh_link_next(tree);
    return 0;
}

void bh_compute_mass_com(Tree *tree, const Vec3 *pos, const double *mass)
{
    /* Los hijos siempre tienen índice mayor que el padre: basta recorrer hacia atrás. */
    for (int64_t node = tree->n - 1; node >= 0; node--) {
        double total = 0.0;
        double mx = 0.0;
        double my = 0.0;
        double mz = 0.0;

        if (tree->node_n_children[node] == 0) {
            if (tree->node_body_count[node] == 0) continue;  /* pseudo-hoja del LET */
            int64_t start = tree->node_body_start[node];
            int64_t count = tree->node_body_count[node];
            for (int64_t b = start; b < start + count; b++) {
                double m = mass[b];
                total += m;
                mx += m * pos[b].x;
                my += m * pos[b].y;
                mz += m * pos[b].z;
            }
        } else {
            int64_t first = tree->node_first_child[node];
            int64_t n_children = tree->node_n_children[node];
            for (int64_t c = first; c < first + n_children; c++) {
                double m = tree->node_mass[c];
                total += m;
                mx += m * tree->node_com[c].x;
                my += m * tree->node_com[c].y;
                mz += m * tree->node_com[c].z;
            }
        }

        tree->node_mass[node] = total;
        if (total > 0.0) {
            tree->node_com[node].x = mx / total;
            tree->node_com[node].y = my / total;
            tree->node_com[node].z = mz / total;
        } else {
            tree->node_com[node] = tree->node_center[node];
        }
    }
}

void bh_link_next(Tree *tree)
{
    tree->node_next[0] = -1;
    int64_t *stack = (int64_t *)malloc((size_t)tree->n * sizeof(int64_t));
    if (stack == NULL) abort();
    int64_t top = 0;
    stack[top++] = 0;
    while (top > 0) {
        int64_t node = stack[--top];
        int64_t n_children = tree->node_n_children[node];
        int64_t first = tree->node_first_child[node];
        for (int64_t k = 0; k < n_children; k++) {
            int64_t child = first + k;
            tree->node_next[child] = (k < n_children - 1) ? child + 1 : tree->node_next[node];
            stack[top++] = child;
        }
    }
    free(stack);
}
