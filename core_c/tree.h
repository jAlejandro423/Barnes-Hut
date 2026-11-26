#ifndef BH_TREE_H
#define BH_TREE_H

#include "common.h"

/* Los 9 arreglos del contrato (mismos nombres que core_py/tree.py) más datos internos de gestión. */
typedef struct {
    Vec3 *node_center;
    double *node_half;
    double *node_mass;
    Vec3 *node_com;
    int64_t *node_first_child;
    int64_t *node_n_children;
    int64_t *node_body_start;
    int64_t *node_body_count;
    int64_t *node_next;

    int64_t n;               /* nodos usados; la raíz es el índice 0 */
    int64_t capacity;        /* nodos reservados */
    int64_t *node_level;     /* interno: nivel de cada nodo, para la construcción por niveles */
} Tree;

int bh_tree_init(Tree *tree, int64_t capacity);
void bh_tree_free(Tree *tree);
int bh_build_tree(const uint64_t *codes, int64_t n, Vec3 origin, double size, Tree *tree);
void bh_compute_mass_com(Tree *tree, const Vec3 *pos, const double *mass);
void bh_link_next(Tree *tree);

#endif
