"""Octree en arreglos planos, construido a partir de las partículas ordenadas por código de Morton.

Idea: con las partículas ordenadas por Morton, cada nodo del árbol es un rango contiguo
[body_start, body_start + body_count) del arreglo. Las partículas de un nodo del nivel L comparten los
L primeros dígitos (octantes) de su código, así que el dígito L viene ordenado dentro del rango: los
hijos son los tramos consecutivos con el mismo dígito. No se mueve ninguna partícula.

Convención de bits (la de morton.py): octante = (bit_x << 2) | (bit_y << 1) | bit_z, y el octante del
nivel L (raíz = 0) es (code >> 3 * (20 - L)) & 7.

Uso típico (partículas ya ordenadas con la permutación de morton_order):
    tree = build_tree(codes, origin, size)      # topología + node_next
    compute_mass_com(tree, pos, mass)           # masa y centro de masa, de abajo hacia arriba
"""
from collections import namedtuple

import numpy as np
from numba import njit

MAX_DEPTH = 21      # PUNTO DE MODIFICACIÓN: nivel máximo (raíz = 0). No puede pasar de 21: el código
                    # de Morton solo tiene 21 dígitos. Al llegar aquí la hoja guarda todas las que queden.
LEAF_SIZE = 1       # PUNTO DE MODIFICACIÓN: un nodo con a lo sumo LEAF_SIZE partículas no se divide.

Tree = namedtuple("Tree", [
    "node_center",       # float64[M,3]  centro geométrico de la celda
    "node_half",         # float64[M]    mitad del lado de la celda
    "node_mass",         # float64[M]    masa total
    "node_com",          # float64[M,3]  centro de masa
    "node_first_child",  # int64[M]      índice del primer hijo (-1 si es hoja); los hijos son contiguos
    "node_n_children",   # int64[M]      número de hijos (0 = hoja)
    "node_body_start",   # int64[M]      primera partícula del rango
    "node_body_count",   # int64[M]      número de partículas del rango
    "node_next",         # int64[M]      siguiente nodo si se salta este subárbol (-1 = fin)
])


@njit
def octant_at_level(code, level):
    """Dígito (octante, 0 a 7) del nivel `level` de un código de Morton.

    Se pasa a int64 antes de desplazar: en Numba, uint64 con int64 da float64 y se perderían bits.
    """
    return (np.int64(code) >> (3 * (20 - level))) & 7


@njit
def build_tree_with_capacity(codes, origin, size, capacity):
    """Construye el árbol con espacio para `capacity` nodos. Devuelve (tree, True), o (None, False) si no
    alcanzó el espacio. Los nodos se procesan en el orden en que se crean (por niveles), y los hijos de
    cada nodo se agregan al final: quedan contiguos y con índice mayor que el de su padre.
    """
    n = codes.shape[0]
    center = np.zeros((capacity, 3))
    half = np.zeros(capacity)
    first_child = np.full(capacity, -1, dtype=np.int64)
    n_children = np.zeros(capacity, dtype=np.int64)
    body_start = np.zeros(capacity, dtype=np.int64)
    body_count = np.zeros(capacity, dtype=np.int64)
    level = np.zeros(capacity, dtype=np.int64)

    # Raíz: el cubo completo con todas las partículas.
    for d in range(3):
        center[0, d] = origin[d] + 0.5 * size
    half[0] = 0.5 * size
    body_count[0] = n
    n_nodes = 1

    node = 0
    while node < n_nodes:
        start = body_start[node]
        count = body_count[node]
        # Hoja: pocas partículas, o se llegó a MAX_DEPTH, o se acabaron los 21 dígitos del código
        # (esto último protege contra un MAX_DEPTH mayor que 21).
        if count <= LEAF_SIZE or level[node] >= MAX_DEPTH or level[node] >= 21:
            node += 1
            continue

        # Recorre el rango y crea un hijo por cada tramo de partículas con el mismo octante.
        first_child[node] = n_nodes
        child_half = 0.5 * half[node]
        k = start
        while k < start + count:
            octant = octant_at_level(codes[k], level[node])
            k_end = k + 1
            while k_end < start + count and octant_at_level(codes[k_end], level[node]) == octant:
                k_end += 1

            if n_nodes == capacity:
                return None, False
            child = n_nodes
            n_nodes += 1
            # El bit del eje decide si el hijo está en la mitad baja (-) o alta (+) de ese eje.
            bit_x = (octant >> 2) & 1
            bit_y = (octant >> 1) & 1
            bit_z = octant & 1
            center[child, 0] = center[node, 0] + (2 * bit_x - 1) * child_half
            center[child, 1] = center[node, 1] + (2 * bit_y - 1) * child_half
            center[child, 2] = center[node, 2] + (2 * bit_z - 1) * child_half
            half[child] = child_half
            body_start[child] = k
            body_count[child] = k_end - k
            level[child] = level[node] + 1
            n_children[node] += 1
            k = k_end
        node += 1

    m = n_nodes
    tree = Tree(
        center[:m].copy(), half[:m].copy(), np.zeros(m), np.zeros((m, 3)),
        first_child[:m].copy(), n_children[:m].copy(),
        body_start[:m].copy(), body_count[:m].copy(), np.full(m, -1, dtype=np.int64),
    )
    return tree, True


@njit
def build_tree(codes, origin, size):
    """codes uint64[N] (ordenados), origin float64[3], size float -> Tree.

    Solo la topología: node_mass y node_com quedan en cero (los llena compute_mass_com).
    Al final enlaza node_next.
    """
    capacity = 2 * codes.shape[0] + 64
    while True:
        tree, ok = build_tree_with_capacity(codes, origin, size, capacity)
        if ok:
            break
        capacity *= 2           # no alcanzó: se repite con el doble de espacio
    link_next(tree)
    return tree


@njit
def compute_mass_com(tree, pos, mass):
    """Masa total y centro de masa de cada nodo, en el lugar, de abajo hacia arriba.

    Recorre los nodos de atrás hacia adelante: como cada hijo tiene índice mayor que su padre, los hijos
    ya están calculados cuando se llega al padre. pos y mass son las partículas ORDENADAS por Morton.
    Una hoja con body_count = 0 (pseudo-hoja del LET) conserva la masa y el centro de masa que trae.
    """
    for node in range(tree.node_half.shape[0] - 1, -1, -1):
        total = 0.0
        mx = 0.0
        my = 0.0
        mz = 0.0
        if tree.node_n_children[node] == 0:
            if tree.node_body_count[node] == 0:
                continue                            # pseudo-hoja: se deja como está
            start = tree.node_body_start[node]
            for b in range(start, start + tree.node_body_count[node]):
                total += mass[b]
                mx += mass[b] * pos[b, 0]
                my += mass[b] * pos[b, 1]
                mz += mass[b] * pos[b, 2]
        else:
            first = tree.node_first_child[node]
            for c in range(first, first + tree.node_n_children[node]):
                m = tree.node_mass[c]
                total += m
                mx += m * tree.node_com[c, 0]
                my += m * tree.node_com[c, 1]
                mz += m * tree.node_com[c, 2]

        tree.node_mass[node] = total
        if total > 0.0:
            tree.node_com[node, 0] = mx / total
            tree.node_com[node, 1] = my / total
            tree.node_com[node, 2] = mz / total
        else:                                       # masa cero: el centro geométrico evita dividir por 0
            tree.node_com[node, 0] = tree.node_center[node, 0]
            tree.node_com[node, 1] = tree.node_center[node, 1]
            tree.node_com[node, 2] = tree.node_center[node, 2]


@njit
def link_next(tree):
    """Llena node_next: el nodo que sigue en el recorrido en profundidad si se SALTA el subárbol del nodo.

    Para un hijo que no es el último, el siguiente es su hermano (los hijos son contiguos). Para el último
    hijo, es el siguiente de su padre. La raíz no tiene siguiente (-1). Así el recorrido es:
        abrir un nodo  -> ir a node_first_child;   aceptarlo o ser hoja -> ir a node_next.
    Usa una pila explícita y no depende del orden de los índices (sirve también para los árboles del LET).
    """
    m = tree.node_half.shape[0]
    tree.node_next[0] = -1
    stack = np.empty(m, dtype=np.int64)
    top = 0
    stack[top] = 0
    top += 1
    while top > 0:
        top -= 1
        node = stack[top]
        n_child = tree.node_n_children[node]
        first = tree.node_first_child[node]
        for k in range(n_child):
            child = first + k
            if k < n_child - 1:
                tree.node_next[child] = child + 1
            else:
                tree.node_next[child] = tree.node_next[node]
            stack[top] = child
            top += 1
