"""Recorrido del octree con el criterio θ, sin recursión (punteros node_next), y costo por partícula.

Aceleración suavizada (G = 1) que ejerce una masa m situada a (dx, dy, dz) de la partícula objetivo:
    a = m (dx, dy, dz) / (dx² + dy² + dz² + ε²)^(3/2)
con (dx, dy, dz) = posición de la fuente - posición del objetivo.

La misma función sirve para el árbol local y para cada árbol del LET (ver CLAUDE.md raíz).
"""
import math

import numpy as np
from numba import njit


@njit
def pair_accel(dx, dy, dz, m, eps2):
    """Aceleración que produce una masa puntual m situada a (dx, dy, dz) del objetivo. -> (ax, ay, az).

    La usan las partículas de las hojas, las celdas aceptadas y las pseudo-hojas del LET.
    Si r²·sqrt(r²) da 0 se salta el par (devuelve 0): pasa con la propia partícula o con dos partículas
    en el mismo punto cuando ε = 0, o con ε tan pequeño que r³ se redondea a 0. Sin esto, Numba lanza
    ZeroDivisionError (y en C saldría NaN en silencio). Con ε > 0 normal nunca se cumple.
    No se comparan índices: fuentes y objetivos son arreglos distintos.
    """
    r2 = dx * dx + dy * dy + dz * dz + eps2
    r3 = r2 * math.sqrt(r2)
    if r3 == 0.0:
        return 0.0, 0.0, 0.0
    factor = m / r3
    return factor * dx, factor * dy, factor * dz


@njit
def accept_cell(half, d2, theta):
    """¿Se puede tratar la celda como una sola masa en su centro de masa? -> bool.

    Criterio: tamaño / distancia < θ, con tamaño = 2·half y d2 = distancia² al centro de masa.
    Se compara al cuadrado para no calcular una raíz: (2·half)² < θ²·d².
    Con θ = 0 nunca se acepta una celda (equivale a la suma directa).

    PUNTO DE MODIFICACIÓN: criterio de apertura θ.
    Ojo: con la distancia al centro de masa, si θ ≥ 1/sqrt(3) ≈ 0.58 una celda puede aceptarse aunque
    contenga a la propia partícula objetivo (el centro de masa puede estar hasta a sqrt(3)·tamaño de ella).
    Alternativas: medir la distancia al centro geométrico (node_center), o exigir además d > tamaño.
    """
    size = 2.0 * half
    return size * size < theta * theta * d2


@njit
def accel_one(tree, src_pos, src_mass, x, y, z, theta, eps):
    """Aceleración sobre un punto (x, y, z) recorriendo el árbol sin recursión. -> (ax, ay, az, n_inter).

    src_pos / src_mass son las partículas a las que apuntan node_body_start y node_body_count.
    n_inter = número de interacciones evaluadas (partículas + celdas): es el costo de esta partícula.

    Recorrido con punteros de salto, empezando en la raíz (nodo 0):
        hoja con partículas -> suma directa sobre ellas,  sigue en node_next
        pseudo-hoja (LET)   -> masa puntual en node_com,   sigue en node_next
        celda aceptada (θ)  -> masa puntual en node_com,   sigue en node_next
        celda rechazada     -> se abre: baja a node_first_child
    Termina cuando node_next es -1 (se pasó el último nodo).
    """
    eps2 = eps * eps
    ax = 0.0
    ay = 0.0
    az = 0.0
    n_inter = 0
    node = 0
    while node != -1:
        if tree.node_n_children[node] == 0:                 # hoja
            count = tree.node_body_count[node]
            if count > 0:                                   # hoja con partículas: suma directa
                start = tree.node_body_start[node]
                for b in range(start, start + count):
                    fx, fy, fz = pair_accel(src_pos[b, 0] - x, src_pos[b, 1] - y, src_pos[b, 2] - z,
                                            src_mass[b], eps2)
                    ax += fx
                    ay += fy
                    az += fz
                    n_inter += 1
            else:                                           # pseudo-hoja del LET: siempre masa puntual
                fx, fy, fz = pair_accel(tree.node_com[node, 0] - x, tree.node_com[node, 1] - y,
                                        tree.node_com[node, 2] - z, tree.node_mass[node], eps2)
                ax += fx
                ay += fy
                az += fz
                n_inter += 1
            node = tree.node_next[node]
            continue

        dx = tree.node_com[node, 0] - x
        dy = tree.node_com[node, 1] - y
        dz = tree.node_com[node, 2] - z
        if accept_cell(tree.node_half[node], dx * dx + dy * dy + dz * dz, theta):
            fx, fy, fz = pair_accel(dx, dy, dz, tree.node_mass[node], eps2)
            ax += fx
            ay += fy
            az += fz
            n_inter += 1
            node = tree.node_next[node]                     # se salta todo el subárbol
        else:
            node = tree.node_first_child[node]              # se abre la celda
    return ax, ay, az, n_inter


@njit
def compute_accelerations(tree, src_pos, src_mass, tgt_pos, theta, eps):
    """Aceleración y costo de cada partícula objetivo. -> (acc float64[M,3], cost int64[M]).

    tree, src_pos, src_mass: el árbol y las partículas a las que apunta (ordenadas por Morton).
    tgt_pos float64[M,3]: dónde se calcula la aceleración. En la versión secuencial es src_pos completo;
    en python_mp y MPI es solo el tramo de partículas de ese proceso.
    Con el LET se llama una vez por árbol (local y cada remoto) y se suman acc y cost.

    Bucle simple, sin parallel=True: el paralelismo lo ponen python_mp y MPI con procesos.
    """
    m = tgt_pos.shape[0]
    acc = np.zeros((m, 3))
    cost = np.zeros(m, dtype=np.int64)
    for i in range(m):
        ax, ay, az, n_inter = accel_one(tree, src_pos, src_mass,
                                        tgt_pos[i, 0], tgt_pos[i, 1], tgt_pos[i, 2], theta, eps)
        acc[i, 0] = ax
        acc[i, 1] = ay
        acc[i, 2] = az
        cost[i] = n_inter
    return acc, cost
