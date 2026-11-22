"""Códigos de Morton de 63 bits (21 bits por eje) y orden de las partículas por ese código.

Convención de bits (acordada con tree.py):
    en cada nivel, el octante es  octante = (bit_x << 2) | (bit_y << 1) | bit_z
    el octante del nivel L (raíz = nivel 0) se lee con  (code >> 3 * (20 - L)) & 7
"""
import numpy as np
from numba import njit

BITS_PER_AXIS = 21          # 3 ejes * 21 bits = 63 bits: cabe en un uint64
CUBE_MARGIN = 1e-6          # margen relativo: la partícula del borde superior no se sale de los 21 bits


def local_bounds(pos):
    """Esquinas mínima y máxima de las partículas.

    pos float64[N,3] -> (lo float64[3], hi float64[3]).
    En MPI cada proceso la llama con sus partículas y luego se hace Allreduce (MIN para lo, MAX para hi).
    """
    lo = pos.min(axis=0)
    hi = pos.max(axis=0)
    return lo, hi


def cube_from_bounds(lo, hi):
    """Cubo que contiene la caja [lo, hi], centrado en ella.

    -> (origin float64[3] = esquina mínima del cubo, size float = lado del cubo).
    Es un cubo y no una caja porque las celdas del octree son cúbicas (node_half es un solo número).
    """
    size = float(np.max(hi - lo))
    if size == 0.0:         # todas las partículas en el mismo punto (o N = 1)
        size = 1.0
    size = size * (1.0 + CUBE_MARGIN)
    center = 0.5 * (lo + hi)
    origin = center - 0.5 * size
    return origin, size


@njit
def to_grid(value, origin, size):
    """Coordenada real -> entero en [0, 2^21 - 1]: índice de la celda más fina en ese eje."""
    n_cells = 1 << BITS_PER_AXIS
    i = int((value - origin) / size * n_cells)
    if i < 0:                   # recorte por seguridad (redondeo en los bordes)
        i = 0
    if i > n_cells - 1:
        i = n_cells - 1
    return i


@njit
def interleave_bits(ix, iy, iz):
    """Intercala los 21 bits de ix, iy, iz en un código de 63 bits, empezando por la raíz.

    En cada nivel se toma un bit de cada eje (del más significativo al menos significativo),
    se arma el octante de 3 bits y se agrega a la derecha del código.
    Se trabaja en int64 (63 bits caben sin signo) para no mezclar uint64 con int64 en Numba.
    """
    code = 0
    for level in range(BITS_PER_AXIS):          # level 0 = raíz
        bit = BITS_PER_AXIS - 1 - level
        bit_x = (ix >> bit) & 1
        bit_y = (iy >> bit) & 1
        bit_z = (iz >> bit) & 1
        octant = (bit_x << 2) | (bit_y << 1) | bit_z
        code = (code << 3) | octant
    return code


@njit
def morton_codes(pos, origin, size):
    """Código de Morton de cada partícula.

    pos float64[N,3], origin float64[3], size float -> codes uint64[N].
    Todas las partículas (y en MPI todos los procesos) deben usar el mismo origin y size.
    """
    n = pos.shape[0]
    codes = np.empty(n, dtype=np.uint64)
    for i in range(n):
        ix = to_grid(pos[i, 0], origin[0], size)
        iy = to_grid(pos[i, 1], origin[1], size)
        iz = to_grid(pos[i, 2], origin[2], size)
        codes[i] = interleave_bits(ix, iy, iz)
    return codes


def morton_order(codes):
    """Permutación que ordena las partículas por código de Morton.

    codes uint64[N] -> perm int64[N]. Uso:
        pos_s = pos[perm]            # ordenar (igual con vel, mass, codes)
        acc_orig[perm] = acc_s       # volver al orden original (para --dump-forces)
    "stable" deja en su orden de entrada a las partículas con el mismo código (misma celda fina),
    así el resultado es reproducible.
    """
    return np.argsort(codes, kind="stable")
