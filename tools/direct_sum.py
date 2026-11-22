"""Suma directa O(N²): aceleraciones exactas, la referencia contra la que se verifica cada versión.

    a_i = Σ_{j≠i}  m_j (r_j - r_i) / (|r_j - r_i|² + ε²)^(3/2)       (G = 1)

Uso:
    python tools/direct_sum.py --input data/prueba/u2k.txt --eps 1e-3 -o acc_directa.txt

Escribe N filas × 3 columnas (ax ay az) en el orden de entrada, el mismo formato de --dump-forces.
Solo para N pequeño (2.000 a 5.000): con N = 200.000 serían 4·10¹⁰ pares por paso.
"""
import argparse
import time

import numpy as np
from numba import njit


@njit
def direct_accelerations(pos, mass, eps):
    """pos float64[N,3], mass float64[N], eps float -> acc float64[N,3].

    Se salta j = i: con ε > 0 esa contribución vale cero, pero con ε = 0 sería 0/0.
    Así la referencia es correcta para cualquier ε.
    """
    n = pos.shape[0]
    eps2 = eps * eps
    acc = np.zeros((n, 3))
    for i in range(n):
        ax = 0.0
        ay = 0.0
        az = 0.0
        for j in range(n):
            if j == i:
                continue
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            r2 = dx * dx + dy * dy + dz * dz + eps2
            factor = mass[j] / (r2 * np.sqrt(r2))      # m_j / (r² + ε²)^(3/2)
            ax += factor * dx
            ay += factor * dy
            az += factor * dz
        acc[i, 0] = ax
        acc[i, 1] = ay
        acc[i, 2] = az
    return acc


def main():
    parser = argparse.ArgumentParser(description="Aceleraciones exactas por suma directa O(N²).")
    parser.add_argument("--input", required=True, help="archivo de 7 columnas x y z vx vy vz m")
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("-o", "--output", required=True, help="archivo de salida N×3")
    args = parser.parse_args()

    data = np.loadtxt(args.input, ndmin=2)
    pos = np.ascontiguousarray(data[:, 0:3])
    mass = np.ascontiguousarray(data[:, 6])

    start = time.perf_counter()
    acc = direct_accelerations(pos, mass, args.eps)      # incluye la compilación de Numba
    seconds = time.perf_counter() - start

    np.savetxt(args.output, acc, fmt="%.17g")
    print(f"N = {pos.shape[0]}, eps = {args.eps}: {seconds:.2f} s (incluye compilar). Escrito en {args.output}")


if __name__ == "__main__":
    main()
