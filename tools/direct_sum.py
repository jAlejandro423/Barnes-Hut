"""Suma directa O(N²): aceleraciones y energía exactas, la referencia contra la que se verifica cada versión.

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

    Regla común del grupo (CLAUDE.md): el par se salta si r2·sqrt(r2) == 0 (ε = 0 con j = i, o dos
    partículas en el mismo punto). No basta con saltar j = i: dos partículas distintas en el mismo punto
    también darían 0/0. Se compara r³ y no r2 porque con ε muy pequeño (1e-160) r2 no es 0 pero r³ sí.
    Así la referencia salta exactamente los mismos pares que las versiones.
    """
    n = pos.shape[0]
    eps2 = eps * eps
    acc = np.zeros((n, 3))
    for i in range(n):
        ax = 0.0
        ay = 0.0
        az = 0.0
        for j in range(n):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            r2 = dx * dx + dy * dy + dz * dz + eps2
            r3 = r2 * np.sqrt(r2)                      # (r² + ε²)^(3/2)
            if r3 == 0.0:
                continue                               # regla común: par saltado
            factor = mass[j] / r3
            ax += factor * dx
            ay += factor * dy
            az += factor * dz
        acc[i, 0] = ax
        acc[i, 1] = ay
        acc[i, 2] = az
    return acc


def direct_kinetic_energy(vel, mass):
    """K = Σ ½ m v²."""
    return 0.5 * float(np.sum(mass * np.sum(vel * vel, axis=1)))


@njit
def direct_potential_energy(pos, mass, eps):
    """U = -Σ_{i<j} m_i m_j / sqrt(r² + ε²). O(N²).

    Salta los mismos pares que direct_accelerations (r2·sqrt(r2) == 0).
    Es la referencia de energía de verify.py. Está aquí, y no se importa de core_py, para que la
    verificación no dependa del código que verifica: si core_py tuviera un bug en la energía,
    la versión y el verificador fallarían igual y la prueba pasaría sin verlo.
    """
    n = pos.shape[0]
    eps2 = eps * eps
    u = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            r2 = dx * dx + dy * dy + dz * dz + eps2
            r = np.sqrt(r2)
            if r2 * r == 0.0:
                continue                               # regla común: los mismos pares que las fuerzas
            u -= mass[i] * mass[j] / r
    return u


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
