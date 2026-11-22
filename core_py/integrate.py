"""Integrador leapfrog (kick-drift-kick) y energías.

El ciclo KDK de un paso va en cada main.py, no aquí:

    kick(vel, acc, 0.5 * dt)          # medio paso de velocidad con las fuerzas viejas
    drift(pos, vel, dt)               # paso completo de posición
    acc = <recalcular fuerzas con las posiciones nuevas>
    kick(vel, acc, 0.5 * dt)          # el otro medio paso con las fuerzas nuevas

Leapfrog es simétrico en el tiempo: la energía no deriva, solo oscila alrededor del valor inicial.
"""
import numpy as np
from numba import njit


def kick(vel, acc, dt):
    """Actualiza velocidades en el lugar: v += a · dt. En KDK se llama con dt/2."""
    vel += acc * dt


def drift(pos, vel, dt):
    """Actualiza posiciones en el lugar: x += v · dt."""
    pos += vel * dt


def kinetic_energy(vel, mass):
    """K = Σ ½ m v²."""
    return 0.5 * float(np.sum(mass * np.sum(vel * vel, axis=1)))


@njit
def potential_energy(pos, mass, eps):
    """U = -Σ_{i<j} m_i m_j / sqrt(r² + ε²). O(N²): solo para N pequeño (verificación).

    Con j > i cada par se cuenta una sola vez y nunca se toma j = i.
    """
    n = pos.shape[0]
    eps2 = eps * eps
    u = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            dx = pos[j, 0] - pos[i, 0]
            dy = pos[j, 1] - pos[i, 1]
            dz = pos[j, 2] - pos[i, 2]
            u -= mass[i] * mass[j] / np.sqrt(dx * dx + dy * dy + dz * dz + eps2)
    return u
