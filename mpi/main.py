"""Barnes-Hut con MPI (mpi4py). Por ahora es la versión secuencial (P = 1): el ciclo, el CLI y las salidas
son los definitivos; el reparto por costo, el LET y la fase `comunicacion` se agregan después.

Uso (secuencial, sin mpirun):
    python mpi/main.py --input data/u20k.txt --steps 100 --dump-forces fuerzas.txt --energy-log energia.csv
"""
import argparse
import os
import sys
import time

import numpy as np

# Raíz del repo calculada desde este archivo: funciona desde cualquier carpeta.
REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from core_py.integrate import drift, kick, kinetic_energy, potential_energy             # noqa: E402
from core_py.morton import cube_from_bounds, local_bounds, morton_codes, morton_order  # noqa: E402
from core_py.traverse import compute_accelerations                                     # noqa: E402
from core_py.tree import build_tree, compute_mass_com                                   # noqa: E402


def parse_args():
    """CLI común a todas las versiones (ver CLAUDE.md raíz). Valores por defecto del contrato."""
    parser = argparse.ArgumentParser(description="Barnes-Hut, versión MPI.")
    parser.add_argument("--input", required=True, help="archivo de 7 columnas: x y z vx vy vz m")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=1e-3)
    parser.add_argument("--workers", type=int, default=1, help="por ahora se ignora (P = 1)")
    parser.add_argument("--partition", choices=["morton", "orb"], default="morton",
                        help="por ahora se ignora (P = 1)")
    parser.add_argument("--dump-forces", help="aceleraciones del paso 0, N×3, en el orden de entrada")
    parser.add_argument("--energy-log", help="CSV step,K,U,E (U es O(N²): solo para N pequeño)")
    parser.add_argument("--timing", help="CSV version,N,P,fase,segundos")
    return parser.parse_args()


def read_input(path):
    """Lee el archivo de partículas. -> (pos float64[N,3], vel float64[N,3], mass float64[N]).

    ascontiguousarray: las columnas de un arreglo 2D no son contiguas, y Numba es más rápido con
    arreglos contiguos.
    """
    data = np.loadtxt(path, ndmin=2)
    pos = np.ascontiguousarray(data[:, 0:3])
    vel = np.ascontiguousarray(data[:, 3:6])
    mass = np.ascontiguousarray(data[:, 6])
    return pos, vel, mass


def sort_particles(pos, vel, mass, ids):
    """Ordena las partículas por código de Morton. Se llama en cada paso, porque se mueven.

    -> (pos, vel, mass, ids, codes, origin, size), todo en el nuevo orden.
    Se reordenan TODOS los arreglos de partícula con la misma permutación: si vel quedara en el orden
    viejo, el kick siguiente mezclaría velocidades de partículas distintas.
    ids[i] = índice de la partícula i en el archivo de entrada (para --dump-forces).
    """
    lo, hi = local_bounds(pos)          # en MPI: Allreduce de lo (MIN) y hi (MAX) aquí
    origin, size = cube_from_bounds(lo, hi)
    codes = morton_codes(pos, origin, size)
    perm = morton_order(codes)
    return pos[perm], vel[perm], mass[perm], ids[perm], codes[perm], origin, size


PHASES = ["ordenamiento", "arbol", "recorrido", "integracion"]     # en MPI se agrega "comunicacion"


def add_time(times, phase, start):
    """Suma a times[phase] los segundos desde start. Devuelve el instante actual (inicio de la siguiente fase)."""
    now = time.perf_counter()
    times[phase] += now - start
    return now


def forces(pos, vel, mass, ids, theta, eps, times):
    """Ordena, construye el árbol y calcula las aceleraciones de todas las partículas.

    -> (pos, vel, mass, ids, acc, cost): arreglos de partícula reordenados y acc/cost en ese mismo orden.
    times: diccionario fase -> segundos acumulados; se le suman ordenamiento, arbol y recorrido.
    """
    t = time.perf_counter()
    pos, vel, mass, ids, codes, origin, size = sort_particles(pos, vel, mass, ids)
    t = add_time(times, "ordenamiento", t)
    tree = build_tree(codes, origin, size)
    compute_mass_com(tree, pos, mass)
    t = add_time(times, "arbol", t)
    acc, cost = compute_accelerations(tree, pos, mass, pos, theta, eps)
    add_time(times, "recorrido", t)
    return pos, vel, mass, ids, acc, cost


def warm_up(pos, vel, mass, ids, theta, eps):
    """Primera llamada a las funciones de Numba: compila. Se hace con pocas partículas (Numba compila
    según los tipos, no según el tamaño) y su tiempo se descarta."""
    k = min(64, pos.shape[0])
    forces(pos[:k], vel[:k], mass[:k], ids[:k], theta, eps, dict.fromkeys(PHASES, 0.0))


def write_timing(path, n, n_procs, times):
    """--timing: CSV version,N,P,fase,segundos (segundos = suma de todos los pasos, sin calentamiento)."""
    with open(path, "w") as f:
        f.write("version,N,P,fase,segundos\n")
        for phase in PHASES:
            f.write(f"mpi,{n},{n_procs},{phase},{times[phase]:.6f}\n")


def write_forces(path, acc, ids):
    """--dump-forces: aceleraciones en el orden del archivo de entrada. La fila i de acc es la partícula
    ids[i], así que acc_orig[ids] = acc la devuelve a su fila original.
    %.17g guarda el float64 exacto (verify.py compara con tolerancias finas)."""
    acc_orig = np.empty_like(acc)
    acc_orig[ids] = acc
    np.savetxt(path, acc_orig, fmt="%.17g")


def energy_row(step, pos, vel, mass, eps):
    """(step, K, U, E). U es O(N²): solo se calcula si piden --energy-log."""
    k = kinetic_energy(vel, mass)
    u = potential_energy(pos, mass, eps)
    return step, k, u, k + u


def write_energy(path, rows):
    """--energy-log: CSV step,K,U,E con todos los dígitos (verify.py compara E0 con tolerancia 1e-8)."""
    with open(path, "w") as f:
        f.write("step,K,U,E\n")
        for step, k, u, e in rows:
            f.write(f"{step},{k:.17g},{u:.17g},{e:.17g}\n")


def main():
    args = parse_args()
    pos, vel, mass = read_input(args.input)
    n = pos.shape[0]
    ids = np.arange(n)
    print(f"N = {n}, pasos = {args.steps}, dt = {args.dt}, theta = {args.theta}, eps = {args.eps}")

    warm_up(pos, vel, mass, ids, args.theta, args.eps)
    times = dict.fromkeys(PHASES, 0.0)
    start_total = time.perf_counter()

    # Paso 0: fuerzas iniciales (KDK necesita la aceleración antes del primer kick).
    pos, vel, mass, ids, acc, cost = forces(pos, vel, mass, ids, args.theta, args.eps, times)
    if args.dump_forces:
        write_forces(args.dump_forces, acc, ids)
    energy_rows = []
    if args.energy_log:
        energy_rows.append(energy_row(0, pos, vel, mass, args.eps))

    # Leapfrog kick-drift-kick.
    half_dt = 0.5 * args.dt
    for step in range(1, args.steps + 1):
        t = time.perf_counter()
        kick(vel, acc, half_dt)                 # medio paso de velocidad con las fuerzas viejas
        drift(pos, vel, args.dt)                # paso completo de posición
        add_time(times, "integracion", t)
        pos, vel, mass, ids, acc, cost = forces(pos, vel, mass, ids, args.theta, args.eps, times)
        t = time.perf_counter()
        kick(vel, acc, half_dt)                 # medio paso con las fuerzas nuevas (mismo orden que vel)
        add_time(times, "integracion", t)
        if args.energy_log:                     # la energía no se mide: es solo verificación
            energy_rows.append(energy_row(step, pos, vel, mass, args.eps))

    if args.energy_log:
        write_energy(args.energy_log, energy_rows)
    if args.timing:
        write_timing(args.timing, n, 1, times)
    measured = sum(times.values())
    print(f"Terminado: {args.steps} pasos en {time.perf_counter() - start_total:.2f} s "
          f"(medido en fases: {measured:.2f} s). "
          + ", ".join(f"{p} {times[p]:.2f} s" for p in PHASES))
    print(f"Interacciones por partícula en el último paso: media {cost.mean():.0f}, máx {cost.max()}")


if __name__ == "__main__":
    main()
