"""Verificación de correctitud de una versión: fuerzas contra la suma directa y conservación de la energía.

Uso:
    python tools/verify.py --version mp --n 5000 --theta 0.5
    python tools/verify.py --version mpi --workers 4 --tipo plummer --n 2000
    python tools/verify.py --version mp --theta 0          # θ = 0 debe dar la suma directa exacta
    python tools/verify.py --cmd "python mi_prueba.py" --n 2000

Pasos:
  1. Genera la distribución (o usa --input) en data/verify/.
  2. Corre la versión con el CLI común y le pide --dump-forces y --energy-log.
  3. Fuerzas: error relativo por partícula |a - a_directa| / |a_directa|. Se juzgan la mediana y el
     percentil 99; el máximo solo se informa (donde |a_directa| ≈ 0 el error relativo se infla).
  4. Energía: el E0 de la versión debe coincidir con el calculado aquí, y la deriva máxima
     |E - E0| / |E0| debe quedar bajo la tolerancia.

Las referencias (fuerzas y energía) salen solo de tools/direct_sum.py; verify.py no importa core_py,
para no verificar un código con él mismo.

Termina con código 0 si todo pasa y 1 si algo falla. Nada entra a main si no pasa.
Ojo con la energía en distribuciones muy densas (concentrada): si dt es mayor que el tiempo dinámico
sqrt(R³ / M) de la región densa, la energía no se conserva ni con fuerzas exactas. Eso no es un bug.
"""
import argparse
import os
import shlex
import subprocess
import sys
import time

import numpy as np

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO)

from tools.direct_sum import direct_accelerations, direct_kinetic_energy, direct_potential_energy  # noqa: E402
from tools.gen_dist import DISTRIBUCIONES, generate, save        # noqa: E402

# PUNTO DE MODIFICACIÓN: cómo se lanza cada versión. {P} se reemplaza por --workers.
VERSION_COMMANDS = {
    "mp": [sys.executable, "python_mp/main.py"],
    "mpi": ["mpirun", "--oversubscribe", "-n", "{P}", sys.executable, "mpi/main.py"],
    "omp": ["./c_omp/bh"],
    "cuda": ["./cuda/bh"],
}

# PUNTO DE MODIFICACIÓN: tolerancias. Calibradas el 22 nov con un Barnes-Hut de prueba (N = 5000, las 4
# distribuciones): el error crece como θ² y lo medido queda 2 a 4 veces por debajo de estos límites.
MEDIAN_TOL_FACTOR = 0.10        # mediana del error relativo < 0.10 · θ²
P99_TOL_FACTOR = 0.25           # percentil 99 del error relativo < 0.25 · θ²
MIN_FORCE_TOL = 1e-6            # piso para θ = 0 (solo queda el redondeo)
ENERGY_E0_TOL = 1e-8            # E0 de la versión contra E0 calculado aquí
ENERGY_DRIFT_TOL = 1e-3         # deriva máxima |E - E0| / |E0| en toda la corrida


def force_tolerances(theta):
    """(tolerancia de la mediana, tolerancia del percentil 99) para un θ dado."""
    median_tol = max(MEDIAN_TOL_FACTOR * theta**2, MIN_FORCE_TOL)
    p99_tol = max(P99_TOL_FACTOR * theta**2, MIN_FORCE_TOL)
    return median_tol, p99_tol


def prepare_input(args):
    """Devuelve la ruta del archivo de entrada: la de --input o una distribución generada en data/verify/."""
    if args.input:
        return args.input
    folder = os.path.join(REPO, "data", "verify")
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, f"{args.tipo}_{args.n}_{args.seed}.txt")
    pos, vel, mass = generate(args.tipo, args.n, args.seed)
    save(path, pos, vel, mass)
    return path


def build_command(args, input_path, forces_path, energy_path):
    """Comando de la versión (tabla o --cmd) más los argumentos del CLI común."""
    if args.cmd:
        base = shlex.split(args.cmd)
    else:
        base = [part.replace("{P}", str(args.workers)) for part in VERSION_COMMANDS[args.version]]
    common = [
        "--input", input_path,
        "--steps", str(args.steps),
        "--dt", str(args.dt),
        "--theta", str(args.theta),
        "--eps", str(args.eps),
        "--workers", str(args.workers),
        "--dump-forces", forces_path,
        "--energy-log", energy_path,
    ]
    return base + common


def check_forces(acc, acc_ref, theta):
    """Compara las aceleraciones con la suma directa. Devuelve True si pasa."""
    if acc.shape != acc_ref.shape:
        print(f"  FALLA: las fuerzas tienen forma {acc.shape} y se esperaba {acc_ref.shape}")
        return False
    if not np.all(np.isfinite(acc)):
        print(f"  FALLA: hay {np.count_nonzero(~np.isfinite(acc))} valores NaN o infinitos")
        return False

    error = np.linalg.norm(acc - acc_ref, axis=1) / np.linalg.norm(acc_ref, axis=1)
    median = np.median(error)
    p99 = np.percentile(error, 99)
    median_tol, p99_tol = force_tolerances(theta)

    median_ok = median < median_tol
    p99_ok = p99 < p99_tol
    print(f"  mediana del error relativo: {median:.2e}  (límite {median_tol:.2e})  {'ok' if median_ok else 'FALLA'}")
    print(f"  percentil 99:               {p99:.2e}  (límite {p99_tol:.2e})  {'ok' if p99_ok else 'FALLA'}")
    print(f"  máximo (solo informativo):  {error.max():.2e}  en la partícula {int(error.argmax())}")
    return median_ok and p99_ok


def check_energy(log, e0_ref):
    """Revisa el CSV step,K,U,E. Devuelve True si pasa."""
    if log.ndim != 2 or log.shape[0] < 2 or log.shape[1] != 4:
        print(f"  FALLA: el log de energía debe tener columnas step,K,U,E y al menos 2 filas (tiene {log.shape})")
        return False
    energy = log[:, 3]
    if not np.all(np.isfinite(energy)):
        print("  FALLA: hay energías NaN o infinitas")
        return False

    e0 = energy[0]
    e0_error = abs(e0 - e0_ref) / abs(e0_ref)
    drift = np.abs(energy - e0) / abs(e0)
    worst = int(drift.argmax())

    e0_ok = e0_error < ENERGY_E0_TOL
    drift_ok = drift.max() < ENERGY_DRIFT_TOL
    print(f"  E0 de la versión = {e0:.10f}, E0 calculado aquí = {e0_ref:.10f}  "
          f"(dif. relativa {e0_error:.1e}, límite {ENERGY_E0_TOL:.0e})  {'ok' if e0_ok else 'FALLA'}")
    print(f"  deriva máxima |E - E0|/|E0| = {drift.max():.2e} en el paso {int(log[worst, 0])}  "
          f"(límite {ENERGY_DRIFT_TOL:.0e})  {'ok' if drift_ok else 'FALLA'}")
    if not drift_ok:
        print("  Pista: si las fuerzas pasan y la distribución es muy densa, puede que dt sea mayor que el\n"
              "  tiempo dinámico sqrt(R³/M) de la región densa (no es un bug). Pruebe con un --dt menor.")
    return e0_ok and drift_ok


def main():
    parser = argparse.ArgumentParser(description="Verifica una versión contra la suma directa O(N²).")
    parser.add_argument("--version", choices=sorted(VERSION_COMMANDS), help="versión a verificar")
    parser.add_argument("--cmd", help="comando propio para lanzar la versión (reemplaza la tabla)")
    parser.add_argument("--workers", type=int, default=2, help="procesos o hilos de la versión")
    parser.add_argument("--input", help="archivo de entrada; si no se da, se genera con --tipo/--n/--seed")
    parser.add_argument("--tipo", choices=sorted(DISTRIBUCIONES), default="uniforme")
    parser.add_argument("--n", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--dt", type=float, default=1e-3)
    parser.add_argument("--theta", type=float, default=0.5)
    parser.add_argument("--eps", type=float, default=1e-3)
    args = parser.parse_args()
    if not args.version and not args.cmd:
        parser.error("hay que dar --version o --cmd")

    input_path = prepare_input(args)
    name = args.version or "cmd"
    out_folder = os.path.join(REPO, "data", "verify")
    os.makedirs(out_folder, exist_ok=True)
    forces_path = os.path.join(out_folder, f"fuerzas_{name}.txt")
    energy_path = os.path.join(out_folder, f"energia_{name}.csv")
    for path in (forces_path, energy_path):     # que no se lea un resultado viejo si la versión no escribe
        if os.path.exists(path):
            os.remove(path)

    data = np.loadtxt(input_path, ndmin=2)
    pos = np.ascontiguousarray(data[:, 0:3])
    vel = np.ascontiguousarray(data[:, 3:6])
    mass = np.ascontiguousarray(data[:, 6])
    n = pos.shape[0]
    if n > 20000:
        print(f"Aviso: N = {n}. La suma directa es O(N²) y puede tardar mucho; lo normal es N entre 2.000 y 5.000.")

    command = build_command(args, input_path, forces_path, energy_path)
    print(f"Verificando '{name}' con N = {n}, theta = {args.theta}, eps = {args.eps}, "
          f"dt = {args.dt}, pasos = {args.steps}, workers = {args.workers}")
    print("$ " + " ".join(command))
    start = time.perf_counter()
    result = subprocess.run(command, cwd=REPO)
    print(f"(la versión tardó {time.perf_counter() - start:.1f} s)")
    if result.returncode != 0:
        print(f"\nRESULTADO: FALLA — la versión terminó con código {result.returncode}")
        sys.exit(1)

    print("\nFuerzas del paso 0 contra la suma directa:")
    if os.path.exists(forces_path):
        acc = np.loadtxt(forces_path, ndmin=2)
        forces_ok = check_forces(acc, direct_accelerations(pos, mass, args.eps), args.theta)
    else:
        print(f"  FALLA: la versión no escribió {forces_path}")
        forces_ok = False

    print("\nConservación de la energía:")
    if os.path.exists(energy_path):
        log = np.loadtxt(energy_path, delimiter=",", skiprows=1, ndmin=2)
        e0_ref = direct_kinetic_energy(vel, mass) + direct_potential_energy(pos, mass, args.eps)
        energy_ok = check_energy(log, e0_ref)
    else:
        print(f"  FALLA: la versión no escribió {energy_path}")
        energy_ok = False

    if forces_ok and energy_ok:
        print("\nRESULTADO: PASA")
        sys.exit(0)
    print("\nRESULTADO: FALLA")
    sys.exit(1)


if __name__ == "__main__":
    main()
