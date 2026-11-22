"""Generador de distribuciones de partículas para las pruebas y las mediciones.

Escribe un archivo de texto con N filas y 7 columnas:  x y z vx vy vz m
Unidades: G = 1. La masa total siempre es 1 (cada partícula pesa 1/N).

Uso:
    python tools/gen_dist.py --tipo uniforme --n 20000 --seed 42 -o data/u20k.txt

Distribuciones:
    uniforme     posiciones al azar en el cubo unitario, v = 0 (la que pide el enunciado)
    plummer      un cúmulo esférico en equilibrio (núcleo denso y halo)
    dos_cumulos  dos cúmulos de Plummer pequeños que se acercan (colisión)
    concentrada  casi todas las partículas en un volumen diminuto, el resto uniforme
"""
import argparse

import numpy as np

# PUNTO DE MODIFICACIÓN: parámetros de cada distribución.
PLUMMER_R_MAX = 10.0             # radio máximo (en unidades del radio de escala a); recorta el halo lejano
CLUSTER_RADIUS = 0.1             # radio de escala a de cada cúmulo en dos_cumulos
CLUSTER_SEPARATION = 2.0         # distancia entre los centros de los dos cúmulos (sobre el eje x)
CLUSTER_OFFSET_Y = 0.2           # desplazamiento en y entre los centros: choque no exactamente frontal
DENSE_FRACTION = 0.9             # fracción de partículas en la nube densa de concentrada
DENSE_SIGMA = 1e-3               # desviación estándar de la nube densa
DENSE_CENTER = (0.7, 0.3, 0.5)   # centro de la nube densa, fuera del centro del cubo a propósito


def uniforme(n, rng):
    """Cubo unitario, velocidades cero, masas iguales."""
    pos = rng.random((n, 3))
    vel = np.zeros((n, 3))
    return pos, vel


def random_directions(n, rng):
    """n vectores unitarios con dirección al azar, repartidos de forma uniforme sobre la esfera."""
    cos_theta = rng.uniform(-1.0, 1.0, n)
    sin_theta = np.sqrt(1.0 - cos_theta**2)
    phi = rng.uniform(0.0, 2.0 * np.pi, n)
    return np.column_stack((sin_theta * np.cos(phi), sin_theta * np.sin(phi), cos_theta))


def plummer(n, rng):
    """Esfera de Plummer con masa total 1 y radio de escala 1 (método de Aarseth, Hénon y Wielen, 1974).

    Radio: la masa encerrada es M(r) = r³ / (1 + r²)^(3/2); se elige una fracción de masa X al azar
    y se despeja r = 1 / sqrt(X^(-2/3) - 1).
    Velocidad: v = q · v_escape(r), con v_escape = sqrt(2) · (1 + r²)^(-1/4), y q se elige por rechazo
    con densidad proporcional a q² (1 - q²)^(7/2).
    """
    # Radios: se descartan los mayores que PLUMMER_R_MAX y se vuelven a sortear.
    r = np.empty(n)
    filled = 0
    while filled < n:
        x = rng.uniform(1e-10, 1.0, n - filled)
        candidates = 1.0 / np.sqrt(x ** (-2.0 / 3.0) - 1.0)
        ok = candidates[candidates < PLUMMER_R_MAX]
        r[filled:filled + ok.size] = ok
        filled += ok.size

    # Fracción q de la velocidad de escape, por rechazo. 0.1 es el máximo de q² (1 - q²)^3.5 (≈ 0.092).
    q = np.empty(n)
    filled = 0
    while filled < n:
        q_try = rng.random(n - filled)
        y_try = 0.1 * rng.random(n - filled)
        ok = q_try[y_try < q_try**2 * (1.0 - q_try**2) ** 3.5]
        q[filled:filled + ok.size] = ok
        filled += ok.size

    v_escape = np.sqrt(2.0) * (1.0 + r**2) ** (-0.25)
    pos = r[:, None] * random_directions(n, rng)
    vel = (q * v_escape)[:, None] * random_directions(n, rng)

    # Centro de masa en el origen y momento total cero (todas las masas son iguales).
    pos -= pos.mean(axis=0)
    vel -= vel.mean(axis=0)
    return pos, vel


def plummer_scaled(n, rng, mass, radius):
    """Plummer de masa total `mass` y radio de escala `radius`.

    Con G = 1, las posiciones escalan con el radio y las velocidades con sqrt(mass / radius).
    """
    pos, vel = plummer(n, rng)
    return pos * radius, vel * np.sqrt(mass / radius)


def dos_cumulos(n, rng):
    """Dos cúmulos de Plummer, cada uno con la mitad de la masa, que caen uno hacia el otro sobre el eje x."""
    n1 = n // 2
    n2 = n - n1
    pos1, vel1 = plummer_scaled(n1, rng, 0.5, CLUSTER_RADIUS)
    pos2, vel2 = plummer_scaled(n2, rng, 0.5, CLUSTER_RADIUS)

    # Velocidad de acercamiento: la de caída libre desde el infinito hasta la separación inicial,
    # v_rel = sqrt(2 G M_total / d). Cada cúmulo lleva la mitad, en sentidos opuestos.
    v_half = 0.5 * np.sqrt(2.0 * 1.0 / CLUSTER_SEPARATION)
    shift = np.array([0.5 * CLUSTER_SEPARATION, 0.5 * CLUSTER_OFFSET_Y, 0.0])
    pos1 -= shift
    pos2 += shift
    vel1[:, 0] += v_half
    vel2[:, 0] -= v_half

    return np.vstack((pos1, pos2)), np.vstack((vel1, vel2))


def concentrada(n, rng):
    """Casi todo en una nube gaussiana diminuta dentro del cubo unitario; el resto, uniforme en el cubo."""
    n_dense = int(DENSE_FRACTION * n)
    n_rest = n - n_dense
    dense = np.array(DENSE_CENTER) + DENSE_SIGMA * rng.standard_normal((n_dense, 3))
    rest = rng.random((n_rest, 3))
    pos = np.vstack((dense, rest))
    rng.shuffle(pos)                 # mezcla las filas: el archivo no viene ordenado por región
    vel = np.zeros((n, 3))
    return pos, vel


DISTRIBUCIONES = {
    "uniforme": uniforme,
    "plummer": plummer,
    "dos_cumulos": dos_cumulos,
    "concentrada": concentrada,
}


def generate(tipo, n, seed):
    """Devuelve (pos float64[N,3], vel float64[N,3], mass float64[N]). Misma semilla, mismo resultado."""
    rng = np.random.default_rng(seed)
    pos, vel = DISTRIBUCIONES[tipo](n, rng)
    mass = np.full(n, 1.0 / n)
    return pos, vel, mass


def save(path, pos, vel, mass):
    """Escribe las 7 columnas. %.17g guarda el float64 exacto: todas las versiones leen los mismos números."""
    data = np.column_stack((pos, vel, mass))
    np.savetxt(path, data, fmt="%.17g")


def main():
    parser = argparse.ArgumentParser(description="Genera una distribución de partículas.")
    parser.add_argument("--tipo", choices=sorted(DISTRIBUCIONES), default="uniforme")
    parser.add_argument("--n", type=int, required=True, help="número de partículas")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("-o", "--output", required=True, help="archivo de salida")
    args = parser.parse_args()

    pos, vel, mass = generate(args.tipo, args.n, args.seed)
    save(args.output, pos, vel, mass)

    lo = pos.min(axis=0)
    hi = pos.max(axis=0)
    print(f"{args.tipo}: N = {args.n}, masa total = {mass.sum():.6f}, semilla = {args.seed}")
    print(f"caja: min = {np.round(lo, 4)}, max = {np.round(hi, 4)}")
    print(f"escrito en {args.output}")


if __name__ == "__main__":
    main()
