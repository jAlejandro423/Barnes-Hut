# Ficha del integrante B — núcleo Python (morton, traverse) y versión MPI

Decisiones tomadas, alternativas descartadas y por qué. Lo que está en el `CLAUDE.md` raíz no se repite;
aquí queda el razonamiento para poder defenderlo en la sustentación.

## Máquina de medición

- Intel Core i5-10300H: **4 núcleos físicos, 8 hilos** (2 por núcleo, Hyper-Threading), 1 socket, en WSL2.
  Predicción: hasta P = 4 debería escalar bien; de 4 a 8 la ganancia es mucho menor porque los hilos
  comparten las unidades de cálculo de cada núcleo; con P > 8 (`--oversubscribe`) los procesos se turnan
  el CPU y el tiempo empeora por cambios de contexto y por esperas en las llamadas colectivas.
- Entorno: Python 3.14.4, numpy 2.5.5, numba 0.68.1, mpi4py 4.1.2, matplotlib 3.11.2, Open MPI del sistema.

## Decisiones

### 2026-11-21 — Interfaz del núcleo Python (con A)

Ver la sección "Interfaz del núcleo Python (acordada por A y B)" del `CLAUDE.md` raíz.

1. **El árbol viaja como `namedtuple` `Tree`** con los 9 arreglos `node_*`.
   - Descartado: arreglos sueltos (firmas de más de 12 argumentos, fácil equivocarse de orden en la demo)
     y tupla simple (desempaquetado por posición, mismo riesgo).
   - Verificado (2026-11-21, numba 0.68.1): un `@njit` puede crear y devolver el `Tree`, pasarlo a otra
     función `@njit`, modificar sus arreglos en el lugar y recibir un `Tree` armado en Python (caso LET
     recibido o `shared_memory`). No hace falta el plan B (desempaquetar dentro de las `@njit`).
2. **Caja contenedora en dos funciones** (`local_bounds` y `cube_from_bounds`) para que MPI haga el
   `Allreduce` de `lo`/`hi` entre ambas y todos los procesos usen los mismos códigos de Morton.
3. **Fuentes y objetivos separados en el recorrido** (`src_pos`/`src_mass` y `tgt_pos`): python_mp y MPI
   calculan solo un subconjunto de partículas, y el mismo recorrido sirve para cada LET.
   - La autointeracción no se excluye: con suavizado, `r_j - r_i = 0` da fuerza 0.
4. **`integrate.py` lo escribe A**: son unas 30 líneas y va junto a la verificación de energía de
   `tools/verify.py`, que es de A. B ya tiene la versión más difícil.
   - Descartado: que lo escriba B porque su paso 1 lo necesita; basta con acordar la firma, y B revisa
     el PR de A.
5. **LET con la misma función de recorrido**: cada envío es un `Tree` independiente; los nodos podados son
   pseudo-hojas (masa y centro de masa, sin partículas). El receptor suma aceleraciones y costos del árbol
   local y de cada LET. Es correcto porque la fuerza total es la suma de las fuerzas de cada árbol.
   - Descartado: función aparte para el LET (dos recorridos que mantener; un cambio de θ en la demo se
     tendría que hacer en dos lugares).
   - Descartado: fusionar el LET con el árbol local (hay que reenlazar índices y `node_next`; frágil).
   - Con θ = 0 el criterio del emisor abre todo, no hay pseudo-hojas y MPI coincide con la suma directa.

## Decisiones pendientes (proponer antes de programar)

- Distribución de partículas al inicio y entre pasos (quién lee el archivo, `Scatterv` / `Alltoallv`).
- Criterio exacto del LET (θ contra la caja del dominio remoto, distancia mínima).
- Cómo se reúnen las aceleraciones para `--dump-forces` (arreglo `ids` por partícula + `Gatherv`).
