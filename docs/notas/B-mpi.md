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
   - La autointeracción no se excluye por índice (fuentes y objetivos son arreglos distintos, y en el LET
     ni siquiera son las mismas partículas). Con ε > 0 su fuerza es 0. Con ε = 0, o ε tan pequeño que
     r²·sqrt(r²) da 0, se salta el par: sin eso Numba lanza ZeroDivisionError y en C sale NaN en silencio.
     Lo detectó A probando `direct_sum.py` con ε = 0 (yo había escrito que el suavizado bastaba siempre).
     A propuso `r2 == 0`; se cambió a `r2·sqrt(r2) == 0` porque con ε = 1e-160 queda ε² = 1e-320 ≠ 0,
     pero r³ sí da 0.
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

### 2026-11-21 — `core_py/morton.py`

1. **Cubo, no caja**: lado = mayor lado de la caja, centrado en ella, con margen relativo `1e-6`.
   El octree necesita celdas cúbicas (`node_half` es un solo número). Sin margen, la partícula del borde
   superior se normalizaría a 2²¹ exacto, que necesita 22 bits.
2. **Intercalado con un bucle de 21 niveles** desde la raíz: `code = (code << 3) | octante`.
   - Descartado: "números mágicos" (máscaras y desplazamientos). Algo más rápido, pero ilegible y no se
     reconstruye de memoria. El bucle tarda ~0,006 s para N = 200.000.
   - Se calcula en `int64` y se guarda en `uint64`: en Numba, `uint64` con `int64` da `float64` y se
     perderían bits.
3. **Convención de bits fija** (`octante = (bit_x << 2) | (bit_y << 1) | bit_z`), escrita en el `CLAUDE.md`
   raíz. Descartado dejarla implícita: si `tree.py` usara otro orden, el árbol saldría mal sin ningún error.
4. **`np.argsort(kind="stable")`** fuera de Numba: reproducible con códigos repetidos. Descartado radix sort
   propio (más código sin ganancia en Python). ~0,018 s para N = 200.000.
5. Riesgo para la demo: con `concentrada` y pocas partículas lejanas, el cubo es enorme, muchas partículas
   comparten código en el nivel 21 y quedan hojas grandes de suma directa (O(k²)). Ver `MAX_DEPTH`.

### 2026-11-25 — `core_py/traverse.py`

1. **Tres funciones pequeñas**: `pair_accel` (fórmula y protección de ε = 0), `accept_cell` (criterio θ,
   `PUNTO DE MODIFICACIÓN`) y `accel_one` (recorrido). Todas las interacciones pasan por `pair_accel`:
   hojas, celdas aceptadas y pseudo-hojas, así que la protección y la fórmula están en un solo lugar.
   - Descartado: todo dentro de `accel_one` (un cambio de θ o de fórmula en la demo quedaría enterrado).
2. **Recorrido con `node_next`, sin pila ni recursión**: hoja o celda aceptada → `node_next`; celda
   rechazada → `node_first_child`. Termina en -1.
   - Descartado: pila explícita (funciona, pero `node_next` ya existe y es lo que porta CUDA) y recursión
     (el enunciado pide recorrido sin recursión; CUDA la evita).
3. **Hojas con partículas siempre por suma directa**, sin probar θ: cubre las hojas grandes de `MAX_DEPTH`.
4. **Costo = interacciones evaluadas** (partículas + celdas) por objetivo; es lo que reparten python_mp y MPI.
5. **`compute_accelerations` sin `parallel=True`**: el paralelismo lo ponen python_mp y MPI con procesos;
   mezclar hilos de Numba con procesos confundiría las mediciones.
6. **Riesgo con θ grande**: con la distancia al centro de masa, si θ ≥ 1/sqrt(3) ≈ 0,58 una celda puede
   aceptarse aunque contenga a la propia partícula (el centro de masa puede estar a una diagonal de ella).
   Con θ = 0,5 no pasa. Alternativas documentadas en `accept_cell`: distancia al centro geométrico, o
   exigir además d > tamaño.
7. Resultados (N = 2.000, contra `direct_sum.py` del PR #3, criterio de `verify.py`):
   - θ = 0: error ~1e-15 en las 4 distribuciones (no 0 exacto: se suma en orden de Morton y la referencia
     en orden de entrada).
   - θ = 0,5: mediana entre 2,8e-3 (plummer) y 1,3e-2 (concentrada, la más cercana al límite 2,5e-2,
     porque la nube densa mide lo mismo que ε).
   - Costo con θ = 0,5 en plummer: mínimo 52, medio 514, máximo 872 interacciones. La partícula más cara
     cuesta 17 veces la más barata: repartir por número de partículas desbalancea.
   - Fuentes partidas en 2 árboles y sumadas (como el LET): misma precisión contra la suma directa
     (mediana 3,3e-3 contra 2,8e-3 con un solo árbol).
   - N = 20.000 uniforme, θ = 0,5: árbol + recorrido en 0,11 s por paso (sin compilación).

### 2026-11-25 — `mpi/main.py`, versión secuencial (P = 1)

1. **El secuencial es el esqueleto de `mpi/main.py`**: el ciclo, el CLI y las salidas son los definitivos;
   el reparto por costo, el LET y `comunicacion` se agregan encima. Se corre sin `mpirun`.
   - Descartado: un archivo secuencial aparte (no está en la estructura, duplicaría código).
2. **Las partículas se reordenan físicamente cada paso con un arreglo `ids`** (índice en el archivo).
   `--dump-forces` usa `acc_orig[ids] = acc`. En MPI las partículas migran y `ids` hace falta igual.
   - Descartado: dejar los arreglos en orden de entrada y ordenar copias (no sirve cuando migran).
   - `sort_particles` reordena pos, vel, mass e ids con la misma permutación: si vel quedara en el orden
     viejo, el kick mezclaría velocidades. Las fuerzas del paso 0 no lo detectarían; la energía sí.
3. **Salidas con `%.17g`**: con `%.6g` falla E0 en `verify.py` (tolerancia 1e-8) aunque el cálculo esté bien.
4. **Tiempos**: suma por fase de todos los pasos; la energía no se mide. El calentamiento de Numba se hace
   con 64 partículas (Numba compila según los tipos, no según el tamaño), así no cuesta un paso completo.
5. Resultados con `verify.py` del PR #3 (N = 2.000, 100 pasos): PASA en uniforme (θ = 0,5 y θ = 0),
   plummer y dos_cumulos. E0 coincide hasta ~1e-14. Deriva de energía en uniforme 3e-4 con θ = 0,5 y
   2,2e-4 con θ = 0: viene del integrador (dt), no del árbol.
6. Tiempos con N = 20.000 uniforme, 20 pasos: recorrido 2,87 s (96 %), ordenamiento 0,07 s, árbol 0,04 s,
   integración 0,01 s. El recorrido es lo que hay que repartir; la fracción secuencial (orden + árbol)
   es ~4 %, lo que limita el speedup teórico (Amdahl) si esa parte no se paraleliza.

## Decisiones pendientes (proponer antes de programar)

- Distribución de partículas al inicio y entre pasos (quién lee el archivo, `Scatterv` / `Alltoallv`).
- Criterio exacto del LET (θ contra la caja del dominio remoto, distancia mínima).
- Cómo se reúnen las aceleraciones para `--dump-forces` (arreglo `ids` por partícula + `Gatherv`).
