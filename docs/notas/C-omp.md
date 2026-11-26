# Ficha del integrante C — núcleo C y versión OpenMP

Decisiones tomadas, alternativas descartadas y resultados de verificación. Lo que ya está en el
`CLAUDE.md` raíz no se repite.

## Estado

- `core_c/`: `morton.c`, `tree.c`, `traverse.c`, `integrate.c`, `io.c` (+ cabeceras y `common.h`).
- `c_omp/`: `main.c` y `Makefile`. Compila con `make -C c_omp` y produce `c_omp/bh`.
- Rama de trabajo: `feat/omp-nucleo` (creada desde `main`, sin tocar ramas ajenas).

## Decisiones

### Núcleo C: mismos nombres y mismo contrato que `core_py/`

1. **`Vec3 {x, y, z}` en arreglo contiguo** en vez de `double (*pos)[3]`.
   Es el mismo layout y hace que `pos[i].x` se lea igual que `pos[i, 0]` en Python.
2. **Orden de Morton con `qsort` y desempate por índice**, en vez de radix sort.
   Radix es más rápido pero más largo; el desempate reproduce `np.argsort(kind="stable")` y da
   resultados reproducibles. Si en la demo hace falta velocidad, se cambia solo `bh_morton_order`.
3. **`Tree` con los 9 arreglos `node_*` del contrato** más dos campos internos (`n`, `capacity`,
   `node_level`). Los campos internos no se usan en el recorrido; sirven para construir por niveles.
4. **`node_next` enlazado una sola vez al construir**, igual que en Python. El recorrido no usa pila:
   abre con `node_first_child` y salta con `node_next`. Eso facilita el port a CUDA (punteros de salto).
5. **Pseudo-hojas del LET ya soportadas** en `bh_accel_one`: `n_children == 0` y `body_count == 0`
   se tratan como masa puntual en `node_com`. No hay una segunda función de recorrido.

### OpenMP

1. **Construcción del árbol por niveles**: en cada nivel se cuentan los tramos de octantes de cada
   padre en paralelo, se asignan rangos contiguos de hijos en orden de padre, y se crean los hijos en
   paralelo. Alternativa descartada: un solo hilo construyendo el árbol (la fase `arbol` era el 40 %
   del tiempo en `plummer` con 8 hilos). La topología y los índices son idénticos a `core_py/tree.py`.
2. **Reparto del recorrido por costo** (`core_c/traverse.c`): se parten las partículas ordenadas en
   P tramos de igual interacción acumulada del paso anterior. `PUNTO DE MODIFICACIÓN`.
   Alternativa: `#pragma omp parallel for schedule(dynamic, 64)`, más simple pero con peor localidad
   y sin aprovechar el historial de costo.
3. **`--workers` manda sobre `OMP_NUM_THREADS`** mediante `omp_set_num_threads`. Así el CLI común
   funciona igual en las cuatro versiones. `--partition` se acepta pero se ignora en OpenMP.
4. **Cálculo de masa y centro de masa en un solo hilo**: el bucle va de atrás hacia adelante porque
   los hijos tienen índice mayor que el padre. Alternativa: paralelizar por nivel en sentido inverso
   (requiere guardar los límites de cada nivel); no valió la pena, es O(N) frente al recorrido.
5. **Medición con `omp_get_wtime`**, fases `ordenamiento`, `arbol`, `recorrido`, `integracion`.
   El calentamiento usa 64 partículas y no se mide, como pide el contrato.
6. **Energía solo si N ≤ `BH_MAX_N_ENERGY` (5000)**: si piden `--energy-log` con N mayor se avisa y no
   se escribe el CSV, porque U es O(N²). `PUNTO DE MODIFICACIÓN` en `core_c/common.h`. Descartado:
   reportar U = 0 o solo K, que rompe el contrato `step,K,U,E`.

## Verificación (2026-11-26, WSL2, 4 núcleos físicos / 8 hilos)

| caso | resultado |
| --- | --- |
| `uniforme`, N=5000, θ=0.5, P=4 | PASA; mediana 3.3e-3, p99 1.8e-2, deriva 1.3e-5 |
| `plummer`, N=5000, θ=0.5, P=8 | PASA; mediana 2.1e-3, p99 1.5e-2, deriva 4.9e-6 |
| `dos_cumulos`, N=5000, θ=0.5, P=4, dt=1e-4 | PASA; mediana 2.4e-3, p99 1.4e-2, deriva 3.9e-6 |
| `concentrada`, N=5000, θ=0.5, P=4, dt=1e-6 | FALLA por energía: deriva 3.8e-3; con dt=1e-7 PASA (2.7e-5) |
| `uniforme`, N=2000, θ=0, P=2 | PASA; mediana 1.4e-15 (coincide con suma directa) |

Comando: `.venv/bin/python tools/verify.py --version omp --n 5000 --theta 0.5 --workers 4`.

La falla de `concentrada` con dt=1e-6 no es del árbol: las fuerzas pasan y E0 coincide con la suma
directa. El tiempo dinámico de la nube densa (σ=1e-3, M≈0.9) es del orden de 3e-5, así que hay que
bajar a dt=1e-7 para que la deriva quede bajo 1e-3. El 2026-11-26 se regeneró
`data/verify/energia_omp.csv` con `--tipo concentrada --dt 1e-7`: E0 = -181.7212013555 coincide con
la suma directa (dif. rel. 1.8e-13) y la deriva máxima quedó en 2.72e-5.

**Ojo al verificar:** `verify.py` usa los mismos archivos de salida para todas las corridas de una
misma versión. Si se lanzan dos `verify.py --version omp` en paralelo se sobrescriben
`fuerzas_omp.txt` y `energia_omp.csv`; hay que ejecutarlos en serie.

## Escalado medido (2026-11-26)

`results/omp_escalado.csv`: uniforme N=20000, 20 pasos, θ=0.5, P=1,2,4,8.
Totales 2.19 / 1.34 / 0.95 / 0.91 s (speedup 1.0 / 1.6 / 2.3 / 2.4). El recorrido escala mejor
(2.05 / 1.21 / 0.83 / 0.61 s); la fase `arbol` empeora con P=8 (0.21 s contra 0.06 s con P=4) por
la sobrecarga de la construcción paralela con más hilos que núcleos físicos.

El 2026-11-26 se repitió con las otras dos distribuciones (N=20000, 20 pasos, θ=0.5):

| archivo | totales P=1/2/4/8 (s) | speedup | recorrido P=1→8 (s) |
| --- | --- | --- | --- |
| `results/omp_plummer.csv` | 5.11 / 2.99 / 2.08 / 1.76 | 1.0 / 1.7 / 2.5 / 2.9 | 4.96 → 1.53 (3.2×) |
| `results/omp_dos_cumulos.csv` | 4.52 / 2.56 / 1.89 / 1.56 | 1.0 / 1.8 / 2.4 / 2.9 | 4.36 → 1.32 (3.3×) |

En ambas, `arbol` vuelve a empeorar con P=8 (0.12-0.13 s contra 0.06 s con P=4): el límite está en
la sobrecarga de la construcción paralela, no en el reparto por costo del recorrido.

## Pendiente

- Repetir los tres escalados con `bench.py` cuando A lo termine (los CSV ya están medidos con `--timing`).
- Probar CUDA en Colab T4.

## Revisión de ramas de B (2026-11-26, solo lectura)

Revisé `git diff main..origin/feat/core-py-morton`, `..feat/core-py-tree` y
`..feat/core-py-traverse`. No encontré cambios bloqueantes: la convención de bits, el orden contiguo de
los hijos y el recorrido con `node_next` coinciden con lo que porté a C.

Preguntas para B:

1. Cuando podés una rama para armar un LET, ¿recalculás `link_next` sobre el árbol podado o copiás el
   `node_next` original? Si el nodo podado pasa a `n_children = 0`, el salto debe seguir yendo a su
   hermano, no al primer hijo.
2. Para `--partition morton` por costo: ¿el partido se hace con el `cost` del paso anterior acumulado
   sobre las partículas ya ordenadas, o se reparte por número de partículas y luego se corrige?

## Preguntas probables en la sustentación

- ¿Por qué el reparto es por costo y no por número de partículas? (cúmulos: una partícula del centro
  hace muchas más interacciones que una del borde).
- ¿Cómo se asegura que el orden de los hijos sea el mismo que en Python? (códigos ordenados + tramos
  de octante creciente).
- ¿Qué pasa con partículas casi coincidentes al llegar a `MAX_DEPTH`? (una hoja guarda todas; la
  fuerza directa con suavizado evita la singularidad).
