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

## Verificación (2026-11-26, WSL2, 4 núcleos físicos / 8 hilos)

| caso | resultado |
| --- | --- |
| `uniforme`, N=5000, θ=0.5, P=4 | PASA; mediana 3.3e-3, p99 1.8e-2, deriva 1.3e-5 |
| `plummer`, N=5000, θ=0.5, P=8 | PASA; mediana 2.1e-3, p99 1.5e-2, deriva 4.9e-6 |
| `uniforme`, N=2000, θ=0, P=2 | PASA; mediana 1.4e-15 (coincide con suma directa) |

Comando: `.venv/bin/python tools/verify.py --version omp --n 5000 --theta 0.5 --workers 4`.

**Ojo al verificar:** `verify.py` usa los mismos archivos de salida para todas las corridas de una
misma versión. Si se lanzan dos `verify.py --version omp` en paralelo se sobrescriben
`fuerzas_omp.txt` y `energia_omp.csv`; hay que ejecutarlos en serie.

## Pendiente

- Probar `dos_cumulos` y `concentrada` con `dt` reducido (la deriva puede venir del integrador, no
  del árbol).
- Medir escalado P=1,2,4,8 con `bench.py` cuando A lo termine.

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
