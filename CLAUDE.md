# Proyecto Barnes-Hut — Programación Concurrente, Paralela y Distribuida

Simulación gravitacional de N cuerpos con el algoritmo de Barnes-Hut, paralelizada en 4 versiones:
Python con multiprocessing, MPI con mpi4py, C con OpenMP y CUDA en Google Colab (GPU T4).
N va de 20.000 a 200.000 partículas. Proyecto de grupo de 3 integrantes (A, B, C).

## Contexto de evaluación (define cómo se escribe el código)

- El 90 % de la nota es sustentación y demo en vivo **sin internet ni IA**. En la demo nos dan una
  distribución desconocida (cúmulos densos, colisiones) y piden modificar el código en pocos minutos.
- Por eso: **código legible antes que código ingenioso.** Funciones cortas, nombres claros en español
  o inglés (consistentes por archivo), sin abstracciones innecesarias, sin metaprogramación.
- Marca con un comentario `PUNTO DE MODIFICACIÓN` los lugares donde se toma una decisión que
  podrían pedirnos cambiar: estrategia de particionado, criterio de apertura θ, asignación de
  partículas a warps, balanceo de carga, tamaño de hoja, profundidad máxima.
- Antes de escribir código no trivial, explica el diseño y las alternativas en 3 a 5 líneas.
  Al terminar, resume qué hace y lista 2 o 3 preguntas que nos podrían hacer sobre ese código.
- **No hagas commits ni push por tu cuenta.** Propón el mensaje de commit; el integrante lo revisa y lo hace.
  Solo si el integrante lo pide explícitamente en la sesión, puedes hacer el commit o el push tú.
- No agregues dependencias nuevas sin decirlo. Todo debe funcionar sin internet el día de la demo.

## Física y parámetros

- G = 1. Fuerza suavizada: `F_ij = m_i m_j (r_j - r_i) / (|r_j - r_i|² + ε²)^(3/2)`.
- Integrador leapfrog en forma kick-drift-kick (KDK).
- Energía: `K = Σ ½ m v²`, `U = -Σ_{i<j} m_i m_j / sqrt(r² + ε²)`. U se calcula con suma directa
  solo para N pequeño (verificación); para N grande no se reporta energía.
- Criterio de apertura: una celda se acepta como partícula única si `tamaño_celda / distancia < θ`,
  con la distancia medida al centro de masa de la celda. θ = 0 equivale a la suma directa.
- Valores por defecto: `--theta 0.5 --eps 1e-3 --dt 1e-3 --steps 100`.

## Arquitectura: dos núcleos con el mismo diseño

- `core_py/` (NumPy + Numba): lo usan `python_mp/` y `mpi/`.
- `core_c/`: lo usa `c_omp/`; `cuda/` lo porta casi función por función.
- Diseño común del árbol:
  1. Calcular la caja contenedora en cada paso (no asumir el cubo unitario).
  2. Código de Morton de 63 bits (21 bits por eje) para cada partícula y ordenar las partículas por él.
  3. Construir el octree a partir del orden de Morton: cada nodo cubre un rango contiguo
     `[body_start, body_start + body_count)` del arreglo ordenado.
  4. Hojas con a lo sumo 1 partícula, salvo al llegar a `MAX_DEPTH` (partículas casi coincidentes),
     donde la hoja guarda todas las que queden.
  5. Masa y centro de masa calculados de abajo hacia arriba.
- Árbol en **arreglos planos** (structure of arrays), mismos nombres en Python y C:
  `node_center[3]`, `node_half`, `node_mass`, `node_com[3]`, `node_first_child`, `node_n_children`,
  `node_body_start`, `node_body_count`, `node_next` (puntero de salto para recorrido sin recursión).
- El reparto de trabajo entre procesos o hilos es **por costo** (interacciones por partícula del paso
  anterior), no por número de partículas. Las distribuciones de la demo castigan el reparto ingenuo.

## Contrato común (no cambiar sin acuerdo del grupo)

- Entrada: archivo de texto con N filas y 7 columnas `x y z vx vy vz m`.
- CLI igual en todas las versiones:
  `--input --steps --dt --theta --eps --workers --partition --dump-forces --energy-log --timing`
- Salidas:
  - `--dump-forces`: aceleraciones del paso 0, N filas × 3 columnas, en el orden original de entrada.
  - `--energy-log`: CSV `step,K,U,E`.
  - `--timing`: CSV `version,N,P,fase,segundos`.
- Fases de tiempo: `arbol`, `ordenamiento`, `recorrido`, `integracion`, y `comunicacion` en MPI.
- En Numba, la primera llamada compila: hacer un paso de calentamiento y no medirlo.

## Requisitos por versión

- **python_mp/**: comparar dos estrategias y elegir una con mediciones:
  `shared` (árbol plano en `multiprocessing.shared_memory`) y `local` (cada proceso reconstruye el árbol).
- **mpi/** (mpi4py): particionado del espacio con curva de Morton por costo (`--partition morton`) y
  bisección recursiva (`--partition orb`). Cada proceso recibe solo las ramas remotas que necesita
  (locally essential tree, intercambio con `Alltoallv`), **nunca el árbol completo**.
- **c_omp/**: construcción del árbol en paralelo a partir del orden de Morton; balanceo de carga entre
  hilos (por costo o `schedule(dynamic)`).
- **cuda/**: árbol aplanado, partículas ordenadas por Morton, recorrido sin recursión (pila explícita o
  punteros de salto), mínima divergencia dentro del warp, tiempos por fase con `cudaEvent`, sin
  errores en `compute-sanitizer`. Se desarrolla y ejecuta en Colab.

## Correctitud (obligatoria)

- Una versión es correcta si sus fuerzas coinciden con la suma directa N² sobre N pequeño (2.000 a 5.000)
  con error relativo aceptable para el θ usado, y si la energía total se conserva dentro de una tolerancia.
- `tools/verify.py` comprueba ambas cosas. **Nada entra a `main` si no pasa.**
- Distribuciones en `tools/gen_dist.py`: `uniforme` (cubo unitario, v = 0, m = 1/N, semilla fija),
  `plummer`, `dos_cumulos` (en colisión) y `concentrada` (casi todo en un volumen muy pequeño).

## Estructura del repositorio

```
Barnes-Hut/
  CLAUDE.md, README.md, requirements.txt, .gitignore
  tools/      gen_dist.py, direct_sum.py, verify.py, bench.py
  core_py/    morton.py, tree.py, traverse.py, integrate.py
  python_mp/  main.py
  mpi/        main.py, partition.py, let.py
  core_c/     morton.c/.h, tree.c/.h, traverse.c/.h, io.c/.h
  c_omp/      main.c, Makefile
  cuda/       barnes_hut.cu, barnes_hut_colab.ipynb
  data/       distribuciones generadas (ignoradas por git); las pequeñas de prueba van en data/prueba/
  results/    CSV de mediciones
  docs/       notas/ (una ficha por integrante), informe/
```

## Git

- `main` protegida: solo entra por PR con 1 aprobación y `verify.py` en verde.
- Ramas `feat/<versión>-<tema>`, `fix/<versión>-<tema>`, `bench/<tema>`, `docs/<tema>`. Vida de 1 a 3 días.
- Commits pequeños, en español: `tipo(alcance): qué cambia y por qué`.
  - Tipos: `feat`, `fix`, `perf`, `test`, `bench`, `refactor`, `docs`, `chore`.
  - Alcances: `tools`, `core-py`, `mp`, `mpi`, `core-c`, `omp`, `cuda`, `informe`, `repo`.
  - Ejemplo: `feat(mpi): corta la curva de Morton por costo acumulado`.
- Revisión cruzada de PR: B revisa a A, C revisa a B, A revisa a C.
- Tags: `v0.1-correcto`, `v1.0-freeze`, `v1.0-entrega`.

## Entorno y comandos

- Python 3.10 a 3.14 en un entorno virtual: `python -m venv .venv` y `pip install -r requirements.txt`
  (numpy, numba, mpi4py, matplotlib). En Windows, todo el proyecto se trabaja dentro de WSL2.
- mpi4py necesita MPI del sistema: `sudo apt install openmpi-bin libopenmpi-dev` (antes del `pip install`).
- Generar: `python tools/gen_dist.py --tipo uniforme --n 20000 --seed 42 -o data/u20k.txt`
- Verificar: `python tools/verify.py --version <mp|mpi|omp|cuda> --n 5000 --theta 0.5`
- Medir: `python tools/bench.py --version <...> --procs 1 2 4 8`
- MPI: `mpirun -n 4 python mpi/main.py --input data/u20k.txt ...` (`--oversubscribe` para más procesos que núcleos)
- C: `make -C c_omp && OMP_NUM_THREADS=4 ./c_omp/bh --input data/u20k.txt ...`
