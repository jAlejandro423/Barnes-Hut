# Ficha del integrante C — port a CUDA (Colab T4)

Plan y decisiones para la versión CUDA. Todavía no hay código; el port se hace después de que
`core_c/` y `c_omp/` pasen `verify.py`.

## Decisiones previstas

1. **Portar casi función por función desde `core_c/`**: `Vec3` → `float3`/`double3`, `Tree` →
   arreglos `float*` o `double*` en memoria global del device. Mantener los mismos nombres de campo
   para poder comparar fuerzas contra la versión OpenMP con `--dump-forces`.
2. **Árbol aplanado y partículas ordenadas por Morton en el device**. El orden es el mismo que en
   `core_py` y `core_c` (21 bits por eje), así que las fuerzas del paso 0 deben coincidir con las de
   OpenMP dentro del redondeo. Las cajas se recalculan cada paso con una reducción en el device.
3. **Recorrido sin recursión con `node_next`**, igual que en C. Alternativa: pila explícita por hilo;
   sirve si hace falta podar ramas con criterios distintos, pero consume más registros y memoria local
   y aumenta la divergencia. Empezar con `node_next`.
4. **Asignación de partículas a warps por costo**, en la misma idea que OpenMP: reparto por
   interacciones del paso anterior (`PUNTO DE MODIFICACIÓN`). Alternativas: una partícula por hilo
   (simple, pero el coste varía mucho entre cúmulos y borde) o una partícula por warp (menos
   divergencia de control pero más serialized).
5. **Fases con `cudaEvent`**: `ordenamiento`, `arbol`, `recorrido`, `integracion` (y `comunicacion`
   si hiciera falta). El calentamiento se descarta como en las otras versiones.
6. **Sin errores en `compute-sanitizer`**: revisar especialmente el acceso a `node_body_start` en
   hojas con `body_count == 0` (pseudo-hojas) y los índices de `node_next` al salir del subárbol.

## Riesgos conocidos

- Precisión: usar `double` primero para pasar tolerancias; cambiar a `float` solo si el error sigue
  por debajo de `0.10·θ²` en mediana y `0.25·θ²` en p99.
- Energía: `U` es O(N²) y caro en GPU; solo calcularlo si piden `--energy-log` y N es pequeño.
- Colab: el repo se clona en la sesión, `cuda/barnes_hut_colab.ipynb` compila el `.cu` con `nvcc`
  y ejecuta `verify.py` con `--version cuda` (o `--cmd` apuntando a `./cuda/bh`).

## Preguntas probables en la sustentación

- ¿Por qué `node_next` y no recursión? (CUDA no tiene pila por hilo barata; el salto es un arreglo).
- ¿Cómo se evita la divergencia dentro del warp? (partículas con coste parecido, misma rama de poda
  la mayor parte del tiempo).
- ¿Cómo se valida si Colab no tiene el repo mergeado? (comparar `--dump-forces` contra `c_omp/bh`).
