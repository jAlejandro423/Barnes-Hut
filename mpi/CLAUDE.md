# Versión MPI (mpi4py) — dueño: integrante B (Jesus)

Este archivo complementa el `CLAUDE.md` de la raíz; no lo reemplaza. Enunciado completo del profesor:
@../docs/enunciado.md

## Por qué esta versión es crítica
- Según el enunciado, **los cambios grandes de particionado en la demo se piden sobre MPI o C**.
  El particionado debe poder cambiarse en minutos (por ejemplo, Morton → ORB o cambiar cómo se mide el costo).
- En la sustentación hay un fragmento de **código MPI con deadlock** que se busca con papel y lápiz.
  Cada vez que uses una llamada MPI, explícame si bloquea y en qué caso podría causar deadlock.

## Lo que me toca
- Núcleo Python compartido con A (integrante A): yo hago `core_py/morton.py` (códigos de Morton de 63 bits,
  21 por eje, y ordenamiento) y `core_py/traverse.py` (recorrido con θ sin recursión usando `node_next`).
  A hace `core_py/tree.py` y `core_py/integrate.py`. Las firmas y la decisión del LET están en la sección
  "Interfaz del núcleo Python (acordada por A y B)" del `CLAUDE.md` raíz; respétalas.
- Toda la carpeta `mpi/`: `main.py`, `partition.py` (Morton por costo y ORB), `let.py` (ramas remotas).
- Revisa mis PR: C. Yo reviso los de A. Soy segundo dueño de CUDA (debo poder modificarla en la demo).

## Orden de trabajo
1. Morton + recorrido secuencial que, junto con el árbol de A, pase `tools/verify.py` con N = 2.000.
2. `--partition morton`: cortar la curva de Morton por costo acumulado (interacciones del paso anterior;
   en el primer paso, costo uniforme).
3. LET con `Alltoallv`: cada proceso envía a los demás solo las ramas que ellos necesitan, nunca el árbol completo.
4. `--partition orb`: bisección recursiva por costo.
5. Distribuciones difíciles (`dos_cumulos`, `concentrada`): revisar balanceo y profundidad máxima.
6. Mediciones: P = 1, 2, 4, 8 y P = 12, 16 con `--oversubscribe`; tiempos por fase incluida `comunicacion`.

## Decisiones que debes proponerme antes de programar (3 a 5 líneas, con alternativas)
- Cómo se distribuyen las partículas al inicio y entre pasos (quién lee el archivo, Scatterv / Alltoallv).
- Caja contenedora global con `Allreduce` (min/max), para que todos usen los mismos códigos de Morton.
- Criterio para decidir qué ramas enviar a cada proceso (θ contra la caja del dominio remoto).
- Cómo se reúnen las aceleraciones en el orden original de entrada para `--dump-forces`.

## Reglas para mí
- Yo debo poder reescribir de memoria lo esencial de cada cambio: ve paso a paso y no generes archivos enteros de golpe.
- Marca `PUNTO DE MODIFICACIÓN` en: elección de particionado, cálculo del costo y criterio del LET.
- Mantén mi ficha en `docs/notas/B-mpi.md`: decisiones, alternativas descartadas y por qué.
- Al final de cada tarea, propón el mensaje de commit con alcance `core-py` o `mpi`.
