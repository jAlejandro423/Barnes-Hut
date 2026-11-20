# Proyecto final: simulación de N cuerpos con Barnes-Hut (enunciado)

> Transcripción del PDF del enunciado del profesor, para que Claude Code tenga el contexto completo.

El proyecto se desarrolla en grupos de máximo tres estudiantes y consiste en implementar una simulación
gravitacional de N cuerpos con el algoritmo de Barnes-Hut y paralelizarla en cuatro versiones: Python con
multiprocessing, MPI, C con OpenMP o pthreads, y CUDA en Google Colab.

Se simula un conjunto de N partículas puntuales que se atraen por gravedad. Cada partícula tiene una masa y una
posición y una velocidad en tres dimensiones. En cada paso de tiempo se calcula la fuerza gravitacional que todas
las demás ejercen sobre cada partícula, se actualizan velocidades y posiciones con el integrador leapfrog y se
repite durante un número fijo de pasos, por ejemplo 100 para las mediciones. Se usa G igual a 1 y un suavizado
gravitacional ε en la distancia, para evitar fuerzas infinitas cuando dos partículas se acercan demasiado. Calcular
las fuerzas sumando todos los pares cuesta del orden de N al cuadrado operaciones por paso, lo que se vuelve
inviable para N grande.

El algoritmo de Barnes-Hut reduce ese costo a del orden de N log N. En cada paso las partículas se organizan en un
octree: un cubo que contiene a todas se divide recursivamente en ocho subcubos hasta que cada celda contiene a lo
sumo una partícula. Cada celda guarda su masa total y su centro de masa. Para calcular la fuerza sobre una
partícula se recorre el árbol desde la raíz. Si una celda está lo bastante lejos, es decir, si su tamaño dividido
por la distancia a la partícula es menor que un parámetro θ, toda la celda se trata como una sola partícula
situada en su centro de masa. Si no, se abre y se examinan sus ocho hijos. Un θ menor da más precisión y más
costo, y θ igual a cero equivale a la suma directa. El árbol se reconstruye en cada paso porque las partículas se
mueven.

No se usa una base de datos externa: las partículas se generan. Cada simulación parte de un archivo con N filas y
siete columnas: posición (x, y, z), velocidad (vx, vy, vz) y masa. Para desarrollar y probar, cada grupo genera
una distribución uniforme, con posiciones al azar dentro de un cubo unitario, velocidades iguales a cero y masas
iguales a 1/N, con semilla fija para que las corridas sean reproducibles. La distribución usada en la
demostración se entrega en este mismo formato.

Cada versión debe resolver su propia dificultad de paralelización. En Python, el árbol debe compartirse o
reconstruirse entre procesos mediante arreglos planos en memoria compartida, reconstrucción local o comunicación
explícita, y la decisión debe sustentarse con mediciones. En MPI, el espacio debe particionarse, por ejemplo con
una curva de Morton o con bisección recursiva, y cada proceso debe recibir solo las partes remotas del árbol que
necesita, sin replicarlo completo. En C, la construcción del árbol debe ser paralela, ya sea con inserción
protegida por candados o a partir de un ordenamiento por Morton, y la carga debe balancearse entre hilos. En CUDA,
el árbol debe aplanarse, las partículas ordenarse y el recorrido hacerse sin recursión, minimizando la divergencia
dentro de cada warp.

El número de partículas va de 20 mil a 200 mil. MPI se ejecuta en una sola máquina con mpirun (en Windows, con
WSL2). La versión CUDA se desarrolla y se ejecuta en Colab con la GPU asignada, normalmente una T4. Como Colab no
incluye nsys ni ncu, el tiempo debe medirse con cudaEvent y reportarse por fase: construcción del árbol,
ordenamiento, recorrido e integración. Para detectar errores de memoria y condiciones de carrera se usa
compute-sanitizer.

Una versión es correcta si sus fuerzas coinciden con las de la suma directa de orden N al cuadrado sobre un N
pequeño, con un error relativo aceptable para el valor de θ elegido, y si la energía total se conserva dentro de
una tolerancia razonable durante la simulación. Una versión que no cumpla esta verificación se considera
incorrecta.

Antes de la sustentación se entregan un informe de máximo seis páginas, el repositorio de git con su historial de
commits y el notebook de CUDA con la sesión ya probada. El informe describe el diseño de cada versión y presenta
las mediciones: escalado con 1, 2, 4 y 8 procesos o hilos, efecto de usar más procesos que núcleos y desglose por
fase en la GPU, junto con el análisis de por qué las curvas se aplanan donde lo hacen, considerando núcleos
físicos frente a hilos virtuales, ancho de banda de memoria y fracción secuencial. La nota es individual: el
informe se califica igual para todos los integrantes del grupo, y la sustentación y la demostración se califican
por integrante según lo que cada uno responda y haga.

| Componente | Peso | Se evalúa |
| --- | --- | --- |
| Informe | 10 % | Diseño de cada versión, mediciones, análisis de escalado, verificación de correctitud e historial de git |
| Sustentación | 35 % | Predicción oral, hallazgo del error en un fragmento de código y justificación de las decisiones de diseño |
| Demostración en vivo | 55 % | Ejecución sobre una distribución desconocida, diagnóstico y modificación del código en el momento |

La sustentación es en grupo, dura unos 25 minutos y se hace sin internet ni asistentes de IA. Tiene tres partes.
Primero, una predicción que se responde antes de ejecutar nada, por ejemplo qué ocurre al pasar de cuatro a ocho
procesos y por qué. Segundo, un fragmento de código MPI o CUDA con un deadlock o una condición de carrera, que
debe encontrarse leyendo, con papel y lápiz. Tercero, preguntas sobre las decisiones de diseño del informe, que
deben poder justificarse. Las preguntas se dirigen a integrantes específicos, por lo que todos deben conocer todo
el proyecto, y cada uno es calificado por sus propias respuestas.

La demostración en vivo se hace en el computador del grupo, sin internet ni asistentes de IA, con el código
entregado. Ese día se entrega una distribución de partículas desconocida que castiga un reparto ingenuo del
trabajo, por ejemplo dos cúmulos densos en colisión o casi todas las partículas concentradas en un volumen muy
pequeño. Cada versión se ejecuta sobre ella con una herramienta de monitoreo abierta, y el grupo explica lo que
observa y diagnostica lo que falle. A continuación se pide una modificación al código, definida en el momento,
como cambiar la estrategia de particionado, el criterio de apertura θ o la forma de asignar partículas a warps. La
distribución y la modificación no se informan con anticipación, por lo que el código debe estar escrito de forma
que pueda leerse y cambiarse en pocos minutos. Las partes de la demostración se asignan a integrantes específicos
en el momento, y cada uno es calificado por lo que hace y explica.

Si Colab no conecta el día de la demostración, el comportamiento de la GPU se explica con el desglose de tiempos
entregado, con menor puntaje en esa parte. La modificación pedida sobre CUDA es un cambio pequeño y localizable;
los cambios grandes de particionado se piden sobre la versión de C o de MPI, que se ejecutan en local.
