# Barnes-Hut paralelo

Simulación gravitacional de N cuerpos con el algoritmo de Barnes-Hut (octree, criterio de apertura θ,
suavizado ε, G = 1, integrador leapfrog), paralelizada en cuatro versiones:

| Carpeta      | Versión                                   |
| ------------ | ----------------------------------------- |
| `python_mp/` | Python con `multiprocessing`              |
| `mpi/`       | MPI con mpi4py                            |
| `c_omp/`     | C con OpenMP                              |
| `cuda/`      | CUDA en Google Colab (GPU T4)             |

Las versiones de Python comparten el núcleo `core_py/` (NumPy + Numba); las de C y CUDA comparten el
diseño de `core_c/`. Las herramientas comunes (generador de distribuciones, suma directa N²,
verificación y mediciones) están en `tools/`.

## Instalación (Linux o WSL2)

```bash
sudo apt install openmpi-bin libopenmpi-dev build-essential
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Uso

```bash
# Generar una distribución
python tools/gen_dist.py --tipo uniforme --n 20000 --seed 42 -o data/u20k.txt

# Verificar una versión contra la suma directa
python tools/verify.py --version mp --n 5000 --theta 0.5

# Medir escalado
python tools/bench.py --version mp --procs 1 2 4 8

# MPI
mpirun -n 4 python mpi/main.py --input data/u20k.txt --steps 100

# C con OpenMP
make -C c_omp && OMP_NUM_THREADS=4 ./c_omp/bh --input data/u20k.txt --steps 100
```

Todas las versiones aceptan la misma interfaz:
`--input --steps --dt --theta --eps --workers --partition --dump-forces --energy-log --timing`.

## Formato de entrada

Archivo de texto con N filas y 7 columnas: `x y z vx vy vz m`.

## Equipo

| Integrante | Rol principal                                        |
| ---------- | ---------------------------------------------------- |
| A          | Herramientas comunes, Python multiprocessing, informe |
| B          | MPI                                                  |
| C          | C con OpenMP y CUDA                                  |
