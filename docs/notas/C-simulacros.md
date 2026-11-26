# Fragmentos con bugs para simulacros

Dos errores típicos para practicar la detección en vivo. Se pueden pegar en `core_c/traverse.c` o
`core_c/tree.c` y corregir en pocos minutos. **No están aplicados al código real.**

## Bug 1 — criterio de apertura al revés

Cambiar en `accept_cell`:

```c
return size * size > theta * theta * d2;   /* BUG: > en vez de < */
```

Síntoma: las fuerzas salen mal y el número de interacciones cae mucho; con θ = 0 casi todas las
celdas se aceptan como masa puntual y el resultado se parece muy poco a la suma directa.
Corrección: `size * size < theta * theta * d2`.

Pregunta para el grupo: ¿por qué con θ = 0 este bug no da directamente la suma directa?

## Bug 2 — salto equivocado al podar una celda

Cambiar en `bh_accel_one`, dentro de la rama de celda aceptada:

```c
node = tree->node_first_child[node];       /* BUG: abre el subárbol que se acaba de podar */
```

Síntoma: la aceleración cuenta dos veces (o más) las mismas celdas; la mediana del error relativo
sube aunque la energía siga más o menos estable. Corrección: `node = tree->node_next[node]`.

Pregunta para el grupo: ¿cómo se detectaría esto mirando solo `cost` y no las fuerzas?
