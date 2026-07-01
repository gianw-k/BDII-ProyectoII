# Mini-manual de usuario

Guía rápida para usar el **Sistema Multimodal de Recuperación y Búsqueda**. Para
la arquitectura interna y los resultados experimentales, ver el
[README](../README.md).

---

## 1. Levantar el sistema

Requisito: Docker.

```bash
docker compose up --build      # db (5432) + backend (8000) + frontend (5173)
```

Antes de buscar hay que construir los índices una vez (ver el
[README](../README.md#instalación-y-uso) para el `ingest` de cada app).

Comprobación rápida de que el backend responde:

```bash
curl http://localhost:8000/health      # -> {"status":"ok"}
```

Direcciones:

| Servicio | URL |
|----------|-----|
| Interfaz gráfica (GUI) | http://localhost:5173 |
| Documentación de la API | http://localhost:8000/docs |


---

## 2. La interfaz

La GUI tiene tres pestañas en la parte superior:

1. **Visual E-commerce** — búsqueda de productos por imagen.
2. **Búsqueda Musical** — búsqueda de canciones por letra o por audio.
3. **Consola SQL** — el mismo motor desde un mini-lenguaje tipo SQL.

---

## 3. Búsqueda Visual E-commerce (imagen)

1. Abre la pestaña **Visual E-commerce**.
2. Pulsa **Elegir archivo** y selecciona la foto de un producto.
3. Se muestra una vista previa de tu consulta.
4. (Opcional) ajusta **Top-N**: cuántos resultados traer.
5. Pulsa **Buscar**.

El motor extrae SIFT + histograma de color HSV, los cuantiza en visual words
(K-Means) y busca por coseno sobre el índice invertido. Cada resultado muestra su
posición (`#1`, `#2`, …), una barra de similitud y el valor de coseno.


---

## 4. Búsqueda Musical (texto y audio)

Dos sub-modos dentro de la pestaña:

### Por letra (TF-IDF)
Escribe un fragmento de letra y pulsa **Buscar**. El motor usa el índice
invertido sobre las letras y devuelve las canciones más parecidas, con el
fragmento que hizo *match*.

### Por similitud acústica (MFCC)
- **Subir archivo**: sube un `.wav`/`.mp3`; el servidor extrae los MFCC y busca
  pistas musicalmente similares.
- **Elegir pista del dataset**: selecciona un género y una pista de ejemplo;
  puedes escuchar la consulta y los resultados con el reproductor integrado.


---

## 5. Consola SQL (ParserSQL)

La consola permite consultar las tres modalidades con una sola sintaxis tipo
SQL. Escribe la consulta y pulsa **Ejecutar** (o `Ctrl+Enter`). Hay botones de
ejemplo para cargar consultas típicas.

### Gramática

```
SELECT <campos> FROM <colección> WHERE <campo> <operador> '<valor>' [LIMIT <n>]
```

- **campos**: `*` o una lista separada por comas (`title, artist`). `score`
  siempre se incluye.
- **colección** → modalidad:
  - `songs` / `lyrics` → texto
  - `tracks` / `audio` → audio (por nombre de pista)
- **operadores**: `LIKE`, `@@`, `<->`, `=` (todos se interpretan como
  "parecido a" / "contiene").

> La búsqueda por imagen se hace subiendo una foto en la pestaña **Visual
> E-commerce**, no desde la consola SQL.
- **LIMIT**: entero positivo (tope 100; por defecto 10).

### Ejemplos

```sql
SELECT * FROM songs   WHERE lyrics @@ 'love you baby'   LIMIT 10
SELECT title, artist FROM songs WHERE lyrics LIKE 'midnight rain'
SELECT * FROM tracks  WHERE audio <-> 'blues.00000.wav' LIMIT 5
```

El resultado muestra cómo se parseó la consulta (modalidad, colección, operador,
límite) y una tabla con las filas recuperadas.

Errores de sintaxis se reportan con un mensaje claro, p. ej. *"coleccion
desconocida 'foo'"* o *"se esperaba 'WHERE'..."*.


---

## 6. Problemas comunes

| Síntoma | Causa probable | Solución |
|---------|----------------|----------|
| `503 indice no construido` | No se corrió el `ingest` | Construye el índice de esa app (ver README) |
| `404 pista/producto no encontrado` | El `filename` no está en el índice | Usa un nombre exacto del dataset |
| Imágenes rotas en los resultados | El backend no encuentra `/data/raw/fashion/images` | Verifica el montaje del volumen de imágenes |
