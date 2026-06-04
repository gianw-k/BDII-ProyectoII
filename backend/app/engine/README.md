# Engine — arquitectura unificada

Contratos en `base.py` (`Splitter`, `Extractor`, `Codebook`). Cada modalidad
los implementa; apps/persistencia/búsqueda son agnósticas.

| Subpaquete | Módulos planeados | Modalidad |
|-----------|-------------------|-----------|
| `split/` | `paragraph.py` (texto), `patch.py` (imagen), `window.py` (audio) | todas |
| `extractor/` | `tfidf.py`, `sift.py`, `mfcc.py` | texto / imagen / audio |
| `codebook/` | `linguistic.py` (top-k palabras), `kmeans.py` (centroides) | texto / imagen+audio |
| `index/` | `spimi.py` (obligatorio texto), `inverted.py`, `histogram.py` | todas |
| `search/` | `similarity.py` (cosine/L2/intersección), `fusion.py` (multimodal) | todas |

Flujo offline (pipeline): split, extract, codebook.build, quantize, index, DB.
Flujo online (request): query, extract, codebook.quantize, search, top-N.
