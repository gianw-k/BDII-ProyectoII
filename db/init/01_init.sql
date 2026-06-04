-- Inicializacion DB multimodal. Corre 1 vez al crear el volumen.
CREATE EXTENSION IF NOT EXISTS vector;

-- Catalogo de items (un producto de moda o una cancion)
CREATE TABLE IF NOT EXISTS items (
    id          BIGSERIAL PRIMARY KEY,
    modality    TEXT NOT NULL,          -- 'image' | 'audio' | 'text'
    app         TEXT NOT NULL,          -- 'ecommerce' | 'music'
    external_id TEXT,                   -- id en el dataset original
    metadata    JSONB DEFAULT '{}'::jsonb
);

-- Chunks: unidades atomicas tras el split
CREATE TABLE IF NOT EXISTS chunks (
    id        BIGSERIAL PRIMARY KEY,
    item_id   BIGINT REFERENCES items(id) ON DELETE CASCADE,
    modality  TEXT NOT NULL,
    position  INT,                      -- orden dentro del item
    content   TEXT                      -- texto del chunk (solo modalidad texto)
);

-- Codebook: palabras visuales/acusticas (centroides) o textuales
CREATE TABLE IF NOT EXISTS codebook (
    id        BIGSERIAL PRIMARY KEY,
    modality  TEXT NOT NULL,
    word_idx  INT NOT NULL,             -- indice de la codeword (0..k-1)
    term      TEXT,                     -- palabra (modalidad texto)
    centroid  vector,                   -- centroide (imagen/audio)
    UNIQUE (modality, word_idx)
);

-- Histogramas: vector de frecuencias de codewords por chunk
-- Se guardan 2 formas: como vector (pgvector) y como columna para tu motor
CREATE TABLE IF NOT EXISTS histograms (
    chunk_id  BIGINT PRIMARY KEY REFERENCES chunks(id) ON DELETE CASCADE,
    modality  TEXT NOT NULL,
    hist      vector NOT NULL           -- histograma normalizado para pgvector
);

-- Indice pgvector (crear tras carga; ejemplo HNSW coseno)
-- CREATE INDEX ON histograms USING hnsw (hist vector_cosine_ops);

-- Indice invertido propio (tu implementacion, via SPIMI para texto)
-- codeword: postings (chunk_id, freq)
CREATE TABLE IF NOT EXISTS inverted_index (
    modality  TEXT NOT NULL,
    word_idx  INT NOT NULL,
    chunk_id  BIGINT REFERENCES chunks(id) ON DELETE CASCADE,
    freq      REAL NOT NULL,
    PRIMARY KEY (modality, word_idx, chunk_id)
);
CREATE INDEX IF NOT EXISTS idx_inverted_word ON inverted_index (modality, word_idx);

-- Comparativa texto nativa: tsvector + GIN (letras de canciones)
ALTER TABLE chunks ADD COLUMN IF NOT EXISTS tsv tsvector;
CREATE INDEX IF NOT EXISTS idx_chunks_tsv ON chunks USING gin (tsv);
