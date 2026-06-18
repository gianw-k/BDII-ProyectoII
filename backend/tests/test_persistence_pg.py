"""Persistencia en PostgreSQL real (texto + imagen).

Necesita una DB Postgres+pgvector corriendo (docker compose up -d db). Si no la
encuentra, los tests se SALTAN en vez de fallar, asi la suite sigue verde en
maquinas sin DB. La URL se puede pasar por TEST_DATABASE_URL.
"""
import os
import json
from pathlib import Path

import numpy as np
import pytest

URL = os.environ.get("TEST_DATABASE_URL", "postgresql://bdii:bdii@localhost:5432/multimodal")


@pytest.fixture(scope="module")
def conn():
    psycopg2 = pytest.importorskip("psycopg2")
    try:
        c = psycopg2.connect(URL, connect_timeout=3)
    except psycopg2.OperationalError as e:
        pytest.skip(f"no hay Postgres en {URL}: {e}")
    from pgvector.psycopg2 import register_vector
    register_vector(c)
    yield c
    c.close()


def _count(conn, sql, params=()):
    with conn.cursor() as cur:
        cur.execute(sql, params)
        return cur.fetchone()[0]


def test_persist_text(conn):
    from app.apps.music.text_index import build
    from app.db.adapters import text_index_to_data
    from app.db.repository import persist_index

    songs = json.loads(
        (Path(__file__).parent / "fixtures" / "lyrics_sample.json").read_text(encoding="utf-8")
    )
    idx = build(songs, pytest.importorskip("tempfile").mkdtemp(), k=128, block_size=3)
    summary = persist_index(conn, text_index_to_data(idx))
    conn.commit()

    assert summary["items"] == 8
    assert _count(conn, "SELECT count(*) FROM items WHERE app='music'") == 8
    assert _count(conn, "SELECT count(*) FROM codebook WHERE modality='text'") == len(idx.codebook.terms)
    assert _count(conn, "SELECT count(*) FROM inverted_index WHERE modality='text'") == summary["postings"]
    # el tsvector quedo poblado (para la comparativa GIN)
    assert _count(conn, "SELECT count(*) FROM chunks WHERE modality='text' AND tsv IS NOT NULL") > 0


def test_gin_fulltext_query(conn):
    # consulta full-text nativa sobre las letras ya persistidas (Fase 3, GIN)
    with conn.cursor() as cur:
        cur.execute(
            "SELECT count(*) FROM chunks "
            "WHERE modality='text' AND tsv @@ plainto_tsquery('spanish', %s)",
            ("fuego",),
        )
        assert cur.fetchone()[0] > 0


def test_persist_image(conn):
    from app.apps.ecommerce.visual_index import build
    from app.db.adapters import visual_index_to_data
    from app.db.repository import persist_index
    import tempfile

    rng = np.random.default_rng(0)
    centers = np.eye(3, 8, dtype=np.float32) * 10
    items, descs = [], []
    for i in range(9):
        c = i % 3
        items.append({"external_id": f"p{i}", "tipo": ["a", "b", "c"][c]})
        descs.append(centers[c] + rng.normal(0, 0.5, size=(40, 8)).astype("float32"))
    idx = build(items, descs, tempfile.mkdtemp(), k=3, block_size=4)

    summary = persist_index(conn, visual_index_to_data(idx))
    conn.commit()
    assert _count(conn, "SELECT count(*) FROM items WHERE app='ecommerce'") == 9
    assert _count(conn, "SELECT count(*) FROM codebook WHERE modality='image'") == 3
    # los centroides se guardaron como vector de pgvector con la dimension correcta
    with conn.cursor() as cur:
        cur.execute("SELECT centroid FROM codebook WHERE modality='image' LIMIT 1")
        centroid = cur.fetchone()[0]
        assert len(centroid) == 8


def test_pgvector_knn(conn):
    # busqueda por similitud con pgvector sobre los histogramas (Fase 3, vector)
    with conn.cursor() as cur:
        cur.execute("SELECT hist FROM histograms WHERE modality='image' LIMIT 1")
        probe = cur.fetchone()[0]
        cur.execute(
            "SELECT chunk_id FROM histograms WHERE modality='image' "
            "ORDER BY hist <=> %s LIMIT 3",
            (np.asarray(probe, dtype=np.float32),),
        )
        vecinos = [r[0] for r in cur.fetchall()]
        assert len(vecinos) == 3
