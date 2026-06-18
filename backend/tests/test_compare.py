"""Comparativas Fase 3 contra Postgres real (texto + imagen).

Se salta si no hay DB. Verifica que los 3 (texto) / 2 (imagen) enfoques corren
sobre los mismos datos, devuelven resultados coherentes y reportan latencia.
"""
import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

URL = "postgresql://bdii:bdii@localhost:5432/multimodal"
FIXT = Path(__file__).parent / "fixtures" / "lyrics_sample.json"


@pytest.fixture(scope="module")
def conn():
    psycopg2 = pytest.importorskip("psycopg2")
    try:
        c = psycopg2.connect(URL, connect_timeout=3)
    except psycopg2.OperationalError as e:
        pytest.skip(f"no hay Postgres: {e}")
    from pgvector.psycopg2 import register_vector
    register_vector(c)
    yield c
    c.close()


@pytest.fixture(scope="module")
def text_index(conn):
    from app.apps.music.text_index import build
    from app.db.adapters import text_index_to_data
    from app.db.repository import persist_index

    songs = json.loads(FIXT.read_text(encoding="utf-8"))
    idx = build(songs, tempfile.mkdtemp(), k=128, block_size=3)
    persist_index(conn, text_index_to_data(idx))
    conn.commit()
    return idx


@pytest.fixture(scope="module")
def visual_index(conn):
    from app.apps.ecommerce.visual_index import build
    from app.db.adapters import visual_index_to_data
    from app.db.repository import persist_index

    rng = np.random.default_rng(0)
    centers = np.eye(3, 8, dtype=np.float32) * 10
    items, descs = [], []
    for i in range(9):
        c = i % 3
        items.append({"external_id": f"p{i}", "tipo": ["a", "b", "c"][c], "cluster": c})
        descs.append(centers[c] + rng.normal(0, 0.5, size=(40, 8)).astype("float32"))
    idx = build(items, descs, tempfile.mkdtemp(), k=3, block_size=4)
    persist_index(conn, visual_index_to_data(idx))
    conn.commit()
    return idx


def test_text_three_methods_agree(conn, text_index):
    from app.comparisons import text as cmp

    q = "fuego pasion ciudad"
    own = cmp.own_search(text_index, q, top_n=3)
    gin = cmp.gin_search(conn, q, top_n=3)
    vec = cmp.pgvector_search(conn, text_index.codebook, q, top_n=3)

    for m in (own, gin, vec):
        assert m["count"] > 0, f"{m['method']} no devolvio nada"
        assert m["latency_ms"] >= 0

    # las 3 tecnicas deben coincidir en que una cancion de fuego es relevante
    def titles(m):
        return {r["title"] for r in m["results"]}
    fuego = {"Fuego en la Ciudad", "Corazon de Fuego"}
    assert titles(own) & fuego
    assert titles(gin) & fuego
    assert titles(vec) & fuego


def test_image_two_methods_same_cluster(conn, visual_index):
    from app.comparisons import image as cmp

    # p0 es del cluster 0; sus vecinos (p3, p6) tambien
    own = cmp.own_search(visual_index, "p0", top_n=3)
    vec = cmp.pgvector_search(conn, "p0", top_n=3)

    assert own["count"] > 0 and vec["count"] > 0
    # el vecino top por ambas vias debe ser del mismo cluster (otro 'a')
    assert own["results"][0]["external_id"] in {"p3", "p6"}
    assert vec["results"][0]["external_id"] in {"p3", "p6"}


def test_compare_text_http(conn, text_index, monkeypatch):
    # smoke test del endpoint real /compare/text
    from app.api import compare

    # compare importo _text_index a su namespace, asi que se parchea ahi
    monkeypatch.setattr(compare, "_text_index", lambda: text_index)
    compare._conn.cache_clear()

    from fastapi.testclient import TestClient
    from app.main import app
    client = TestClient(app)

    r = client.get("/compare/text", params={"q": "ocean city lights", "top_n": 3})
    assert r.status_code == 200
    body = r.json()
    methods = {m["method"] for m in body["methods"]}
    assert methods == {"inverted_index", "gin_fulltext", "pgvector_cosine"}
