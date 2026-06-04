"""Slice de texto end-to-end: split, codebook, SPIMI, busqueda.

Construye el indice desde la muestra de letras y verifica que buscar por
letra devuelve las canciones correctas.
"""
import json
from pathlib import Path

import pytest

from app.apps.music.text_index import build, MusicTextIndex

FIXTURE = Path(__file__).parent / "fixtures" / "lyrics_sample.json"


@pytest.fixture(scope="module")
def index(tmp_path_factory) -> MusicTextIndex:
    songs = json.loads(FIXTURE.read_text(encoding="utf-8"))
    out = tmp_path_factory.mktemp("music_text")
    build(songs, out, k=128, block_size=3)          # block_size pequeño = ejercita merge SPIMI
    return MusicTextIndex.load(out)


def _titles(results):
    return [r["title"] for r in results]


def test_index_built(index):
    assert len(index.items) == 8
    assert len(index.chunks) > 8                      # cada cancion da varios parrafos
    assert index.index.num_terms > 0
    assert index.index.num_postings > 0


def test_search_spanish_fuego(index):
    res = index.search("fuego pasion ciudad", top_n=3)
    titles = _titles(res)
    # las 2 canciones de fuego/pasion deben rankear arriba
    assert "Fuego en la Ciudad" in titles
    assert "Corazon de Fuego" in titles


def test_search_english_ocean(index):
    res = index.search("ocean city lights night", top_n=3)
    titles = _titles(res)
    assert "Ocean Drive" in titles or "City Lights" in titles
    assert res[0]["score"] > 0


def test_search_specific_song(index):
    res = index.search("electric dreams robots wires", top_n=1)
    assert res[0]["title"] == "Electric Dreams"


def test_no_match_returns_empty(index):
    # palabras inexistentes en el corpus, sin resultados
    res = index.search("zxqw wxyz qqqq", top_n=5)
    assert res == []
