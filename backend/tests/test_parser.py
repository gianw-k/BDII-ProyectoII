"""Tests del ParserSQL: el mini-lenguaje SQL para recuperacion multimodal.

Solo se prueba el parseo (no la ejecucion contra indices reales): que la
sintaxis valida produzca el `ParsedQuery` correcto y que la invalida levante
`QueryParseError` con un mensaje util.
"""
import pytest

from app.engine.query.parser import parse, QueryParseError


# ─────────────────────────── consultas validas ────────────────────────────────

def test_text_select_star():
    q = parse("SELECT * FROM songs WHERE lyrics @@ 'love you baby' LIMIT 10")
    assert q.modality == "text"
    assert q.field == "lyrics"
    assert q.op == "@@"
    assert q.value == "love you baby"
    assert q.limit == 10
    assert q.fields == ["*"]


def test_text_like_sin_limit_usa_default():
    q = parse("SELECT title, artist FROM songs WHERE lyrics LIKE 'midnight rain'")
    assert q.modality == "text"
    assert q.fields == ["title", "artist"]
    assert q.op == "LIKE"
    assert q.limit == 10               # default cuando no hay LIMIT


def test_audio_similar_operator():
    q = parse("SELECT * FROM tracks WHERE audio <-> 'blues.00000.wav' LIMIT 5")
    assert q.modality == "audio"
    assert q.op == "<->"
    assert q.value == "blues.00000.wav"
    assert q.limit == 5


def test_image_tilde_operator():
    q = parse("SELECT * FROM products WHERE image ~ '1163.jpg' LIMIT 8")
    assert q.modality == "image"
    assert q.op == "~"
    assert q.value == "1163.jpg"
    assert q.limit == 8


def test_keywords_case_insensitive():
    q = parse("select * from SONGS where lyrics like 'hello'")
    assert q.modality == "text"
    assert q.op == "LIKE"


def test_comillas_dobles():
    q = parse('SELECT * FROM songs WHERE lyrics @@ "with \'inner\' word"')
    assert q.value == "with 'inner' word"


def test_limit_se_capa_a_100():
    q = parse("SELECT * FROM songs WHERE lyrics @@ 'x' LIMIT 9999")
    assert q.limit == 100


# ─────────────────────────── consultas invalidas ──────────────────────────────

@pytest.mark.parametrize("sql", [
    "",
    "   ",
    "SELECT * FROM songs",                                   # falta WHERE
    "SELECT FROM songs WHERE lyrics @@ 'x'",                 # falta campos
    "SELECT * FROM unknown WHERE x @@ 'y'",                  # coleccion invalida
    "SELECT * FROM songs WHERE lyrics ?? 'x'",               # operador invalido
    "SELECT * FROM songs WHERE lyrics @@ noquotes",          # literal sin comillas
    "SELECT * FROM songs WHERE lyrics @@ 'x' LIMIT abc",     # limit no numerico
    "SELECT * FROM songs WHERE lyrics @@ 'x' LIMIT 0",       # limit no positivo
    "SELECT * FROM songs WHERE lyrics @@ 'x' EXTRA",         # texto sobrante
])
def test_consultas_invalidas_levantan_error(sql):
    with pytest.raises(QueryParseError):
        parse(sql)


def test_mensaje_de_error_es_legible():
    with pytest.raises(QueryParseError) as exc:
        parse("SELECT * FROM unknown WHERE x @@ 'y'")
    assert "coleccion desconocida" in str(exc.value).lower()
