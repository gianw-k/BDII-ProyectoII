"""ParserSQL: un mini-lenguaje tipo SQL para recuperacion de texto y multimedia.

La idea es que el usuario consulte el motor con una sintaxis familiar en vez de
un formulario, dejando claro que por debajo es el mismo paradigma agnostico de
modalidad: una coleccion, un campo de contenido y un operador de similitud.

Gramatica soportada (palabras clave case-insensitive):

    SELECT  <campos>  FROM <coleccion>  WHERE <campo> <op> <literal>  [LIMIT <n>]

    campos     := '*' | ident (',' ident)*
    coleccion  := songs | tracks | products | ...   (alias por modalidad)
    op         := LIKE | @@ | ~ | <-> | =
    literal    := 'texto entre comillas'  ó  "texto"

Ejemplos:

    SELECT * FROM songs   WHERE lyrics @@ 'love you baby'      LIMIT 10
    SELECT title, artist FROM songs WHERE lyrics LIKE 'midnight rain'
    SELECT * FROM tracks  WHERE audio <-> 'blues.00000.wav'    LIMIT 5
    SELECT * FROM products WHERE image ~ '1163.jpg'            LIMIT 8

El parser es un descenso recursivo escrito a mano (sin dependencias externas):
tokeniza y construye un `ParsedQuery`. La ejecucion vive en la capa de API, que
despacha el `ParsedQuery` al indice de la modalidad correspondiente.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field


class QueryParseError(ValueError):
    """Error de sintaxis en la consulta SQL del usuario."""


# ─────────────────────────── alias de coleccion -> modalidad ──────────────────

# Cada coleccion (la "tabla") cae en una de las tres modalidades del motor.
_COLLECTIONS: dict[str, str] = {
    "songs": "text", "song": "text", "lyrics": "text", "music_text": "text",
    "tracks": "audio", "track": "audio", "audio": "audio", "music_audio": "audio",
    "products": "image", "product": "image", "images": "image",
    "image": "image", "ecommerce": "image",
}

# Operadores de recuperacion validos. Todos se interpretan como "parecido a" /
# "contiene"; se conserva el operador tal cual por si el informe lo necesita.
_OPERATORS = {"LIKE", "@@", "~", "<->", "="}


# ─────────────────────────── resultado del parseo ─────────────────────────────

@dataclass
class ParsedQuery:
    modality: str               # text | audio | image
    collection: str             # coleccion normalizada tal como la escribio el user
    field: str                  # campo de contenido (lyrics / audio / image)
    op: str                     # operador de recuperacion usado
    value: str                  # literal de la consulta (texto o filename)
    limit: int = 10             # top-N
    fields: list[str] = field(default_factory=lambda: ["*"])  # columnas proyectadas

    def to_dict(self) -> dict:
        return {
            "modality": self.modality,
            "collection": self.collection,
            "field": self.field,
            "op": self.op,
            "value": self.value,
            "limit": self.limit,
            "fields": self.fields,
        }


# ─────────────────────────── tokenizador ──────────────────────────────────────

# Orden importante: los operadores multi-caracter (<->, @@) antes que los de un
# solo caracter para que el regex no los parta.
_TOKEN_RE = re.compile(
    r"""
      (?P<ws>\s+)                         # espacios (se descartan)
    | (?P<string>'[^']*'|"[^"]*")         # literal entre comillas
    | (?P<op><->|@@|~|=)                  # operadores simbolicos
    | (?P<punct>[*,])                     # estrella y coma
    | (?P<ident>[A-Za-z_][A-Za-z0-9_]*)   # identificadores / palabras clave
    | (?P<number>\d+)                     # enteros (LIMIT)
    """,
    re.VERBOSE,
)


@dataclass
class _Tok:
    kind: str
    value: str


def _tokenize(sql: str) -> list[_Tok]:
    toks: list[_Tok] = []
    pos = 0
    n = len(sql)
    while pos < n:
        m = _TOKEN_RE.match(sql, pos)
        if not m:
            raise QueryParseError(f"caracter inesperado en la posicion {pos}: '{sql[pos]}'")
        pos = m.end()
        kind = m.lastgroup
        if kind == "ws":
            continue
        text = m.group()
        if kind == "string":
            toks.append(_Tok("string", text[1:-1]))   # sin comillas
        elif kind == "number":
            toks.append(_Tok("number", text))
        elif kind == "op" or kind == "punct":
            toks.append(_Tok("symbol", text))
        else:  # ident -> puede ser palabra clave; se decide en el parser
            toks.append(_Tok("ident", text))
    return toks


# ─────────────────────────── parser (descenso recursivo) ──────────────────────

class _Parser:
    def __init__(self, toks: list[_Tok]):
        self.toks = toks
        self.i = 0

    # -- utilidades de cursor --------------------------------------------------
    def _peek(self) -> _Tok | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def _next(self) -> _Tok:
        tok = self._peek()
        if tok is None:
            raise QueryParseError("consulta incompleta: se esperaba mas entrada")
        self.i += 1
        return tok

    def _expect_keyword(self, kw: str) -> None:
        tok = self._next()
        if tok.kind != "ident" or tok.value.upper() != kw:
            raise QueryParseError(f"se esperaba '{kw}' pero se encontro '{tok.value}'")

    # -- regla principal -------------------------------------------------------
    def parse(self) -> ParsedQuery:
        self._expect_keyword("SELECT")
        fields = self._parse_fields()
        self._expect_keyword("FROM")
        collection = self._parse_ident("nombre de coleccion")
        modality = _COLLECTIONS.get(collection.lower())
        if modality is None:
            raise QueryParseError(
                f"coleccion desconocida '{collection}'. "
                f"Validas: {', '.join(sorted(set(_COLLECTIONS)))}"
            )
        self._expect_keyword("WHERE")
        field_name = self._parse_ident("campo de la condicion")
        op = self._parse_operator()
        value = self._parse_string()
        limit = self._parse_optional_limit()

        if self._peek() is not None:
            extra = self._peek().value
            raise QueryParseError(f"texto sobrante tras la consulta: '{extra}'")

        return ParsedQuery(
            modality=modality,
            collection=collection,
            field=field_name,
            op=op,
            value=value,
            limit=limit,
            fields=fields,
        )

    # -- sub-reglas ------------------------------------------------------------
    def _parse_fields(self) -> list[str]:
        tok = self._next()
        if tok.kind == "symbol" and tok.value == "*":
            return ["*"]
        if tok.kind != "ident":
            raise QueryParseError(f"se esperaba '*' o un campo, no '{tok.value}'")
        fields = [tok.value]
        while self._peek() and self._peek().kind == "symbol" and self._peek().value == ",":
            self._next()  # consume ','
            nxt = self._next()
            if nxt.kind != "ident":
                raise QueryParseError(f"se esperaba un campo tras ',', no '{nxt.value}'")
            fields.append(nxt.value)
        return fields

    def _parse_ident(self, what: str) -> str:
        tok = self._next()
        if tok.kind != "ident":
            raise QueryParseError(f"se esperaba {what}, no '{tok.value}'")
        return tok.value

    def _parse_operator(self) -> str:
        tok = self._next()
        candidate = tok.value.upper() if tok.kind == "ident" else tok.value
        if candidate not in _OPERATORS:
            raise QueryParseError(
                f"operador no soportado '{tok.value}'. "
                f"Usa uno de: {', '.join(sorted(_OPERATORS))}"
            )
        return candidate

    def _parse_string(self) -> str:
        tok = self._next()
        if tok.kind != "string":
            raise QueryParseError(
                f"se esperaba un literal entre comillas, no '{tok.value}'"
            )
        if not tok.value.strip():
            raise QueryParseError("el valor de busqueda esta vacio")
        return tok.value

    def _parse_optional_limit(self) -> int:
        tok = self._peek()
        if tok is None:
            return 10
        if tok.kind == "ident" and tok.value.upper() == "LIMIT":
            self._next()
            num = self._next()
            if num.kind != "number":
                raise QueryParseError(f"LIMIT espera un entero, no '{num.value}'")
            n = int(num.value)
            if n <= 0:
                raise QueryParseError("LIMIT debe ser un entero positivo")
            return min(n, 100)   # techo de seguridad
        return 10


def parse(sql: str) -> ParsedQuery:
    """Parsea una consulta SQL del mini-lenguaje y devuelve un `ParsedQuery`.

    Lanza `QueryParseError` con un mensaje legible si la sintaxis es invalida.
    """
    if not sql or not sql.strip():
        raise QueryParseError("consulta vacia")
    toks = _tokenize(sql)
    if not toks:
        raise QueryParseError("consulta vacia")
    return _Parser(toks).parse()
