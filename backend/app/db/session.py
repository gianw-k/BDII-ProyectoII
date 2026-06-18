"""Conexion a PostgreSQL.

Usamos psycopg2 directo (sin ORM): el esquema ya esta escrito a mano en
db/init/01_init.sql y las consultas son simples, asi que un ORM solo
estorbaria. `register_vector` engancha el tipo `vector` de pgvector para poder
mandar/recibir arrays de numpy tal cual.
"""
from __future__ import annotations
from contextlib import contextmanager

import psycopg2
from pgvector.psycopg2 import register_vector

from app.core.config import settings


def connect(url: str | None = None):
    """Abre una conexion y deja listo el tipo vector de pgvector."""
    conn = psycopg2.connect(url or settings.database_url)
    register_vector(conn)
    return conn


@contextmanager
def get_conn(url: str | None = None):
    """Conexion como context manager: commitea si todo va bien, rollback si no."""
    conn = connect(url)
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
