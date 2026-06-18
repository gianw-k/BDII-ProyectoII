"""Prueba del camino vectorial completo: K-Means, SPIMI y busqueda.

No usa OpenCV ni dataset: inventamos 3 clusters bien separados y armamos items
que tiran a uno de ellos. Lo que queremos comprobar es que el motor (K-Means ->
histograma BoVW -> SPIMI -> indice invertido -> coseno) devuelve los items del
mismo cluster que la consulta. Si esto pasa, queda demostrado que el indice y la
busqueda de texto sirven igual para vectores sin tocar nada.
"""
import numpy as np
import pytest

from app.engine.codebook.kmeans import KMeansCodebook
from app.apps.ecommerce.visual_index import build, VisualIndex

DIM = 8
N_CLUSTERS = 3
CENTERS = np.array([
    [10, 0, 0, 0, 0, 0, 0, 0],
    [0, 10, 0, 0, 0, 0, 0, 0],
    [0, 0, 10, 0, 0, 0, 0, 0],
], dtype=np.float32)


def _descriptors(rng, cluster: int, n: int = 40) -> np.ndarray:
    """n descriptores cerca del centro `cluster`, con un poco de ruido encima."""
    return CENTERS[cluster] + rng.normal(0, 0.5, size=(n, DIM)).astype(np.float32)


@pytest.fixture(scope="module")
def planted():
    rng = np.random.default_rng(42)
    items, descriptors, labels = [], [], []
    for i in range(15):
        c = i % N_CLUSTERS
        items.append({"name": f"item-{i}", "cluster": c})
        descriptors.append(_descriptors(rng, c))
        labels.append(c)
    return items, descriptors, labels


@pytest.fixture(scope="module")
def index(planted, tmp_path_factory) -> VisualIndex:
    items, descriptors, _ = planted
    out = tmp_path_factory.mktemp("ecommerce_image")
    # block_size chico a proposito, asi SPIMI tiene que mergear varios bloques
    build(items, descriptors, out, k=N_CLUSTERS, block_size=4)
    return VisualIndex.load(out)


def test_codebook_shape(index):
    assert index.codebook.centroids is not None
    assert index.codebook.centroids.shape == (N_CLUSTERS, DIM)


def test_index_built(index, planted):
    items, _, _ = planted
    assert len(index.items) == len(items)
    assert index.index.num_postings > 0


def test_search_returns_same_cluster(index, planted):
    items, _, labels = planted
    rng = np.random.default_rng(7)
    for query_cluster in range(N_CLUSTERS):
        q = _descriptors(rng, query_cluster, n=30)
        results = index.search(q, top_n=5)
        assert results, "la busqueda no devolvio resultados"
        # el primer resultado tiene que ser del mismo cluster que preguntamos
        assert results[0]["cluster"] == query_cluster
        # y con score alto, porque los histogramas quedan casi calcados
        assert results[0]["score"] > 0.9


def test_persistence_roundtrip(index, tmp_path):
    # guardar y volver a cargar el codebook no debe cambiar los centroides
    p = tmp_path / "cb"
    index.codebook.save(p)
    reloaded = KMeansCodebook.load(p)
    assert np.allclose(reloaded.centroids, index.codebook.centroids)
