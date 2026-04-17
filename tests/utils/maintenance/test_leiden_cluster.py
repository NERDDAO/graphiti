"""Tests for the Leiden community-detection helper.

Covers the ``leiden_cluster`` function in isolation (no Neo4j, no driver).
The Neo4j driver's ``get_community_clusters`` is tested separately via
integration tests that require a live database.
"""

from __future__ import annotations

import pytest

from graphiti_core.driver.operations.graph_utils import (
    Neighbor,
    label_propagation,
    leiden_cluster,
)


def _make_projection(edges: list[tuple[str, str, int]], nodes: list[str] | None = None):
    """Build a symmetric projection from an undirected edge list.

    Matches the shape ``Neo4jGraphMaintenanceOperations.get_community_clusters``
    passes in (each edge appears in both endpoints' neighbor lists).
    """
    all_nodes: set[str] = set(nodes or [])
    for a, b, _ in edges:
        all_nodes.update((a, b))
    projection: dict[str, list[Neighbor]] = {n: [] for n in all_nodes}
    for a, b, w in edges:
        projection[a].append(Neighbor(node_uuid=b, edge_count=w))
        projection[b].append(Neighbor(node_uuid=a, edge_count=w))
    return projection


class TestLeidenCluster:
    def test_empty_projection_returns_empty(self) -> None:
        assert leiden_cluster({}) == []

    def test_all_singletons_dropped_by_min_size(self) -> None:
        # Five disconnected nodes — default min_community_size=5 drops all
        projection = _make_projection([], nodes=[f'n{i}' for i in range(5)])
        assert leiden_cluster(projection, min_community_size=5) == []

    def test_singletons_kept_when_min_size_is_one(self) -> None:
        projection = _make_projection([], nodes=[f'n{i}' for i in range(3)])
        clusters = leiden_cluster(projection, min_community_size=1)
        # Each isolated node becomes its own cluster of size 1
        assert sorted(len(c) for c in clusters) == [1, 1, 1]

    def test_two_dense_cliques_are_separated(self) -> None:
        # Two 5-cliques connected by a single weak edge should split into
        # two communities under Leiden (LPA also handles this, but Leiden
        # is strictly stricter about community boundaries).
        a_nodes = [f'a{i}' for i in range(5)]
        b_nodes = [f'b{i}' for i in range(5)]
        edges: list[tuple[str, str, int]] = []
        # Clique A (all pairs, weight 10)
        for i, n1 in enumerate(a_nodes):
            for n2 in a_nodes[i + 1:]:
                edges.append((n1, n2, 10))
        # Clique B (all pairs, weight 10)
        for i, n1 in enumerate(b_nodes):
            for n2 in b_nodes[i + 1:]:
                edges.append((n1, n2, 10))
        # Single weak bridge
        edges.append((a_nodes[0], b_nodes[0], 1))

        clusters = leiden_cluster(_make_projection(edges), min_community_size=5, seed=42)
        assert len(clusters) == 2
        as_sets = [set(c) for c in clusters]
        # Each clique should land in its own community
        assert set(a_nodes) in as_sets
        assert set(b_nodes) in as_sets

    def test_deterministic_with_same_seed(self) -> None:
        a_nodes = [f'a{i}' for i in range(6)]
        b_nodes = [f'b{i}' for i in range(6)]
        edges: list[tuple[str, str, int]] = []
        for i, n1 in enumerate(a_nodes):
            for n2 in a_nodes[i + 1:]:
                edges.append((n1, n2, 5))
        for i, n1 in enumerate(b_nodes):
            for n2 in b_nodes[i + 1:]:
                edges.append((n1, n2, 5))
        edges.append((a_nodes[0], b_nodes[0], 1))
        proj = _make_projection(edges)

        r1 = [sorted(c) for c in leiden_cluster(proj, seed=42)]
        r2 = [sorted(c) for c in leiden_cluster(proj, seed=42)]
        assert sorted(r1) == sorted(r2)

    def test_min_community_size_filters_small_clusters(self) -> None:
        # Mixed sizes: one 6-clique, one 3-clique, two isolates.
        # min_size=5 should keep only the 6-clique.
        big = [f'big{i}' for i in range(6)]
        small = [f'sm{i}' for i in range(3)]
        edges: list[tuple[str, str, int]] = []
        for i, n1 in enumerate(big):
            for n2 in big[i + 1:]:
                edges.append((n1, n2, 5))
        for i, n1 in enumerate(small):
            for n2 in small[i + 1:]:
                edges.append((n1, n2, 5))
        proj = _make_projection(edges, nodes=big + small + ['iso1', 'iso2'])

        clusters = leiden_cluster(proj, min_community_size=5)
        assert len(clusters) == 1
        assert set(clusters[0]) == set(big)

    def test_returns_same_type_as_label_propagation(self) -> None:
        """Contract check — downstream callers must not care which algo ran."""
        a = [f'a{i}' for i in range(5)]
        edges = [(a[i], a[j], 3) for i in range(5) for j in range(i + 1, 5)]
        proj = _make_projection(edges)
        leiden_result = leiden_cluster(proj, min_community_size=1)
        lpa_result = label_propagation(proj)
        # Both return list[list[str]]
        assert isinstance(leiden_result, list)
        assert isinstance(lpa_result, list)
        assert all(isinstance(c, list) and all(isinstance(u, str) for u in c) for c in leiden_result)
        assert all(isinstance(c, list) and all(isinstance(u, str) for u in c) for c in lpa_result)


class TestLeidenClusterImportError:
    """Regression: driver code relies on ImportError to fall back to LPA.

    If ``leidenalg`` or ``python-igraph`` go missing at runtime (e.g. a minimal
    Docker image), we must raise ImportError — not AttributeError — so the
    Neo4j driver's ``_cluster`` try/except can catch it and call LPA instead.
    """

    def test_raises_import_error_when_libs_missing(self, monkeypatch: pytest.MonkeyPatch) -> None:
        import builtins

        real_import = builtins.__import__

        def _fake_import(name: str, *args, **kwargs):
            if name in ('igraph', 'leidenalg'):
                raise ImportError(f'mocked missing {name}')
            return real_import(name, *args, **kwargs)

        monkeypatch.setattr(builtins, '__import__', _fake_import)

        with pytest.raises(ImportError, match='leidenalg'):
            leiden_cluster({'x': []})
