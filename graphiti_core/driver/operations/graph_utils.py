"""
Copyright 2024, Zep Software, Inc.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import logging
from collections import defaultdict

from pydantic import BaseModel

logger = logging.getLogger(__name__)


class Neighbor(BaseModel):
    node_uuid: str
    edge_count: int


def label_propagation(projection: dict[str, list[Neighbor]]) -> list[list[str]]:
    community_map = {uuid: i for i, uuid in enumerate(projection.keys())}

    while True:
        no_change = True
        new_community_map: dict[str, int] = {}

        for uuid, neighbors in projection.items():
            curr_community = community_map[uuid]

            community_candidates: dict[int, int] = defaultdict(int)
            for neighbor in neighbors:
                community_candidates[community_map[neighbor.node_uuid]] += neighbor.edge_count
            community_lst = [
                (count, community) for community, count in community_candidates.items()
            ]

            community_lst.sort(reverse=True)
            candidate_rank, community_candidate = community_lst[0] if community_lst else (0, -1)
            if community_candidate != -1 and candidate_rank > 1:
                new_community = community_candidate
            else:
                new_community = max(community_candidate, curr_community)

            new_community_map[uuid] = new_community

            if new_community != curr_community:
                no_change = False

        if no_change:
            break

        community_map = new_community_map

    community_cluster_map: dict[int, list[str]] = defaultdict(list)
    for uuid, community in community_map.items():
        community_cluster_map[community].append(uuid)

    return list(community_cluster_map.values())


def leiden_cluster(
    projection: dict[str, list[Neighbor]],
    min_community_size: int = 5,
    resolution: float = 1.0,
    seed: int = 42,
) -> list[list[str]]:
    """Cluster nodes via the Leiden algorithm (Traag et al. 2019).

    Higher-quality alternative to label propagation: tighter modularity, no
    "mega-dump" failure mode on scale-free graphs with hub entities, and
    deterministic given ``seed``. Runs in-process via the ``leidenalg`` +
    ``python-igraph`` libraries.

    Args:
        projection: Same format as ``label_propagation`` — uuid →
            list of ``Neighbor(node_uuid, edge_count)``. Expected to be
            symmetric (undirected); each edge appears once in each
            endpoint's neighbor list.
        min_community_size: Communities with fewer members are dropped
            from the result. Default 5 — below this, an LLM community
            summary is token-waste.
        resolution: Leiden resolution parameter (``gamma``). 1.0 is
            balanced, <1.0 = fewer/larger, >1.0 = more/smaller.
        seed: Random seed for deterministic output.

    Returns:
        List of clusters, each a list of node uuids. Disconnected nodes
        and under-size clusters are omitted (keeps the downstream
        ``build_community`` LLM summary step focused on meaningful groups).

    Raises:
        ImportError: leidenalg / python-igraph not installed. Caller
            should fall back to ``label_propagation``.
    """
    try:
        import igraph as ig
        import leidenalg as la
    except ImportError as e:
        raise ImportError(
            "leidenalg + python-igraph required for leiden_cluster. "
            "Install via: pip install 'leidenalg>=0.10.0' 'python-igraph>=0.11.0'"
        ) from e

    uuids = list(projection.keys())
    if not uuids:
        return []

    name_to_idx = {u: i for i, u in enumerate(uuids)}

    # Dedupe edges (projection is symmetric — each edge appears twice).
    # Keep the first-seen edge_count for weight; graphiti's projection
    # stores the same count in both directions, so this is lossless.
    seen: set[tuple[int, int]] = set()
    edge_list: list[tuple[int, int]] = []
    weights: list[int] = []
    for src_uuid, neighbors in projection.items():
        src_idx = name_to_idx[src_uuid]
        for nb in neighbors:
            tgt_idx = name_to_idx.get(nb.node_uuid)
            if tgt_idx is None:
                continue  # neighbor outside the projection; skip defensively
            a, b = (src_idx, tgt_idx) if src_idx < tgt_idx else (tgt_idx, src_idx)
            if a == b or (a, b) in seen:
                continue
            seen.add((a, b))
            edge_list.append((a, b))
            weights.append(nb.edge_count)

    graph = ig.Graph(n=len(uuids), directed=False)
    if edge_list:
        graph.add_edges(edge_list)
        graph.es['weight'] = weights

    partition = la.find_partition(
        graph,
        la.RBConfigurationVertexPartition,
        weights='weight' if edge_list else None,
        resolution_parameter=resolution,
        seed=seed,
    )

    clusters: list[list[str]] = []
    for member_indices in partition:
        if len(member_indices) < min_community_size:
            continue
        clusters.append([uuids[i] for i in member_indices])
    return clusters
