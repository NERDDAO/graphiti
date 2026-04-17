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
import os
from typing import Any

from graphiti_core.driver.driver import GraphProvider
from graphiti_core.driver.operations.graph_ops import GraphMaintenanceOperations
from graphiti_core.driver.operations.graph_utils import (
    Neighbor,
    label_propagation,
    leiden_cluster,
)
from graphiti_core.driver.query_executor import QueryExecutor
from graphiti_core.driver.record_parsers import community_node_from_record, entity_node_from_record
from graphiti_core.graph_queries import get_fulltext_indices, get_range_indices
from graphiti_core.helpers import semaphore_gather
from graphiti_core.models.nodes.node_db_queries import (
    COMMUNITY_NODE_RETURN,
    get_entity_node_return_query,
)
from graphiti_core.nodes import CommunityNode, EntityNode, EpisodicNode

logger = logging.getLogger(__name__)

# Algorithm selection + tuning via env. Default: Leiden via leidenalg.
# Set GRAPHITI_COMMUNITY_ALGO=lpa to force the legacy label-propagation path.
_COMMUNITY_ALGO = os.getenv('GRAPHITI_COMMUNITY_ALGO', 'leiden').lower()
_LEIDEN_MIN_SIZE = int(os.getenv('GRAPHITI_LEIDEN_MIN_COMMUNITY_SIZE', '5'))
_LEIDEN_RESOLUTION = float(os.getenv('GRAPHITI_LEIDEN_RESOLUTION', '1.0'))
_LEIDEN_SEED = int(os.getenv('GRAPHITI_LEIDEN_SEED', '42'))


class Neo4jGraphMaintenanceOperations(GraphMaintenanceOperations):
    async def clear_data(
        self,
        executor: QueryExecutor,
        group_ids: list[str] | None = None,
    ) -> None:
        if group_ids is None:
            await executor.execute_query('MATCH (n) DETACH DELETE n')
        else:
            for label in ['Entity', 'Episodic', 'Community']:
                await executor.execute_query(
                    f"""
                    MATCH (n:{label})
                    WHERE n.group_id IN $group_ids
                    DETACH DELETE n
                    """,
                    group_ids=group_ids,
                )

    async def build_indices_and_constraints(
        self,
        executor: QueryExecutor,
        delete_existing: bool = False,
    ) -> None:
        if delete_existing:
            await self.delete_all_indexes(executor)

        range_indices = get_range_indices(GraphProvider.NEO4J)
        fulltext_indices = get_fulltext_indices(GraphProvider.NEO4J)
        index_queries = range_indices + fulltext_indices

        await semaphore_gather(*[executor.execute_query(q) for q in index_queries])

    async def delete_all_indexes(
        self,
        executor: QueryExecutor,
    ) -> None:
        await executor.execute_query('CALL db.indexes() YIELD name DROP INDEX name')

    async def get_community_clusters(
        self,
        executor: QueryExecutor,
        group_ids: list[str] | None = None,
    ) -> list[Any]:
        community_clusters: list[list[EntityNode]] = []

        if group_ids is None:
            group_id_values, _, _ = await executor.execute_query(
                """
                MATCH (n:Entity)
                WHERE n.group_id IS NOT NULL
                RETURN
                    collect(DISTINCT n.group_id) AS group_ids
                """
            )
            group_ids = group_id_values[0]['group_ids'] if group_id_values else []

        resolved_group_ids: list[str] = group_ids or []
        for group_id in resolved_group_ids:
            # Single-query batch projection: fetch every (src_uuid, tgt_uuid,
            # edge_count) triple in one round trip. The prior N+1 implementation
            # issued one Cypher per entity — ~19k RTTs for a medium bonfire.
            edge_records, _, _ = await executor.execute_query(
                """
                MATCH (n:Entity {group_id: $group_id})-[e:RELATES_TO]-(m:Entity {group_id: $group_id})
                RETURN n.uuid AS src, m.uuid AS tgt, count(e) AS edge_count
                """,
                group_id=group_id,
                routing_='r',
            )

            # Seed projection with every entity so isolated nodes appear too
            # (matters for the caller, which needs to see zero-neighbor nodes
            # even though Leiden/LPA will drop them).
            entity_records, _, _ = await executor.execute_query(
                """
                MATCH (n:Entity {group_id: $group_id})
                RETURN n.uuid AS uuid
                """,
                group_id=group_id,
                routing_='r',
            )
            projection: dict[str, list[Neighbor]] = {
                r['uuid']: [] for r in entity_records
            }
            for rec in edge_records:
                src, tgt, count = rec['src'], rec['tgt'], rec['edge_count']
                if src in projection:
                    projection[src].append(Neighbor(node_uuid=tgt, edge_count=count))

            cluster_uuids = self._cluster(projection)

            # Fetch full node objects for each cluster
            for cluster in cluster_uuids:
                if not cluster:
                    continue
                cluster_records, _, _ = await executor.execute_query(
                    """
                    MATCH (n:Entity)
                    WHERE n.uuid IN $uuids
                    RETURN
                    """
                    + get_entity_node_return_query(GraphProvider.NEO4J),
                    uuids=cluster,
                    routing_='r',
                )
                community_clusters.append([entity_node_from_record(r) for r in cluster_records])

        return community_clusters

    @staticmethod
    def _cluster(projection: dict[str, list[Neighbor]]) -> list[list[str]]:
        """Run the configured community algorithm, fall back to LPA on error.

        Default is Leiden (``GRAPHITI_COMMUNITY_ALGO=leiden``); set the env to
        ``lpa`` to force label-propagation (or when leidenalg / igraph are not
        installed, which raises ImportError and auto-falls-back with a warning).
        """
        if _COMMUNITY_ALGO == 'lpa':
            return label_propagation(projection)
        try:
            return leiden_cluster(
                projection,
                min_community_size=_LEIDEN_MIN_SIZE,
                resolution=_LEIDEN_RESOLUTION,
                seed=_LEIDEN_SEED,
            )
        except ImportError as e:
            logger.warning(
                'Leiden requested but leidenalg/igraph unavailable (%s); '
                'falling back to label-propagation. Install with `pip install '
                'leidenalg python-igraph` or set GRAPHITI_COMMUNITY_ALGO=lpa '
                'to silence this warning.',
                e,
            )
            return label_propagation(projection)

    async def remove_communities(
        self,
        executor: QueryExecutor,
    ) -> None:
        await executor.execute_query(
            """
            MATCH (c:Community)
            DETACH DELETE c
            """
        )

    async def determine_entity_community(
        self,
        executor: QueryExecutor,
        entity: EntityNode,
    ) -> None:
        # Check if the node is already part of a community
        records, _, _ = await executor.execute_query(
            """
            MATCH (c:Community)-[:HAS_MEMBER]->(n:Entity {uuid: $entity_uuid})
            RETURN
            """
            + COMMUNITY_NODE_RETURN,
            entity_uuid=entity.uuid,
        )

        if len(records) > 0:
            return

        # If the node has no community, find the mode community of surrounding entities
        records, _, _ = await executor.execute_query(
            """
            MATCH (c:Community)-[:HAS_MEMBER]->(m:Entity)-[:RELATES_TO]-(n:Entity {uuid: $entity_uuid})
            RETURN
            """
            + COMMUNITY_NODE_RETURN,
            entity_uuid=entity.uuid,
        )

    async def get_mentioned_nodes(
        self,
        executor: QueryExecutor,
        episodes: list[EpisodicNode],
    ) -> list[EntityNode]:
        episode_uuids = [episode.uuid for episode in episodes]

        records, _, _ = await executor.execute_query(
            """
            MATCH (episode:Episodic)-[:MENTIONS]->(n:Entity)
            WHERE episode.uuid IN $uuids
            RETURN DISTINCT
            """
            + get_entity_node_return_query(GraphProvider.NEO4J),
            uuids=episode_uuids,
            routing_='r',
        )

        return [entity_node_from_record(r) for r in records]

    async def get_communities_by_nodes(
        self,
        executor: QueryExecutor,
        nodes: list[EntityNode],
    ) -> list[CommunityNode]:
        node_uuids = [node.uuid for node in nodes]

        records, _, _ = await executor.execute_query(
            """
            MATCH (c:Community)-[:HAS_MEMBER]->(m:Entity)
            WHERE m.uuid IN $uuids
            RETURN DISTINCT
            """
            + COMMUNITY_NODE_RETURN,
            uuids=node_uuids,
            routing_='r',
        )

        return [community_node_from_record(r) for r in records]
