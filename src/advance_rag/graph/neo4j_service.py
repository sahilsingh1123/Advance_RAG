"""Neo4j graph database service for GraphRAG."""

import json
from typing import Any, Dict, List, Optional, Tuple

import neo4j
from neo4j import GraphDatabase

from advance_rag.core.config import get_settings
from advance_rag.core.logging import get_logger, log_query
from advance_rag.models import Document, GraphCommunity, GraphEntity, GraphRelation

logger = get_logger(__name__)
settings = get_settings()


class Neo4jService:
    """Neo4j service for graph operations."""

    def __init__(
        self,
        uri: str = settings.NEO4J_URI,
        user: str = settings.NEO4J_USER,
        password: str = settings.NEO4J_PASSWORD,
    ):
        """Initialize Neo4j service."""
        self.driver = GraphDatabase.driver(uri, auth=(user, password))
        logger.info("Initialized Neo4j service")

    async def close(self):
        """Close Neo4j driver."""
        if self.driver:
            await self.driver.close()
            logger.info("Closed Neo4j connection")

    def _create_session(self):
        """Create Neo4j session."""
        return self.driver.session(database="neo4j")

    async def create_constraints(self):
        """Create database constraints."""
        constraints = [
            "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT relation_id_unique IF NOT EXISTS FOR ()-[r:RELATION]-() REQUIRE r.id IS UNIQUE",
            "CREATE CONSTRAINT community_id_unique IF NOT EXISTS FOR (c:Community) REQUIRE c.id IS UNIQUE",
            "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",
            "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
            "CREATE INDEX relation_type_idx IF NOT EXISTS FOR ()-[r:RELATION]-() ON (r.type)",
            "CREATE INDEX community_level_idx IF NOT EXISTS FOR (c:Community) ON (c.level)",
        ]

        with self._create_session() as session:
            for constraint in constraints:
                try:
                    session.run(constraint)
                    logger.info(f"Created constraint/index: {constraint}")
                except Exception as e:
                    logger.warning(f"Failed to create constraint: {e}")

    async def create_entity(self, entity: GraphEntity) -> bool:
        """Create a graph entity."""
        with self._create_session() as session:
            try:
                result = session.run(
                    """
                    MERGE (e:Entity {id: $id})
                    SET e.type = $type,
                        e.name = $name,
                        e.properties = $properties,
                        e.created_at = datetime()
                    RETURN e
                """,
                    {
                        "id": entity.id,
                        "type": entity.type,
                        "name": entity.name,
                        "properties": json.dumps(entity.properties),
                    },
                )

                return result.single() is not None
            except Exception as e:
                logger.error(f"Failed to create entity {entity.id}: {e}")
                return False

    async def create_relation(self, relation: GraphRelation) -> bool:
        """Create a graph relation."""
        with self._create_session() as session:
            try:
                result = session.run(
                    """
                    MATCH (source:Entity {id: $source_id})
                    MATCH (target:Entity {id: $target_id})
                    MERGE (source)-[r:RELATION {id: $id}]->(target)
                    SET r.type = $type,
                        r.properties = $properties,
                        r.created_at = datetime()
                    RETURN r
                """,
                    {
                        "id": relation.id,
                        "source_id": relation.source_id,
                        "target_id": relation.target_id,
                        "type": relation.type,
                        "properties": json.dumps(relation.properties),
                    },
                )

                return result.single() is not None
            except Exception as e:
                logger.error(f"Failed to create relation {relation.id}: {e}")
                return False

    async def find_entities(
        self,
        entity_type: Optional[str] = None,
        name_pattern: Optional[str] = None,
        limit: int = 100,
    ) -> List[GraphEntity]:
        """Find entities matching criteria."""
        with self._create_session() as session:
            query = "MATCH (e:Entity)"
            params = {}

            conditions = []
            if entity_type:
                conditions.append("e.type = $entity_type")
                params["entity_type"] = entity_type

            if name_pattern:
                conditions.append("e.name =~ $name_pattern")
                params["name_pattern"] = f"(?i).*{name_pattern}.*"

            if conditions:
                query += " WHERE " + " AND ".join(conditions)

            query += " RETURN e ORDER BY e.name LIMIT $limit"
            params["limit"] = limit

            result = session.run(query, params)

            entities = []
            for record in result:
                node = record["e"]
                entities.append(
                    GraphEntity(
                        id=node["id"],
                        type=node["type"],
                        name=node["name"],
                        properties=json.loads(node.get("properties", "{}")),
                    )
                )

            return entities

    async def find_related_entities(
        self,
        entity_id: str,
        relation_type: Optional[str] = None,
        direction: str = "both",  # "in", "out", or "both"
        max_hops: int = settings.GRAPH_MAX_HOPS,
    ) -> List[Tuple[GraphEntity, GraphRelation, GraphEntity]]:
        """Find entities related to the given entity."""
        with self._create_session() as session:
            if direction == "out":
                pattern = f"(source)-[r:RELATION*1..{max_hops}]->(target)"
            elif direction == "in":
                pattern = f"(source)<-[r:RELATION*1..{max_hops}]-(target)"
            else:  # both
                pattern = f"(source)-[r:RELATION*1..{max_hops}]-(target)"

            query = f"""
                MATCH (source:Entity {{id: $entity_id}})
                MATCH {pattern}
                WHERE source <> target
            """

            params = {"entity_id": entity_id}

            if relation_type:
                query += " AND ALL(rel in r WHERE rel.type = $relation_type)"
                params["relation_type"] = relation_type

            query += """
                RETURN source, r, target
                ORDER BY length(r)
                LIMIT 100
            """

            result = session.run(query, params)

            triples = []
            for record in result:
                source_node = record["source"]
                target_node = record["target"]
                relations = record["r"]

                # Create entities
                source_entity = GraphEntity(
                    id=source_node["id"],
                    type=source_node["type"],
                    name=source_node["name"],
                    properties=json.loads(source_node.get("properties", "{}")),
                )

                target_entity = GraphEntity(
                    id=target_node["id"],
                    type=target_node["type"],
                    name=target_node["name"],
                    properties=json.loads(target_node.get("properties", "{}")),
                )

                # For each relation in the path
                for rel in relations:
                    relation = GraphRelation(
                        id=rel["id"],
                        source_id=rel.start_node["id"],
                        target_id=rel.end_node["id"],
                        type=rel["type"],
                        properties=json.loads(rel.get("properties", "{}")),
                    )
                    triples.append((source_entity, relation, target_entity))

            return triples

    async def create_community(self, community: GraphCommunity) -> bool:
        """Create a graph community."""
        with self._create_session() as session:
            try:
                # Create community node
                result = session.run(
                    """
                    CREATE (c:Community {
                        id: $id,
                        summary: $summary,
                        level: $level,
                        created_at: datetime()
                    })
                    RETURN c
                """,
                    {
                        "id": community.id,
                        "summary": community.summary,
                        "level": community.level,
                    },
                )

                # Connect community to entities
                if community.entities:
                    session.run(
                        """
                        UNWIND $entity_ids AS entity_id
                        MATCH (c:Community {id: $community_id})
                        MATCH (e:Entity {id: entity_id})
                        MERGE (e)-[:IN_COMMUNITY]->(c)
                    """,
                        {
                            "community_id": community.id,
                            "entity_ids": community.entities,
                        },
                    )

                return result.single() is not None
            except Exception as e:
                logger.error(f"Failed to create community {community.id}: {e}")
                return False

    async def find_communities(
        self, level: Optional[int] = None, limit: int = 50
    ) -> List[GraphCommunity]:
        """Find graph communities."""
        with self._create_session() as session:
            query = "MATCH (c:Community)"
            params = {}

            if level is not None:
                query += " WHERE c.level = $level"
                params["level"] = level

            query += """
                OPTIONAL MATCH (e:Entity)-[:IN_COMMUNITY]->(c)
                WITH c, COLLECT(e.id) as entities
                RETURN c.id, c.summary, c.level, entities
                ORDER BY c.level, c.id
                LIMIT $limit
            """
            params["limit"] = limit

            result = session.run(query, params)

            communities = []
            for record in result:
                communities.append(
                    GraphCommunity(
                        id=record["c.id"],
                        entities=record["entities"],
                        summary=record["c.summary"],
                        level=record["c.level"],
                    )
                )

            return communities

    async def get_graph_statistics(self) -> Dict[str, Any]:
        """Get graph statistics."""
        with self._create_session() as session:
            stats = {}

            # Node counts by type
            result = session.run(
                """
                MATCH (e:Entity)
                RETURN e.type as type, COUNT(*) as count
                ORDER BY count DESC
            """
            )
            stats["entity_types"] = {
                record["type"]: record["count"] for record in result
            }

            # Relation counts by type
            result = session.run(
                """
                MATCH ()-[r:RELATION]->()
                RETURN r.type as type, COUNT(*) as count
                ORDER BY count DESC
            """
            )
            stats["relation_types"] = {
                record["type"]: record["count"] for record in result
            }

            # Community statistics
            result = session.run(
                """
                MATCH (c:Community)
                RETURN c.level as level, COUNT(*) as count
                ORDER BY level
            """
            )
            stats["communities"] = {
                record["level"]: record["count"] for record in result
            }

            # Overall counts
            stats["total_entities"] = session.run(
                "MATCH (e:Entity) RETURN COUNT(e) as count"
            ).single()["count"]
            stats["total_relations"] = session.run(
                "MATCH ()-[r:RELATION]->() RETURN COUNT(r) as count"
            ).single()["count"]
            stats["total_communities"] = session.run(
                "MATCH (c:Community) RETURN COUNT(c) as count"
            ).single()["count"]

            return stats

    async def run_community_detection(
        self, resolution: float = settings.GRAPH_COMMUNITY_RESOLUTION
    ):
        """Run community detection algorithm."""
        import time

        start_time = time.time()

        with self._create_session() as session:
            # Use Neo4j Graph Data Science library
            try:
                # Run Louvain algorithm
                result = session.run(
                    """
                    CALL gds.louvain.stream({
                        nodeProjection: 'Entity',
                        relationshipProjection: {
                            RELATION: {
                                type: 'RELATION',
                                orientation: 'UNDIRECTED'
                            }
                        },
                        includeIntermediateCommunities: true,
                        seedProperty: 'seed',
                        randomSeed: 42
                    })
                    YIELD nodeId, communityId, intermediateCommunityIds
                    RETURN gds.util.asNode(nodeId).id as entity_id,
                           communityId as community,
                           intermediateCommunityIds as intermediate_communities
                """
                )

                # Group entities by community
                communities = {}
                for record in result:
                    community_id = str(record["community"])
                    entity_id = record["entity_id"]

                    if community_id not in communities:
                        communities[community_id] = []
                    communities[community_id].append(entity_id)

                # Create community nodes
                for community_id, entity_ids in communities.items():
                    community = GraphCommunity(
                        id=f"community_{community_id}",
                        entities=entity_ids,
                        summary="",  # Will be generated later
                        level=0,
                    )
                    await self.create_community(community)

                duration_ms = (time.time() - start_time) * 1000
                log_query(
                    query_type="community_detection",
                    query="gds.louvain",
                    duration_ms=duration_ms,
                    num_results=len(communities),
                )

                logger.info(f"Detected {len(communities)} communities")
                return communities

            except Exception as e:
                logger.error(f"Community detection failed: {e}")
                return {}
