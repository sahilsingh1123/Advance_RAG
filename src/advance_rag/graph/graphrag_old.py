"""GraphRAG implementation for clinical data reasoning."""

import json
import re
from typing import Any, Dict, List, Optional, Tuple

import networkx as nx
from sklearn.cluster import KMeans

from advance_rag.core.config import get_settings
from advance_rag.core.logging import get_logger
from advance_rag.graph.neo4j_service import Neo4jService
from advance_rag.models import Document, GraphCommunity, GraphEntity, GraphRelation

logger = get_logger(__name__)
settings = get_settings()


class EntityExtractor:
    """Extract entities and relations from clinical text."""

    def __init__(self):
        """Initialize entity extractor."""
        # Clinical entity patterns
        self.entity_patterns = {
            "subject": [
                r"subject\s+(\w+)",
                r"patient\s+(\w+)",
                r"participant\s+(\w+)",
                r'usubjid\s*=\s*[\'"]?(\w+)[\'"]?',
            ],
            "treatment": [
                r"treatment\s+(\w+)",
                r"drug\s+(\w+)",
                r"medication\s+(\w+)",
                r'trtd[a-z]*\s*=\s*[\'"]?(\w+)[\'"]?',
            ],
            "lab_test": [
                r"lab\s+(\w+)",
                r"test\s+(\w+)",
                r'lb[a-z]*\s*=\s*[\'"]?(\w+)[\'"]?',
                r"parameter\s+(\w+)",
            ],
            "adverse_event": [
                r"adverse\s+event\s+(\w+)",
                r'ae[a-z]*\s*=\s*[\'"]?(\w+)[\'"]?',
                r"event\s+(\w+)",
            ],
            "medication": [
                r"medication\s+(\w+)",
                r'cm[a-z]*\s*=\s*[\'"]?(\w+)[\'"]?',
                r"concomitant\s+(\w+)",
            ],
            "study": [
                r"study\s+(\w+)",
                r"protocol\s+(\w+)",
                r"trial\s+(\w+)",
            ],
            "domain": [
                r"domain\s+(\w+)",
                r"sdtm\s+(\w+)",
                r"adam\s+(\w+)",
            ],
        }

        # Relation patterns
        self.relation_patterns = [
            (r"(\w+)\s+(?:received|took|was\s+given)\s+(\w+)", "received"),
            (r"(\w+)\s+(?:had|experienced|reported)\s+(\w+)", "experienced"),
            (r"(\w+)\s+(?:showed|demonstrated)\s+(\w+)", "showed"),
            (r"(\w+)\s+(?:treated\s+with|managed\s+with)\s+(\w+)", "treated_with"),
            (r"(\w+)\s+(?:associated\s+with|linked\s+to)\s+(\w+)", "associated_with"),
            (r"(\w+)\s+(?:derived\s+from|based\s+on)\s+(\w+)", "derived_from"),
        ]

    def extract_entities(self, text: str) -> List[GraphEntity]:
        """Extract entities from text."""
        entities = []
        entity_counter = 0

        # Extract entities using patterns
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE)
                for match in matches:
                    entity_name = match.group(1).upper()

                    # Create entity
                    entity = GraphEntity(
                        id=f"entity_{entity_counter}",
                        type=entity_type,
                        name=entity_name,
                        properties={
                            "source": "pattern_extraction",
                            "context": text[
                                max(0, match.start() - 50) : match.end() + 50
                            ],
                        },
                    )
                    entities.append(entity)
                    entity_counter += 1

        # Extract SDTM/ADaM specific entities
        sdtm_domains = [
            "DM",
            "AE",
            "CM",
            "DA",
            "DS",
            "DV",
            "EX",
            "FA",
            "IE",
            "IN",
            "MH",
            "MO",
            "PE",
            "PR",
            "QS",
            "SC",
            "SU",
            "TA",
            "TE",
            "TI",
            "TR",
            "TV",
            "VS",
        ]

        adam_datasets = [
            "ADSL",
            "ADAE",
            "ADCM",
            "ADDA",
            "ADDS",
            "ADLB",
            "ADMB",
            "ADMH",
            "ADPC",
            "ADPE",
            "ADRG",
            "ADRS",
            "ADTE",
            "ADTI",
            "ADTR",
            "ADVS",
            "ADYD",
        ]

        # Check for SDTM domains
        for domain in sdtm_domains:
            if domain in text:
                entity = GraphEntity(
                    id=f"entity_{entity_counter}",
                    type="sdtm_domain",
                    name=domain,
                    properties={"source": "sdtm_domain"},
                )
                entities.append(entity)
                entity_counter += 1

        # Check for ADaM datasets
        for dataset in adam_datasets:
            if dataset in text:
                entity = GraphEntity(
                    id=f"entity_{entity_counter}",
                    type="adam_dataset",
                    name=dataset,
                    properties={"source": "adam_dataset"},
                )
                entities.append(entity)
                entity_counter += 1

        return entities

    def extract_relations(
        self, text: str, entities: List[GraphEntity]
    ) -> List[GraphRelation]:
        """Extract relations between entities."""
        relations = []
        relation_counter = 0

        # Create entity lookup
        entity_lookup = {e.name.lower(): e for e in entities}

        # Extract relations using patterns
        for pattern, relation_type in self.relation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE)
            for match in matches:
                source_name = match.group(1).lower()
                target_name = match.group(2).lower()

                # Find matching entities
                source_entity = entity_lookup.get(source_name)
                target_entity = entity_lookup.get(target_name)

                if source_entity and target_entity:
                    relation = GraphRelation(
                        id=f"relation_{relation_counter}",
                        source_id=source_entity.id,
                        target_id=target_entity.id,
                        type=relation_type,
                        properties={
                            "source": "pattern_extraction",
                            "context": text[
                                max(0, match.start() - 50) : match.end() + 50
                            ],
                        },
                    )
                    relations.append(relation)
                    relation_counter += 1

        return relations


class CommunityDetector:
    """Detect communities in the knowledge graph."""

    def __init__(self):
        """Initialize community detector."""
        self.algorithm = settings.GRAPH_COMMUNITY_ALGORITHM
        self.resolution = settings.GRAPH_COMMUNITY_RESOLUTION

    def detect_communities_networkx(
        self, entities: List[GraphEntity], relations: List[GraphRelation]
    ) -> List[GraphCommunity]:
        """Detect communities using NetworkX."""
        # Create NetworkX graph
        G = nx.Graph()

        # Add nodes
        for entity in entities:
            G.add_node(entity.id, type=entity.type, name=entity.name)

        # Add edges
        for relation in relations:
            G.add_edge(relation.source_id, relation.target_id, type=relation.type)

        # Detect communities
        if self.algorithm == "louvain":
            try:
                import community as community_louvain

                partition = community_louvain.best_partition(
                    G, resolution=self.resolution
                )
            except ImportError:
                logger.warning("python-louvain not installed, using label propagation")
                communities = nx.algorithms.community.label_propagation_communities(G)
                partition = {}
                for i, community in enumerate(communities):
                    for node in community:
                        partition[node] = i
        else:
            # Use Leiden algorithm (simplified as KMeans on node embeddings)
            communities = nx.algorithms.community.label_propagation_communities(G)
            partition = {}
            for i, community in enumerate(communities):
                for node in community:
                    partition[node] = i

        # Group entities by community
        community_groups = {}
        for node_id, community_id in partition.items():
            if community_id not in community_groups:
                community_groups[community_id] = []
            community_groups[community_id].append(node_id)

        # Create community objects
        graph_communities = []
        for community_id, entity_ids in community_groups.items():
            community = GraphCommunity(
                id=f"community_{community_id}",
                entities=entity_ids,
                summary="",  # Will be generated later
                level=0,
            )
            graph_communities.append(community)

        return graph_communities


class GraphRAGService:
    """GraphRAG service for knowledge graph-based retrieval."""

    def __init__(self, neo4j_service: Neo4jService):
        """Initialize GraphRAG service."""
        self.neo4j = neo4j_service
        self.entity_extractor = EntityExtractor()
        self.community_detector = CommunityDetector()
        logger.info("Initialized GraphRAG service")

    async def extract_and_store_entities(self, document: Document) -> None:
        """Extract and store entities from document."""
        # Extract entities and relations
        entities = self.entity_extractor.extract_entities(document.content)
        relations = self.entity_extractor.extract_relations(document.content, entities)

        # Store in Neo4j
        for entity in entities:
            await self.neo4j.create_entity(entity)

        for relation in relations:
            await self.neo4j.create_relation(relation)

        logger.info(
            f"Extracted {len(entities)} entities and {len(relations)} relations"
        )

    async def detect_and_store_communities(self) -> None:
        """Detect and store communities in the graph."""
        # Run community detection
        communities = await self.neo4j.run_community_detection()

        # Generate summaries for communities
        for community_id, entity_ids in communities.items():
            summary = await self._generate_community_summary(entity_ids)

            community = GraphCommunity(
                id=f"community_{community_id}",
                entities=entity_ids,
                summary=summary,
                level=0,
            )

            await self.neo4j.create_community(community)

        logger.info(f"Detected and stored {len(communities)} communities")

    async def _generate_community_summary(self, entity_ids: List[str]) -> str:
        """Generate summary for a community."""
        # Get entities in the community
        entities = await self.neo4j.find_entities(limit=1000)
        community_entities = [e for e in entities if e.id in entity_ids]

        if not community_entities:
            return "Empty community"

        # Group entities by type
        entity_groups = {}
        for entity in community_entities:
            if entity.type not in entity_groups:
                entity_groups[entity.type] = []
            entity_groups[entity.type].append(entity.name)

        # Generate summary
        summary_parts = []
        for entity_type, names in entity_groups.items():
            if len(names) <= 3:
                summary_parts.append(f"{entity_type}: {', '.join(names)}")
            else:
                summary_parts.append(f"{entity_type}: {len(names)} items")

        return " | ".join(summary_parts)

    async def global_search(self, query: str, top_k: int = 5) -> List[str]:
        """Perform global search using community summaries."""
        # Find relevant communities
        communities = await self.neo4j.find_communities(limit=50)

        # Simple relevance scoring based on keyword matching
        relevant_communities = []
        query_words = set(query.lower().split())

        for community in communities:
            summary_words = set(community.summary.lower().split())
            overlap = len(query_words & summary_words)

            if overlap > 0:
                relevant_communities.append((community, overlap))

        # Sort by relevance
        relevant_communities.sort(key=lambda x: x[1], reverse=True)

        # Return top community summaries
        return [c[0].summary for c in relevant_communities[:top_k]]

    async def local_search(
        self, query: str, entity_name: str, max_hops: int = settings.GRAPH_MAX_HOPS
    ) -> List[str]:
        """Perform local search around an entity."""
        # Find the entity
        entities = await self.neo4j.find_entities(name_pattern=entity_name, limit=1)

        if not entities:
            return []

        entity = entities[0]

        # Find related entities
        triples = await self.neo4j.find_related_entities(entity.id, max_hops=max_hops)

        # Extract relevant information
        context_parts = []
        context_parts.append(f"Entity: {entity.name} ({entity.type})")

        for source_entity, relation, target_entity in triples[:10]:
            context_parts.append(
                f"{source_entity.name} {relation.type} {target_entity.name}"
            )

        return context_parts

    async def drift_search(
        self, query: str, entity_name: str, top_k: int = 5
    ) -> List[str]:
        """Perform DRIFT search (entity + community context)."""
        # Get local context
        local_context = await self.local_search(query, entity_name)

        # Get global context
        global_context = await self.global_search(query, top_k)

        # Combine contexts
        combined_context = local_context + global_context

        return combined_context[: top_k * 2]  # Limit results

    async def get_graph_context(self, query: str) -> Dict[str, Any]:
        """Get graph context for a query."""
        context = {"entities": [], "relations": [], "communities": [], "paths": []}

        # Extract entities from query
        query_entities = self.entity_extractor.extract_entities(query)

        # Find matching entities in graph
        for query_entity in query_entities:
            entities = await self.neo4j.find_entities(
                entity_type=query_entity.type, name_pattern=query_entity.name, limit=5
            )
            context["entities"].extend([e.name for e in entities])

            # Get relations for each entity
            for entity in entities:
                triples = await self.neo4j.find_related_entities(entity.id, max_hops=2)
                context["relations"].extend(
                    [f"{s.name} {r.type} {t.name}" for s, r, t in triples[:5]]
                )

        # Get relevant communities
        community_summaries = await self.global_search(query, top_k=3)
        context["communities"] = community_summaries

        return context
