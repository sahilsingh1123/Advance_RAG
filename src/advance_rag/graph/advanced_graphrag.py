"""Advanced GraphRAG implementation with automated graph construction."""

import re
import json
from typing import Any, Dict, List, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum
import asyncio

import networkx as nx
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer

from advance_rag.core.config import get_settings
from advance_rag.core.logging import get_logger
from advance_rag.graph.neo4j_service import Neo4jService
from advance_rag.models import Document, GraphCommunity, GraphEntity, GraphRelation

logger = get_logger(__name__)
settings = get_settings()


class EntityType(Enum):
    """Standard entity types for clinical data."""

    SUBJECT = "subject"
    TREATMENT = "treatment"
    LAB_TEST = "lab_test"
    ADVERSE_EVENT = "adverse_event"
    MEDICATION = "medication"
    STUDY = "study"
    DOMAIN = "domain"
    PROTOCOL = "protocol"
    VISIT = "visit"
    PARAMETER = "parameter"


class RelationType(Enum):
    """Standard relation types."""

    RECEIVED = "received"
    EXPERIENCED = "experienced"
    TREATED_WITH = "treated_with"
    ASSOCIATED_WITH = "associated_with"
    DERIVED_FROM = "derived_from"
    PARTICIPATED_IN = "participated_in"
    MEASURED_IN = "measured_in"
    REPORTED_IN = "reported_in"


@dataclass
class Entity:
    """Enhanced entity with confidence and context."""

    id: str
    type: EntityType
    name: str
    confidence: float
    context: str
    properties: Dict[str, Any]
    source_spans: List[Tuple[int, int]]


@dataclass
class Relation:
    """Enhanced relation with confidence."""

    id: str
    subject: str
    object: str
    relation_type: RelationType
    confidence: float
    context: str
    properties: Dict[str, Any]


class AdvancedEntityExtractor:
    """Advanced entity extraction using multiple strategies."""

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initialize entity extractor."""
        self.model = SentenceTransformer(model_name)
        self._load_entity_patterns()
        self._load_relation_patterns()

    def _load_entity_patterns(self):
        """Load comprehensive entity patterns."""
        self.entity_patterns = {
            EntityType.SUBJECT: [
                r"\b(?:subject|patient|participant)\s+[A-Z]{2}\d{4}\b",
                r'\busubjid\s*=\s*[\'"]?([A-Z]{2}\d{4})[\'"]?\b',
                r"\bsubject\s*#?\s*(\w+)\b",
            ],
            EntityType.TREATMENT: [
                r"\b(?:treatment|drug|medication|therapy)\s+([A-Za-z0-9\s-]+?)(?:\s|$)",
                r'\btrt[a-z]*\s*=\s*[\'"]?([^\'"\n]+)[\'"]?\b',
                r"\b(?:received|took|was\s+given)\s+([A-Za-z0-9\s-]+?)(?:\s|$)",
            ],
            EntityType.LAB_TEST: [
                r"\b(?:lab|test|parameter|measurement)\s+([A-Za-z0-9\s_\-]+?)(?:\s|$)",
                r'\blb[a-z]*\s*=\s*[\'"]?([^\'"\n]+)[\'"]?\b',
                r"\b(?:hemoglobin|creatinine|alt|ast|wbc|rbc)\b",
            ],
            EntityType.ADVERSE_EVENT: [
                r"\b(?:adverse\s+event|ae|side\s+effect)\s*:?\s*([A-Za-z0-9\s_\-]+?)(?:\s|$|\n)",
                r'\bae[a-z]*\s*=\s*[\'"]?([^\'"\n]+)[\'"]?\b',
                r"\b(?:headache|nausea|vomiting|dizziness|fatigue)\b",
            ],
            EntityType.MEDICATION: [
                r"\b(?:concomitant|background)\s+(?:medication|drug)\s+([A-Za-z0-9\s-]+?)(?:\s|$)",
                r'\bcm[a-z]*\s*=\s*[\'"]?([^\'"\n]+)[\'"]?\b',
            ],
            EntityType.STUDY: [
                r"\b(?:study|protocol|trial)\s+([A-Z]{2}\d{4})\b",
                r"\bprotocol\s*#?\s*(\w+)\b",
            ],
            EntityType.DOMAIN: [
                r"\b(?:domain|sdtm|adam)\s+([A-Za-z]+)\b",
                r"\b(?:demographics|adverse\s+events|efficacy|safety)\b",
            ],
        }

    def _load_relation_patterns(self):
        """Load relation extraction patterns."""
        self.relation_patterns = [
            (
                r"(\w+(?:\s+\w+)*)\s+(?:received|took|was\s+given)\s+(\w+(?:\s+\w+)*)",
                RelationType.RECEIVED,
                0.8,
            ),
            (
                r"(\w+(?:\s+\w+)*)\s+(?:had|experienced|reported)\s+(\w+(?:\s+\w+)*)",
                RelationType.EXPERIENCED,
                0.8,
            ),
            (
                r"(\w+(?:\s+\w+)*)\s+(?:treated\s+with|managed\s+with)\s+(\w+(?:\s+\w+)*)",
                RelationType.TREATED_WITH,
                0.7,
            ),
            (
                r"(\w+(?:\s+\w+)*)\s+(?:associated\s+with|linked\s+to)\s+(\w+(?:\s+\w+)*)",
                RelationType.ASSOCIATED_WITH,
                0.6,
            ),
            (
                r"(\w+(?:\s+\w+)*)\s+(?:participated\s+in|enrolled\s+in)\s+(\w+(?:\s+\w+)*)",
                RelationType.PARTICIPATED_IN,
                0.7,
            ),
            (
                r"(\w+(?:\s+\w+)*)\s+(?:measured\s+in|assessed\s+in)\s+(\w+(?:\s+\w+)*)",
                RelationType.MEASURED_IN,
                0.6,
            ),
        ]

    def extract_entities(self, text: str) -> List[Entity]:
        """Extract entities using pattern matching and context analysis."""
        entities = []
        entity_counter = 0

        # Extract entities using patterns
        for entity_type, patterns in self.entity_patterns.items():
            for pattern in patterns:
                matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
                for match in matches:
                    entity_name = match.group(1) if match.groups() else match.group(0)
                    entity_name = re.sub(r"\s+", " ", entity_name.strip()).upper()

                    # Calculate confidence based on pattern specificity
                    confidence = self._calculate_entity_confidence(
                        entity_name, entity_type, match
                    )

                    if confidence > 0.5:  # Threshold for entity acceptance
                        entity = Entity(
                            id=f"entity_{entity_counter}",
                            type=entity_type,
                            name=entity_name,
                            confidence=confidence,
                            context=self._extract_context(
                                text, match.start(), match.end()
                            ),
                            properties={
                                "source": "pattern_extraction",
                                "pattern": pattern,
                                "position": (match.start(), match.end()),
                            },
                            source_spans=[(match.start(), match.end())],
                        )
                        entities.append(entity)
                        entity_counter += 1

        # Deduplicate entities
        entities = self._deduplicate_entities(entities)

        logger.info(f"Extracted {len(entities)} entities from text")
        return entities

    def _calculate_entity_confidence(
        self, name: str, entity_type: EntityType, match
    ) -> float:
        """Calculate confidence score for entity."""
        base_confidence = 0.6

        # Boost confidence for specific patterns
        if "=" in match.group(0):  # Variable assignment pattern
            base_confidence += 0.2

        # Boost for standard formats
        if entity_type == EntityType.SUBJECT and re.match(r"^[A-Z]{2}\d{4}$", name):
            base_confidence += 0.2
        elif entity_type == EntityType.STUDY and re.match(r"^[A-Z]{2}\d{4}$", name):
            base_confidence += 0.2

        # Boost for common clinical terms
        common_terms = {
            "HEMOGLOBIN",
            "CREATININE",
            "ALT",
            "AST",
            "WBC",
            "RBC",
            "HEADACHE",
            "NAUSEA",
            "VOMITING",
            "DIZZINESS",
        }
        if name in common_terms:
            base_confidence += 0.1

        return min(base_confidence, 1.0)

    def _extract_context(
        self, text: str, start: int, end: int, window: int = 100
    ) -> str:
        """Extract context around entity match."""
        context_start = max(0, start - window)
        context_end = min(len(text), end + window)
        return text[context_start:context_end]

    def _deduplicate_entities(self, entities: List[Entity]) -> List[Entity]:
        """Deduplicate entities based on name and type."""
        seen = set()
        deduplicated = []

        for entity in entities:
            key = (entity.type, entity.name)
            if key not in seen:
                seen.add(key)
                deduplicated.append(entity)
            else:
                # Merge with existing entity if higher confidence
                for i, existing in enumerate(deduplicated):
                    if (existing.type, existing.name) == key:
                        if entity.confidence > existing.confidence:
                            deduplicated[i] = entity
                        break

        return deduplicated

    def extract_relations(self, text: str, entities: List[Entity]) -> List[Relation]:
        """Extract relations between entities."""
        relations = []
        relation_counter = 0

        # Create entity lookup
        entity_map = {(e.name, e.type.value): e for e in entities}

        for pattern, relation_type, confidence in self.relation_patterns:
            matches = re.finditer(pattern, text, re.IGNORECASE | re.MULTILINE)
            for match in matches:
                if len(match.groups()) >= 2:
                    subject_name = match.group(1).strip().upper()
                    object_name = match.group(2).strip().upper()

                    # Find matching entities
                    subject_entity = self._find_entity_by_name(subject_name, entities)
                    object_entity = self._find_entity_by_name(object_name, entities)

                    if subject_entity and object_entity:
                        relation = Relation(
                            id=f"relation_{relation_counter}",
                            subject=subject_entity.id,
                            object=object_entity.id,
                            relation_type=relation_type,
                            confidence=confidence,
                            context=self._extract_context(
                                text, match.start(), match.end()
                            ),
                            properties={
                                "source": "pattern_extraction",
                                "pattern": pattern,
                                "position": (match.start(), match.end()),
                            },
                        )
                        relations.append(relation)
                        relation_counter += 1

        logger.info(f"Extracted {len(relations)} relations from text")
        return relations

    def _find_entity_by_name(
        self, name: str, entities: List[Entity]
    ) -> Optional[Entity]:
        """Find entity by name with fuzzy matching."""
        # Exact match first
        for entity in entities:
            if entity.name == name:
                return entity

        # Fuzzy match
        for entity in entities:
            if name in entity.name or entity.name in name:
                return entity

        return None


class GraphBuilder:
    """Automated graph construction with community detection."""

    def __init__(self):
        """Initialize graph builder."""
        self.entity_extractor = AdvancedEntityExtractor()
        self.sentence_model = SentenceTransformer("all-MiniLM-L6-v2")

    async def build_graph_from_documents(
        self, documents: List[Document], neo4j_service: Neo4jService
    ) -> Dict[str, Any]:
        """Build knowledge graph from documents."""
        logger.info(f"Building graph from {len(documents)} documents")

        all_entities = []
        all_relations = []

        # Extract entities and relations from all documents
        for doc in documents:
            entities = self.entity_extractor.extract_entities(doc.content)
            relations = self.entity_extractor.extract_relations(doc.content, entities)

            # Add document context
            for entity in entities:
                entity.properties["document_id"] = doc.id
                entity.properties["study_id"] = doc.study_id

            for relation in relations:
                relation.properties["document_id"] = doc.id
                relation.properties["study_id"] = doc.study_id

            all_entities.extend(entities)
            all_relations.extend(relations)

        # Deduplicate across documents
        deduplicated_entities = self._deduplicate_entities_across_docs(all_entities)
        deduplicated_relations = self._deduplicate_relations_across_docs(all_relations)

        # Build NetworkX graph for analysis
        nx_graph = self._build_networkx_graph(
            deduplicated_entities, deduplicated_relations
        )

        # Detect communities
        communities = self._detect_communities(nx_graph)

        # Calculate centrality measures
        centrality = self._calculate_centrality(nx_graph)

        # Store in Neo4j
        await self._store_in_neo4j(
            deduplicated_entities, deduplicated_relations, communities, neo4j_service
        )

        return {
            "entities_count": len(deduplicated_entities),
            "relations_count": len(deduplicated_relations),
            "communities_count": len(communities),
            "graph_density": nx.density(nx_graph),
            "centrality": centrality,
        }

    def _deduplicate_entities_across_docs(self, entities: List[Entity]) -> List[Entity]:
        """Deduplicate entities across all documents."""
        entity_map = {}

        for entity in entities:
            key = (entity.type, entity.name)
            if key not in entity_map:
                entity_map[key] = entity
            else:
                # Merge document sources
                existing = entity_map[key]
                if "document_ids" not in existing.properties:
                    existing.properties["document_ids"] = [
                        existing.properties.get("document_id")
                    ]
                existing.properties["document_ids"].append(
                    entity.properties.get("document_id")
                )
                existing.properties["document_ids"] = list(
                    set(existing.properties["document_ids"])
                )

                # Update confidence if higher
                if entity.confidence > existing.confidence:
                    existing.confidence = entity.confidence

        return list(entity_map.values())

    def _deduplicate_relations_across_docs(
        self, relations: List[Relation]
    ) -> List[Relation]:
        """Deduplicate relations across all documents."""
        relation_map = {}

        for relation in relations:
            key = (relation.subject, relation.object, relation.relation_type)
            if key not in relation_map:
                relation_map[key] = relation
            else:
                # Merge document sources
                existing = relation_map[key]
                if "document_ids" not in existing.properties:
                    existing.properties["document_ids"] = [
                        existing.properties.get("document_id")
                    ]
                existing.properties["document_ids"].append(
                    relation.properties.get("document_id")
                )
                existing.properties["document_ids"] = list(
                    set(existing.properties["document_ids"])
                )

        return list(relation_map.values())

    def _build_networkx_graph(
        self, entities: List[Entity], relations: List[Relation]
    ) -> nx.Graph:
        """Build NetworkX graph for analysis."""
        G = nx.Graph()

        # Add nodes
        for entity in entities:
            G.add_node(
                entity.id,
                type=entity.type.value,
                name=entity.name,
                confidence=entity.confidence,
                properties=entity.properties,
            )

        # Add edges
        for relation in relations:
            G.add_edge(
                relation.subject,
                relation.object,
                type=relation.relation_type.value,
                confidence=relation.confidence,
                properties=relation.properties,
            )

        return G

    def _detect_communities(self, graph: nx.Graph) -> List[Dict[str, Any]]:
        """Detect communities using multiple algorithms."""
        communities = []

        # Try DBSCAN first (good for arbitrary shapes)
        if len(graph) > 10:
            # Get adjacency matrix
            adj_matrix = nx.adjacency_matrix(graph).todense()

            # Use DBSCAN on adjacency matrix
            clustering = DBSCAN(eps=0.5, min_samples=2, metric="cosine")
            labels = clustering.fit_predict(adj_matrix)

            # Create community objects
            for label in set(labels):
                if label != -1:  # Not noise
                    nodes = [
                        node
                        for i, node in enumerate(graph.nodes())
                        if labels[i] == label
                    ]
                    if len(nodes) >= 2:
                        communities.append(
                            {
                                "id": f"community_{label}",
                                "nodes": nodes,
                                "size": len(nodes),
                                "algorithm": "dbscan",
                            }
                        )

        # Fallback to K-means if DBSCAN doesn't work well
        if not communities and len(graph) > 5:
            n_clusters = min(5, len(graph) // 2)
            clustering = KMeans(n_clusters=n_clusters, random_state=42)

            # Use node embeddings for clustering
            node_embeddings = self._get_node_embeddings(graph)
            labels = clustering.fit_predict(node_embeddings)

            for label in range(n_clusters):
                nodes = [
                    node for i, node in enumerate(graph.nodes()) if labels[i] == label
                ]
                if len(nodes) >= 2:
                    communities.append(
                        {
                            "id": f"community_{label}",
                            "nodes": nodes,
                            "size": len(nodes),
                            "algorithm": "kmeans",
                        }
                    )

        return communities

    def _get_node_embeddings(self, graph: nx.Graph) -> np.ndarray:
        """Get embeddings for nodes using graph structure."""
        # Simple approach: use node degrees and clustering coefficients
        features = []
        for node in graph.nodes():
            degree = graph.degree(node)
            clustering = nx.clustering(graph, node)
            features.append([degree, clustering])

        return np.array(features)

    def _calculate_centrality(self, graph: nx.Graph) -> Dict[str, Dict[str, float]]:
        """Calculate various centrality measures."""
        centrality = {}

        # Betweenness centrality
        betweenness = nx.betweenness_centrality(graph)

        # Closeness centrality
        closeness = nx.closeness_centrality(graph)

        # PageRank
        pagerank = nx.pagerank(graph)

        # Combine for each node
        for node in graph.nodes():
            centrality[node] = {
                "betweenness": betweenness.get(node, 0),
                "closeness": closeness.get(node, 0),
                "pagerank": pagerank.get(node, 0),
            }

        return centrality

    async def _store_in_neo4j(
        self,
        entities: List[Entity],
        relations: List[Relation],
        communities: List[Dict[str, Any]],
        neo4j_service: Neo4jService,
    ):
        """Store graph data in Neo4j."""
        # Store entities
        for entity in entities:
            await neo4j_service.create_entity(
                {
                    "id": entity.id,
                    "type": entity.type.value,
                    "name": entity.name,
                    "confidence": entity.confidence,
                    "context": entity.context,
                    "properties": entity.properties,
                }
            )

        # Store relations
        for relation in relations:
            await neo4j_service.create_relation(
                {
                    "id": relation.id,
                    "subject": relation.subject,
                    "object": relation.object,
                    "type": relation.relation_type.value,
                    "confidence": relation.confidence,
                    "context": relation.context,
                    "properties": relation.properties,
                }
            )

        # Store communities
        for community in communities:
            await neo4j_service.create_community(
                {
                    "id": community["id"],
                    "nodes": community["nodes"],
                    "size": community["size"],
                    "algorithm": community["algorithm"],
                    "properties": community,
                }
            )


class AdvancedGraphRAGService:
    """Advanced GraphRAG service with automated construction."""

    def __init__(self, neo4j_service: Neo4jService):
        """Initialize GraphRAG service."""
        self.neo4j_service = neo4j_service
        self.graph_builder = GraphBuilder()

    async def build_knowledge_graph(self, documents: List[Document]) -> Dict[str, Any]:
        """Build knowledge graph from documents."""
        return await self.graph_builder.build_graph_from_documents(
            documents, self.neo4j_service
        )

    async def query_graph(
        self, query_text: str, top_k: int = 10
    ) -> List[Dict[str, Any]]:
        """Query knowledge graph for relevant information."""
        # Extract entities from query
        entities = self.graph_builder.entity_extractor.extract_entities(query_text)

        if not entities:
            return []

        # Find related entities and relations
        results = []
        for entity in entities[:3]:  # Limit to top 3 entities
            related = await self.neo4j_service.find_related_entities(
                entity.name, entity.type.value
            )
            results.extend(related)

        # Rank and return top results
        results = self._rank_graph_results(results, entities)
        return results[:top_k]

    def _rank_graph_results(
        self, results: List[Dict[str, Any]], query_entities: List[Entity]
    ) -> List[Dict[str, Any]]:
        """Rank graph results based on relevance to query."""
        scored_results = []

        for result in results:
            score = 0.0

            # Boost score if result matches query entities
            for q_entity in query_entities:
                if result.get("name") == q_entity.name:
                    score += 1.0
                elif result.get("type") == q_entity.type.value:
                    score += 0.5

            # Add centrality score
            centrality = result.get("centrality", {})
            score += centrality.get("pagerank", 0) * 0.3
            score += centrality.get("betweenness", 0) * 0.2

            scored_results.append({**result, "relevance_score": score})

        return sorted(scored_results, key=lambda x: x["relevance_score"], reverse=True)
