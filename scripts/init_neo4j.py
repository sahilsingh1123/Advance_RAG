"""Initialize Neo4j database with constraints and indexes."""

from neo4j import GraphDatabase
from advance_rag.core.config import get_settings

settings = get_settings()

# Connect to Neo4j
driver = GraphDatabase.driver(
    settings.NEO4J_URI, auth=(settings.NEO4J_USER, settings.NEO4J_PASSWORD)
)

constraints = [
    # Entity constraints
    "CREATE CONSTRAINT entity_id_unique IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
    "CREATE INDEX entity_type_idx IF NOT EXISTS FOR (e:Entity) ON (e.type)",
    "CREATE INDEX entity_name_idx IF NOT EXISTS FOR (e:Entity) ON (e.name)",
    # Relation constraints
    "CREATE CONSTRAINT relation_id_unique IF NOT EXISTS FOR ()-[r:RELATION]-() REQUIRE r.id IS UNIQUE",
    "CREATE INDEX relation_type_idx IF NOT EXISTS FOR ()-[r:RELATION]-() ON (r.type)",
    # Community constraints
    "CREATE CONSTRAINT community_id_unique IF NOT EXISTS FOR (c:Community) REQUIRE c.id IS UNIQUE",
    "CREATE INDEX community_level_idx IF NOT EXISTS FOR (c:Community) ON (c.level)",
    # Document constraints
    "CREATE CONSTRAINT document_id_unique IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
    "CREATE INDEX document_type_idx IF NOT EXISTS FOR (d:Document) ON (d.type)",
    # Study constraints
    "CREATE CONSTRAINT study_id_unique IF NOT EXISTS FOR (s:Study) REQUIRE s.id IS UNIQUE",
]

# Create sample data
sample_data = [
    # Create sample study
    """
    MERGE (s:Study {id: 'STUDY001'})
    SET s.title = 'Sample Clinical Trial',
        s.description = 'A randomized controlled trial',
        s.phase = 'Phase II',
        s.indication = 'Hypertension'
    """,
    # Create sample entities
    """
    MERGE (e1:Entity {id: 'entity_001'})
    SET e1.type = 'subject',
        e1.name = 'STUDY001001',
        e1.properties = {age: 45, sex: 'M'}
    """,
    """
    MERGE (e2:Entity {id: 'entity_002'})
    SET e2.type = 'treatment',
        e2.name = 'DRUG_A',
        e2.properties = {dose: '100mg', frequency: 'daily'}
    """,
    """
    MERGE (e3:Entity {id: 'entity_003'})
    SET e3.type = 'lab_test',
        e3.name = 'ALT',
        e3.properties = {unit: 'U/L', normal_range: '10-50'}
    """,
    # Create sample relations
    """
    MATCH (subject:Entity {id: 'entity_001'})
    MATCH (treatment:Entity {id: 'entity_002'})
    MERGE (subject)-[r1:RELATION {id: 'rel_001'}]->(treatment)
    SET r1.type = 'received',
        r1.properties = {start_date: '2024-01-01', duration: '12 weeks'}
    """,
    """
    MATCH (subject:Entity {id: 'entity_001'})
    MATCH (lab:Entity {id: 'entity_003'})
    MERGE (subject)-[r2:RELATION {id: 'rel_002'}]->(lab)
    SET r2.type = 'had',
        r2.properties = {value: 35, date: '2024-01-15'}
    """,
]

try:
    with driver.session(database="neo4j") as session:
        print("Creating constraints and indexes...")
        for constraint in constraints:
            try:
                session.run(constraint)
                print(f"✓ Created: {constraint}")
            except Exception as e:
                print(f"✗ Failed: {constraint} - {e}")

        print("\nCreating sample data...")
        for query in sample_data:
            try:
                session.run(query)
                print(f"✓ Created sample data")
            except Exception as e:
                print(f"✗ Failed: {e}")

        print("\nNeo4j initialization complete!")

finally:
    driver.close()
