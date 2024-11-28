import logging
from floki.storage.graphstores import GraphStoreBase
from floki.storage.graphstores.neo4j.client import Neo4jClient
from floki.storage.graphstores.neo4j.utils import value_sanitize, get_current_time
from floki.types import Node, Relationship
from neo4j import Query
from neo4j.exceptions import Neo4jError, CypherSyntaxError
from pydantic import Field
from typing import Any, Dict, Optional, List

logger = logging.getLogger(__name__)

class Neo4jGraphStore(GraphStoreBase):
    """
    Neo4j-based graph store implementation using Pydantic.
    """

    uri: str = Field(..., description="The URI of the Neo4j database.")
    user: str = Field(..., description="The username for authentication.")
    password: str = Field(..., description="The password for authentication.")
    database: str = Field(default="neo4j", description="The Neo4j database to use.")
    sanitize: bool = Field(default=True, description="Whether to sanitize the results.")
    timeout: int = Field(default=30, description="Query timeout in seconds.")
    graph_schema: Dict[str, Any] = Field(default_factory=dict, description="Schema of the graph structure.")

    # Client initialized in model_post_init, not during regular initialization
    client: Optional[Neo4jClient] = Field(default=None, init=False, description="Client for interacting with the Neo4j database.")
    
    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization to set up the Neo4j client after model instantiation.
        """
        self.client = Neo4jClient(self.uri, self.user, self.password, self.database)
        logger.info(f"Neo4jGraphStore initialized with database {self.database}")

        # Complete post-initialization
        super().model_post_init(__context)

    def add_node(self, node: Node) -> None:
        """
        Add a node to the Neo4j database with specified label and properties.

        Args:
            node (Node): The node to add.

        Raises:
            ValueError: If there is an issue with the query execution.
        """
        query = f"""MERGE (n: {node.label} {{id: {node.id}}})
        ON CREATE SET n.createdAt = $current_time
        SET n.updatedAt = $current_time, n += apoc.map.clean($props, [], [])
        WITH n
        CALL apoc.create.addLabels(n, $additional_labels)
        YIELD node
        """
        if node.embedding is not None:
            query += """
            WITH node, $embedding AS embedding
            CALL db.create.setNodeVectorProperty(node, 'embedding', embedding)
            YIELD node
            """
        query += "RETURN node"

        params = {
            'props': node.properties,
            'additional_labels': node.additional_labels,
            'embedding': node.embedding,
            'current_time': get_current_time()
        }
        try:
            with self.client.driver.session(database=self.client.database) as session:
                session.run(query, params)
                logger.info("Node with label %s and properties %s added or updated successfully", node.label, node.properties)
        except Neo4jError as e:
            logger.error("Failed to add or update node: %s", str(e))
            raise ValueError(f"Failed to add or update node: {str(e)}")

    def add_nodes(self, nodes: List[Node]) -> None:
        """
        Add multiple nodes to the Neo4j database with specified label and properties.

        Args:
            nodes (List[Node]): A list of nodes to add.

        Raises:
            ValueError: If there is an issue with the query execution.
        """
        query = """
        UNWIND $nodes AS node
        MERGE (n:{node.label} {id: node.id})
        ON CREATE SET n.createdAt = $current_time
        SET n.updatedAt = $current_time, n += apoc.map.clean(node.properties, [], [])
        WITH n, node.additional_labels AS additional_labels, node.embedding AS embedding
        CALL apoc.create.addLabels(n, additional_labels)
        YIELD node
        WITH node, embedding
        WHERE embedding IS NOT NULL
        CALL db.create.setNodeVectorProperty(node, 'embedding', embedding)
        YIELD node
        RETURN node
        """
        
        params = {
            'nodes': [n.model_dump() for n in nodes],
            'current_time': get_current_time()
        }
        try:
            with self.client.driver.session(database=self.client.database) as session:
                session.run(query, params)
                logger.info("Nodes added or updated successfully")
        except Neo4jError as e:
            logger.error("Failed to add or update nodes: %s", str(e))
            raise ValueError(f"Failed to add or update nodes: {str(e)}")
    
    def add_relationship(self, relationship: Relationship) -> None:
        """
        Create a relationship between two nodes in the Neo4j database.

        Args:
            relationship (Relationship): The relationship to create.

        Raises:
            ValueError: If there is an issue with the query execution.
        """
        query = f"""
        MATCH (a {{id: $source_node_id}}), (b {{id: $target_node_id}})
        MERGE (a)-[r:{relationship.type}]->(b)
        ON CREATE SET r.createdAt = $current_time
        SET r.updatedAt = $current_time, r += $properties
        RETURN r
        """
        params = {
            'source_node_id': relationship.source_node_id,
            'target_node_id': relationship.target_node_id,
            'properties': relationship.properties or {},
            'current_time': get_current_time()
        }
        try:
            with self.client.driver.session(database=self.client.database) as session:
                session.run(query, params)
                logger.info("Relationship with label %s between %s and %s created or updated successfully", relationship.type, relationship.source_node_id, relationship.target_node_id)
        except Neo4jError as e:
            logger.error("Failed to create or update relationship: %s", str(e))
            raise ValueError(f"Failed to create or update relationship: {str(e)}")
    
    def add_relationships(self, relationships: List[Relationship]) -> None:
        """
        Create multiple relationships between nodes in the Neo4j database.

        Args:
            relationships (List[Relationship]): A list of relationships to create.

        Raises:
            ValueError: If there is an issue with the query execution.
        """
        query = """
        UNWIND $relationships AS rel
        MATCH (a {id: rel.source_node_id}), (b {id: rel.target_node_id})
        MERGE (a)-[r:{rel.label}]->(b)
        ON CREATE SET r.createdAt = $current_time
        SET r.updatedAt = $current_time, r += rel.properties
        RETURN r
        """
        params = {
            'relationships': [r.model_dump() for r in relationships],
            'current_time': get_current_time()
        }
        try:
            with self.client.driver.session(database=self.client.database) as session:
                session.run(query, params)
                logger.info("Relationships created or updated successfully")
        except Neo4jError as e:
            logger.error("Failed to create or update relationships: %s", str(e))
            raise ValueError(f"Failed to create or update relationships: {str(e)}")
    
    def query(self, query: str, params: Dict[str, Any] = None, sanitize: bool = None) -> List[Dict[str, Any]]:
        """
        Execute a Cypher query against a Neo4j database and optionally sanitize the results.

        Args:
            query (str): The Cypher query to execute.
            params (Dict[str, Any]): Optional dictionary of parameters for the query.
            sanitize (bool): Whether to sanitize the results.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries representing the query results.

        Raises:
            ValueError: If there is a syntax error in the Cypher query.
            Neo4jError: If any other Neo4j-related error occurs.
        
        References:
            `https://github.com/langchain-ai/langchain/blob/master/libs/community/langchain_community/graphs/neo4j_graph.py`_
        """
        params = params or {}
        sanitize = sanitize if sanitize is not None else self.sanitize
        try:
            with self.client.driver.session(database=self.client.database) as session:
                result = session.run(Query(text=query, timeout=self.timeout), parameters=params)
                json_data = [record.data() for record in result]
                if sanitize:
                    json_data = [value_sanitize(el) for el in json_data]
                logger.info("Query executed successfully: %s", query)
                return json_data
        except CypherSyntaxError as e:
            logger.error("Syntax error in the Cypher query: %s", str(e))
            raise ValueError(f"Syntax error in the Cypher query: {str(e)}")
        except Neo4jError as e:
            logger.error("An error occurred during the query execution: %s", str(e))
            raise ValueError(f"An error occurred during the query execution: {str(e)}")
    
    def reset(self):
        """
        Reset the Neo4j database by deleting all nodes and relationships.

        Raises:
            ValueError: If there is an issue with the query execution.
        """
        try:
            with self.client.driver.session() as session:
                session.run("CALL apoc.schema.assert({}, {})")
                session.run("CALL apoc.periodic.iterate('MATCH (n) RETURN n', 'DETACH DELETE n', {batchSize:1000, iterateList:true})")
                logger.info("Database reset successfully")
        except Neo4jError as e:
            logger.error("Failed to reset database: %s", str(e))
            raise ValueError(f"Failed to reset database: {str(e)}")

    def refresh_schema(self) -> None:
        """
        Refresh the database schema, including node properties, relationship properties, constraints, and indexes.

        Raises:
            ValueError: If there is an issue with the query execution.
        """
        try:
            # Refresh node properties
            node_properties = self.query(
                """
                CALL apoc.meta.data()
                YIELD label, property, type
                WHERE type <> 'RELATIONSHIP'
                RETURN label, collect({property: property, type: type}) AS properties
                """
            )

            # Refresh relationship properties
            relationship_properties = self.query(
                """
                CALL apoc.meta.data()
                YIELD label, property, type
                WHERE type = 'RELATIONSHIP'
                RETURN label, collect({property: property, type: type}) AS properties
                """
            )

            # Refresh constraints
            constraints = self.query("SHOW CONSTRAINTS")

            # Refresh indexes
            indexes = self.query(
                """
                CALL apoc.schema.nodes() 
                YIELD label, properties, type, size, valuesSelectivity 
                WHERE type = 'RANGE' 
                RETURN *, size * valuesSelectivity as distinctValues
                """
            )

            self.graph_schema = {
                "node_props": {record['label']: record['properties'] for record in node_properties},
                "rel_props": {record['label']: record['properties'] for record in relationship_properties},
                "constraints": constraints,
                "indexes": indexes,
            }
            logger.info("Schema refreshed successfully")
        except Neo4jError as e:
            logger.error("Failed to refresh schema: %s", str(e))
            raise ValueError(f"Failed to refresh schema: {str(e)}")

    def get_schema(self, refresh: bool = False) -> Dict[str, Any]:
        """
        Get the schema of the Neo4jGraph store.

        Args:
            refresh (bool): Whether to refresh the schema before returning it. Defaults to False.

        Returns:
            Dict[str, Any]: The schema of the Neo4jGraph store.
        """
        if not self.graph_schema or refresh:
            self.refresh_schema()
        return self.graph_schema
    
    def create_vector_index(self, label: str, property: str, dimensions: int, similarity_function: str = 'cosine'):
        """
        Creates a vector index for a specified label and property in the Neo4j database.

        Args:
            label (str): The label of the nodes to index.
            property (str): The property of the nodes to index.
            dimensions (int): The number of dimensions of the vector.
            similarity_function (str): The similarity function to use (default is 'cosine').

        Raises:
            ValueError: If there is an issue with the query execution.
        """
        query = f"""
        CREATE VECTOR INDEX {label.lower()}_embedding_index IF NOT EXISTS
        FOR (n:{label})
        ON (n.{property})
        OPTIONS {{
            indexConfig: {{
                'vector.dimensions': {dimensions},
                'vector.similarity_function': '{similarity_function}'
            }}
        }}
        """
        try:
            with self.client.driver.session() as session:
                session.run(query)
                logger.info("Vector index for %s on property %s created successfully", label, property)
        except Neo4jError as e:
            logger.error("Failed to create vector index: %s", str(e))
            raise ValueError(f"Failed to create vector index: {str(e)}")