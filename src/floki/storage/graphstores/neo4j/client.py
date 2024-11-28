from neo4j import GraphDatabase
from typing import Optional
import os
import logging

logger = logging.getLogger(__name__)

class Neo4jClient:
    def __init__(self, uri: Optional[str] = None, user: Optional[str] = None, password: Optional[str] = None, database: Optional[str] = 'neo4j'):
        """
        Initializes the Neo4j client with the given connection parameters.

        Args:
            uri (Optional[str]): The URI of the Neo4j database.
            user (Optional[str]): The username for authentication.
            password (Optional[str]): The password for authentication.
            database (Optional[str]): The database to use. Defaults to 'neo4j'.
        """
        self.uri = os.getenv('NEO4J_URI', uri)
        self.user = os.getenv('NEO4J_USERNAME', user)
        self.password = os.getenv('NEO4J_PASSWORD', password)
        self.database = database
        self.driver = None

        try:
            self.driver = GraphDatabase.driver(self.uri, auth=(self.user, self.password))
            logger.info("Successfully created the driver for URI: %s", self.uri)
        except Exception as e:
            logger.error("Failed to create the driver: %s", str(e))
            exit(1)
    
    def close(self):
        """Closes the Neo4j driver connection."""
        if self.driver is not None:
            self.driver.close()
            logger.info("Neo4j driver connection closed")
    
    def test_connection(self):
        """Tests the connection to the Neo4j database.

        Returns:
            bool: True if the connection is successful, False otherwise.
        
        Raises:
            ValueError: If there is an error testing the connection.
        """
        try:
            with self.driver.session() as session:
                result = session.run("CALL dbms.components() YIELD name, versions, edition")
                record = result.single()
                if record:
                    logger.info("Connected to %s version %s (%s edition)", record['name'], record['versions'][0], record['edition'])
                    return True
                else:
                    logger.warning("No record found during the connection test")
                    return False
        except Exception as e:
            logger.error("Error testing connection: %s", str(e))
            raise ValueError(f"Error testing connection: {str(e)}")