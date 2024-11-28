from floki.storage.daprstores.statestore import DaprStateStore
from floki.service.fastapi.base import ServiceBase
from dapr.conf import settings as dapr_settings
from dapr.ext.fastapi import DaprApp
from dapr.clients import DaprClient
from pydantic import Field
from typing import Optional, Any, Dict
from abc import ABC
import json
import os
import logging

logger = logging.getLogger(__name__)

class DaprEnabledService(ServiceBase, ABC):
    """
    A Dapr-enabled service class extending ServiceBase with Dapr-specific functionality.
    Provides pub/sub, and handles Dapr-specific configurations.
    """
    
    message_bus_name: str = Field(..., description="The name of the Dapr message bus component, defining the pub/sub base.")
    daprGrpcAddress: Optional[str] = Field(None, description="Full address of the Dapr sidecar.")
    daprGrpcHost: Optional[str] = Field(None, description="Host address for the Dapr gRPC endpoint.")
    daprGrpcPort: Optional[int] = Field(None, description="Port number for the Dapr gRPC endpoint.")

    # Initialized in model_post_init
    dapr_app: Optional[DaprApp] = Field(default=None, init=False, description="DaprApp for pub/sub integration.")
    dapr_client: Optional[DaprClient] = Field(default=None, init=False, description="Dapr client for state and pub/sub.")
    dapr_app: Optional[DaprApp] = Field(default=None, init=False, description="DaprApp for pub/sub integration.")

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization to configure the FastAPI app and Dapr-specific settings.
        """
        # Initialize inherited FastAPI app setup
        super().model_post_init(__context)

        # Configure Dapr gRPC settings, using environment variables if provided
        env_daprGrpcHost = os.getenv('DAPR_RUNTIME_HOST')
        env_daprGrpcPort = os.getenv('DAPR_GRPC_PORT')

        # Resolve final values and persist them in the instance
        self.daprGrpcHost = self.daprGrpcHost or env_daprGrpcHost or dapr_settings.DAPR_RUNTIME_HOST
        self.daprGrpcPort = int(self.daprGrpcPort or env_daprGrpcPort or dapr_settings.DAPR_GRPC_PORT)
        
        # Set the Dapr gRPC address based on finalized settings
        self.daprGrpcAddress = self.daprGrpcAddress or f"{self.daprGrpcHost}:{self.daprGrpcPort}"

        # Initialize DaprApp for pub/sub and DaprClient for state and messaging
        self.dapr_app = DaprApp(self.app)
        self.dapr_client = DaprClient(address=self.daprGrpcAddress)

        logger.info(f"{self.name} DaprEnabledService initialized with Dapr gRPC address {self.daprGrpcAddress}.")
    
    async def get_metadata_from_store(self, store: DaprStateStore, key: str) -> Optional[dict]:
        """
        Retrieve metadata from a specified Dapr state store based on a provided key.

        Args:
            store (DaprStateStore): The Dapr state store instance to query.
            key (str): The key in the state store under which metadata is stored.

        Returns:
            Optional[dict]: The metadata stored under the specified key if found; otherwise, None.
        """
        try:
            metadata = store.get_state(key).data
            return json.loads(metadata) if metadata else None
        except Exception as e:
            logger.error(f"Error retrieving metadata for key '{key}' from store '{store.store_name}': {e}")
            return None
    
    async def publish_message(self, topic_name: str, pubsub_name: str, message: Any, metadata: Optional[Dict[str, Any]] = None):
        """
        Helper function to publish a message to a specific topic with optional metadata.

        Args:
            topic_name (str): The topic to publish the message to.
            pubsub_name (str): The pub/sub component to use.
            message (Any): The message content, can be None or any JSON-serializable type.
            metadata (Optional[Dict[str, Any]]): Additional metadata to include in the publish event.

        Raises:
            Exception: If publishing the message fails.
        """
        try:
            # Convert message to JSON string
            json_message = json.dumps(message if message is not None else {})

            with DaprClient(address=self.daprGrpcAddress) as client:
                client.publish_event(
                    pubsub_name=pubsub_name,
                    topic_name=topic_name,
                    data=json_message,
                    data_content_type='application/json',
                    publish_metadata=metadata or {}
                )
        except Exception as e:
            logger.error(
                f"Error publishing message to topic '{topic_name}' on pub/sub '{pubsub_name}'. "
                f"Message: {message}, Metadata: {metadata}, Error: {e}"
            )
            raise Exception(f"Failed to publish message to topic '{topic_name}' on pub/sub '{pubsub_name}': {str(e)}")
    
    async def publish_event_message(self, topic_name: str, pubsub_name: str, source: str, message_type: str, message: dict, **kwargs) -> None:
        """
        Publishes an event message to a specified topic with dynamic metadata.

        Args:
            topic_name (str): The topic to publish the message to.
            pubsub_name (str): The pub/sub component to use.
            source (str): The source of the message (e.g., service or agent name).
            message_type (str): The type of the message (e.g., "AgentActionResultMessage").
            message (dict): The content of the message to publish.
            **kwargs: Additional metadata fields to include in the message.
        """
        # Base metadata
        base_metadata = {
            "cloudevent.type": message_type,
            "cloudevent.source": source,
        }

        # Merge additional metadata from kwargs
        metadata = {**base_metadata, **kwargs}
        
        # Publish the message
        await self.publish_message(
            topic_name=topic_name,
            pubsub_name=pubsub_name,
            message=message,
            metadata=metadata,
        )

        logger.info(f"{source} successfully published {message_type} to topic '{topic_name}'!.")
        logger.debug(f"Message: {message}, Metadata: {metadata}")