from dapr.actor.runtime.config import ActorRuntimeConfig, ActorTypeConfig, ActorReentrancyConfig
from fastapi import FastAPI, HTTPException, Request, Response, status
from floki.storage.daprstores.statestore import DaprStateStore
from floki.agent.actor import AgentActorBase, AgentActorInterface
from floki.service.fastapi import DaprEnabledService
from floki.types.agent import AgentActorMessage
from floki.types.message import UserMessage
from floki.agent import AgentBase
from cloudevents.http.conversion import from_http
from cloudevents.http.event import CloudEvent
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from dapr.actor.runtime.runtime import ActorRuntime
from dapr.ext.fastapi import DaprActor
from dapr.actor import ActorProxy, ActorId
from pydantic import Field, model_validator, ConfigDict
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Optional, Any
import json
import logging

logger = logging.getLogger(__name__)

class AgentService(DaprEnabledService):
    """
    A Pydantic-based class for managing services and exposing FastAPI routes with Dapr pub/sub and actor support.
    """

    agent: AgentBase
    agent_topic_name: Optional[str] = Field(None, description="The topic name dedicated to this specific agent, derived from the agent's name if not provided.")
    broadcast_topic_name: Optional[str] = Field("beacon_channel", description="The default topic used for broadcasting messages to all agents.")
    task_results_topic_name: Optional[str] = Field("task_results_channel", description="The default topic used for sending the results of a task executed by an agent.")
    agents_state_store_name: str = Field(..., description="The name of the Dapr state store component used to store and share agent metadata centrally.")

    # Fields initialized in model_post_init
    actor: Optional[DaprActor] = Field(default=None, init=False, description="DaprActor for actor lifecycle support.")
    actor_name: Optional[str] = Field(default=None, init=False, description="Actor name")
    actor_proxy: Optional[ActorProxy] = Field(default=None, init=False, description="Proxy for invoking methods on the agent's actor.")
    actor_class: Optional[type] = Field(default=None, init=False, description="Dynamically created actor class for the agent")
    agent_metadata: Optional[dict] = Field(default=None, init=False, description="Agent's metadata")
    agent_metadata_store: Optional[DaprStateStore] = Field(default=None, init=False, description="Dapr state store instance for accessing and managing centralized agent metadata.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    @model_validator(mode="before")
    def set_service_name_and_topic(cls, values: dict):
        # Derive the service name from the agent's name or role
        if not values.get("name") and "agent" in values:
            values["name"] = values["agent"].name or values["agent"].role
        # Derive agent_topic_name from service name if not provided
        if not values.get("agent_topic_name") and values.get("name"):
            values["agent_topic_name"] = values["name"]
        return values

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization to configure the Dapr settings, FastAPI app, and other components.
        """

        # Proceed with base model setup
        super().model_post_init(__context)

        # Initialize the Dapr state store for agent metadata
        self.agent_metadata_store = DaprStateStore(store_name=self.agents_state_store_name, address=self.daprGrpcAddress)

        # Dynamically create the actor class based on the agent's name
        actor_class_name = f"{self.agent.name}Actor"

        # Create the actor class dynamically using the 'type' function
        self.actor_class = type(actor_class_name, (AgentActorBase,), {
            '__init__': lambda self, ctx, actor_id: AgentActorBase.__init__(self, ctx, actor_id),
            'agent': self.agent
        })

        # Prepare agent metadata
        self.agent_metadata = {
            "name": self.agent.name,
            "role": self.agent.role,
            "goal": self.agent.goal,
            "topic_name": self.agent_topic_name,
            "pubsub_name": self.message_bus_name
        }

        # Proxy for actor methods
        self.actor_name = self.actor_class.__name__
        self.actor_proxy = ActorProxy.create(self.actor_name, ActorId(self.agent.name), AgentActorInterface)
        
        # DaprActor for actor support
        self.actor = DaprActor(self.app)

        # Subscribe to the main agent-specific topic
        self.dapr_app.subscribe(pubsub=self.message_bus_name, topic=self.agent_topic_name)(self.process_agent_message)
        logger.info(f"{self.name} subscribed to topic {self.agent_topic_name} on {self.message_bus_name}")

        # Subscribe to the broadcast topic to handle shared messages
        self.dapr_app.subscribe(pubsub=self.message_bus_name, topic=self.broadcast_topic_name)(self.process_broadcast_message)
        logger.info(f"{self.name} subscribed to topic {self.broadcast_topic_name} on {self.message_bus_name}")

        # Adding other API Routes
        self.app.add_api_route("/GetMessages", self.get_messages, methods=["GET"]) # Get messages from conversation history state

        logger.info(f"Dapr Actor class {self.actor_class.__name__} initialized.")

    @asynccontextmanager
    async def lifespan(self, app: FastAPI):
        """
        Extended lifespan to manage actor registration and metadata setup at startup
        and cleanup on shutdown.
        """
        # Actor Runtime Configuration (e.g., reentrancy)
        actor_runtime_config = ActorRuntimeConfig()
        actor_runtime_config.update_actor_type_configs([
            ActorTypeConfig(
                actor_type=self.actor_class.__name__,
                actor_idle_timeout=timedelta(hours=1),
                actor_scan_interval=timedelta(seconds=30),
                drain_ongoing_call_timeout=timedelta(minutes=1),
                drain_rebalanced_actors=True,
                reentrancy=ActorReentrancyConfig(enabled=True))
        ])
        ActorRuntime.set_actor_config(actor_runtime_config)
        
        # Register actor class during startup            
        await self.actor.register_actor(self.actor_class)
        logger.info(f"{self.actor_name} Dapr actor registered.")
        
        # Register agent metadata
        await self.register_agent_metadata()

        try:
            yield  # Continue with FastAPI's main lifespan context
        finally:
            # Perform any required cleanup, such as metadata removal
            await self.stop()
    
    async def get_agents_metadata(self) -> dict:
        """
        Retrieve metadata for all agents except the orchestrator itself.
        """
        key = "agents_metadata"
        try:
            agents_metadata = await self.get_metadata_from_store(self.agent_metadata_store, key) or {}
            # Exclude the orchestrator's own metadata
            return {name: metadata for name, metadata in agents_metadata.items() if name != self.agent.name}
        except Exception as e:
            logger.error(f"Failed to retrieve agents metadata: {e}")
            return {}
    
    async def register_agent_metadata(self) -> None:
        """
        Registers the agent's metadata in the Dapr state store under 'agents_metadata'.
        """
        key = "agents_metadata"
        try:
            # Retrieve existing metadata or initialize as an empty dictionary
            agents_metadata = await self.get_metadata_from_store(self.agent_metadata_store, key) or {}
            agents_metadata[self.name] = self.agent_metadata

            # Save the updated metadata back to Dapr store
            self.agent_metadata_store.save_state(key, json.dumps(agents_metadata), {"contentType": "application/json"})
            logger.info(f"{self.name} registered its metadata under key '{key}'")

        except Exception as e:
            logger.error(f"Failed to register metadata for agent {self.name}: {e}")
    
    async def invoke_task(self, task: Optional[str]) -> Response:
        """
        Use the actor to invoke a task by running the InvokeTask method through ActorProxy.

        Args:
            task (Optional[str]): The task string to invoke on the actor.

        Returns:
            Response: A FastAPI Response containing the result or an error message.
        """
        try:
            response = await self.actor_proxy.InvokeTask(task)
            return Response(content=response, status_code=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Failed to run task for {self.actor_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Error invoking task: {str(e)}")
    
    async def add_message(self, message: AgentActorMessage) -> None:
        """
        Adds a message to the conversation history in the actor's state.
        """
        try:
            await self.actor_proxy.AddMessage(message.model_dump())
        except Exception as e:
            logger.error(f"Failed to add message to {self.actor_name}: {e}")
    
    async def get_messages(self) -> Response:
        """
        Retrieve the conversation history from the actor.
        """
        try:
            messages = await self.actor_proxy.GetMessages()
            return JSONResponse(content=jsonable_encoder(messages), status_code=status.HTTP_200_OK)
        except Exception as e:
            logger.error(f"Failed to retrieve messages for {self.actor_name}: {e}")
            raise HTTPException(status_code=500, detail=f"Error retrieving messages: {str(e)}")
    
    async def publish_message_to_all(self, message_type: Any, message: dict, **kwargs) -> None:
        """
        Publishes a message to all agents on the configured broadcast topic.

        Args:
            message_type (str): The type of the message (e.g., "AgentActionResultMessage").
            message (dict): The content of the message to broadcast.
            **kwargs: Additional metadata fields to include in the message.
        """
        try:
            # Retrieve metadata for all agents
            agents_metadata = await self.get_agents_metadata()
            if not agents_metadata:
                logger.warning("No agents available for broadcast.")
                return
            
            logger.info(f"{self.agent.name} sending {message_type} to all agents.")

            # Use publish_event_message for broadcasting
            await self.publish_event_message(
                topic_name=self.broadcast_topic_name,
                pubsub_name=self.message_bus_name,
                source=self.agent.name,
                message_type=message_type,
                message=message,
                **kwargs,
            )
        
        except Exception as e:
            logger.error(f"Failed to send broadcast message of type {message_type}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error broadcasting message: {str(e)}")
    
    async def publish_message_to_agent(self, name: str, message_type: Any, message: dict, **kwargs) -> None:
        """
        Publishes a message to a specific agent.

        Args:
            name (str): The name of the target agent.
            message_type (str): The type of the message (e.g., "TriggerActionMessage").
            message (dict): The content of the message.
            **kwargs: Additional metadata fields to include in the message.
        """
        try:
            # Retrieve metadata for all agents
            agents_metadata = await self.get_agents_metadata()
            if name not in agents_metadata:
                raise HTTPException(status_code=404, detail=f"Agent {name} not found.")

            # Extract agent-specific metadata
            agent_metadata = agents_metadata[name]

            logger.info(f"{self.agent.name} sending {message_type} to agent {name}.")

            # Use publish_event_message for targeting a specific agent
            await self.publish_event_message(
                topic_name=agent_metadata["topic_name"],
                pubsub_name=agent_metadata["pubsub_name"],
                source=self.agent.name,
                message_type=message_type,
                message=message,
                **kwargs,
            )

        except Exception as e:
            logger.error(f"Failed to publish message to agent {name}: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error publishing message to agent: {str(e)}")
    
    async def publish_results_message(self, message_type: Any, message: dict, **kwargs) -> None:
        """
        Publishes a message with results to a specific topic.

        Args:
            message_type (str): The type of the message (e.g., "TaskResultMessage").
            message (dict): The content of the message to publish.
            **kwargs: Additional metadata fields to include in the message.
        """
        try:
            logger.info(f"{self.agent.name} sending {message_type} to task results topic.")

            # Use publish_event_message for publishing results
            await self.publish_event_message(
                topic_name=self.task_results_topic_name,
                pubsub_name=self.message_bus_name,
                source=self.agent.name,
                message_type=message_type,
                message=message,
                **kwargs,
            )

        except Exception as e:
            logger.error(f"Failed to publish results message of type {message_type} to topic '{self.task_results_topic_name}': {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error publishing results message: {str(e)}")
    
    async def process_agent_message(self, request: Request) -> Response:
        """
        Processes messages sent directly to the agent's topic.
        Handles various message types (e.g., TriggerAction) and broadcasts responses if necessary.

        Args:
            request (Request): The incoming pub/sub request containing a task.

        Returns:
            Response: The agent's response after processing the task.
        """
        try:
            # Parse the incoming CloudEvent
            body = await request.body()
            headers = request.headers
            event: CloudEvent = from_http(dict(headers), body)

            message_type = event.get("type")
            message_data: dict = event.data
            source = event.get("source")

            logger.info(f"{self.agent.name} received {message_type} message from {source}.")

            # Extract workflow_instance_id from headers if available
            workflow_instance_id = headers.get("workflow_instance_id")

            # Handle TriggerAction type messages
            if message_type == "TriggerActionMessage":
                task = message_data.get("task")

                # Execute the task, defaulting to the agent's internal memory if no task is provided
                if not task:
                    logger.info(f"{self.agent.name} running a task from memory.")

                # Execute the task or no input
                response = await self.invoke_task(task)

                # Prepare and broadcast the result as AgentActionResultMessage
                response_message = UserMessage(
                    name=self.agent.name,
                    content=response.body.decode()
                )

                # Broadcast results to all agents
                await self.publish_message_to_all(
                    message_type="ActionResponseMessage",
                    message=response_message.model_dump()
                )

                # Send results to task results topic
                additional_metadata = {'ttlInSeconds': '120'}
                if workflow_instance_id:
                    additional_metadata["event_name"] = "AgentCompletedTask"
                    additional_metadata["workflow_instance_id"] = workflow_instance_id
                
                await self.publish_results_message(
                    message_type="ActionResponseMessage",
                    message=response_message.model_dump(),
                    **additional_metadata
                )

                return response

            else:
                # Log unsupported message types
                logger.warning(f"Unsupported message type '{message_type}' received by {self.agent.name}.")
                return Response(content=json.dumps({"error": f"Unsupported message type: {message_type}"}), status_code=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            logger.error(f"Error processing agent message: {e}", exc_info=True)
            return Response(content=json.dumps({"error": f"Error processing message: {str(e)}"}), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    async def process_broadcast_message(self, request: Request) -> Response:
        """
        Processes a message from the broadcast topic.
        Ensures the agent does not process messages sent by itself,
        and adds user messages to both the agent's memory and the actor's state.

        Args:
            request (Request): The incoming broadcast request.

        Returns:
            Response: Acknowledgment of the broadcast processing.
        """
        try:
            # Parse the CloudEvent from the request
            body = await request.body()
            headers = request.headers
            event: CloudEvent = from_http(dict(headers), body)
            
            message_type = event.get("type")
            broadcast_message: dict = event.data
            source = event.get("source")
            message_content = broadcast_message.get("content")
            
            if not message_content:
                logger.warning(f"Broadcast message missing 'content': {broadcast_message}")
                return Response(content="Invalid broadcast message: 'content' is required.", status_code=status.HTTP_400_BAD_REQUEST)

            # Ignore messages sent by this agent (based on CloudEvent source)
            if source == self.agent.name:
                logger.info(f"{self.agent.name} ignored its own broadcast message of type {message_type}.")
                return Response(status_code=status.HTTP_204_NO_CONTENT)

            # Log and process the valid broadcast message
            logger.info(f"{self.agent.name} is processing broadcast message {message_type} from '{source}'")
            logger.debug(f"Message: {message_content}")

            # Add the message to the agent's memory
            self.agent.memory.add_message(broadcast_message)

            # Add the message to the actor's state
            actor_message = AgentActorMessage(**broadcast_message)
            await self.add_message(actor_message)

            return Response(content="Broadcast message added to memory and actor state.", status_code=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error processing broadcast message: {e}", exc_info=True)
            return Response(content=f"Error processing message: {str(e)}", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)