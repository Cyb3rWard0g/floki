from dapr.actor.runtime.config import ActorRuntimeConfig, ActorTypeConfig, ActorReentrancyConfig
from fastapi import FastAPI, HTTPException, Response, status
from floki.storage.daprstores.statestore import DaprStateStore
from floki.agent.actor import AgentActorBase, AgentActorInterface
from floki.service.fastapi import DaprEnabledService
from floki.types.agent import AgentActorMessage
from floki.agent import AgentBase
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from dapr.actor.runtime.runtime import ActorRuntime
from dapr.ext.fastapi import DaprActor
from dapr.actor import ActorProxy, ActorId
from pydantic import Field, model_validator, ConfigDict
from contextlib import asynccontextmanager
from datetime import timedelta
from typing import Optional, Any, Callable, TypeVar, Union, List
import json
import logging

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=Callable)

class AgentServiceBase(DaprEnabledService):
    """
    A Pydantic-based class for managing services and exposing FastAPI routes with Dapr pub/sub and actor support.
    """

    agent: AgentBase
    agent_topic_name: Optional[str] = Field(None, description="The topic name dedicated to this specific agent, derived from the agent's name if not provided.")
    broadcast_topic_name: str = Field("beacon_channel", description="The default topic used for broadcasting messages to all agents.")
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

        # Registering App routes and subscriping to topics dynamically
        self.register_message_routes()

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
    
    def register_message_routes(self) -> None:
        """
        Dynamically register message handlers and the Dapr /subscribe endpoint.
        """
        for method_name in dir(self):
            method = getattr(self, method_name, None)
            if callable(method) and hasattr(method, "_is_message_handler"):
                # Retrieve metadata from the decorator
                router_data = method._message_router_data
                pubsub_name = router_data.get("pubsub") or self.message_bus_name
                is_broadcast = router_data.get("is_broadcast", False)

                # Dynamically assign topic_name to self.agent.name if not explicitly provided
                topic_name = router_data.get("topic")
                if not topic_name and not is_broadcast:
                    topic_name = self.agent.name
                elif is_broadcast:
                    topic_name = self.broadcast_topic_name

                # Extend route with method name for uniqueness
                route = router_data.get("route") or (f"/events/{pubsub_name}/{topic_name}/{method_name}" if topic_name else None)
                message_type = router_data.get("message_type")
                dead_letter_topic = router_data.get("dead_letter_topic")
                custom_rules = router_data.get("rules")

                # Validation: Ensure a route is provided for standalone handlers
                if not route:
                    raise ValueError(
                        f"Method '{method_name}' must define a 'route' or be tied to a pub/sub topic."
                    )

                # Register the route in FastAPI
                self.app.add_api_route(route, method, methods=["POST"], tags=["PubSub"])
                handler_type = "broadcast" if is_broadcast else "standard"
                logger.info(f"Registered {handler_type} POST route for '{method_name}'")
                logger.info(f"Route: {route}")

                # Register the subscription only if `topic_name` is explicitly provided or set dynamically
                if topic_name:
                    subscription = next(
                        (sub for sub in self.dapr_app._subscriptions if sub["pubsubname"] == pubsub_name and sub["topic"] == topic_name),
                        None,
                    )
                    if subscription is None:
                        subscription = {
                            "pubsubname": pubsub_name,
                            "topic": topic_name,
                            "routes": {"rules": []},  # Default route removed
                            **({"deadLetterTopic": dead_letter_topic} if dead_letter_topic else {}),
                        }
                        self.dapr_app._subscriptions.append(subscription)

                    # Add routing rule for `message_type` or `rules`
                    if isinstance(message_type, list):
                        # Use JSON formatting for CEL list rules
                        rule = {"match": f"event.type in {json.dumps(message_type)}", "path": route}
                        subscription["routes"]["rules"].append(rule)
                    elif message_type:
                        rule = {"match": f"event.type == '{message_type}'", "path": route}
                        subscription["routes"]["rules"].append(rule)
                    elif custom_rules:
                        rule = {"match": custom_rules["match"], "path": route}
                        subscription["routes"]["rules"].append(rule)

                    logger.info(
                        f"Subscribed '{method_name}' to topic '{topic_name}' with "
                        f"rules '{custom_rules or f'event.type == {message_type}'}'"
                    )
        
        logger.debug(f"Subscription Routes: {json.dumps(self.dapr_app._get_subscriptions(), indent=2)}")

def message_router(
    pubsub: Optional[str] = None,
    topic: Optional[str] = None,
    message_type: Optional[Union[str, List[str]]] = None,
    route: Optional[str] = None,
    dead_letter_topic: Optional[str] = None,
    broadcast: bool = False,
    rules: Optional[dict] = None,
) -> Callable[[T], T]:
    def decorator(func: T) -> T:
        # Attach metadata for dynamic registration
        func._is_message_handler = True
        func._message_router_data = {
            "pubsub": pubsub,
            "topic": topic if not broadcast else None,
            "message_type": message_type,
            "route": route,
            "dead_letter_topic": dead_letter_topic,
            "rules": rules,
            "is_broadcast": broadcast,
        }

        # Validation
        if not message_type and not route and not rules and not broadcast:
            raise ValueError(
                "If 'message_type', 'rules', or 'broadcast' are not specified, a 'route' must be defined."
            )
        if broadcast and route is None:
            # Allow broadcast handlers to omit route, as it can be dynamically set
            pass
        elif rules and not isinstance(rules, dict):
            raise ValueError(
                "'rules' must be a dictionary with valid CEL match logic. See: https://github.com/google/cel-spec"
            )

        return func

    return decorator