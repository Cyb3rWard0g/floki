from floki.storage.daprstores.statestore import DaprStateStore
from floki.agent.utils.text_printer import ColorTextFormatter
from floki.workflow.service import WorkflowAppService
from fastapi import HTTPException
from pydantic import BaseModel, Field
from typing import Any, Optional, Union
import logging

logger = logging.getLogger(__name__)

class AgenticWorkflowService(WorkflowAppService):
    """
    Abstract base class for agentic workflows, providing a template for common workflow operations.
    """
    broadcast_topic_name: Optional[str] = Field("beacon_channel", description="The default topic used for broadcasting messages to all agents.")
    task_results_topic_name: Optional[str] = Field("task_results_channel", description="The default topic used for sending the results of a task executed by an agent.")
    agents_state_store_name: str = Field(..., description="The name of the Dapr state store component used to store and share agent metadata centrally.")
    max_iterations: int = Field(default=10,description="Maximum number of iterations for workflows. Must be greater than 0.", ge=1)

    # Fields initialized later
    agent_metadata_store: Optional[DaprStateStore] = Field(default=None, init=False, description="Dapr state store instance for accessing and managing centralized agent metadata.")
    formatter: Optional[ColorTextFormatter] = Field(default=None, init=False, description="Formatter for text output.")
    current_speaker: Optional[str] = Field(default=None, init=False, description="Current speaker in the conversation.")

    def model_post_init(self, __context: Any) -> None:
        """
        Configure workflows and initialize AgentService and WorkflowApp.
        """
        super().model_post_init(__context)

        # Initialize the Dapr state store for agent metadata
        self.agent_metadata_store = DaprStateStore(store_name=self.agents_state_store_name, address=self.daprGrpcAddress)

        # Initialize the text formatter
        self.formatter = ColorTextFormatter()

        # Subscribe to tasks results topic
        self.dapr_app.subscribe(pubsub=self.message_bus_name, topic=self.task_results_topic_name)(self.raise_workflow_event_from_request)
        logger.info(f"{self.name} subscribed to topic {self.task_results_topic_name} on {self.message_bus_name}")
    
    async def get_agents_metadata(self) -> dict:
        """
        Retrieve metadata for all agents registered.
        """
        key = "agents_metadata"
        try:
            agents_metadata = await self.get_metadata_from_store(self.agent_metadata_store, key) or {}
            return agents_metadata
        except Exception as e:
            logger.error(f"Failed to retrieve agents metadata: {e}")
            return {}
    
    async def broadcast_message(self, message: Union[BaseModel, dict], **kwargs) -> None:
        """
        Sends a message to all agents.

        Args:
            message (Union[BaseModel, dict]): The message content as a Pydantic model or dictionary.
            **kwargs: Additional metadata fields to include in the message.
        """
        try:
            agents_metadata = await self.get_agents_metadata()
            if not agents_metadata:
                logger.warning("No agents available for broadcast.")
                return

            logger.info(f"{self.name} preparing to broadcast message to all agents.")

            await self.publish_event_message(
                topic_name=self.broadcast_topic_name,
                pubsub_name=self.message_bus_name,
                source=self.name,
                message=message,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Failed to broadcast message: {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error broadcasting message: {str(e)}")

    async def send_message_to_agent(self, name: str, message: Union[BaseModel, dict], **kwargs) -> None:
        """
        Sends a message to a specific agent.

        Args:
            name (str): The name of the target agent.
            message (Union[BaseModel, dict]): The message content as a Pydantic model or dictionary.
            **kwargs: Additional metadata fields to include in the message.
        """
        try:
            agents_metadata = await self.get_agents_metadata()
            if name not in agents_metadata:
                raise HTTPException(status_code=404, detail=f"Agent {name} not found.")

            agent_metadata = agents_metadata[name]
            logger.info(f"{self.name} preparing to send message to agent '{name}'.")

            await self.publish_event_message(
                topic_name=agent_metadata["topic_name"],
                pubsub_name=agent_metadata["pubsub_name"],
                source=self.name,
                message=message,
                **kwargs,
            )
        except Exception as e:
            logger.error(f"Failed to send message to agent '{name}': {e}", exc_info=True)
            raise HTTPException(status_code=500, detail=f"Error sending message to agent '{name}': {str(e)}")
    
    def print_interaction(self, sender_agent_name: str, recipient_agent_name: str, message: str) -> None:
        """
        Prints the interaction between two agents in a formatted and colored text.

        Args:
            sender_agent_name (str): The name of the agent sending the message.
            recipient_agent_name (str): The name of the agent receiving the message.
            message (str): The message content to display.
        """
        separator = "-" * 80
        
        # Print sender -> recipient and the message
        interaction_text = [
            (sender_agent_name, "floki_mustard"),
            (" -> ", "floki_teal"),
            (f"{recipient_agent_name}\n\n", "floki_mustard"),
            (message + "\n\n", None),
            (separator + "\n", "floki_teal"),
        ]

        # Print the formatted text
        self.formatter.print_colored_text(interaction_text)