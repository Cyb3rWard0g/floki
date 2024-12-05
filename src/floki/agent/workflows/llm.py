from floki.agent.workflows.base import AgenticWorkflowService
from floki.types import DaprWorkflowContext, BaseMessage
from floki.llm import LLMClientBase, OpenAIChatClient
from floki.prompt import ChatPromptTemplate
from typing import Dict, Any, Optional
from datetime import timedelta
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class TriggerActionMessage(BaseModel):
    task: Optional[str] = None

class LLMWorkflowService(AgenticWorkflowService):
    """
    Implements a workflow where the next speaker is dynamically selected using an LLM.
    """
    llm: Optional[LLMClientBase] = Field(default_factory=OpenAIChatClient, description="Language model client for generating responses.")

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes and configures the LLM-based workflow service.
        """
        super().model_post_init(__context)

        self.workflow_name = "llm_workflow"

        # Register workflows and tasks
        self.workflow(self.llm_workflow, name=self.workflow_name)
        # Built-in tasks
        self.task(self.add_message)
        # Custom tasks
        self.task(self.process_input)
        self.task(self.broadcast_input_message)
        self.task(self.select_next_speaker)
        self.task(self.trigger_agent)

    def llm_workflow(self, ctx: DaprWorkflowContext, input: Dict[str, Any]):
        """
        Executes a workflow where an LLM selects the next speaker dynamically.

        Steps:
        1. Processes input and broadcasts the initial message.
        2. Iterates through agents, selecting the next speaker using an LLM.
        3. Waits for agent responses or handles timeouts.
        4. Logs interactions until max iterations are reached.
        """
        # Save Instance ID
        instance_id = ctx.instance_id

        # Process the input
        message_input = yield ctx.call_activity(self.process_input, input=input)

        logger.info(f"Initial message from {message_input['role']} -> {self.name}")
        
        # Add the initial message to the workflow history
        yield ctx.call_activity(self.add_message, input={"instance_id": instance_id, "message": message_input})

        # Broadcast the initial message
        yield ctx.call_activity(self.broadcast_input_message, input=message_input)

        # Start the LLM-driven conversation loop
        for i in range(self.max_iterations):
            logger.info(f"LLM iteration {i + 1}/{self.max_iterations} (Instance ID: {instance_id}).")

            # Select the next speaker using the LLM
            next_speaker = yield ctx.call_activity(self.select_next_speaker, input={"iteration": i + 1, "instance_id": instance_id})
            
            # Trigger the next speaker's activity
            yield ctx.call_activity(self.trigger_agent, input={"name": next_speaker, "instance_id": instance_id})

            # Wait for response or timeout
            logger.info("Waiting for agent response...")
            event_data = ctx.wait_for_external_event("AgentCompletedTask")
            timeout_task = ctx.create_timer(timedelta(seconds=self.workflow_timeout))
            any_results = yield self.when_any([event_data, timeout_task])

            if any_results == timeout_task:
                logger.warning(f"Agent response timed out (Iteration: {i + 1}, Instance ID: {instance_id}).")
            else:
                task_results = yield event_data
                logger.info(f"{task_results['name']} -> {self.name}")

                # Record the agent's response in the workflow history
                yield ctx.call_activity(self.add_message, input={"instance_id": instance_id, "message": task_results})

        logger.info(f"Max iterations reached. Ending LLM-driven workflow (Instance ID: {instance_id}).")

        # Return the last output
        return task_results

    async def process_input(self, message: str):
        """
        Processes the input message for the workflow.

        Args:
            message (str): The user-provided input message.
        Returns:
            dict: Serialized UserMessage with the content.
        """
        return {"role":"user", "content": message}

    async def broadcast_input_message(self, **kwargs):
        """
        Broadcasts a message to all agents.
        """
        message = {key: value for key, value in kwargs.items()}
        await self.broadcast_message(message=BaseMessage(**message))

    async def select_next_speaker(self, iteration: int, instance_id: str) -> str:
        """
        Uses the LLM to select the next speaker dynamically.

        Args:
            iteration (int): Current iteration of the conversation.
        Returns:
            str: Name of the selected agent.
        """
        agents_metadata = await self.get_agents_metadata()
        if not agents_metadata:
            logger.warning("No agents available for selection.")
            raise ValueError("No agents found for speaker selection.")

        # Format the list of agents for the prompt
        lines = [
            f"Agent: {name}, Role: {metadata.get('role', 'Unknown role')}, Goal: {metadata.get('goal', 'Unknown goal')}"
            for name, metadata in agents_metadata.items()
        ]
        agents_list = "\n".join(lines)

        # Fetch message history for the current workflow instance
        workflow_entry = self.state.instances.get(instance_id)
        message_history = workflow_entry.messages if workflow_entry and workflow_entry.messages else []

        # Format the historical messages
        formatted_messages = [
            {"role": msg.role, "content": msg.content, "name": msg.name} if msg.name else {"role": msg.role, "content": msg.content}
            for msg in message_history
        ]

        # Build the ChatPromptTemplate
        prompt_messages = [
            {'role': 'system', 'content': f"You are managing a conversation.\nThe agents in this conversation are:\n{agents_list}"}
        ]

        # Extend prompt_messages with the formatted historical messages
        prompt_messages.extend(formatted_messages)

        # Add the final instruction to select the next speaker
        prompt_messages.append(
            {'role': 'system', 'content': f"Based on the chat history, select the next speaker for iteration {iteration}. Return only the name of the selected agent, and nothing else."}
        )

        # Create the prompt from messages
        prompt = ChatPromptTemplate.from_messages(prompt_messages)
        formatted_prompt = prompt.format_prompt()

        logger.info(f"{self.name} selecting next speaker for iteration {iteration}.")

        # Use the LLM to determine the next speaker
        response = self.llm.generate(messages=formatted_prompt)
        next_speaker = response.get_content().strip()
        logger.info(f"LLM selected next speaker: {next_speaker}")
        return next_speaker

    async def trigger_agent(self, name: str, instance_id: str) -> None:
        """
        Triggers the specified agent to perform its activity.

        Args:
            name (str): Name of the agent to trigger.
            instance_id (str): Workflow instance ID for context.
        """
        await self.send_message_to_agent(
            name=name,
            message=TriggerActionMessage(task=None),
            workflow_instance_id=instance_id,
        )