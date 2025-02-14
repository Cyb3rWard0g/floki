from floki.types.workflow import LLMWorkflowState, LLMWorkflowEntry, LLMWorkflowMessage
from floki.agent.workflows.base import AgenticWorkflowService
from floki.types import DaprWorkflowContext, BaseMessage
from floki.llm import LLMClientBase, OpenAIChatClient
from floki.prompt import ChatPromptTemplate
from typing import Dict, Any, Optional
from dataclasses import dataclass
from datetime import timedelta
from datetime import datetime
from pydantic import BaseModel, Field
import logging

logger = logging.getLogger(__name__)

class TriggerActionMessage(BaseModel):
    """
    Represents a message used to trigger an agent's activity within the workflow.
    """
    task: Optional[str] = None

@dataclass
class ChatLoop:
    """
    Represents the state of the chat loop, which is updated after each iteration.

    Attributes:
        message (str): The latest message in the conversation.
        iteration (int): The current iteration of the workflow loop.
    """
    message: str
    iteration: int

class LLMOrchestrator(AgenticWorkflowService):
    """
    Implements an agentic workflow where an LLM dynamically selects the next speaker.
    The workflow iterates through conversations, updating its state and persisting messages.

    Uses the `continue_as_new` pattern to restart the workflow with updated input at each iteration.
    """
    llm: Optional[LLMClientBase] = Field(default_factory=OpenAIChatClient, description="Language model client for generating responses.")

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes and configures the LLM-based workflow service.
        """

        # Initializes local LLM Workflow State
        self.state = LLMWorkflowState()

        super().model_post_init(__context)

        self.workflow_name = "llm_workflow"

        # Register workflows and tasks
        self.workflow(self.llm_workflow, name=self.workflow_name)
        # Custom tasks
        self.task(self.initialize_wf_state)
        self.task(self.update_wf_state)
        self.task(self.process_input)
        self.task(self.broadcast_input_message)
        self.task(self.select_next_speaker)
        self.task(self.trigger_agent)

    def llm_workflow(self, ctx: DaprWorkflowContext, input: ChatLoop):
        """
        Executes a workflow where an LLM selects the next speaker dynamically.
        Uses `continue_as_new` to persist iteration state.

        Workflow Steps:
        1. Initializes workflow state on first iteration.
        2. Broadcasts the initial message to all agents.
        3. Selects the next speaker using an LLM.
        4. Triggers the selected agent's response.
        5. Waits for an agent's response or handles a timeout.
        6. Updates the workflow state and continues the loop.

        The workflow runs until the `max_iterations` threshold is reached.

        Args:
            ctx (DaprWorkflowContext): The workflow execution context.
            input (ChatLoop): The current workflow state containing `message` and `iteration`.

        Returns:
            str: The last processed message when the workflow terminates.
        """
        message = input.get("message")
        iteration = input.get("iteration", 0)
        instance_id = ctx.instance_id

        if not ctx.is_replaying:
            logger.info(f"Workflow iteration {iteration + 1} started (Instance ID: {instance_id}).")

        # Check Termination Condition
        if iteration >= self.max_iterations:
            logger.info(f"Max iterations reached. Ending LLM-driven workflow (Instance ID: {instance_id}).")
            # Record Output
            yield ctx.call_activity(self.update_wf_state, input={"instance_id": instance_id, "message": message})
            # Return output to workflow
            return message

        # First iteration: Initialze workflow state, process initial input and broadcast
        if iteration == 0:
            yield ctx.call_activity(self.initialize_wf_state, input={"instance_id": instance_id, "message": message})

            message_input = yield ctx.call_activity(self.process_input, input=input)
            logger.info(f"Initial message from {message_input['role']} -> {self.name}")

            # Add message to history
            yield ctx.call_activity(self.update_wf_state, input={"instance_id": instance_id, "message": message_input})

            # Broadcast initial message
            yield ctx.call_activity(self.broadcast_input_message, input=message_input)
        
        # Select next speaker using LLM
        next_speaker = yield ctx.call_activity(self.select_next_speaker, input={"iteration": iteration + 1, "instance_id": instance_id})

        # Trigger agent
        yield ctx.call_activity(self.trigger_agent, input={"name": next_speaker, "instance_id": instance_id})

        # Wait for response or timeout
        logger.info("Waiting for agent response...")
        event_data = ctx.wait_for_external_event("AgentCompletedTask")
        timeout_task = ctx.create_timer(timedelta(seconds=self.timeout))
        any_results = yield self.when_any([event_data, timeout_task])
        
        if any_results == timeout_task:
            logger.warning(f"Agent response timed out (Iteration: {iteration + 1}, Instance ID: {instance_id}).")
            task_results = {"name": "timeout", "content": "Timeout occurred. Continuing..."}
        else:
            task_results = yield event_data
            logger.info(f"{task_results['name']} -> {self.name}")

            # Record agent response in history
            yield ctx.call_activity(self.update_wf_state, input={"instance_id": instance_id, "message": task_results})
        
        # Update ChatLoop
        input["message"] = task_results
        input["iteration"] = iteration + 1

        # Restart workflow with updated state
        ctx.continue_as_new(input)

    async def initialize_wf_state(self, instance_id: str, message: str):
        """
        Initializes the workflow state for a new instance.

        Args:
            instance_id (str): The ID of the workflow instance.
            message (str): The initial input message for the workflow.
        """
        workflow_entry = LLMWorkflowEntry(input=message)
        self.state["instances"][instance_id] = workflow_entry.model_dump(mode="json")
        self.save_state()
    
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
        Broadcasts a message to all registered agents.

        Args:
            **kwargs: The message content and additional metadata.
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
        agents_metadata: dict = await self.get_agents_metadata()
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
        workflow_entry = self.state["instances"].get(instance_id)
        message_history = workflow_entry["messages"] if workflow_entry and workflow_entry["messages"] else []

        # Format the historical messages
        formatted_messages = [
            {"role": msg["role"], "content": msg["content"], "name": msg["name"]} if msg["name"] else {"role": msg["role"], "content": msg["content"]}
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
    
    def update_wf_state(self, instance_id: str, message: Optional[Dict[str, Any]] = None, final_output: Optional[str] = None) -> None:
        """
        Updates the workflow state by appending a new message or marking the workflow as completed.

        This function retrieves the workflow entry associated with `instance_id` and updates:
        - If `message` is provided, it adds it to the conversation history and updates `last_message`.
        - If `final_output` is provided (indicating workflow completion), it updates `output` and `end_time`.
        
        Args:
            instance_id (str): The unique identifier of the workflow instance.
            message (Optional[Dict[str, Any]]): The message to be recorded in the workflow state.
            final_message (Optional[str]): The final output message indicating the workflow's completion.
        
        Side Effects:
            - Modifies the workflow entry in `self.state["instances"]`.
            - Updates iteration count if a new message is added.
            - Persists the updated state using `self.save_state()`.
        """
        
        # Retrieve or initialize the workflow entry
        workflow_entry = self.state["instances"].get(instance_id)

        if not workflow_entry:
            raise ValueError(f"No workflow entry found for instance_id: {instance_id}")

        if message:
            # Convert the message into an LLMWorkflowMessage object
            workflow_message = LLMWorkflowMessage(**message).model_dump(mode="json")

            # Update the message history
            workflow_entry["messages"].append(workflow_message)

            # Update last_message for quick access
            workflow_entry["last_message"] = workflow_message

        if final_output:
            # Mark the workflow as completed
            workflow_entry["output"] = final_output
            workflow_entry["end_time"] = datetime.now()

        # Save state
        self.save_state()