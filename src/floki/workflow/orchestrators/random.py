from floki.workflow.agentic import AgenticWorkflow
from floki.types import DaprWorkflowContext, BaseMessage
from typing import Any, Optional
from dataclasses import dataclass
from datetime import timedelta
from pydantic import BaseModel
import random
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

class RandomOrchestrator(AgenticWorkflow):
    """
    Implements a random workflow where agents are selected randomly to perform tasks.
    The workflow iterates through conversations, selecting a random agent at each step.

    Uses `continue_as_new` to persist iteration state.
    """
    def model_post_init(self, __context: Any) -> None:
        """
        Initializes and configures the random workflow service.
        Registers tasks and workflows, then starts the workflow runtime.
        """
        super().model_post_init(__context)

        self.workflow_name = "random_workflow"

        # Register workflows and tasks
        self.workflow(self.random_workflow, name=self.workflow_name)
        # Custom tasks
        self.task(self.process_input)
        self.task(self.broadcast_input_message)
        self.task(self.select_random_speaker)
        self.task(self.trigger_agent)

    def random_workflow(self, ctx: DaprWorkflowContext, input: ChatLoop):
        """
        Executes a random workflow where agents are selected randomly for interactions.

        Workflow Steps:
        1. Processes input and broadcasts the initial message.
        2. Iterates through agents, selecting a random speaker each round.
        3. Waits for agent responses or handles timeouts.
        4. Updates the workflow state and continues the loop.
        5. Terminates when max iterations are reached.

        Uses `continue_as_new` to persist iteration state.

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
            logger.info(f"Random workflow iteration {iteration + 1} started (Instance ID: {instance_id}).")

        # Check Termination Condition
        if iteration >= self.max_iterations:
            logger.info(f"Max iterations reached. Ending random workflow (Instance ID: {instance_id}).")
            return message

        # First iteration: Process input and broadcast
        if iteration == 0:
            message_input = yield ctx.call_activity(self.process_input, input={"message": message})
            logger.info(f"Initial message from {message_input['role']} -> {self.name}")

            # Broadcast initial message
            yield ctx.call_activity(self.broadcast_input_message, input=message_input)

        # Select a random speaker
        random_speaker = yield ctx.call_activity(self.select_random_speaker, input={"iteration": iteration})

        # Trigger agent
        yield ctx.call_activity(self.trigger_agent, input={"name": random_speaker, "instance_id": instance_id})

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

        # Update ChatLoop for next iteration
        input["message"] = task_results
        input["iteration"] = iteration + 1

        # Restart workflow with updated state
        ctx.continue_as_new(input)

    async def process_input(self, message: str):
        """
        Processes the input message for the workflow.

        Args:
            message (str): The user-provided input message.
        Returns:
            dict: Serialized UserMessage with the content.
        """
        return {"role": "user", "content": message}

    async def broadcast_input_message(self, **kwargs):
        """
        Broadcasts a message to all agents.

        Args:
            **kwargs: The message content and additional metadata.
        """
        message = {key: value for key, value in kwargs.items()}
        await self.broadcast_message(message=BaseMessage(**message))

    async def select_random_speaker(self, iteration: int) -> str:
        """
        Selects a random speaker, ensuring that a different agent is chosen if possible.

        Args:
            iteration (int): The current iteration number.
        Returns:
            str: The name of the randomly selected agent.
        """
        agents_metadata = await self.get_agents_metadata()
        if not agents_metadata:
            logger.warning("No agents available for selection.")
            raise ValueError("Agents metadata is empty. Cannot select a random speaker.")

        agent_names = list(agents_metadata.keys())

        # Handle single-agent scenarios
        if len(agent_names) == 1:
            random_speaker = agent_names[0]
            logger.info(f"Only one agent available: {random_speaker}. Using the same agent.")
            return random_speaker

        # Select a random speaker, avoiding repeating the previous speaker when possible
        previous_speaker = getattr(self, "current_speaker", None)
        if previous_speaker in agent_names and len(agent_names) > 1:
            agent_names.remove(previous_speaker)

        random_speaker = random.choice(agent_names)
        self.current_speaker = random_speaker
        logger.info(f"{self.name} randomly selected agent {random_speaker} (Iteration: {iteration}).")
        return random_speaker

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