from floki.agent.workflows.base import AgenticWorkflowService
from floki.types import DaprWorkflowContext, BaseMessage
from typing import Dict, Any, Optional
from datetime import timedelta
from pydantic import BaseModel
import random
import logging

logger = logging.getLogger(__name__)

class TriggerActionMessage(BaseModel):
    task: Optional[str] = None

class RandomWorkflowService(AgenticWorkflowService):
    """
    Implements a random workflow where agents are selected randomly to perform tasks.
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
        # Built-in tasks
        self.task(self.add_message)
        # Custom tasks
        self.task(self.process_input)
        self.task(self.broadcast_input_message)
        self.task(self.select_random_speaker)
        self.task(self.trigger_agent)
    
    def random_workflow(self, ctx: DaprWorkflowContext, input: Dict[str, Any]):
        """
        Executes a random workflow where agents are selected randomly for interactions.

        Steps:
        1. Processes input and broadcasts the initial message.
        2. Iterates through random agent selections per round.
        3. Waits for agent responses or handles timeouts.
        4. Logs interactions until max iterations are reached.
        """
        # Save Instance ID
        instance_id = ctx.instance_id

        # Process Input
        message_input = yield ctx.call_activity(self.process_input, input=input)

        logger.info(f"Initial message from {message_input['role']} -> {self.name}")
        
        # Broadcast first message
        yield ctx.call_activity(self.broadcast_input_message, input=message_input)

        # Start Iteration
        for i in range(self.max_iterations):
            logger.info(f"Random workflow iteration {i + 1}/{self.max_iterations} (Instance ID: {instance_id}).")

            # Select a random speaker
            random_speaker = yield ctx.call_activity(self.select_random_speaker, input={"iteration": i + 1})

            # Trigger the agent activity for the randomly selected speaker
            yield ctx.call_activity(
                self.trigger_agent, 
                input={"name": random_speaker, "instance_id": ctx.instance_id}
            )

            # Block the workflow on either agent response or a timeout
            logger.info("Waiting for agent response...")
            event_data = ctx.wait_for_external_event("AgentCompletedTask")
            timeout_expired_task = ctx.create_timer(timedelta(seconds=self.workflow_timeout))
            any_results = yield self.when_any([event_data, timeout_expired_task])
            
            # Handle timeout or log response
            if any_results == timeout_expired_task:
                logger.warning(f"Agent task execution timed out (Iteration: {i + 1}, Instance ID: {instance_id}).")
            else:
                task_results = yield event_data
                logger.info(f"{task_results['name']} -> {self.name}")
                yield ctx.call_activity(
                    self.add_message, 
                    input={"instance_id": instance_id, "message": task_results}
                )
        
        logger.info(f"Max iterations reached. Ending random workflow (Instance ID: {instance_id}).")

        # Return last output
        return task_results
    
    async def process_input(self, message: str):
        """
        Validates and serializes the input message for the workflow.

        Args:
            message (str): The input message to process.
        Returns:
            dict: Serialized UserMessage.
        """
        return {"role":"user", "content": message}

    async def broadcast_input_message(self, **kwargs):
        """
        Broadcasts a message to all agents.
        """
        message = {key: value for key, value in kwargs.items()}
        await self.broadcast_message(message=BaseMessage(**message))
    
    async def select_random_speaker(self, iteration: int) -> str:
        """
        Selects a random speaker, avoiding the current speaker if possible.

        Returns:
            str: The name of the randomly selected agent.
        """
        agents_metadata = await self.get_agents_metadata()
        if not agents_metadata:
            logger.warning("No agents available for selection.")
            raise ValueError("Agents metadata is empty. Cannot select a random speaker.")

        agent_names = [name for name in agents_metadata.keys()]

        # Handle single-agent scenarios
        if len(agent_names) == 1:
            random_speaker = agent_names[0]
            self.current_speaker = random_speaker
            logger.info(f"Only one agent available: {random_speaker}. Continuing with the same agent.")
            return random_speaker

        # Avoid selecting the current speaker if multiple agents are available
        if len(agent_names) > 1 and self.current_speaker in agent_names:
            agent_names.remove(self.current_speaker)

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