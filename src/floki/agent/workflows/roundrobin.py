from floki.agent.workflows.base import AgenticWorkflowService
from floki.types import DaprWorkflowContext, BaseMessage
from typing import Dict, Any, Optional
from datetime import timedelta
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class TriggerActionMessage(BaseModel):
    task: Optional[str] = None

class RoundRobinWorkflowService(AgenticWorkflowService):
    """
    Implements a round-robin workflow where agents take turns performing tasks.
    """
    def model_post_init(self, __context: Any) -> None:
        """
        Initializes and configures the round-robin workflow service.
        Registers tasks and workflows, then starts the workflow runtime.
        """
        super().model_post_init(__context)
        
        self.workflow_name = "round_robin_workflow"

        # Register workflows and tasks
        self.workflow(self.round_robin_workflow, name=self.workflow_name)
        # Built-in tasks
        self.task(self.add_message)
        # Custom tasks
        self.task(self.process_input)
        self.task(self.broadcast_input_message)
        self.task(self.select_next_speaker)
        self.task(self.trigger_agent)
    
    def round_robin_workflow(self, ctx: DaprWorkflowContext, input: Dict[str, Any]):
        """
        Executes a round-robin workflow where agents interact iteratively.

        Steps:
        1. Processes input and broadcasts the initial message.
        2. Iterates through agents, selecting a speaker each round.
        3. Waits for agent responses or handles timeouts.
        4. Logs interactions until max iterations are reached.
        """
        # Save Instance ID
        instance_id = ctx.instance_id

        # Record the initial input message
        message = yield ctx.call_activity(self.process_input, input=input)

        # Record the first message in the workflow
        yield ctx.call_activity(self.add_message, input={"instance_id": instance_id, "message": message})

        logger.info(f"{message['role']} -> {self.name}")
        
        # Broadcast first message
        yield ctx.call_activity(self.broadcast_input_message, input=message)

        # Start Iteration
        for i in range(self.max_iterations):
            logger.info(f"Round-robin iteration {i + 1}/{self.max_iterations}.")
            
            # Select the next speaker
            next_speaker = yield ctx.call_activity(self.select_next_speaker, input={"iteration": i + 1})
            
            # Trigger the agent activity for the selected speaker
            yield ctx.call_activity(self.trigger_agent, input={"name": next_speaker, "instance_id": instance_id})

            # Block the workflow on either agent response or a timeout
            logger.info("Waiting for agent response...")
            event_data = ctx.wait_for_external_event("AgentCompletedTask")
            timeout_expired_task = ctx.create_timer(timedelta(seconds=self.workflow_timeout))
            any_results = yield self.when_any([event_data, timeout_expired_task])
            
            # Handle timeout or log response
            if any_results == timeout_expired_task:
                logger.info("Agent task execution timed out.")
            else:
                task_results = yield event_data
                yield ctx.call_activity(self.add_message, input={"instance_id": instance_id, "message": task_results})
                logger.info(f"{task_results['name']} -> {self.name}")
        
        # End the workflow and record the final output
        logger.info("Max iterations reached. Ending round-robin workflow.")

        # Return last output
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
    
    async def select_next_speaker(self, iteration: int) -> str:
        """
        Selects the next speaker in round-robin order, avoiding the current speaker if possible.

        Args:
            iteration (int): The current iteration number.
        Returns:
            str: The name of the selected agent.
        """
        agents_metadata = await self.get_agents_metadata()
        if not agents_metadata:
            logger.warning("No agents available for selection.")
            raise ValueError("Agents metadata is empty. Cannot select next speaker.")

        agent_names = [name for name in agents_metadata.keys()]

        # Handle single-agent scenarios
        if len(agent_names) == 1:
            next_speaker = agent_names[0]
            self.current_speaker = next_speaker
            logger.info(f"Only one agent available: {next_speaker}. Continuing with the same agent.")
            return next_speaker

        # Avoid selecting the current speaker if possible
        if self.current_speaker in agent_names:
            agent_names.remove(self.current_speaker)
        
        next_speaker = agent_names[iteration % len(agent_names)]
        self.current_speaker = next_speaker
        logger.info(f"{self.name} selected agent {next_speaker} for iteration {iteration}.")
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