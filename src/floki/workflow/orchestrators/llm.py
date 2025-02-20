from floki.types.workflow import LLMWorkflowState, LLMWorkflowEntry, LLMWorkflowMessage, NextStep, TaskResult, TaskPlan, PlanStep, ProgressCheckOutput
from floki.workflow.agentic import AgenticWorkflow
from floki.workflow.orchestrators.prompts import TASK_PLANNING_PROMPT, TASK_INITIAL_PROMPT, PROGRESS_CHECK_PROMPT, NEXT_STEP_PROMPT
from floki.types import DaprWorkflowContext, BaseMessage
from floki.llm import LLMClientBase, OpenAIChatClient
from typing import Dict, Any, Optional, List
from floki.llm.utils import StructureHandler
from dataclasses import dataclass
from datetime import timedelta
from datetime import datetime
from pydantic import BaseModel, Field
import json
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
    """
    message: str = Field(..., description="The latest message in the conversation.")
    iteration: int = Field(..., description="The current iteration of the workflow loop.")
    verdict: str = Field(..., description="Task status.")

# Output Schemas
plan_schema = json.dumps(StructureHandler.enforce_strict_json_schema(TaskPlan.model_json_schema())["properties"]["plan"])
progress_check_schema = json.dumps(StructureHandler.enforce_strict_json_schema(ProgressCheckOutput.model_json_schema()))
next_step_schema = json.dumps(NextStep.model_json_schema())

class LLMOrchestrator(AgenticWorkflow):
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
        # Bult-in tasks
        self.task(self.get_agents_metadata_as_string)
        # Custom tasks
        self.task(self.init_workflow_state)
        self.task(self.generate_plan, description=TASK_PLANNING_PROMPT)
        self.task(self.broadcast_initial_message)
        self.task(self.generate_progress_check, description=PROGRESS_CHECK_PROMPT, include_chat_history=True)
        self.task(self.update_plan)
        self.task(self.generate_next_step, description=NEXT_STEP_PROMPT, include_chat_history=True)
        self.task(self.broadcast_task_message)
        self.task(self.trigger_agent)
        self.task(self.update_workflow_state)
        self.task(self.update_task_history)

    def llm_workflow(self, ctx: DaprWorkflowContext, input: ChatLoop):
        """
        Executes an LLM-driven agentic workflow where the next agent is dynamically selected
        based on task progress. Manages iterative execution, updating state, handling agent 
        responses, and ensuring task completion.

        Args:
            ctx (DaprWorkflowContext): The workflow execution context.
            input (ChatLoop): The current workflow state containing `message`, `iteration`, and `verdict`.

        Returns:
            str: The final processed message when the workflow terminates.

        Raises:
            RuntimeError: If the LLM determines the task is `failed`.
        """
        # Step 1: Check Termination Conditions from the previous iteration
        message = input.get("message")
        iteration = input.get("iteration", 0)
        instance_id = ctx.instance_id
        plan = self.state["instances"].get(instance_id, {}).get("plan", [])

        if iteration >= self.max_iterations:
            logger.info(f"Workflow ending: Max iterations reached.")
            yield ctx.call_activity(self.update_workflow_state, input={"instance_id": instance_id, "final_output": message})
            return message
        
        if not ctx.is_replaying:
            logger.info(f"Workflow iteration {iteration + 1} started (Instance ID: {instance_id}).")

        # Step 2: Retrieve available agents
        agents = yield ctx.call_activity(self.get_agents_metadata_as_string)

        # Step 3: First iteration setup
        if iteration == 0:
            if not ctx.is_replaying:
                logger.info(f"Initial message from User -> {self.name}")

            # Initialize workflow state with user input
            yield ctx.call_activity(self.init_workflow_state, input={"instance_id": instance_id, "message": message})

            # Generate the plan using an LLM
            plan = yield ctx.call_activity(self.generate_plan, input={"task": message, "agents": agents, "plan_schema": plan_schema})

            # Generate and broadcast initial message
            yield ctx.call_activity(self.broadcast_initial_message, input={"instance_id": instance_id, "task": message, "agents": agents, "plan": plan})
        
        # Step 4: Run Progress Check
        progress = yield ctx.call_activity(self.generate_progress_check, input={"task": message, "plan": plan, "progress_check_schema": progress_check_schema})

        if progress["plan_needs_update"]:
            plan = yield ctx.call_activity(self.update_plan, input={
                "instance_id": instance_id,
                "plan": plan,
                "status_updates": progress.get("plan_status_update", []), 
                "plan_updates": progress.get("plan_restructure", [])
            })

        # Step 5: Run Next Step Logic with current plan
        next_step = yield ctx.call_activity(self.generate_next_step, input={"task": message, "agents": agents, "plan": plan, "next_step_schema": next_step_schema})

        # Extract Verdict from NextStep
        verdict = next_step["verdict"]

        # Step 6: Check Termination Conditions after LLM decision
        if verdict == "completed":
            logger.info("Workflow marked as completed by LLM.")
            yield ctx.call_activity(self.update_workflow_state, input={"instance_id": instance_id, "final_output": message})
            return message

        if verdict == "failed":
            logger.error("LLM determined the task cannot be completed.")
            raise RuntimeError("Workflow execution failed due to unresolved issues.")

        # Extract Additional Properties from NextStep
        next_agent = next_step["next_agent"]
        instruction = next_step["instruction"]

        # Step 7: Broadcast Task to all Agents
        yield ctx.call_activity(self.broadcast_task_message, input={"instance_id": instance_id, "message": instruction})

        # Step 8: Trigger next agent
        yield ctx.call_activity(self.trigger_agent, input={"name": next_agent, "instance_id": instance_id})

        # Step 9: Wait for agent response or timeout
        if not ctx.is_replaying:
            logger.info(f"Waiting for {next_agent}'s response...")
        
        event_data = ctx.wait_for_external_event("AgentCompletedTask")
        timeout_task = ctx.create_timer(timedelta(seconds=self.timeout))
        any_results = yield self.when_any([event_data, timeout_task])
        
        if any_results == timeout_task:
            logger.warning(f"Agent response timed out (Iteration: {iteration + 1}, Instance ID: {instance_id}).")
            task_results = {"name": "timeout", "content": "Timeout occurred. Continuing..."}
        else:
            task_results = yield event_data
            if not ctx.is_replaying:
                logger.info(f"{task_results['name']} sent a response.")

            # Step 10: Record agent response in history
            yield ctx.call_activity(self.update_workflow_state, input={"instance_id": instance_id, "message": task_results})
            
            # Step 11: Save the task execution to task_history
            step_id = next_step.get("step", 0)
            substep_id = next_step.get("substep", 0)

            yield ctx.call_activity(
                self.update_task_history, 
                input={
                    "instance_id": instance_id,
                    "agent": next_agent,
                    "step": step_id,
                    "substep": substep_id,
                    "result": task_results["content"]
                }
            )
        
        # Step 12: Update ChatLoop state and continue workflow
        input["message"] = task_results
        input["iteration"] = iteration + 1

        # Restart workflow with updated state
        ctx.continue_as_new(input)

    async def init_workflow_state(self, instance_id: str, message: str):
        """
        Initializes the workflow state for a new instance.

        Args:
            instance_id (str): The ID of the workflow instance.
            message (str): The initial input message for the workflow.
        """
        workflow_entry = LLMWorkflowEntry(input=message)
        self.state["instances"][instance_id] = workflow_entry.model_dump(mode="json")
        self.save_state()
    
    async def generate_plan(self, task: str, agents: str, plan_schema: str) -> List[PlanStep]:
        """
        Generates a structured execution plan for the given task.

        Args:
            task (str): The description of the task to be executed.
            agents (str): Formatted list of available agents and their roles.
            plan_schema (sr): Schema of the plan

        Returns:
            List[Dict[str, Any]]: A list of steps for the overall plan.
        """
        pass

    async def broadcast_initial_message(self, instance_id: str, task: str, agents: str, plan: List[Dict[str, Any]]):
        """
        Constructs and broadcasts the initial task message to all agents.

        Args:
            instance_id (str): The ID of the workflow instance.
            task (str): Description of the task to be completed.
            agents (str): Formatted list of available agents and their roles.
            plan (List[Dict[str, Any]]): Execution plan outlining agent assignments.
        """
        initial_prompt = TASK_INITIAL_PROMPT.format(task=task, agents=agents, plan=plan)
        message = {"name": self.name, "role": "user", "content": initial_prompt}

        # Update state message history
        await self.update_workflow_state(instance_id=instance_id, message=message, plan=plan)

        # Broadcast message
        initial_message = BaseMessage(**message)
        await self.broadcast_message(message=initial_message)
    
    async def generate_progress_check(self, task: str, plan: str, progress_check_schema: str) -> ProgressCheckOutput:
        """
        Evaluates the current plan's progress and determines necessary updates.

        Args:
            task (str): The current task description.
            plan (str): The structured execution plan.
            progress_check_schema (str): The schema of the progress check

        Returns:
            ProgressCheckOutput: The plan update details, including status changes and restructuring if needed.
        """
        pass

    async def update_plan(self, instance_id: str, plan: List[Dict[str, Any]], status_updates: Optional[List[Dict[str, Any]]] = None, plan_updates: Optional[List[Dict[str, Any]]] = None) -> List[Dict[str, Any]]:
        """
        Updates the execution plan based on status changes and/or plan restructures.

        Args:
            instance_id (str): The workflow instance ID.
            plan (List[Dict[str, Any]]): The current execution plan.
            status_updates (Optional[List[Dict[str, Any]]]): List of updates for step statuses.
            plan_updates (Optional[List[Dict[str, Any]]]): List of full step modifications.

        Returns:
            List[Dict[str, Any]]: The updated execution plan.
        """

        # Step 1: Apply status updates directly to `plan`
        if status_updates:
            for update in status_updates:
                step_id = update["step"]
                substep_id = update.get("substep")
                new_status = update["status"]

                # Ensure the step exists before modifying it
                matching_steps = [s for s in plan if s["step"] == step_id]
                if not matching_steps:
                    logger.warning(f"Step {step_id} not found in the current plan. Skipping update.")
                    continue

                for step in matching_steps:
                    if substep_id is not None:
                        matching_substeps = [ss for ss in step["substeps"] if ss["substep"] == substep_id]
                        if not matching_substeps:
                            logger.warning(f"Substep {substep_id} not found in Step {step_id}. Skipping update.")
                            continue
                        for substep in matching_substeps:
                            substep["status"] = new_status
                    else:
                        step["status"] = new_status

        # Step 1.5: Check if all substeps are completed and update the main step accordingly
        for step in plan:
            if step["substeps"]:  # Only check if there are substeps
                substep_statuses = {ss["status"] for ss in step["substeps"]}

                if all(status == "completed" for status in substep_statuses):
                    step["status"] = "completed"
                elif "in_progress" in substep_statuses:
                    step["status"] = "in_progress"
                else:
                    step["status"] = "not_started"  # Default fallback

        # Step 2: Apply plan restructuring updates (if provided)
        if plan_updates:
            for updated_step in plan_updates:
                step_id = updated_step["step"]

                # Ensure the step exists before modifying it
                matching_indices = [i for i, s in enumerate(plan) if s["step"] == step_id]
                if not matching_indices:
                    logger.warning(f"Step {step_id} not found in the current plan. Skipping restructure.")
                    continue

                for i in matching_indices:
                    plan[i].update(updated_step)

        # Step 3: Save the updated plan back to state and persist changes
        self.state["instances"][instance_id]["plan"] = plan
        await self.update_workflow_state(instance_id=instance_id, plan=plan)

        return plan

    async def generate_next_step(self, task: str, agents: str, plan: str, next_step_schema: str) -> NextStep:
        """
        Determines the next agent to respond in a workflow.

        Args:
            task (str): The current task description.
            agents (str): A list of available agents.
            plan (str): The structured execution plan.
            next_step_schema (str): The next step schema.

        Returns:
            Dict: A structured response with the next agent, an instruction, and a status verdict.
        """
        pass

    async def broadcast_task_message(self, instance_id: str, message: str):
        """
        Saves message to workflow state and broadcasts it to all registered agents.

        Args:
            instance_id (str): Workflow instance ID for context.
            message (str): A message to append to the workflow state and broadcast to all agents.
        """
        # Store message to local cache and workflow state
        message = {"name": self.name, "role": "user", "content": message}
        await self.update_workflow_state(instance_id=instance_id, message=message)

        # Broadcast message
        task_message = BaseMessage(**message)
        await self.broadcast_message(message=task_message)

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
    
    async def update_workflow_state(
        self, 
        instance_id: str, 
        message: Optional[Dict[str, Any]] = None, 
        final_output: Optional[str] = None, 
        plan: Optional[List[Dict[str, Any]]] = None
    ) -> None:
        """
        Updates the workflow state with a new message, execution plan, or final output.

        Args:
            instance_id (str): The unique ID of the workflow instance.
            message (Optional[Dict[str, Any]]): A message to append to the workflow state.
            final_output (Optional[str]): The final output of the workflow if completed.
            plan (Optional[List[Dict[str,Any]]]): The structured execution plan for the workflow.

        Raises:
            ValueError: If no workflow entry exists for the given `instance_id`.
        """
        workflow_entry = self.state["instances"].get(instance_id)

        if not workflow_entry:
            raise ValueError(f"No workflow entry found for instance_id: {instance_id}")

        if plan:
            workflow_entry["plan"] = plan

        if message:
            # Store dictionary to local cache
            await self.append_message(BaseMessage(**message))

            # Update workflow state
            workflow_message = LLMWorkflowMessage(**message).model_dump(mode="json")
            workflow_entry["messages"].append(workflow_message)
            workflow_entry["last_message"] = workflow_message

        if final_output:
            workflow_entry["output"] = final_output
            workflow_entry["end_time"] = datetime.now().isoformat()

        self.save_state()
    
    async def update_task_history(self, instance_id: str, agent: str, step: int, substep: Optional[float], result: str):
        """
        Updates the task history for a workflow instance by recording the results of an agent's execution.

        Args:
            instance_id (str): The unique workflow instance ID.
            agent (str): The name of the agent who performed the task.
            step (int): The step number associated with the task.
            substep (Optional[float]): The substep number, if applicable.
            result (str): The result or response generated by the agent.

        Raises:
            ValueError: If the instance ID does not exist in the workflow state.
        """
        workflow_entry = self.state["instances"].get(instance_id)

        if not workflow_entry:
            raise ValueError(f"No workflow entry found for instance_id: {instance_id}")

        task_result = TaskResult(
            agent=agent,
            step=step,
            substep=substep,
            result=result
        )

        # Append to task history
        workflow_entry["task_history"].append(task_result.model_dump(mode="json"))
        
        # Save state
        self.save_state()