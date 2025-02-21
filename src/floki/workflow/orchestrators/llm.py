from floki.workflow.orchestrators.prompts import TASK_PLANNING_PROMPT, TASK_INITIAL_PROMPT, PROGRESS_CHECK_PROMPT, NEXT_STEP_PROMPT, SUMMARY_GENERATION_PROMPT
from floki.types.workflow import LLMWorkflowState, LLMWorkflowEntry, LLMWorkflowMessage, NextStep, TaskResult, TaskPlan, PlanStep, ProgressCheckOutput
from floki.workflow.agentic import AgenticWorkflow
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
        self.task(self.generate_plan, description=TASK_PLANNING_PROMPT)
        self.task(self.prepare_initial_message)
        self.task(self.broadcast_message_to_agents)
        self.task(self.generate_next_step, description=NEXT_STEP_PROMPT, include_chat_history=True)
        self.task(self.trigger_agent)
        self.task(self.update_task_history)
        self.task(self.check_progress, description=PROGRESS_CHECK_PROMPT, include_chat_history=True)
        self.task(self.update_plan),
        self.task(self.generate_summary, description=SUMMARY_GENERATION_PROMPT, include_chat_history=True)
        self.task(self.finish_workflow)

    def llm_workflow(self, ctx: DaprWorkflowContext, input: ChatLoop):
        """
        Executes an LLM-driven agentic workflow where the next agent is dynamically selected 
        based on task progress. The workflow iterates through execution cycles, updating state, 
        handling agent responses, and determining task completion.

        Args:
            ctx (DaprWorkflowContext): The workflow execution context.
            input (ChatLoop): The current workflow state containing `message`, `iteration`, and `verdict`.

        Returns:
            str: The final processed message when the workflow terminates.

        Raises:
            RuntimeError: If the LLM determines the task is `failed`.
        """
        # Step 1: Check iteration inputs
        message = input.get("message")
        iteration = input.get("iteration", 0)
        instance_id = ctx.instance_id
        plan = self.state["instances"].get(instance_id, {}).get("plan", [])
        
        if not ctx.is_replaying:
            logger.info(f"Workflow iteration {iteration + 1} started (Instance ID: {instance_id}).")

        # Step 2: Retrieve available agents
        agents = yield ctx.call_activity(self.get_agents_metadata_as_string)

        # Step 3: First iteration setup
        if iteration == 0:
            if not ctx.is_replaying:
                logger.info(f"Initial message from User -> {self.name}")
            
            # Generate the plan using a language model
            plan = yield ctx.call_activity(self.generate_plan, input={"task": message, "agents": agents, "plan_schema": plan_schema})

            # Initialize workflow state and format message with plan
            initial_message = yield ctx.call_activity(self.prepare_initial_message, input={"instance_id": instance_id, "task": message, "agents": agents, "plan": plan})

            # broadcast initial message to all agents
            yield ctx.call_activity(self.broadcast_message_to_agents, input={"instance_id": instance_id, "message": initial_message})
        
        # Step 4: Identify agent and instruction for the next step
        next_step = yield ctx.call_activity(self.generate_next_step, input={"task": message, "agents": agents, "plan": plan, "next_step_schema": next_step_schema})

        # Extract Additional Properties from NextStep
        next_agent = next_step["next_agent"]
        instruction = next_step["instruction"]
        step_id = next_step.get("step", None)
        substep_id = next_step.get("substep", None)

        # Step 5: Broadcast Task to all Agents
        yield ctx.call_activity(self.broadcast_message_to_agents, input={"instance_id": instance_id, "message": instruction})

        # Step 6: Trigger next agent
        plan = yield ctx.call_activity(self.trigger_agent, input={"instance_id": instance_id, "name": next_agent, "step": step_id, "substep": substep_id})

        # Step 7: Wait for agent response or timeout
        if not ctx.is_replaying:
            logger.info(f"Waiting for {next_agent}'s response...")
        
        event_data = ctx.wait_for_external_event("AgentCompletedTask")
        timeout_task = ctx.create_timer(timedelta(seconds=self.timeout))
        any_results = yield self.when_any([event_data, timeout_task])
        
        if any_results == timeout_task:
            logger.warning(f"Agent response timed out (Iteration: {iteration + 1}, Instance ID: {instance_id}).")
            task_results = {"name": self.name, "role": "user", "content": f"Timeout occurred. {next_agent} did not respond on time. We need to try again..."}
        else:
            task_results = yield event_data
            if not ctx.is_replaying:
                logger.info(f"{task_results['name']} sent a response.")
            
        # Step 8: Save the task execution results to chat and task history
        yield ctx.call_activity(self.update_task_history, input={"instance_id": instance_id, "agent": next_agent, "step": step_id, "substep": substep_id, "results": task_results})
        
        # Step 9: Check progress
        progress = yield ctx.call_activity(self.check_progress, input={ "task": message, "plan": plan, "step": step_id, "substep": substep_id, "results": task_results["content"], "progress_check_schema": progress_check_schema})
        
        verdict = progress["verdict"]
        status_updates = progress.get("plan_status_update", [])
        plan_updates = progress.get("plan_restructure", [])

        # Step 10: Process progress suggestions and iteration condition
        if verdict != "continue" or iteration + 1 >= self.max_iterations:
            if iteration >= self.max_iterations:
                verdict = "max_iterations_reached"
            
            if not ctx.is_replaying:
                logger.info(f"Workflow ending with verdict: {verdict}")
            
            # Generate final summary based on execution
            summary = yield ctx.call_activity(self.generate_summary, input={"task": message, "verdict": verdict, "plan": plan, "step": step_id, "substep": substep_id, "agent": next_agent, "result": task_results["content"]})

            # Finalize the workflow properly
            yield ctx.call_activity(self.finish_workflow, input={"instance_id": instance_id, "plan": plan, "step": step_id, "substep": substep_id, "verdict": verdict, "summary": summary})

            if not ctx.is_replaying:
                logger.info(f"Workflow {instance_id} has been finalized with verdict: {verdict}")
            
            return summary
        
        if status_updates or plan_updates:
            yield ctx.call_activity(self.update_plan, input={"instance_id": instance_id, "plan": plan, "status_updates": status_updates, "plan_updates": plan_updates})
        
        # Step 11: Update ChatLoop state and continue workflow
        input["message"] = task_results
        input["iteration"] = iteration + 1

        # Restart workflow with updated ChatLoop state
        ctx.continue_as_new(input)
    
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

    async def prepare_initial_message(self, instance_id: str, task: str, agents: str, plan: List[Dict[str, Any]]) -> str:
        """
        Initializes the workflow entry and sends the first task briefing to all agents.

        Args:
            instance_id (str): The ID of the workflow instance.
            task (str): The initial input message describing the task.
            agents (str): The formatted list of available agents.
            plan (List[Dict[str, Any]]): The structured execution plan generated beforehand.
        """
        # Format Initial Message with the Plan
        formatted_message = TASK_INITIAL_PROMPT.format(task=task, agents=agents, plan=plan)

        # Save initial input and plan to initial workflow entry
        workflow_entry = LLMWorkflowEntry(input=task, plan=plan)

        # Save initial workflow in local and remote state
        self.state["instances"][instance_id] = workflow_entry.model_dump(mode="json")
        self.save_state()

        return formatted_message
    
    async def broadcast_message_to_agents(self, instance_id: str, message: str):
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
    
    async def generate_next_step(self, task: str, agents: str, plan: str, next_step_schema: str) -> NextStep:
        """
        Determines the next agent to respond in a workflow.

        Args:
            task (str): The current task description.
            agents (str): A list of available agents.
            plan (str): The structured execution plan.
            next_step_schema (str): The next step schema.

        Returns:
            Dict: A structured response with the next agent, an instruction, and step ids.
        """
        pass

    async def update_step_statuses(self, plan: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Ensures step and sub-step statuses follow logical progression:
        - A step should only be marked "completed" if all substeps are "completed".
        - If a sub-step is "in_progress", its parent step must also be "in_progress".
        - If a sub-step is "completed" but its parent step is "not_started", set the parent to "in_progress".
        
        Args:
            plan (List[Dict[str, Any]]): The current execution plan.

        Returns:
            List[Dict[str, Any]]: The updated execution plan with correct statuses.
        """
        for step in plan:
            if "substeps" in step and step["substeps"]:
                substep_statuses = {ss["status"] for ss in step["substeps"]}

                # Case 1: If ALL substeps are "completed", parent step must be "completed".
                if all(status == "completed" for status in substep_statuses):
                    step["status"] = "completed"

                # Case 2: If ANY substep is "in_progress", parent step must be "in_progress".
                elif "in_progress" in substep_statuses:
                    step["status"] = "in_progress"

                # Case 3: If a sub-step was completed, but the step is still "not_started", update it.
                elif "completed" in substep_statuses and step["status"] == "not_started":
                    step["status"] = "in_progress"

        return plan
    
    async def trigger_agent(self, instance_id: str, name: str, step: int, substep: Optional[float]) -> List[Dict[str, Any]]:
        """
        Updates step status and triggers the specified agent to perform its activity.

        Args:
            instance_id (str): Workflow instance ID for context.
            name (str): Name of the agent to trigger.
            step (int): The step number associated with the task.
            substep (Optional[float]): The substep number, if applicable.

        Returns:
            List[Dict[str, Any]]: The updated execution plan.
        """
        workflow_entry = self.state["instances"].get(instance_id)
        plan = workflow_entry["plan"]

        # Ensure the step exists
        matching_steps = [s for s in plan if s["step"] == step]
        if not matching_steps:
            raise ValueError(f"Step {step} not found in the current plan.")

        for step_entry in matching_steps:
            if substep is not None:
                matching_substeps = [ss for ss in step_entry.get("substeps", []) if ss["substep"] == substep]
                if not matching_substeps:
                    raise ValueError(f"Substep {substep} not found in Step {step}.")
                for sub in matching_substeps:
                    sub["status"] = "in_progress"
            else:
                step_entry["status"] = "in_progress"

        # Apply global status updates to maintain consistency
        plan = await self.update_step_statuses(plan)

        # Save to state and update workflow
        self.state["instances"][instance_id]["plan"] = plan
        await self.update_workflow_state(instance_id=instance_id, plan=plan)

        # Send message to agent
        await self.send_message_to_agent(
            name=name,
            message=TriggerActionMessage(task=None),
            workflow_instance_id=instance_id,
        )

        return plan
        
    async def update_workflow_state(self, instance_id: str, message: Optional[Dict[str, Any]] = None, final_output: Optional[str] = None, plan: Optional[List[Dict[str, Any]]] = None):
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

        # Retrieve workflow state
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
    
    async def update_task_history(self, instance_id: str, agent: str, step: int, substep: Optional[float], results: Dict[str, Any]):
        """
        Updates the task history for a workflow instance by recording the results of an agent's execution.

        Args:
            instance_id (str): The unique workflow instance ID.
            agent (str): The name of the agent who performed the task.
            step (int): The step number associated with the task.
            substep (Optional[float]): The substep number, if applicable.
            results (Dict[str, Any]): The result or response generated by the agent.

        Raises:
            ValueError: If the instance ID does not exist in the workflow state.
        """
        # Record agent response in message history
        await self.update_workflow_state(instance_id=instance_id, message=results)

        # Retrieve Workflow state
        workflow_entry = self.state["instances"].get(instance_id)

        if not workflow_entry:
            raise ValueError(f"No workflow entry found for instance_id: {instance_id}")

        # Record agent response in task history
        task_result = TaskResult(
            agent=agent,
            step=step,
            substep=substep,
            result=results["content"]
        )

        # Append to task history
        workflow_entry["task_history"].append(task_result.model_dump(mode="json"))
        
        # Save state
        self.save_state()
    
    async def check_progress(self, task: str, plan: str, step: int, substep: Optional[float], results: str, progress_check_schema: str) -> ProgressCheckOutput:
        """
        Evaluates the current plan's progress and determines necessary updates.

        Args:
            task (str): The current task description.
            plan (str): The structured execution plan.
            step (int): The step number associated with the task.
            substep (Optional[float]): The substep number, if applicable.
            results (str): The result or response generated by the agent.
            progress_check_schema (str): The schema of the progress check

        Returns:
            ProgressCheckOutput: The plan update details, including status changes and restructuring if needed.
        """
        pass

    async def update_plan(self, instance_id: str, plan: List[Dict[str, Any]], status_updates: Optional[List[Dict[str, Any]]] = None, plan_updates: Optional[List[Dict[str, Any]]] = None):
        """
        Updates the execution plan based on status changes and/or plan restructures.

        Args:
            instance_id (str): The workflow instance ID.
            plan (List[Dict[str, Any]]): The current execution plan.
            status_updates (Optional[List[Dict[str, Any]]]): List of updates for step statuses.
            plan_updates (Optional[List[Dict[str, Any]]]): List of full step modifications.

        Raises:
            ValueError: If a specified step or substep is not found.
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
                    raise ValueError(f"Step {step_id} not found in the current plan. Cannot proceed with update.")

                for step in matching_steps:
                    if substep_id is not None:
                        matching_substeps = [ss for ss in step["substeps"] if ss["substep"] == substep_id]
                        if not matching_substeps:
                            raise ValueError(f"Substep {substep_id} not found in Step {step_id}. Cannot proceed with update.")
                        
                        # Apply update
                        for substep in matching_substeps:
                            substep["status"] = new_status
                    else:
                        # Update main step status
                        step["status"] = new_status

        # Step 2: Apply plan restructuring updates (if provided)
        if plan_updates:
            for updated_step in plan_updates:
                step_id = updated_step["step"]
                matching_indices = [i for i, s in enumerate(plan) if s["step"] == step_id]
                if not matching_indices:
                    raise ValueError(f"Step {step_id} not found in the current plan. Cannot proceed with restructure.")

                for i in matching_indices:
                    plan[i].update(updated_step)

        # Step 3: Apply global consistency checks for statuses
        plan = await self.update_step_statuses(plan)

        # Save to state and update workflow
        self.state["instances"][instance_id]["plan"] = plan
        await self.update_workflow_state(instance_id=instance_id, plan=plan)
    
    async def generate_summary(self, task: str, verdict: str, plan: str, step: int, substep: Optional[float], agent: str, result: str) -> str:
        """
        Generates a structured summary of task execution based on conversation history, execution results, and the task plan.

        Args:
            task (str): The original task description.
            verdict (str): The overall task status (e.g., "continue", "completed", or "failed").
            plan (str): The structured execution plan detailing task progress.
            step (int): The step number associated with the most recent action.
            substep (Optional[float]): The substep number, if applicable.
            agent (str): The name of the agent who executed the last action.
            result (str): The response or outcome generated by the agent.

        Returns:
            str: A concise but informative summary of task progress and results, structured for user readability.
        """
        pass

    async def finish_workflow(self, instance_id: str, plan: List[Dict[str, Any]], step: int, substep: Optional[float], verdict: str, summary: str):
        """
        Finalizes the workflow by updating the plan, marking the provided step/substep as completed if applicable,
        and storing the summary and verdict.

        Args:
            instance_id (str): The workflow instance ID.
            plan (List[Dict[str, Any]]): The current execution plan.
            step (int): The step that was last worked on.
            substep (Optional[float]): The substep that was last worked on (if applicable).
            verdict (str): The final workflow verdict (`completed`, `failed`, or `max_iterations_reached`).
            summary (str): The generated summary of the workflow execution.

        Returns:
            None
        """

        status_updates = []

        if verdict == "completed":
            matching_steps = [s for s in plan if s["step"] == step]
            if not matching_steps:
                raise ValueError(f"Step {step} not found in the current plan. Cannot mark as completed.")

            for step_entry in matching_steps:
                if substep is not None:
                    # Ensure substep exists before updating
                    matching_substeps = [ss for ss in step_entry["substeps"] if ss["substep"] == substep]
                    if not matching_substeps:
                        raise ValueError(f"Substep {substep} not found in Step {step}. Cannot mark as completed.")

                    # Mark the substep as completed
                    for sub in matching_substeps:
                        sub["status"] = "completed"
                    status_updates.append({"step": step, "substep": substep, "status": "completed"})

                    # **Check if all substeps are completed**
                    all_substeps_completed = all(ss["status"] == "completed" for ss in step_entry["substeps"])
                    if all_substeps_completed:
                        step_entry["status"] = "completed"
                        status_updates.append({"step": step, "status": "completed"})

                else:
                    # If no substeps exist, mark the step as completed
                    step_entry["status"] = "completed"
                    status_updates.append({"step": step, "status": "completed"})

        # Apply updates while keeping parent step untouched if not all substeps are completed
        if status_updates:
            await self.update_plan(instance_id=instance_id, plan=plan, status_updates=status_updates)

        # Store the final summary and verdict in workflow state
        await self.update_workflow_state(instance_id=instance_id, final_output=summary)