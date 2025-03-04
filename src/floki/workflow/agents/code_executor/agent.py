from floki.workflow.agents.code_executor.state import CodeExecWorkflowState, CodeExecWorkflowEntry, CodeExecWorkflowMessage
from floki.workflow.agents.code_executor.schemas import AgentTaskResponse, TriggerAction
from floki.types.executor import ExecutionRequest, ExecutionResult, CodeSnippet
from floki.types.message import BaseMessage, EventMessageMetadata
from floki.executors import CodeExecutorBase, LocalCodeExecutor
from floki.workflow.agents.base import AgentServiceBase
from floki.workflow.decorators import workflow, task
from floki.types import DaprWorkflowContext
from floki.messaging import message_router
from fastapi import Response, status
from typing import Any, List, Dict, Optional
from pydantic import Field
from datetime import datetime
import logging
import re

logger = logging.getLogger(__name__)

class CodeExecutorAgent(AgentServiceBase):
    """
    Orchestrates the execution of code extracted from user messages.
    Uses agentic workflows to iterate and optimize execution.
    """

    goal : str = Field(default="Extract code from text and execute Python and Bash scripts as provided, returning outputs, errors, and execution details without modification or interpretation.")
    executor: CodeExecutorBase = Field(default_factory=LocalCodeExecutor, description="Code execution backend.")

    def model_post_init(self, __context: Any) -> None:
        """Initializes the workflow with agentic execution capabilities."""
        
        # Initialize Agent State
        self.state = CodeExecWorkflowState()

        # Name of main Workflow
        self.workflow_name = "CodeExecutionWorkflow"

        super().model_post_init(__context)
    
    @message_router
    @workflow(name="CodeExecutionWorkflow")
    def execute_code_workflow(self, ctx: DaprWorkflowContext, message: TriggerAction):
        """
        Executes a workflow that extracts, runs, and returns code execution results.

        Args:
            ctx (DaprWorkflowContext): The workflow execution context.
            message (TriggerAction): The input message containing task instructions.

        Returns:
            str: The formatted output of executed code or an error message if no valid code is found.
        """
        # Step 0: Retrieve metadata and task details
        task = message.get("task")
        instance_id = ctx.instance_id
        metadata = message.get("_message_metadata") or {}

        # Extract workflow metadata with proper defaults
        source_agent = metadata.get("source") or None
        source_workflow_instance_id = metadata.get("headers", {}).get("workflow_instance_id") or None

        if not ctx.is_replaying:
            logger.info(f"Workflow started (Instance ID: {instance_id}).")

        # Step 1: Initialize workflow state
        self.state.setdefault("instances", {})
        workflow_entry = CodeExecWorkflowEntry(
            input=task or "Triggered without input.",
            source_agent=source_agent,
            source_workflow_instance_id=source_workflow_instance_id,
        )
        self.state["instances"].setdefault(instance_id, workflow_entry.model_dump(mode="json"))

        # Step 2: Update chat history
        yield ctx.call_activity(self.update_message_history, input={"instance_id": instance_id, "task": task})

        if not ctx.is_replaying:
            logger.info(f"Message from {source_agent} -> {self.name}")

        # Step 3: Extract Code Blocks
        code_blocks = yield ctx.call_activity(self.extract_code, input={"task": task})
        if not code_blocks:
            formatted_output = "No valid code blocks found."
        else:
            # Step 4: Execute Code
            execution_request = ExecutionRequest(snippets=code_blocks, timeout=5).model_dump()
            execution_results = yield ctx.call_activity(self.run_code, input={"execution_request": execution_request})

            # Step 5: Format Results
            formatted_output = yield ctx.call_activity(self.format_execution_results, input={"execution_results": execution_results})

        # Step 6: Respond to source agent if available
        if source_agent:
            if not ctx.is_replaying:
                logger.info(f"Sending response to {source_agent}..")
            
            yield ctx.call_activity(self.send_response_back, input={
                "response": formatted_output,
                "target_agent": source_agent,
                "target_instance_id": source_workflow_instance_id
            })

        # Step 7: Finalize workflow
        yield ctx.call_activity(self.finish_workflow, input={"instance_id": instance_id, "summary": formatted_output})
        
        return formatted_output

    @task
    async def update_message_history(self, instance_id: str, task: str):
        """
        Updates workflow state by storing the latest message.

        Args:
            instance_id (str): Unique workflow instance ID.
            task (str): The task input message.
        """
        messages = self.construct_messages(task or {})
        user_message = next((msg | {"content": msg["content"].strip()} for msg in reversed(messages) if msg.get("role") == "user"), None)

        if user_message:
            await self.update_workflow_state(instance_id=instance_id, message=user_message)
    
    @task
    def extract_code(self, task: str) -> List[CodeSnippet]:
        """
        Extracts code snippets from a message.

        Args:
            task (str): The message containing possible code blocks.

        Returns:
            List[CodeSnippet]: Extracted code snippets with language and content.
        """
        messages = self.construct_messages(task or {})
        user_message = next((msg | {"content": msg["content"].strip()} for msg in reversed(messages) if msg.get("role") == "user"), None)
        pattern = re.compile(r"```(\w+)?\n([\s\S]+?)\n```")
        return [CodeSnippet(language=match[0] or "plaintext", code=match[1]) for match in pattern.findall(user_message["content"])]

    @task
    async def run_code(self, execution_request: ExecutionRequest) -> List[ExecutionResult]:
        """
        Executes extracted code snippets.

        Args:
            execution_request (ExecutionRequest): Request object containing code snippets.

        Returns:
            List[ExecutionResult]: Execution results for each code snippet.
        """
        return await self.executor.execute(execution_request)
    
    @task
    def format_execution_results(self, execution_results: List[Dict[str, Any]]) -> str:
        """
        Formats execution results into a readable response.

        Args:
            execution_results (List[Dict[str, Any]]): List of execution outputs.

        Returns:
            str: Formatted execution results.
        """
        return "\n".join(
            f"**Execution Result:**\n{res['output']}" if res["status"] == "success"
            else f"**Error:**\n{res['output']}"
            for res in execution_results
        )

    @task
    async def send_response_back(self, response: str, target_agent: str, target_instance_id: str):
        """
        Sends execution results back to the requesting agent.

        Args:
            response (str): The formatted execution output.
            target_agent (str): The requesting agent's name.
            target_instance_id (str): The workflow instance ID of the requesting agent.
        """
        logger.info(f"Sending execution results to {target_agent} (Instance: {target_instance_id})")

        agent_response = AgentTaskResponse(name=self.name, role="assistant", content=response)
        await self.send_message_to_agent(
            name=target_agent,
            message=agent_response,
            event_name="AgentTaskResponse",
            workflow_instance_id=target_instance_id
        )

    @task
    async def finish_workflow(self, instance_id: str, summary: str):
        """
        Marks the workflow as complete and stores the final output.

        Args:
            instance_id (str): Unique workflow instance ID.
            summary (str): The final output of the execution.
        """
        await self.update_workflow_state(instance_id=instance_id, final_output=summary)
    
    async def update_workflow_state(self, instance_id: str, message: Optional[Dict[str, Any]] = None, final_output: Optional[str] = None):
        """
        Updates the workflow state by appending a new message or setting the final output.

        Args:
            instance_id (str): The unique identifier of the workflow instance.
            message (Optional[Dict[str, Any]]): A dictionary representing a message to be added to the workflow state.
            final_output (Optional[str]): The final output of the workflow, marking its completion.

        Raises:
            ValueError: If no workflow entry is found for the given instance_id.
        """
        workflow_entry = self.state["instances"].get(instance_id)
        if not workflow_entry:
            raise ValueError(f"No workflow entry found for instance_id {instance_id} in local state.")

        # Only update the provided fields
        if message is not None:
            serialized_message = CodeExecWorkflowMessage(**message).model_dump(mode="json")

            # Update workflow state messages
            workflow_entry["messages"].append(serialized_message)
            workflow_entry["last_message"] = serialized_message

            # Update the local chat history
            self.memory.add_message(message)

        if final_output is not None:
            workflow_entry["output"] = final_output
            workflow_entry["end_time"] = datetime.now().isoformat()

        # Persist updated state
        self.save_state()
    
    @message_router(broadcast=True)
    async def process_broadcast_message(self, message: BaseMessage, metadata: EventMessageMetadata) -> Response:
        """
        Processes a broadcast message, filtering out messages sent by the same agent 
        and updating local memory with valid messages.

        Args:
            message (BaseMessage): The received broadcast message.
            metadata (EventMessageMetadata): Metadata associated with the broadcast event.

        Returns:
            Response: HTTP response indicating success or failure.
        """
        try:
            logger.info(f"{self.name} received broadcast message of type '{metadata.type}' from '{metadata.source}'.")

            # Ignore messages sent by this agent
            if metadata.source == self.name:
                logger.info(f"{self.name} ignored its own broadcast message of type '{metadata.type}'.")
                return Response(status_code=status.HTTP_204_NO_CONTENT)

            # Log and process the valid broadcast message
            logger.debug(f"{self.name} is processing broadcast message of type '{metadata.type}' from '{metadata.source}'.")
            logger.debug(f"Message content: {message.content}")

            # Update the local chat history
            self.memory.add_message(message)

            return Response(content="Broadcast message added to memory and actor state.", status_code=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error processing broadcast message: {e}", exc_info=True)
            return Response(content=f"Error processing message: {str(e)}", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)