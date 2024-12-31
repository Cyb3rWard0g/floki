from dapr.ext.workflow import WorkflowRuntime, WorkflowActivityContext, DaprWorkflowContext, DaprWorkflowClient
from dapr.ext.workflow.workflow_state import WorkflowState
from floki.types.workflow import WorkflowStatus, WorkflowStateMap, WorkflowMessage, WorkflowEntry
from typing import Any, Callable, Generator, Optional, Dict, TypeVar, Union, List
from floki.storage.daprstores.statestore import DaprStateStore
from floki.workflow.task import Task, TaskWrapper
from pydantic import BaseModel, Field, ValidationError, ConfigDict
from dapr.conf import settings as dapr_settings
from dapr.clients import DaprClient
from durabletask import task as dtask
from datetime import datetime
import asyncio
import functools
import logging
import uuid
import json
import os

logger = logging.getLogger(__name__)

T = TypeVar('T')
TInput = TypeVar('TInput')
TOutput = TypeVar('TOutput')
Workflow = Callable[..., Union[Generator[dtask.Task, Any, Any], TOutput]]

class WorkflowApp(BaseModel):
    """
    A Pydantic-based class to encapsulate a Dapr Workflow runtime and manage workflows and tasks.

    Attributes:
        daprGrpcHost (Optional[str]): Host address for the Dapr gRPC endpoint.
        daprGrpcPort (Optional[int]): Port number for the Dapr gRPC endpoint.
        wf_runtime (WorkflowRuntime): The Dapr Workflow runtime instance.
        wf_client (DaprWorkflowClient): The Dapr Workflow client instance for invoking and interacting with workflows.
        tasks (Dict[str, Callable]): A dictionary storing registered task functions by name.
        workflows (Dict[str, Callable]): A dictionary storing registered workflows by name.
        timeout (Optional[int]): Timeout for workflow completion in seconds. Defaults to 300.
    """

    daprGrpcHost: Optional[str] = Field(None, description="Host address for the Dapr gRPC endpoint.")
    daprGrpcPort: Optional[int] = Field(None, description="Port number for the Dapr gRPC endpoint.")
    workflow_state_store_name: str = Field(default="workflowstatestore", description="The name of the Dapr state store component used to store workflow metadata.")
    workflow_timeout: int = Field(default=300, description="Default timeout duration in seconds for workflow tasks.")

    # Initialized in model_post_init
    wf_runtime: Optional[WorkflowRuntime] = Field(default=None, init=False, description="Workflow runtime instance.")
    wf_client: Optional[DaprWorkflowClient] = Field(default=None, init=False, description="Workflow client instance.")
    wf_state_store: Optional[DaprStateStore] = Field(default=None, init=False, description="Dapr state store instance for accessing and managing workflow state.")
    wf_state_key: str = Field(default="workflow_state", init=False, description="Dapr state store key for the workflow state.")
    state: WorkflowStateMap = Field(default=None, init=False, description="Workflow Dapr state.")
    tasks: Dict[str, Callable] = Field(default_factory=dict, description="Dictionary of registered tasks.")
    workflows: Dict[str, Callable] = Field(default_factory=dict, description="Dictionary of registered workflows.")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization to configure Dapr Workflow runtime and client.
        """
        # Configure Dapr gRPC settings, using environment variables if provided
        env_daprGrpcHost = os.getenv('DAPR_RUNTIME_HOST')
        env_daprGrpcPort = os.getenv('DAPR_GRPC_PORT')

        # Resolve final values for Dapr settings
        self.daprGrpcHost = self.daprGrpcHost or env_daprGrpcHost or dapr_settings.DAPR_RUNTIME_HOST
        self.daprGrpcPort = int(self.daprGrpcPort or env_daprGrpcPort or dapr_settings.DAPR_GRPC_PORT)

        # Initialize WorkflowRuntime and DaprWorkflowClient
        self.wf_runtime = WorkflowRuntime(host=self.daprGrpcHost, port=self.daprGrpcPort)
        self.wf_client = DaprWorkflowClient(host=self.daprGrpcHost, port=self.daprGrpcPort)

        # Initialize Workflow state store
        self.wf_state_store = DaprStateStore(store_name=self.workflow_state_store_name, address=f"{self.daprGrpcHost}:{self.daprGrpcPort}")

        # Register workflow
        self.register_workflow_metadata()

        logger.info(f"Initialized WorkflowApp with Dapr gRPC host '{self.daprGrpcHost}' and port '{self.daprGrpcPort}'.")

        # Proceed with base model setup
        super().model_post_init(__context)
    
    def task(self,func: Optional[Callable] = None, *, name: Optional[str] = None, description: Optional[str] = None, agent: Optional[Any] = None, agent_method: Optional[Union[str, Callable]] = "run", llm: Optional[Any] = None, llm_method: Optional[Union[str, Callable]] = "generate") -> Callable:
        """
        Custom decorator to create and register a workflow task, supporting async and extended capabilities.

        This decorator allows for the creation and registration of tasks that can be executed
        as part of a Dapr workflow. The task can optionally integrate with an agent or LLM 
        for enhanced functionality. It supports both synchronous and asynchronous functions.

        Args:
            func (Callable, optional): The function to be decorated as a workflow task. Defaults to None.
            name (Optional[str]): The name to register the task with. Defaults to the function's name.
            description (Optional[str]): A textual description of the task. Defaults to None.
            agent (Optional[Any]): The agent to use for executing the task if a description is provided. Defaults to None.
            agent_method (Optional[Union[str, Callable]]): The method or callable to invoke the agent. Defaults to "run".
            llm (Optional[Any]): The LLM client to use for executing the task if a description is provided. Defaults to None.
            llm_method (Optional[Union[str, Callable]]): The method or callable to invoke the LLM client. Defaults to "generate".

        Returns:
            Callable: The decorated function wrapped with task logic and registered as an activity.
        """
        # Check if the first argument is a string, implying it's the description
        if isinstance(func, str) is True:
            description = func
            func = None
        
        def decorator(f: Callable):
            """
            Decorator to wrap a function as a Dapr workflow activity task.
            
            Args:
                f (Callable): The function to be wrapped and registered as a Dapr task.

            Returns:
                Callable: A decorated function wrapped with the task execution logic.
            """
            # Wrap the original function with Task logic
            task_instance = Task(
                func=f,
                description=description,
                agent=agent,
                agent_method=agent_method,
                llm=llm,
                llm_method=llm_method,
            )

            @functools.wraps(f)
            def task_wrapper(ctx: WorkflowActivityContext, input: Any = None):
                """
                Wrapper function for executing tasks in a Dapr workflow.
                Handles both sync and async tasks.
                """
                async def async_execution():
                    """
                    Handles the actual asynchronous execution of the task.
                    """
                    try:
                        result = await task_instance(ctx, input)
                        return result
                    except Exception as e:
                        logger.error(f"Async task execution failed: {e}")
                        raise

                def run_in_event_loop(coroutine):
                    """
                    Helper function to run a coroutine in the current or a new event loop.
                    """
                    try:
                        loop = asyncio.get_event_loop()
                    except RuntimeError:
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)

                    return loop.run_until_complete(coroutine)

                try:
                    if asyncio.iscoroutinefunction(f) or asyncio.iscoroutinefunction(task_instance.__call__):
                        # Handle async tasks
                        return run_in_event_loop(async_execution())
                    else:
                        # Handle sync tasks
                        result = task_instance(ctx, input)
                        if asyncio.iscoroutine(result):
                            logger.warning("Sync task returned a coroutine. Running it in the event loop.")
                            return run_in_event_loop(result)

                        logger.info(f"Sync task completed.")
                        return result

                except Exception as e:
                    logger.error(f"Task execution failed: {e}")
                    raise

            # Register the task with Dapr Workflow
            activity_decorator = self.wf_runtime.activity(name=name or f.__name__)
            registered_activity = activity_decorator(task_wrapper)

            # Optionally, store the task in the registry for easier access
            task_name = name or f.__name__
            self.tasks[task_name] = registered_activity

            return registered_activity

        return decorator(func) if func else decorator
    
    def workflow(self, func: Optional[Callable] = None, *, name: Optional[str] = None) -> Callable:
        """
        Custom decorator to register a function as a workflow, building on top of Dapr's workflow decorator.
        
        This decorator allows you to register a function as a Dapr workflow while injecting additional
        context and custom logic. It leverages the existing Dapr `workflow` decorator to ensure compatibility
        with the Dapr workflow runtime and adds the workflow to an internal registry for easy management.

        Args:
            func (Callable, optional): The function to be decorated as a workflow. Defaults to None.
            name (Optional[str]): The name to register the workflow with. Defaults to the function's name.

        Returns:
            Callable: The decorated function with context injection and registered as a workflow.
        """
        def decorator(f: Callable):
            @functools.wraps(f)
            def workflow_wrapper(ctx: DaprWorkflowContext, *args, **kwargs):
                # Inject the context into the function's closure
                return f(ctx, *args, **kwargs)

            # Use the original workflow decorator to register the task_wrapper
            workflow_decorator = self.wf_runtime.workflow(name=name)
            registered_workflow = workflow_decorator(workflow_wrapper)

            # Optionally, store the task in your task registry
            workflow_name = name or f.__name__
            self.workflows[workflow_name] = registered_workflow
            return workflow_wrapper

        return decorator(func) if func else decorator

    def create_task(
        self,
        *,
        name: Optional[str],
        description: Optional[str],
        agent: Optional[Any] = None,
        agent_method: Optional[Union[str, Callable]] = "run",
        llm: Optional[Any] = None,
        llm_method: Optional[Union[str, Callable]] = "generate"
    ) -> Callable:
        """
        Method to create and register a task directly, without using it as a decorator.

        Args:
            name (Optional[str]): The name to register the task with.
            description (Optional[str]): A textual description of the task, which can be used by an agent or LLM.
            agent (Optional[Any]): The agent to use for executing the task if a description is provided. Defaults to None.
            agent_method (Optional[Union[str, Callable]]): The method or callable to invoke the agent. Defaults to "run".
            llm (Optional[Any]): The LLM client to use for executing the task if a description is provided. Defaults to None.
            llm_method (Optional[Union[str, Callable]]): The method or callable to invoke the LLM client. Defaults to "generate".

        Returns:
            Callable: The wrapped Task object, ready to be used in a workflow.
        """
        # Create the Task instance directly
        task_instance = Task(None, description, agent, agent_method, llm, llm_method)

        # Wrap the Task instance with a TaskWrapper that provides a __name__
        wrapped_task = TaskWrapper(task_instance, name)

        # Register the wrapped Task instance with the provided name
        self.wf_runtime.register_activity(wrapped_task, name=name)

        # Store the wrapped task in your task registry
        self.tasks[name] = wrapped_task
        return wrapped_task
    
    def register_workflow_metadata(self) -> None:
        """
        Initializes or loads the workflow metadata from the Dapr state store.
        """
        logger.info("Registering Workflow metadata.")
        
        # Attempt to retrieve existing state
        has_state, state_data = self.wf_state_store.try_get_state(self.wf_state_key)

        if not has_state:
            # No existing state, initialize with default values
            logger.info("Initializing state for workflow.")
            self.state = WorkflowStateMap()
            # Save newly initialized state
            self.save_state(self.state)
        else:
            # Load the existing state
            logger.info("Loading existing workflow state.")
            logger.debug(f"Existing state data: {state_data}")
            try:
                self.state = WorkflowStateMap(**state_data)
            except ValidationError as e:
                # Handle invalid existing state
                logger.error(f"Failed to validate existing state: {e}")
                # Reinitialize with default values and save
                self.state = WorkflowStateMap()
                self.save_state(self.state)
    
    def save_state(self, value: Optional[WorkflowStateMap] = None) -> None:
        """
        Saves the workflow state to the Dapr state store using the predefined workflow state key.

        Args:
            value (Optional[WorkflowStateMap]): The state data to save. If not provided, uses `self.state`.
        """
        try:
            # Use the provided state or fallback to the local state
            state_to_save = value or self.state
            if not state_to_save:
                raise ValueError("No state to save. Both `value` and `self.state` are None.")

            self.wf_state_store.save_state(self.wf_state_key, state_to_save.model_dump_json())
            logger.info(f"Successfully saved state for key '{self.wf_state_key}'.")
        except Exception as e:
            logger.error(f"Failed to save state for key '{self.wf_state_key}': {e}")
            raise
    
    def add_message(self, instance_id: str, message: Dict[str, Any]) -> None:
        """
        Adds a message to the workflow entry for the given instance ID.

        Args:
            instance_id (str): The workflow instance ID.
            message (Dict[str, Any]): The message to add.
        """
        workflow_entry = self.state.instances.get(instance_id)
        if not workflow_entry:
            logger.error(f"Workflow with instance ID {instance_id} not found.")
            return

        workflow_message = WorkflowMessage(**message)
        workflow_entry.messages.append(workflow_message)
        self.save_state()
    
    def resolve_task(self, task_name: str) -> Callable:
        """
        Resolve the task function by its registered name.

        Args:
            task_name (str): The name of the task to resolve.

        Returns:
            Callable: The resolved task function.

        Raises:
            AttributeError: If the task function is not found.
        """
        task_func = self.tasks.get(task_name)
        if not task_func:
            raise AttributeError(f"Task '{task_name}' not found.")
        return task_func
    
    def resolve_workflow(self, workflow_name: str) -> Callable:
        """
        Resolve the workflow function by its registered name.

        Args:
            workflow_name (str): The name of the workflow to resolve.

        Returns:
            Callable: The resolved workflow function.

        Raises:
            AttributeError: If the workflow function is not found.
        """
        workflow_func = self.workflows.get(workflow_name)
        if not workflow_func:
            raise AttributeError(f"Workflow '{workflow_name}' not found.")
        return workflow_func

    def run_workflow(self, workflow: Union[str, Callable], input: Union[str, Dict[str, Any]] = None) -> str:
        """
        Start a workflow and manage its lifecycle.

        Args:
            workflow (Union[str, Callable]): Workflow name or callable instance.
            input (Union[str, Dict[str, Any]], optional): Input for the workflow. Defaults to None.

        Returns:
            str: Workflow instance ID.
        """
        try:
            # Start Workflow Runtime
            self.start_runtime()

            # Generate unique instance ID
            instance_id = str(uuid.uuid4()).replace("-", "")

            # Check for existing workflows
            if instance_id in self.state.instances:
                logger.warning(f"Workflow instance {instance_id} already exists.")
                return

            # Prepare workflow input
            entry_input = input if isinstance(input, str) else json.dumps(input) if input else ""

            # Initialize workflow entry
            workflow_entry = WorkflowEntry(input=entry_input, status=WorkflowStatus.RUNNING)
            self.state.instances[instance_id] = workflow_entry
            self.save_state()

            # Resolve the workflow function
            workflow_func = self.resolve_workflow(workflow) if isinstance(workflow, str) else workflow

            # Schedule workflow execution
            instance_id = self.wf_client.schedule_new_workflow(
                workflow=workflow_func,
                input=input,
                instance_id=instance_id
            )

            logger.info(f"Started workflow with instance ID {instance_id}.")
            return instance_id
        except Exception as e:
            logger.error(f"Failed to start workflow {workflow}: {e}")
            raise
    
    async def monitor_workflow_completion(self, instance_id: str):
        """
        Monitor workflow instance in the background and handle its final state.
        """
        try:
            logger.info(f"Starting to monitor workflow '{instance_id}'...")

            state: WorkflowState = await asyncio.to_thread(
                self.wait_for_workflow_completion,
                instance_id,
                fetch_payloads=True,
                timeout_in_seconds=self.workflow_timeout,
            )

            if not state:
                logger.error(f"Workflow '{instance_id}' not found.")
                self.handle_workflow_output(instance_id, "Workflow not found.", WorkflowStatus.FAILED)
                return

            # Directly map runtime status to WorkflowStatus
            workflow_status = WorkflowStatus[state.runtime_status.name]

            if workflow_status == WorkflowStatus.COMPLETED:
                logger.info(f"Workflow '{instance_id}' completed successfully!")
                logger.debug(f"Output: {state.serialized_output}")
                self.handle_workflow_output(instance_id, state.serialized_output, WorkflowStatus.COMPLETED)
            else:
                logger.error(f"Workflow '{instance_id}' ended with status '{workflow_status.value}'.")
                self.handle_workflow_output(
                    instance_id,
                    f"Workflow ended with status: {workflow_status.value}.",
                    workflow_status,
                )

        except TimeoutError:
            logger.error(f"Workflow '{instance_id}' monitoring timed out.")
            self.handle_workflow_output(instance_id, "Workflow monitoring timed out.", WorkflowStatus.FAILED)
        except Exception as e:
            logger.error(f"Error monitoring workflow '{instance_id}': {e}")
            self.handle_workflow_output(instance_id, f"Error monitoring workflow: {e}", WorkflowStatus.FAILED)
        finally:
            logger.info(f"Finished monitoring workflow '{instance_id}'.")
            self.stop_runtime()
    
    def run_and_monitor_workflow(self, workflow: Union[str, Callable], input: Optional[Union[str, Dict[str, Any]]] = None) -> WorkflowState:
        """
        Run a workflow synchronously and handle its completion.

        Args:
            workflow (Union[str, Callable]): Workflow name or callable instance.
            input (Optional[Union[str, Dict[str, Any]]]): The input for the workflow.

        Returns:
            WorkflowState: The final state of the workflow after completion.
        """
        try:

            # Schedule the workflow
            instance_id = self.run_workflow(workflow, input=input)

            # Wait for workflow completion
            state: WorkflowState = self.wait_for_workflow_completion(
                instance_id,
                fetch_payloads=True,
                timeout_in_seconds=self.workflow_timeout,
            )

            if not state:
                logger.error(f"Workflow '{instance_id}' not found.")
                self.handle_workflow_output(instance_id, "Workflow not found.", WorkflowStatus.FAILED)
                raise RuntimeError(f"Workflow '{instance_id}' not found.")

            # Determine workflow status
            try:
                workflow_status = WorkflowStatus[state.runtime_status.name]
            except KeyError:
                workflow_status = WorkflowStatus.UNKNOWN
                logger.warning(f"Unrecognized workflow status '{state.runtime_status.name}'. Defaulting to UNKNOWN.")

            if workflow_status == WorkflowStatus.COMPLETED:
                logger.info(f"Workflow '{instance_id}' completed successfully!")
                logger.debug(f"Output: {state.serialized_output}")
                self.handle_workflow_output(instance_id, state.serialized_output, WorkflowStatus.COMPLETED)
            else:
                logger.error(f"Workflow '{instance_id}' ended with status '{workflow_status.value}'.")
                self.handle_workflow_output(
                    instance_id,
                    f"Workflow ended with status: {workflow_status.value}.",
                    workflow_status,
                )

            # Return the final state
            logger.info(f"Returning final output for workflow '{instance_id}'")
            logger.debug(f"Serialized Output: {state.serialized_output}")
            return state.serialized_output

        except TimeoutError:
            logger.error(f"Workflow '{instance_id}' monitoring timed out.")
            self.handle_workflow_output(instance_id, "Workflow monitoring timed out.", WorkflowStatus.FAILED)
            raise
        except Exception as e:
            logger.error(f"Error during workflow '{instance_id}': {e}")
            self.handle_workflow_output(instance_id, f"Error during workflow: {e}", WorkflowStatus.FAILED)
            raise
        finally:
            logger.info(f"Finished workflow with Instance ID: {instance_id}.")
            self.stop_runtime()
    
    def handle_workflow_output(self, instance_id: str, output: Any, status: WorkflowStatus):
        """
        Handle the output of a completed workflow.
        """
        try:
            # Check if workflow exists
            workflow_entry = self.state.instances.get(instance_id)
            if not workflow_entry:
                logger.error(f"Workflow with instance ID {instance_id} not found.")
                return
            
            # Update workflow entry
            workflow_entry.output = output
            workflow_entry.status = status
            workflow_entry.end_time = datetime.now()

            # Persist the updated state
            self.save_state()

            logger.info(f"Workflow '{instance_id}' output persisted successfully.")
        except Exception as e:
            logger.error(f"Failed to persist workflow output for '{instance_id}': {e}")
    
    def terminate_workflow(self, instance_id: str, *, output: Optional[Any] = None) -> None:
        """
        Terminates a running workflow instance.

        Args:
            instance_id (str): The ID of the workflow instance to terminate.
            output (Optional[Any]): The optional output to set for the terminated workflow instance.

        Raises:
            Exception: If the termination request fails.
        """
        try:
            self.wf_client.terminate_workflow(instance_id=instance_id, output=output)
            logger.info(f"Successfully terminated workflow '{instance_id}' with output: {output}")
        except Exception as e:
            logger.error(f"Failed to terminate workflow '{instance_id}'. Error: {e}")
            raise Exception(f"Error terminating workflow '{instance_id}': {e}")
    
    def get_workflow_state(self, instance_id: str) -> Optional[Any]:
        """
        Retrieve the state of the workflow instance with the given ID.

        Args:
            instance_id (str): The ID of the workflow instance to retrieve the state for.

        Returns:
            Optional[Any]: The state of the workflow instance if found, otherwise None.

        Raises:
            RuntimeError: If there is an issue retrieving the workflow state.
        """
        try:
            state = self.wf_client.get_workflow_state(instance_id)
            logger.info(f"Retrieved state for workflow {instance_id}: {state.runtime_status}.")
            return state
        except Exception as e:
            logger.error(f"Failed to retrieve workflow state for {instance_id}: {e}")
            return None
    
    def wait_for_workflow_completion(self, instance_id: str, fetch_payloads: bool = True, timeout_in_seconds: int = 120) -> Optional[WorkflowState]:
        """
        Wait for the workflow instance to complete and retrieve its state.

        Args:
            instance_id (str): The unique ID of the workflow instance to wait for.
            fetch_payloads (bool): Whether to fetch the input, output payloads, 
                                and custom status for the workflow instance. Defaults to True.
            timeout_in_seconds (int): The maximum time in seconds to wait for the workflow instance 
                                    to complete. Defaults to 120 seconds.

        Returns:
            Optional[WorkflowState]: The state of the workflow instance if it completes within the timeout, otherwise None.

        Raises:
            RuntimeError: If there is an issue waiting for the workflow completion.
        """
        try:
            state = self.wf_client.wait_for_workflow_completion(
                instance_id, fetch_payloads=fetch_payloads, timeout_in_seconds=timeout_in_seconds
            )
            if state:
                logger.info(f"Workflow {instance_id} completed with status: {state.runtime_status}.")
            else:
                logger.warning(f"Workflow {instance_id} did not complete within the timeout period.")
            return state
        except Exception as e:
            logger.error(f"Error while waiting for workflow {instance_id} completion: {e}")
            return None
    
    def raise_workflow_event(self, instance_id: str, event_name: str, *, data: Any | None = None) -> None:
        """
        Raises an event for a running Dapr workflow instance.

        Args:
            instance_id (str): The unique identifier of the workflow instance.
            event_name (str): The name of the event to raise in the workflow.
            data (Any | None): The optional data payload for the event.

        Raises:
            Exception: If raising the event fails.
        """
        try:
            logger.info(f"Raising workflow event '{event_name}' for instance '{instance_id}'")
            self.wf_client.raise_workflow_event(instance_id=instance_id, event_name=event_name, data=data)
            logger.info(f"Successfully raised workflow event '{event_name}' for instance '{instance_id}'!")
        except Exception as e:
            logger.error(
                f"Error raising workflow event '{event_name}' for instance '{instance_id}'. "
                f"Data: {data}, Error: {e}"
            )
            raise Exception(f"Failed to raise workflow event '{event_name}' for instance '{instance_id}': {str(e)}")
    
    def call_service(
        self,
        service: str,
        http_method: str = "POST",
        input: Optional[Dict[str, Any]] = None,
        timeout: Optional[int] = None
    ) -> Any:
        """
        Call an external agent service via Dapr.

        Args:
            service (str): The name of the agent service to call.
            http_method (str, optional): The HTTP method to use (e.g., "GET", "POST"). Defaults to "POST".
            input (Optional[Dict[str, Any]], optional): The input data to pass to the agent service. Defaults to None.
            timeout (Optional[int], optional): Timeout for the service call in seconds. Defaults to None.

        Returns:
            Any: The response from the agent service.

        Raises:
            Exception: If there is an error invoking the agent service.
        """
        try:
            with DaprClient() as d:
                resp = d.invoke_method(
                    service, 
                    "generate", 
                    http_verb=http_method, 
                    data=json.dumps(input) if input else None,
                    timeout=timeout
                )
                if resp.status_code != 200:
                    raise Exception(f"Error calling {service} service: {resp.status_code}: {resp.text}")
                agent_response = json.loads(resp.data.decode("utf-8"))
                logger.info(f"Agent's Result: {agent_response}")
                return agent_response
        except Exception as e:
            logger.error(f"Failed to call agent service: {e}")
            raise e
    
    def when_all(self, tasks: List[dtask.Task[T]]) -> dtask.WhenAllTask[T]:
        """
        Returns a task that completes when all of the provided tasks complete or when one of the tasks fails.

        This is useful in orchestrating multiple tasks in a workflow where you want to wait for all tasks
        to either complete or for the first one to fail.

        Args:
            tasks (List[dtask.Task[T]]): A list of task instances that should all complete.

        Returns:
            dtask.WhenAllTask[T]: A task that represents the combined completion of all the provided tasks.
        """
        return dtask.when_all(tasks)

    def when_any(self, tasks: List[dtask.Task[T]]) -> dtask.WhenAnyTask:
        """
        Returns a task that completes when any one of the provided tasks completes or fails.

        This is useful in scenarios where you want to proceed as soon as one of the tasks finishes, without
        waiting for the others to complete.

        Args:
            tasks (List[dtask.Task[T]]): A list of task instances, any of which can complete or fail to trigger completion.

        Returns:
            dtask.WhenAnyTask: A task that represents the completion of the first of the provided tasks to finish.
        """
        return dtask.when_any(tasks)
    
    def start_runtime(self):
        """
        Start the workflow runtime
        """

        logger.info("Starting workflow runtime.")
        self.wf_runtime.start()
    
    def stop_runtime(self):
        """
        Stop the workflow runtime.
        """
        logger.info("Stopping workflow runtime.")
        self.wf_runtime.shutdown()