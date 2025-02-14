from dapr.ext.workflow import WorkflowRuntime, WorkflowActivityContext, DaprWorkflowContext, DaprWorkflowClient
from dapr.ext.workflow.workflow_state import WorkflowState
from floki.types.workflow import WorkflowStatus
from typing import Any, Callable, Generator, Optional, Dict, TypeVar, Union, List, Type
from floki.storage.daprstores.statestore import DaprStateStore
from floki.workflow.task import Task, TaskWrapper
from pydantic import BaseModel, Field, ConfigDict, ValidationError
from dapr.conf import settings as dapr_settings
from dapr.clients import DaprClient
from durabletask import task as dtask
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

    This class provides:
        - Workflow runtime initialization and management.
        - Task and workflow registration using decorators.
        - Workflow state persistence using Dapr state store.
        - Methods to start, monitor, and interact with workflows.

    Attributes:
        daprGrpcHost (Optional[str]): Host address for the Dapr gRPC endpoint.
        daprGrpcPort (Optional[int]): Port number for the Dapr gRPC endpoint.
        state_store_name (str): The Dapr state store component for persisting workflow state.
        state_name (str): The key under which workflow state is stored.
        state (Union[BaseModel, dict]): The current workflow state, which can be a dictionary or Pydantic model.
        state_format (Optional[Type[BaseModel]]): An optional schema for validating workflow state.
        timeout (int): Timeout duration (seconds) for workflow tasks.
        
        wf_runtime (WorkflowRuntime): The Dapr Workflow runtime instance.
        wf_client (DaprWorkflowClient): The Dapr Workflow client instance for managing workflows.
        state_store (Optional[DaprStateStore]): Dapr state store instance for workflow state.
        tasks (Dict[str, Callable]): Internal dictionary of registered task functions.
        workflows (Dict[str, Callable]): Internal dictionary of registered workflows.
    """

    daprGrpcHost: Optional[str] = Field(None, description="Host address for the Dapr gRPC endpoint.")
    daprGrpcPort: Optional[int] = Field(None, description="Port number for the Dapr gRPC endpoint.")
    state_store_name: str = Field(default="workflowstatestore", description="The name of the Dapr state store component used to store workflow metadata.")
    state_name: str = Field(default="workflowstate", description="Dapr state store key for the workflow state.")
    state: Union[BaseModel, dict] = Field(default_factory=dict, description="The current state of the workflow, which can be a dictionary or a Pydantic object.")
    state_format: Optional[Type[BaseModel]] = Field(default=None, description="The schema to enforce state structure and validation.")
    timeout: int = Field(default=300, description="Default timeout duration in seconds for workflow tasks.")

    # Initialized in model_post_init
    wf_runtime: Optional[WorkflowRuntime] = Field(default=None, init=False, description="Workflow runtime instance.")
    wf_client: Optional[DaprWorkflowClient] = Field(default=None, init=False, description="Workflow client instance.")
    state_store: Optional[DaprStateStore] = Field(default=None, init=False, description="Dapr state store instance for accessing and managing workflow state.")
    tasks: Dict[str, Callable] = Field(default_factory=dict, init=False, description="Dictionary of registered tasks.")
    workflows: Dict[str, Callable] = Field(default_factory=dict, init=False, description="Dictionary of registered workflows.")
    
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
        self.state_store = DaprStateStore(store_name=self.state_store_name, address=f"{self.daprGrpcHost}:{self.daprGrpcPort}")

        # Load or initialize state
        self.initialize_state()

        logger.info(f"Initialized WorkflowApp with Dapr gRPC host '{self.daprGrpcHost}' and port '{self.daprGrpcPort}'.")

        # Proceed with base model setup
        super().model_post_init(__context)
    
    def initialize_state(self) -> None:
        """
        Initializes the workflow state:
        - Uses `state` if provided (converting BaseModel to dict if needed).
        - Loads from storage if `state` is not set.
        - Defaults to `{}` if no stored state exists.
        - Always persists the final state.
        """
        try:
            if isinstance(self.state, BaseModel):
                logger.info("User provided state of type BaseModel.")
                self.state = self.state.model_dump()

            if not isinstance(self.state, dict):
                raise TypeError(f"Invalid state type: {type(self.state)}. Expected dict or BaseModel.")

            if not self.state:
                logger.info("No user-provided state. Loading from storage.")
                self.state = self.load_state()

            logger.info(f"Workflow state initialized with {len(self.state)} keys.")
            self.save_state()

        except Exception as e:
            logger.error(f"Failed to initialize workflow state: {e}")
            raise RuntimeError(f"Error initializing workflow state: {e}") from e
    
    def load_state(self) -> dict:
        """
        Loads and validates the workflow state from the Dapr state store.

        Returns:
            dict: The loaded and validated state.

        Raises:
            RuntimeError: If loading the state fails.
            TypeError: If the retrieved state is not a dictionary.
            ValidationError: If schema validation fails.
        """
        try:
            has_state, state_data = self.state_store.try_get_state(self.state_name)

            if has_state and state_data:
                logger.info(f"Existing state found for key '{self.state_name}'. Validating it.")

                if not isinstance(state_data, dict):
                    raise TypeError(f"Invalid state type retrieved: {type(state_data)}. Expected dict.")

                try:
                    return self.validate_state(state_data) if self.state_format else state_data
                except ValidationError as e:
                    logger.error(f"State validation failed: {e}")
                    return {}

            logger.info(f"No existing state found for key '{self.state_name}'. Initializing empty state.")
            return {}

        except Exception as e:
            logger.error(f"Failed to load state for key '{self.state_name}': {e}")
            raise RuntimeError(f"Error loading workflow state: {e}") from e
    
    def validate_state(self, state_data: dict) -> dict:
        """
        Validates workflow state against `state_format`.

        Args:
            state_data (dict): The state data to validate.

        Returns:
            dict: The validated state.

        Raises:
            ValidationError: If schema validation fails.
        """
        try:
            logger.info("Validating workflow state against schema.")
            validated_state: BaseModel = self.state_format(**state_data)
            return validated_state.model_dump()
        except ValidationError as e:
            logger.error(f"State validation failed: {e}")
            raise ValidationError(f"Invalid workflow state: {e}")
    
    def save_state(self, state: Optional[Union[dict, BaseModel, str]] = None) -> None:
        """
        Saves the workflow state to the Dapr state store using the predefined workflow state key.

        Args:
            state (Optional[Union[dict, BaseModel, str]]): The state data to save.
        """
        try:
            # Use provided state or fallback to self.state
            state_to_save = state or self.state
            if not state_to_save:
                logger.warning("Skipping state save: Empty state.")
                return

            # Convert state to JSON string
            if isinstance(state_to_save, BaseModel):
                state_to_save = state_to_save.model_dump_json()  # Convert BaseModel to JSON string
            elif isinstance(state_to_save, dict):
                state_to_save = json.dumps(state_to_save)  # Convert dictionary to JSON string
            elif isinstance(state_to_save, str):
                # Ensure it's a valid JSON string before storing
                try:
                    json.loads(state_to_save)  # Validate that it's a valid JSON string
                except json.JSONDecodeError as e:
                    raise ValueError(f"Invalid JSON string provided as state: {e}")
            else:
                raise TypeError(f"Invalid state type: {type(state_to_save)}. Expected dict, BaseModel, or JSON string.")

            # Save the state in Dapr (Dapr expects str or bytes)
            self.state_store.save_state(self.state_name, state_to_save)
            logger.info(f"Successfully saved state for key '{self.state_name}'.")
        except Exception as e:
            logger.error(f"Failed to save state for key '{self.state_name}': {e}")
            raise
    
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
        Start a workflow.

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
    
    def get_workflow_state(self, instance_id: str) -> Optional[WorkflowState]:
        """
        Retrieves the final state of a workflow instance.

        Args:
            instance_id (str): The ID of the workflow instance.

        Returns:
            Optional[WorkflowState]: The final state of the workflow, or None if not found.
        """
        try:
            state: WorkflowState = self.wait_for_workflow_completion(
                instance_id,
                fetch_payloads=True,
                timeout_in_seconds=self.timeout,
            )

            if not state:
                logger.error(f"Workflow '{instance_id}' not found.")
                return None

            return state
        except TimeoutError:
            logger.error(f"Workflow '{instance_id}' monitoring timed out.")
            return None
        except Exception as e:
            logger.error(f"Error retrieving workflow state for '{instance_id}': {e}")
            return None
    
    async def monitor_workflow_completion(self, instance_id: str):
        """
        Monitors the execution of a workflow instance asynchronously and logs its final state.

        Args:
            instance_id (str): The unique identifier of the workflow instance to monitor.

        This method waits for the workflow to complete, then logs whether it succeeded, failed, or timed out.
        It does not return a value but provides visibility into the workflow’s execution status.
        """
        try:
            logger.info(f"Starting to monitor workflow '{instance_id}'...")

            # Retrieve workflow state
            state = await asyncio.to_thread(self.get_workflow_state, instance_id)

            if not state:
                return  # Error already logged in get_workflow_state

            workflow_status = WorkflowStatus[state.runtime_status.name] if state.runtime_status.name in WorkflowStatus.__members__ else WorkflowStatus.UNKNOWN

            if workflow_status == WorkflowStatus.COMPLETED:
                logger.info(f"Workflow '{instance_id}' completed successfully!")
                logger.debug(f"Output: {state.serialized_output}")
            else:
                logger.error(f"Workflow '{instance_id}' ended with status '{workflow_status.value}'.")

        except Exception as e:
            logger.error(f"Error monitoring workflow '{instance_id}': {e}")
        finally:
            logger.info(f"Finished monitoring workflow '{instance_id}'.")
    
    def run_and_monitor_workflow(self, workflow: Union[str, Callable], input: Optional[Union[str, Dict[str, Any]]] = None) -> Optional[str]:
        """
        Run a workflow synchronously and handle its completion.

        Args:
            workflow (Union[str, Callable]): Workflow name or callable instance.
            input (Optional[Union[str, Dict[str, Any]]]): The input for the workflow.

        Returns:
            Optional[str]: The serialized output of the workflow after completion.
        """
        try:
            # Schedule the workflow
            instance_id = self.run_workflow(workflow, input=input)

            # Retrieve workflow state
            state = self.get_workflow_state(instance_id)

            if not state:
                raise RuntimeError(f"Workflow '{instance_id}' not found.")

            workflow_status = WorkflowStatus[state.runtime_status.name] if state.runtime_status.name in WorkflowStatus.__members__ else WorkflowStatus.UNKNOWN

            if workflow_status == WorkflowStatus.COMPLETED:
                logger.info(f"Workflow '{instance_id}' completed successfully!")
                logger.debug(f"Output: {state.serialized_output}")
            else:
                logger.error(f"Workflow '{instance_id}' ended with status '{workflow_status.value}'.")

            # Return the final state output
            return state.serialized_output

        except Exception as e:
            logger.error(f"Error during workflow '{instance_id}': {e}")
            raise
        finally:
            logger.info(f"Finished workflow with Instance ID: {instance_id}.")
            self.stop_runtime()
    
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