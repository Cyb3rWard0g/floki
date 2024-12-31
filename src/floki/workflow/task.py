from pydantic import BaseModel, Field, ConfigDict, TypeAdapter, ValidationError
from typing import Any, Callable, Optional, Union, get_origin, get_args, List
from floki.types import ChatCompletion, BaseMessage, UserMessage
from floki.llm.utils import StructureHandler
from floki.llm.base import LLMClientBase
from floki.llm.openai import OpenAIChatClient
from dapr.ext.workflow import WorkflowActivityContext
from functools import update_wrapper
from types import SimpleNamespace
from dataclasses import is_dataclass
import asyncio
import inspect
import logging

logger = logging.getLogger(__name__)

class Task(BaseModel):
    """
    Encapsulates task logic for execution by an LLM, agent, or Python function.

    Supports both synchronous and asynchronous tasks, with optional output validation
    using Pydantic models or specified return types.
    """

    func: Optional[Callable] = Field(None, description="The original function to be executed, if provided.")
    description: Optional[str] = Field(None, description="A description template for the task, used with LLM or agent.")
    agent: Optional[Any] = Field(None, description="The agent used for task execution, if applicable.")
    agent_method: Union[str, Callable] = Field("run", description="The method or callable for invoking the agent.")
    llm: Optional[LLMClientBase] = Field(default_factory=OpenAIChatClient, description="The LLM client for executing the task, if applicable.")
    llm_method: Union[str, Callable] = Field("generate", description="The method or callable for invoking the LLM client.")

    # Initialized during setup
    signature: Optional[inspect.Signature] = Field(None, init=False, description="The signature of the provided function.")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization to set up function signatures and default LLM clients.
        """
        if self.description and not self.llm:
            self.llm = OpenAIChatClient()
        
        if self.func:
            update_wrapper(self, self.func)

        self.signature = inspect.signature(self.func) if self.func else None

        # Proceed with base model setup
        super().model_post_init(__context)
    
    async def __call__(self, ctx: WorkflowActivityContext, input: Any = None) -> Any:
        """
        Executes the task and validates its output.
        Ensures all coroutines are awaited before returning.
        """
        input = self._normalize_input(input) if input is not None else {}

        try:
            if self.agent or self.llm:
                # Task requires LLM/Agent
                description = self.description or (self.func.__doc__ if self.func else None)
                logger.info(f"Executing task with LLM/agent")
                result = await self._run_task(self.format_description(description, input))
                result = await self._validate_output_llm(result)
            elif self.func:
                # Task is a Python function
                logger.info(f"Executing Regular Task")
                if asyncio.iscoroutinefunction(self.func):
                    # Await async function
                    result = await self.func(**input)
                else:
                    # Call sync function
                    result = self._execute_function(input)
                result = await self._validate_output(result)
            else:
                raise ValueError("Task must have an LLM, agent, or regular function for execution.")
            
            return result
        except Exception as e:
            logger.error(f"Task execution error: {e}")
            raise

    def _normalize_input(self, input: Any) -> dict:
        """
        Converts input into a normalized dictionary.

        Args:
            input (Any): Input to normalize (e.g., dictionary, dataclass, or object).

        Returns:
            dict: Normalized dictionary representation of the input.
        """
        if is_dataclass(input):
            return input.__dict__
        elif isinstance(input, SimpleNamespace):
            return vars(input)
        elif not isinstance(input, dict):
            return self._single_value_to_dict(input)
        return input

    def _single_value_to_dict(self, value: Any) -> dict:
        """
        Wraps a single input value in a dictionary.

        Args:
            value (Any): Single input value.

        Returns:
            dict: Dictionary with parameter name as the key.
        """
        param_name = list(self.signature.parameters.keys())[0]
        return {param_name: value}
    
    def format_description(self, description: str, input: dict) -> str:
        """
        Formats a description string with input parameters.

        Args:
            description (str): Description template.
            input (dict): Input parameters for formatting.

        Returns:
            str: Formatted description string.
        """
        if self.signature:
            bound_args = self.signature.bind(**input)
            bound_args.apply_defaults()
            return description.format(**bound_args.arguments)
        return description.format(**input)
    
    async def _run_task(self, formatted_description: str) -> Any:
        """
        Determine whether to run the task using an agent or an LLM.

        Args:
            formatted_description (str): The formatted description to pass to the agent or LLM.

        Returns:
            Any: The result of the agent or LLM execution.

        Raises:
            ValueError: If neither an agent nor an LLM is provided.
        """
        logger.debug(f"Task Description: {formatted_description}")

        if self.agent:
            return await self._run_agent(formatted_description)
        elif self.llm:
            return await self._run_llm(formatted_description)
        else:
            raise ValueError("No agent or LLM provided.")

    async def _run_agent(self, description: str) -> Any:
        """
        Execute the task using the provided agent.

        Args:
            description (str): The formatted description to pass to the agent.

        Returns:
            Any: The result of the agent execution.
        """
        logger.info("Running task with agent...")
        if isinstance(self.agent_method, str):
            agent_callable = getattr(self.agent, self.agent_method, None)
            if not agent_callable:
                raise AttributeError(f"Agent does not have a method named '{self.agent_method}'.")
        elif callable(self.agent_method):
            agent_callable = self.agent_method
        else:
            raise ValueError("Invalid agent method provided.")

        # Execute the agent method
        result = await agent_callable({"task": description}) if asyncio.iscoroutinefunction(agent_callable) else agent_callable({"task": description})

        logger.debug(f"Agent result type: {type(result)}, value: {result}")
        return self._convert_result(result)

    async def _run_llm(self, description: Union[str, List[BaseMessage]]) -> Any:
        """
        Execute the task using the provided LLM.

        Args:
            description (Union[str, List[BaseMessage]]): The description to pass to the LLM.

        Returns:
            Any: The result of the LLM execution.

        Raises:
            AttributeError: If the LLM method does not exist.
            ValueError: If the LLM method is not callable.
        """
        logger.info("Running task with LLM...")

        # Ensure description is a list of UserMessages
        if isinstance(description, str):
            description = [UserMessage(description)]

        llm_params = {'messages': description}

        # Add response model if the signature specifies it
        if self.signature and self.signature.return_annotation is not inspect.Signature.empty:
            return_annotation = self.signature.return_annotation
            if isinstance(return_annotation, type) and issubclass(return_annotation, BaseModel):
                llm_params['response_model'] = return_annotation

        # Resolve and call the LLM method
        if isinstance(self.llm_method, str):
            llm_callable = getattr(self.llm, self.llm_method, None)
            if not llm_callable:
                raise AttributeError(f"The LLM does not have a method named '{self.llm_method}'.")
        elif callable(self.llm_method):
            llm_callable = self.llm_method
        else:
            raise ValueError("Invalid LLM method provided.")

        # Execute the LLM method (async or sync)
        result = await llm_callable(**llm_params) if asyncio.iscoroutinefunction(llm_callable) else llm_callable(**llm_params)

        logger.debug(f"LLM result type: {type(result)}, value: {result}")
        return self._convert_result(result)

    def _convert_result(self, result: Any) -> Any:
        """
        Convert the task result to a dictionary if necessary.

        Args:
            result (Any): The raw task result.

        Returns:
            Any: The converted result.
        """
        if isinstance(result, ChatCompletion):
            logger.info("Extracted message content from ChatCompletion.")
            return result.get_content()

        if isinstance(result, BaseModel):
            logger.info("Converting Pydantic model to dictionary.")
            return result.model_dump()

        if isinstance(result, list) and all(isinstance(item, BaseModel) for item in result):
            logger.info("Converting list of Pydantic models to list of dictionaries.")
            return [item.model_dump() for item in result]

        # If no specific conversion is necessary, return as-is
        logger.info("Returning final task result.")
        return result
    
    def _execute_function(self, input: dict) -> Any:
        """
        Execute the wrapped function with the provided input.

        Args:
            input (dict): The input data to pass to the function.

        Returns:
            Any: The result of the function execution.
        """
        return self.func(**input)
    
    async def _validate_output_llm(self, result: Any) -> Any:
        """
        Specialized validation for LLM task outputs.

        Args:
            result (Any): The result to validate.

        Returns:
            Any: The validated result.

        Raises:
            TypeError: If the result does not match the expected type or validation fails.
        """
        if asyncio.iscoroutine(result):
            logger.error("Unexpected coroutine detected during validation.")
            result = await result 

        if self.signature:
            expected_type = self.signature.return_annotation

            if expected_type and expected_type is not inspect.Signature.empty:
                origin = get_origin(expected_type)

                # Handle Union types
                if origin is Union:
                    valid_types = get_args(expected_type)
                    if not isinstance(result, valid_types):
                        raise TypeError(f"Expected return type to be one of {valid_types}, but got {type(result)}")
                    return result

                # Handle Pydantic models
                if isinstance(expected_type, type) and issubclass(expected_type, BaseModel):
                    try:
                        validated_result = StructureHandler.validate_response(result, expected_type)
                        return validated_result.model_dump()
                    except ValidationError as e:
                        raise TypeError(f"Validation failed for type {expected_type}: {e}")

                # Handle lists of Pydantic models
                if origin:
                    args = get_args(expected_type)
                    if origin is list and len(args) == 1 and issubclass(args[0], BaseModel):
                        if not all(isinstance(item, args[0]) for item in result):
                            raise TypeError(f"Expected all items in the list to be of type {args[0]}, but got {type(result)}")
                        return [StructureHandler.validate_response(item, args[0]) for item in result]

        # If no specific validation applies, return the result as-is
        return result
    
    async def _validate_output(self, result: Any) -> Any:
        """
        Validate the output of the task against the expected type.

        Args:
            result (Any): The result to validate.

        Returns:
            Any: The validated result.

        Raises:
            ValidationError: If the result does not match the expected type.
        """
        if self.signature:
            expected_type = self.signature.return_annotation

            if expected_type and expected_type is not inspect.Signature.empty:
                # Use TypeAdapter for validation
                try:
                    adapter = TypeAdapter(expected_type)
                    validated_result = adapter.validate_python(result)
                    return validated_result
                except ValidationError as e:
                    raise TypeError(f"Validation failed for type {expected_type}: {e}")

        # If no specific validation applies, return the result as-is
        return result

class TaskWrapper:
    """
    A wrapper for the Task class that allows it to be used as a callable with a __name__ attribute.
    """

    def __init__(self, task_instance: Task, name: str):
        """
        Initialize the TaskWrapper.

        Args:
            task_instance (Task): The task instance to wrap.
            name (str): The name of the task.
        """
        self.task_instance = task_instance
        self.__name__ = name

    def __call__(self, *args, **kwargs):
        """
        Delegate the call to the wrapped Task instance.

        Args:
            *args: Positional arguments to pass to the Task's __call__ method.
            **kwargs: Keyword arguments to pass to the Task's __call__ method.

        Returns:
            Any: The result of the Task's __call__ method.
        """
        return self.task_instance(*args, **kwargs)

    def __getattr__(self, item):
        """
        Delegate attribute access to the Task instance.

        Args:
            item (str): The attribute to access.

        Returns:
            Any: The value of the attribute on the Task instance.
        """
        return getattr(self.task_instance, item)