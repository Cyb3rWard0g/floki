from typing import Any, Callable, Optional, Union, get_origin, get_args, List
from dapr.ext.workflow import WorkflowActivityContext
from floki.types import ChatCompletion, BaseMessage, UserMessage
from floki.llm.base import LLMClientBase
from floki.llm.utils import StructureHandler
from floki.llm.openai import OpenAIChatClient
from pydantic import BaseModel, Field, ConfigDict
from functools import update_wrapper
from types import SimpleNamespace
from dataclasses import is_dataclass
import inspect
import logging

logger = logging.getLogger(__name__)

class Task(BaseModel):
    """
    A class encapsulating task logic for execution by an LLM, agent, or Python function.
    """

    func: Optional[Callable] = Field(None, description="The original function to be executed, if provided.")
    description: Optional[str] = Field(None, description="A description template for the task, used with LLM or agent.")
    agent: Optional[Any] = Field(None, description="The agent used for task execution, if applicable.")
    agent_method: Union[str, Callable] = Field("run", description="The method or callable for invoking the agent.")
    llm: Optional[LLMClientBase] = Field(default_factory=OpenAIChatClient, description="The LLM client for executing the task, if applicable.")
    llm_method: Union[str, Callable] = Field("generate", description="The method or callable for invoking the LLM client.")

    # Initialized in model_post_init
    signature: Optional[inspect.Signature] = Field(None, init=False, description="The signature of the provided function.")
    
    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        if self.description and not self.llm:
            self.llm = OpenAIChatClient()
        
        if self.func:
            update_wrapper(self, self.func)
        
        self.signature = inspect.signature(self.func) if self.func else None

        # Proceed with base model setup
        super().model_post_init(__context)
    
    def __call__(self, ctx: WorkflowActivityContext, input: Any = None) -> Any:
        """
        Executes the task based on the configured function, agent, or LLM.

        Args:
            ctx (WorkflowActivityContext): Execution context for the task.
            input (Any): Task input, normalized to a dictionary.

        Returns:
            Any: The result of the task execution.

        Raises:
            ValueError: If the task is not properly configured with either a function, an agent, or an LLM.
        """

        # Normalize the input
        input = self._normalize_input(input) if input is not None else {}

        # Determine the description only for agent or LLM tasks
        if self.agent or self.llm:
            description = self.description or (self.func.__doc__ if self.func else None)
        else:
            description = None

        if description:
            result = self._run_task(self.format_description(self.description, input))
        elif self.func:
            result = self._execute_function(input or {})
        else:
            raise ValueError("Task must have a function or description for execution.")
        
        # Validate and return the output
        return self._validate_output(result)

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
            logger.info("FORMAT DESCRIPTION")
            return description.format(**bound_args.arguments)
        return description.format(**input)

    def _run_task(self, formatted_description: str) -> Any:
        """
        Determine whether to run the task using an agent or an LLM.

        This method delegates the execution to either the _run_agent or _run_llm method
        based on whether an agent or LLM is provided.

        Args:
            formatted_description (str): The formatted description to pass to the agent or LLM.

        Returns:
            Any: The result of the agent or LLM execution.

        Raises:
            ValueError: If neither an agent nor an LLM is provided.
        """
        logger.info(f"Running task with description: {formatted_description}")
        if self.agent:
            return self._run_agent(formatted_description)
        elif self.llm:
            return self._run_llm(formatted_description)
        else:
            raise ValueError("No agent or LLM provided.")

    def _execute_function(self, input: dict) -> Any:
        """
        Execute the wrapped function with the provided input.

        Args:
            input (dict): The input data to pass to the function.

        Returns:
            Any: The result of the function execution.
        """
        return self.func(**input)

    def _run_agent(self, description: str) -> Any:
        """
        Execute the task using the provided agent.

        Args:
            description (str): The formatted description to pass to the agent.

        Returns:
            Any: The result of the agent execution.

        Raises:
            AttributeError: If the agent method does not exist.
            ValueError: If the agent method is not callable.
        """
        if isinstance(self.agent_method, str):
            if hasattr(self.agent, self.agent_method):
                agent_callable = getattr(self.agent, self.agent_method)
                return agent_callable({"task": description})
            else:
                raise AttributeError(f"The agent does not have a method named '{self.agent_method}'.")
        elif callable(self.agent_method):
            return self.agent_method(self.agent, {"task": description})
        else:
            raise ValueError("Invalid agent method provided.")
    
    def _run_llm(self, description: Union[str, List[BaseMessage]]) -> Any:
        """
        Execute the task using the provided LLM.

        If the description is a string, it is converted into a UserMessage. 
        If the description is a list of BaseMessage, it is passed as-is.

        Args:
            description (Union[str, List[BaseMessage]]): The description to pass to the LLM.

        Returns:
            Any: The result of the LLM execution.

        Raises:
            AttributeError: If the LLM method does not exist.
            ValueError: If the LLM method is not callable.
        """
        if isinstance(description, str):
            description = [UserMessage(description)]

        # Prepare the parameters for the LLM call
        llm_params = {'messages': description}

        # Add response_model to parameters if it exists
        if self.signature and self.signature.return_annotation is not inspect.Signature.empty:
            return_annotation = self.signature.return_annotation
            if isinstance(return_annotation, type) and issubclass(return_annotation, BaseModel):
                llm_params['response_model'] = return_annotation

        # Determine if llm_method is a string or callable and execute accordingly
        if isinstance(self.llm_method, str):
            if hasattr(self.llm, self.llm_method):
                llm_callable = getattr(self.llm, self.llm_method)
                result = llm_callable(**llm_params)
            else:
                raise AttributeError(f"The LLM does not have a method named '{self.llm_method}'.")
        elif callable(self.llm_method):
            result = self.llm_method(self.llm, **llm_params)
        else:
            raise ValueError("Invalid LLM method provided.")

        # Check if the result is a ChatCompletion and extract the message content
        if isinstance(result, ChatCompletion):
            message_content = result.get_content()
            logger.info(f"Extracted message content: {message_content}")
            return message_content

        # If the result is a Pydantic model, convert it to a dictionary
        if isinstance(result, BaseModel):
            logger.info("Converting Pydantic model to dictionary.")
            return result.model_dump()

        # If none of the above conditions are met
        logger.info(f"Final result: {result}")
        return result

    def _validate_output(self, result: Any) -> Any:
        """
        Validate the output of the task against the expected type.

        This method checks whether the result matches the expected return type
        defined in the function signature. It supports basic types, Pydantic models,
        and complex types like Unions.

        Args:
            result (Any): The result to validate.

        Returns:
            Any: The validated result.

        Raises:
            ValidationError: If the result does not match the expected Pydantic model.
            TypeError: If the return type annotation is not as expected.
        """
        if self.signature:
            expected_type = self.signature.return_annotation

            if expected_type and expected_type is not inspect.Signature.empty:
                origin = get_origin(expected_type)
                if origin is Union:
                    valid_types = get_args(expected_type)
                    if not isinstance(result, valid_types):
                        raise TypeError(f"Expected return type to be one of {valid_types}, but got {type(result)}")
                    return result

                if issubclass(expected_type, BaseModel):
                    if not isinstance(StructureHandler.validate_response(result, expected_type), expected_type):
                        raise TypeError(f"Expected return type to be {expected_type}, but got {type(result)}")
                    return result
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