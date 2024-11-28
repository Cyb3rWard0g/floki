from floki.tool.utils.function_calling import to_function_call_definition
from pydantic import BaseModel, Field, ValidationError, model_validator
from typing import Callable, Type, Optional, Any, Dict
from floki.tool.utils.tool import ToolHelper
from inspect import signature, Parameter
from floki.types import ToolError
import logging

logger = logging.getLogger(__name__)

class AgentTool(BaseModel):
    """
    Base class for agent tools, structuring both class-based and function-based tools.
    
    Attributes:
        name (str): The tool's name.
        description (str): A brief description of the tool's purpose.
        args_model (Optional[Type[BaseModel]]): Model for validating tool arguments.
        func (Optional[Callable]): Function defining tool behavior.
    """
    name: str = Field(..., description="The name of the tool, formatted with capitalization and no spaces.")
    description: str = Field(..., description="A brief description of the tool's functionality.")
    args_model: Optional[Type[BaseModel]] = Field(None, description="Pydantic model for validating tool arguments.")
    func: Optional[Callable] = Field(None, description="Optional function implementing the tool's behavior.")

    @model_validator(mode='before')
    @classmethod
    def set_name_and_description(cls, values: dict) -> dict:
        """
        Validator to dynamically set `name` and `description` before Pydantic validation.
        This ensures that the `name` is formatted and derived either from the class or from the `func`.
        
        Args:
            values (dict): A dictionary of field values before validation.

        Returns:
            dict: Updated field values after processing.
        """
        func = values.get('func')
        if func:
            values['name'] = values.get('name', func.__name__)
            values['description'] = func.__doc__
        return values

    @classmethod
    def from_func(cls, func: Callable) -> 'AgentTool':
        """
        Creates an instance of `AgentTool` from a raw Python function.

        Args:
            func (Callable): The function to wrap in the tool.

        Returns:
            AgentTool: An instance of `AgentTool`.
        """
        ToolHelper.check_docstring(func)
        return cls(func=func)
    
    def model_post_init(self, __context: Any) -> None:
        """
        Handles post-initialization logic for both class-based and function-based tools.
        Ensures `name` formatting and infers `args_model` if necessary.
        """
        self.name = self.name.replace(' ', '_').title().replace('_', '')

        if self.func:
            self._initialize_from_func(self.func)
        else:
            self._initialize_from_run()
        
        return super().model_post_init(__context)

    def _initialize_from_func(self, func: Callable) -> None:
        """Initialize Tool fields from a provided function."""
        if self.args_model is None:
            self.args_model = ToolHelper.infer_func_schema(func)

    def _initialize_from_run(self) -> None:
        """Initialize Tool fields based on the abstract `_run` method."""
        if self.args_model is None:
            self.args_model = ToolHelper.infer_func_schema(self._run)

    def _run(self, *args, **kwargs) -> Any:
        """Provide default implementation of _run to support function-based tools."""
        if self.func:
            return self.func(*args, **kwargs)
        raise NotImplementedError("No function or _run method defined for this tool.")
    
    def run(self, *args, **kwargs) -> Any:
        """
        Executes the tool by running either the class-based `_run` method or the function-based `func`.
        
        Args:
            *args: Positional arguments for the tool.
            **kwargs: Keyword arguments for the tool.

        Returns:
            Any: The result of executing the tool.

        Raises:
            ValidationError: If argument validation fails.
            ToolError: If any other error occurs during execution.
        """
        try:
            if self.func:
                return self._run_function(*args, **kwargs)
            return self._run_method(*args, **kwargs)
        except ValidationError as e:
            self._log_and_raise_error(e)
        except Exception as e:
            self._log_and_raise_error(e)

    def _run_method(self, *args, **kwargs) -> Any:
        """Validates and executes the class-based `_run` method."""
        return self._execute_with_signature(self._run, *args, **kwargs)

    def _run_function(self, *args, **kwargs) -> Any:
        """Validates and executes the provided function `func`."""
        return self._execute_with_signature(self.func, *args, **kwargs)

    def _execute_with_signature(self, func: Callable, *args, **kwargs) -> Any:
        """
        Validates and executes a function (either class-based or function-based) using its signature.
        
        Args:
            func (Callable): The function to execute.
            *args: Positional arguments for the function.
            **kwargs: Keyword arguments for the function.

        Returns:
            Any: The result of executing the function.
        """
        sig = signature(func)
        
        # Update kwargs with args by mapping positional arguments to parameter names
        if args:
            arg_names = list(sig.parameters.keys())
            kwargs.update(dict(zip(arg_names, args)))

        # Validate and execute the function
        if self.args_model:
            validated_args = self.args_model(**kwargs)  # Validate keyword arguments
            kwargs = validated_args.model_dump()  # Convert validated model back to dict

        return func(**kwargs)

    def _log_and_raise_error(self, error: Exception) -> None:
        """Log the error and raise a ToolError."""
        logger.error(f"Error executing tool '{self.name}': {str(error)}")
        raise ToolError(f"An error occurred during the execution of tool '{self.name}': {str(error)}")

    def __call__(self, *args, **kwargs) -> Any:
        """Allow the AgentTool instance to be called like a regular function."""
        return self.run(*args, **kwargs)

    def to_function_call(self, format_type: str = 'openai', use_deprecated: bool = False) -> Dict:
        """
        Converts the tool to a specified function call format.

        Args:
            format_type (str): The format type (e.g., 'openai').
            use_deprecated (bool): Whether to use deprecated format.

        Returns:
            Dict: The function call representation.
        """
        return to_function_call_definition(self.name, self.description, self.args_model, format_type, use_deprecated)

    def __repr__(self) -> str:
        """Returns a string representation of the AgentTool."""
        return f"AgentTool(name={self.name}, description={self.description})"
    
    @property
    def args_schema(self) -> dict:
        """Returns a JSON-serializable dictionary of the tool's function args_model."""
        if self.args_model:
            schema = self.args_model.model_json_schema()
            for property_details in schema.get('properties', {}).values():
                property_details.pop('title', None)
            return schema["properties"]
        return {}

    @property
    def signature(self) -> str:
        """Provides a dynamic and detailed string representation of the tool's function signature."""
        func_to_inspect = self.func if self.func else self._run
        params = signature(func_to_inspect).parameters
        args = [
            f"{name}: {param.annotation.__name__ if param.annotation != Parameter.empty else 'Any'}"
            f"{' = ' + repr(param.default) if param.default != Parameter.empty else ''}"
            for name, param in params.items()
        ]
        return f"{self.name}({', '.join(args)})"

def tool(func: Optional[Callable] = None, *, args_model: Optional[Type[BaseModel]] = None) -> AgentTool:
    """
    A decorator to wrap a function with an `AgentTool` for validation and metadata.

    Args:
        func (Optional[Callable]): The function to wrap.
        args_model (Optional[Type[BaseModel]]): Optional Pydantic model for argument validation.

    Returns:
        AgentTool: The wrapped function as an `AgentTool`.
    """
    def decorator(f: Callable) -> AgentTool:
        ToolHelper.check_docstring(f)
        return AgentTool(func=f, args_model=args_model)
    return decorator(func) if func else decorator