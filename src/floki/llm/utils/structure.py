from typing import Type, TypeVar, Optional, Union, List, get_args, Any
from pydantic import BaseModel, Field, create_model, ValidationError
from floki.tool.utils.function_calling import to_function_call_definition
from floki.types import StructureError
from collections.abc import Iterable
import logging
import json

logger = logging.getLogger(__name__)

T = TypeVar("T", bound=BaseModel)

class StructureHandler:
    @staticmethod
    def is_json_string(input_string: str) -> bool:
        """
        Check if the given input is a valid JSON string.

        Args:
            input_string (str): The string to check.

        Returns:
            bool: True if the input is a valid JSON string, False otherwise.
        """
        try:
            json.loads(input_string)
            return True
        except json.JSONDecodeError:
            return False
    
    @staticmethod
    def generate_request(
        response_model: Union[Type[T],Type[Iterable[T]]],
        llm_provider: str,
        **params
    ) -> dict:
        """
        Generates a structured request that conforms to a specified API format using the given Pydantic model.
        This function prepares a request configuration that includes the model as a tool specification and
        sets the necessary parameters for the API call.

        Args:
            response_model (Union[Type[T], Type[List[T]]]): The Pydantic model that defines the schema of the response data.
            llm_provider: The LLM provider to use (e.g., 'openai')for generating the function call definition.
                                        Supported formats include 'openai' and 'claude'.
            **params: Additional keyword arguments that can be included in the API request.

        Returns:
            dict: A dictionary of parameters enhanced with 'tools' and 'tool_choice' configurations,
                  which define how the API should handle the response structure based on the response model.
        """
        logger.info("Structured response enabled.")

        if isinstance(response_model, Iterable) is True:
            logger.info("Response model is of type Iterable.")
            item_model = get_args(response_model)[0]
            response_model = StructureHandler.create_iterable_model(item_model)

        name = response_model.__name__
        description = response_model.__doc__ or ""
        args_schema = response_model
        
        model_tool_format = to_function_call_definition(name, description, args_schema, llm_provider)
        
        params['tools'] = [model_tool_format]
        params['tool_choice'] = {
            "type": "function",
            "function": {"name": model_tool_format['function']['name']},
        }
        return params

    @staticmethod
    def create_iterable_model(
        model: Type[BaseModel],
        model_name: Optional[str] = None,
        model_description: Optional[str] = None
    ) -> Type[BaseModel]:
        """
        Constructs an iterable Pydantic model for a given Pydantic model.

        Args:
            model (Type[BaseModel]): The original Pydantic model to capture a list of objects of the original model type.
            model_name (Optional[str]): The name of the new iterable model. Defaults to None.
            model_description (Optional[str]): The description of the new iterable model. Defaults to None.

        Returns:
            Type[BaseModel]: A new Pydantic model class representing a list of the original pydantic model.
        """
        model_name = model.__name__ if model_name is None else model_name
        iterable_model_name = f"Iterable{model_name}"

        list_field = (
            List[model],
            Field(
                default_factory=list,
                repr=False,
                description=f"A list of `{model_name}` objects",
            ),
        )

        iterable_model = create_model(
            iterable_model_name,
            objects=list_field,
            __base__=(BaseModel,)
        )

        iterable_model.__doc__ = (
            f"A Pydantic model to capture `{iterable_model_name}` objects"
            if model_description is None
            else model_description
        )
        
        print(type(iterable_model))
        return iterable_model

    @staticmethod
    def extract_structured_response(response: Any, llm_provider: str) -> str:
        """
        Extracts the structured JSON string from the response based on the LLM provider.

        Args:
            response (Any): The API response data to extract.
            llm_provider: The LLM provider to use (e.g., 'openai').

        Returns:
            str: The extracted JSON string.

        Raises:
            StructureError: If extraction fails.
        """
        try:
            if llm_provider == "openai":
                extracted_response = response.choices[0].message.tool_calls[0].function.arguments
                logger.debug(f"Extracted structured response: {extracted_response}")
                return extracted_response
            else:
                raise StructureError(f"Unsupported LLM provider: {llm_provider}")
        except KeyError as e:
            logger.error(f"Key error while extracting structured response: {e}")
            raise StructureError(f"Extraction failed for structured response: {e}")

    @staticmethod
    def validate_response(response: Union[str, dict], model: Type[T]) -> T:
        """
        Validates a JSON string or a dictionary using a specified Pydantic model.

        This method checks whether the response is a JSON string or a dictionary. 
        If the response is a JSON string, it validates it using the `model_validate_json` method.
        If the response is a dictionary, it validates it using the `model_validate` method.

        Args:
            response (Union[str, dict]): The JSON string or dictionary to validate.
            model (Type[T]): The Pydantic model that defines the expected structure of the response.

        Returns:
            T: An instance of the Pydantic model populated with the validated data.

        Raises:
            StructureError: If the validation fails.
        """
        try:
            if isinstance(response, str) and StructureHandler.is_json_string(response):
                # If it's a valid JSON string, use model_validate_json
                return model.model_validate_json(response)
            elif isinstance(response, dict):
                # If it's a dictionary, use model_validate
                return model.model_validate(response)
            else:
                raise ValueError("Response must be a JSON string or a dictionary.")
        except ValidationError as e:
            logger.error(f"Validation error while parsing structured response: {e}")
            raise StructureError(f"Validation failed for structured response: {e}")