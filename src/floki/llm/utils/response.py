from typing import Union, Dict, Any, Type, Optional, Iterator
from floki.llm.utils import StreamHandler, StructureHandler
from dataclasses import is_dataclass, asdict
from floki.types import ChatCompletion
from pydantic import BaseModel
import logging

logger = logging.getLogger(__name__)

class ResponseHandler:
    """
    Handles the processing of responses from language models.
    """

    @staticmethod
    def process_response(
        response: Any,
        llm_provider: str,
        response_model: Optional[Type[BaseModel]] = None,
        stream: bool = False
    ) -> Union[Iterator[Dict[str, Any]], Dict[str, Any]]:
        """
        Process the response from the language model.

        Args:
            response: The response object from the language model.
            llm_provider: The LLM provider to use (e.g., 'openai').
            response_model: A pydantic model to parse and validate the structured response.
            stream: Indicates if the response is a stream.

        Returns:
            Union[Iterator[Dict[str, Any]], Dict[str, Any]]: The processed response.
        """
        if stream:
            return StreamHandler.process_stream(stream=response, llm_provider=llm_provider, response_model=response_model)
        else:
            if response_model:
                structured_response_json = StructureHandler.extract_structured_response(response, llm_provider)
                structured_response_instance = StructureHandler.validate_response(structured_response_json, response_model)
                if isinstance(structured_response_instance, response_model):
                    logger.info("Structured output was successfully validated.")
                    logger.info(f"Returning an instance of {type(structured_response_instance)}.")
                    return structured_response_instance
                else:
                    logger.error("Validation failed for structured response.")
            
            # Convert response to dictionary
            if isinstance(response, dict):
                # Already a dictionary
                response_dict = response
            elif is_dataclass(response):
                # Dataclass instance
                response_dict = asdict(response)
            elif isinstance(response, BaseModel):
                # Pydantic object
                response_dict = response.model_dump()
            else:
                raise ValueError(f"Unsupported response type: {type(response)}")
            
            completion = ChatCompletion(**response_dict)
            logger.debug(f"Chat completion response: {completion}")
            return completion