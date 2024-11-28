from floki.llm.utils import RequestHandler, ResponseHandler
from floki.llm.huggingface.client import HFHubInferenceClient
from floki.types.llm import HFInferenceClientConfig
from floki.prompt.prompty import Prompty
from floki.types.message import BaseMessage
from floki.llm.base import LLMClientBase
from floki.tool import AgentTool
from typing import Union, Optional, Iterable, Dict, Any, List, Iterator, Type
from pydantic import BaseModel, Field
from huggingface_hub import InferenceClient
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class HFHubChatClient(LLMClientBase):
    """
    Concrete class for the Hugging Face Hub's chat completion API using the Inference API.
    This class extends the LLMClientBase and provides the necessary configurations for Hugging Face models.
    """
    model: str = Field(default=None, description="Model ID to use from Hugging Face Hub.")
    api_key: Optional[str] = Field(default=None, description="API key for Hugging Face services. Optional.")
    base_url: Optional[str] = Field(default=None, description="Base URL for Hugging Face API. Optional.")
    headers: Optional[Dict[str, str]] = Field(default=None, description="Additional headers to send to the server. Optional.")
    cookies: Optional[Dict[str, str]] = Field(default=None, description="Cookies to send with the request. Optional.")
    proxies: Optional[Any] = Field(default=None, description="Proxies to use for the request. Optional.")
    timeout: Union[int, float, Dict[str, Any]] = Field(default=1500, description="Timeout for requests. Can be an integer, float, or dictionary.")

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes private attributes for provider, api, config, and client after validation.
        """
        # Set the private provider and api attributes
        self._provider = "huggingface"
        self._api = "chat"

        # Set up the private config and client attributes
        self._config = self.get_config()
        self._client = self.get_client()
        return super().model_post_init(__context)

    def get_config(self) -> HFInferenceClientConfig:
        """
        Returns the appropriate configuration for the Hugging Face Inference API.
        """
        return HFInferenceClientConfig(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url,
            headers=self.headers,
            cookies=self.cookies,
            proxies=self.proxies
        )

    def get_client(self) -> InferenceClient:
        """
        Initialize and return the Hugging Face Inference client.
        """
        config: HFInferenceClientConfig = self.config
        return HFHubInferenceClient(
            model=config.model,
            api_key=config.api_key,
            base_url=config.base_url,
            headers=config.headers,
            cookies=config.cookies,
            proxies=config.proxies,
            timeout=self.timeout
        ).get_client()

    @property
    def config(self) -> Dict[str, Any]:
        return self._config

    @property
    def client(self) -> InferenceClient:
        return self._client

    @classmethod
    def from_prompty(cls, prompty_source: Union[str, Path], timeout: Union[int, float, Dict[str, Any]] = 1500) -> 'HFHubChatClient':
        """
        Initializes an HFHubChatClient client using a Prompty source, which can be a file path or inline content.
        
        Args:
            prompty_source (Union[str, Path]): The source of the Prompty file, which can be a path to a file 
                or inline Prompty content as a string.
            timeout (Union[int, float, Dict[str, Any]], optional): Timeout for requests, defaults to 1500 seconds.

        Returns:
            HFHubChatClient: An instance of HFHubChatClient configured with the model settings from the Prompty source.
        """
        # Load the Prompty instance from the provided source
        prompty_instance = Prompty.load(prompty_source)

        # Generate the prompt template from the Prompty instance
        prompt_template = Prompty.to_prompt_template(prompty_instance)

        # Extract the model configuration from Prompty
        model_config = prompty_instance.model

        # Initialize the HFHubChatClient based on the Prompty model configuration
        return cls.model_validate({
            'model': model_config.configuration.name,
            'api_key': model_config.configuration.api_key,
            'base_url': model_config.configuration.base_url,
            'headers': model_config.configuration.headers,
            'cookies': model_config.configuration.cookies,
            'proxies': model_config.configuration.proxies,
            'timeout': timeout,
            'prompty': prompty_instance,
            'prompt_template': prompt_template,
        })

    def generate(
        self,
        messages: Union[str, Dict[str, Any], BaseMessage, Iterable[Union[Dict[str, Any], BaseMessage]]] = None,
        input_data: Optional[Dict[str, Any]] = None,
        model: Optional[str] = None,
        tools: Optional[List[Union[AgentTool, Dict[str, Any]]]] = None,
        response_model: Optional[Type[BaseModel]] = None,
        **kwargs
    ) -> Union[Iterator[Dict[str, Any]], Dict[str, Any]]:
        """
        Generate chat completions based on provided messages or input_data for prompt templates.

        Args:
            messages (Optional): Either pre-set messages or None if using input_data.
            input_data (Optional[Dict[str, Any]]): Input variables for prompt templates.
            model (str): Specific model to use for the request, overriding the default.
            tools (List[Union[AgentTool, Dict[str, Any]]]): List of tools for the request.
            response_model (Type[BaseModel]): Optional Pydantic model for structured response parsing.
            **kwargs: Additional parameters for the language model.

        Returns:
            Union[Iterator[Dict[str, Any]], Dict[str, Any]]: The chat completion response(s).
        """

        # If input_data is provided, check for a prompt_template
        if input_data:
            if not self.prompt_template:
                raise ValueError("Inputs are provided but no 'prompt_template' is set. Please set a 'prompt_template' to use the input_data.")
            
            logger.info("Using prompt template to generate messages.")
            messages = self.prompt_template.format_prompt(**input_data)

        # Ensure we have messages at this point
        if not messages:
            raise ValueError("Either 'messages' or 'input_data' must be provided.")

        # Process and normalize the messages
        params = {'messages': RequestHandler.normalize_chat_messages(messages)}

        # Merge Prompty parameters if available, then override with any explicit kwargs
        if self.prompty:
            params = {**self.prompty.model.parameters.model_dump(), **params, **kwargs}
        else:
            params.update(kwargs)

        # If a model is provided, override the default model
        params['model'] = model or self.model

        # Prepare and send the request
        params = RequestHandler.process_params(params, llm_provider=self.provider, tools=tools, response_model=response_model)

        try:
            logger.info("Invoking Hugging Face ChatCompletion API.")
            response = self.client.chat_completion(**params)
            logger.info("Chat completion retrieved successfully.")

            return ResponseHandler.process_response(response, llm_provider=self.provider, response_model=response_model, stream=params.get('stream', False))
        except Exception as e:
            logger.error(f"An error occurred during the ChatCompletion API call: {e}")
            raise