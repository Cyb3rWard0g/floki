from floki.types.llm import AzureOpenAIClientConfig, AzureOpenAIModelConfig, OpenAIClientConfig, OpenAIModelConfig
from floki.llm.utils import RequestHandler, ResponseHandler
from floki.llm.openai.openai_client import OpenAIClient
from floki.llm.openai.azure_client import AzureOpenAIClient
from floki.prompt.prompty import Prompty
from floki.types.message import BaseMessage
from floki.llm.chat import ChatClientBase
from floki.tool import AgentTool
from typing import Union, Optional, Iterable, Dict, Any, List, Iterator, Type
from openai.types.chat import ChatCompletionMessage
from pydantic import BaseModel, Field, model_validator
from openai import AzureOpenAI, OpenAI
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class OpenAIChatClient(ChatClientBase):
    """
    Concrete class for the OpenAI chat endpoint with support for OpenAI and Azure OpenAI services.
    Always sets provider to 'openai' and api to 'chat'.
    """
    model: str = Field(default=None,description="Model name to use, e.g., 'gpt-4'")
    api_key: Optional[str] = Field(default=None, description="API key for OpenAI or Azure OpenAI. Optional.")
    base_url: Optional[str] = Field(default=None, description="Base URL for OpenAI API (OpenAI-specific). Optional.")
    azure_endpoint: Optional[str] = Field(default=None, description="Azure endpoint URL (Azure OpenAI-specific). Optional.")
    azure_deployment: Optional[str] = Field(default=None, description="Azure deployment name (Azure OpenAI-specific). Optional.")
    api_version: Optional[str] = Field(default=None, description="Azure API version (Azure OpenAI-specific). Optional.")
    organization: Optional[str] = Field(default=None, description="Organization for OpenAI or Azure OpenAI. Optional.")
    project: Optional[str] = Field(default=None, description="Project for OpenAI or Azure OpenAI. Optional.")
    azure_ad_token: Optional[str] = Field(default=None, description="Azure AD token for authentication (Azure OpenAI-specific). Optional.")
    azure_client_id: Optional[str] = Field(default=None, description="Client ID for Managed Identity authentication (Azure OpenAI-specific). Optional.")
    timeout: Union[int, float, Dict[str, Any]] = Field(default=1500, description="Timeout for requests. Can be an integer, float, or dictionary. Defaults to 1500 seconds.")

    @model_validator(mode='before')
    def validate_and_initialize(cls, values: Dict[str, Any]) -> Dict[str, Any]:
        """
        Ensures the 'model' is set during validation. 
        Uses 'azure_deployment' if no model is specified, defaults to 'gpt-4o'.
        """
        if 'model' not in values or values['model'] is None:
            values['model'] = values.get('azure_deployment', 'gpt-4o')
        return values
    
    def model_post_init(self, __context: Any)-> None:
        """
        Initializes private attributes for provider, api, config, and client after validation.
        """
        # Set the private provider and api attributes
        self._provider = "openai"
        self._api = "chat"

        # Set up the private config and client attributes
        self._config: Union[AzureOpenAIClientConfig, OpenAIClientConfig] = self.get_config()
        self._client: Union[AzureOpenAI, OpenAI] = self.get_client()
        return super().model_post_init(__context)
    
    def get_config(self) -> Union[AzureOpenAIClientConfig, OpenAIClientConfig]:
        """
        Returns the appropriate configuration for OpenAI or Azure OpenAI.
        """
        is_azure = self.azure_endpoint or self.azure_deployment

        if is_azure:
            return AzureOpenAIClientConfig(
                api_key=self.api_key,
                organization=self.organization,
                project=self.project,
                azure_ad_token=self.azure_ad_token,
                azure_endpoint=self.azure_endpoint,
                azure_deployment=self.azure_deployment or self.model,
                api_version=self.api_version
            )
        else:
            return OpenAIClientConfig(
                api_key=self.api_key,
                base_url=self.base_url,
                organization=self.organization,
                project=self.project
            )
    
    def get_client(self) -> Union[AzureOpenAI, OpenAI]:
        """
        Initialize and return the appropriate client (OpenAI or Azure OpenAI).
        """
        config = self.config
        timeout = self.timeout

        if isinstance(config, OpenAIClientConfig):
            logger.info("Initializing OpenAI client...")
            return OpenAIClient(
                api_key=config.api_key,
                base_url=config.base_url,
                organization=config.organization,
                project=config.project,
                timeout=timeout
            ).get_client()

        elif isinstance(config, AzureOpenAIClientConfig):
            logger.info("Initializing Azure OpenAI client...")
            return AzureOpenAIClient(
                api_key=config.api_key,
                azure_ad_token=config.azure_ad_token,
                azure_endpoint=config.azure_endpoint,
                azure_deployment=config.azure_deployment,
                api_version=config.api_version,
                organization=config.organization,
                project=config.project,
                azure_client_id=self.azure_client_id,
                timeout=timeout
            ).get_client()
    
    @property
    def config(self) -> Union[AzureOpenAIClientConfig, OpenAIClientConfig]:
        return self._config

    @property
    def client(self) -> Union[OpenAI, AzureOpenAI]:
        return self._client
    
    @classmethod
    def from_prompty(cls, prompty_source: Union[str, Path], timeout: Union[int, float, Dict[str, Any]] = 1500) -> 'OpenAIChatClient':
        """
        Initializes an OpenAIChatClient client using a Prompty source, which can be a file path or inline content.
        
        Args:
            prompty_source (Union[str, Path]): The source of the Prompty file, which can be a path to a file 
                or inline Prompty content as a string.
            timeout (Union[int, float, Dict[str, Any]], optional): Timeout for requests, defaults to 1500 seconds.

        Returns:
            OpenAIChatClient: An instance of OpenAIChatClient configured with the model settings from the Prompty source.
        """
        # Load the Prompty instance from the provided source
        prompty_instance = Prompty.load(prompty_source)

        # Generate the prompt template from the Prompty instance
        prompt_template = Prompty.to_prompt_template(prompty_instance)

        # Extract the model configuration from Prompty
        model_config = prompty_instance.model

        # Initialize the OpenAIChatClient instance using model_validate
        if isinstance(model_config.configuration, OpenAIModelConfig):
            return cls.model_validate({
                'model': model_config.configuration.name,
                'api_key': model_config.configuration.api_key,
                'base_url': model_config.configuration.base_url,
                'organization': model_config.configuration.organization,
                'project': model_config.configuration.project,
                'timeout': timeout,
                'prompty': prompty_instance,
                'prompt_template': prompt_template,
            })
        elif isinstance(model_config.configuration, AzureOpenAIModelConfig):
            return cls.model_validate({
                'model': model_config.configuration.azure_deployment,
                'api_key': model_config.configuration.api_key,
                'azure_endpoint': model_config.configuration.azure_endpoint,
                'azure_deployment': model_config.configuration.azure_deployment,
                'api_version': model_config.configuration.api_version,
                'organization': model_config.configuration.organization,
                'project': model_config.configuration.project,
                'azure_ad_token': model_config.configuration.azure_ad_token,
                'azure_client_id': model_config.configuration.azure_client_id,
                'timeout': timeout,
                'prompty': prompty_instance,
                'prompt_template': prompt_template,
            })
        else:
            raise ValueError(f"Unsupported model configuration type: {type(model_config.configuration)}")
    
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

        # Merge prompty parameters if available, then override with any explicit kwargs
        if self.prompty:
            params = {**self.prompty.model.parameters.model_dump(), **params, **kwargs}
        else:
            params.update(kwargs)

        # If a model is provided, override the default model
        params['model'] = model or self.model

        # Prepare and send the request
        params = RequestHandler.process_params(params, llm_provider=self.provider, tools=tools, response_model=response_model)

        try:
            logger.info("Invoking ChatCompletion API.")
            logger.debug(f"ChatCompletion API Parameters:{params}")
            response: ChatCompletionMessage = self.client.chat.completions.create(**params, timeout=self.timeout)
            logger.info("Chat completion retrieved successfully.")

            return ResponseHandler.process_response(response, llm_provider=self.provider, response_model=response_model, stream=params.get('stream', False))
        except Exception as e:
            logger.error(f"An error occurred during the ChatCompletion API call: {e}")
            raise