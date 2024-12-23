from floki.llm.openai.openai_client import OpenAIClient
from floki.llm.openai.azure_client import AzureOpenAIClient
from floki.llm.base import LLMClientBase
from floki.types.llm import (
    AzureOpenAIClientConfig, OpenAIClientConfig,
    AudioSpeechRequest, AudioTranscriptionRequest,
    AudioTranslationRequest, AudioTranscriptionResponse, AudioTranslationResponse,
)
from typing import Union, Optional, Dict, Any, Type
from openai import AzureOpenAI, OpenAI
from pydantic import BaseModel, Field, ValidationError
import logging

logger = logging.getLogger(__name__)

def validate_request(request: Union[BaseModel, Dict[str, Any]], request_class: Type[BaseModel]) -> BaseModel:
    """
    Helper function to validate and transform a dictionary into a Pydantic object.

    Args:
        request (Union[BaseModel, Dict[str, Any]]): The request data as a dictionary or a Pydantic object.
        request_class (Type[BaseModel]): The Pydantic model class for validation.

    Returns:
        BaseModel: A validated Pydantic object.

    Raises:
        ValueError: If validation fails.
    """
    # Transform dictionary to Pydantic object if needed
    if isinstance(request, dict):
        try:
            request = request_class(**request)
        except ValidationError as e:
            raise ValueError(f"Validation error: {e}")

    # Validate the request if it's already a Pydantic object
    try:
        validated_request = request_class.model_validate(request)
    except ValidationError as e:
        raise ValueError(f"Validation error: {e}")
    
    return validated_request

class OpenAIAudioClient(LLMClientBase):
    """
    Client for handling OpenAI's audio functionalities, including speech generation, transcription, and translation.
    Supports both OpenAI and Azure OpenAI configurations.
    """
    api_key: Optional[str] = Field(default=None, description="API key for OpenAI or Azure OpenAI. Optional.")
    base_url: Optional[str] = Field(default=None, description="Base URL for OpenAI API (OpenAI-specific). Optional.")
    azure_endpoint: Optional[str] = Field(default=None, description="Azure endpoint URL (Azure OpenAI-specific). Optional.")
    azure_deployment: Optional[str] = Field(default=None, description="Azure deployment name (Azure OpenAI-specific). Optional.")
    api_version: Optional[str] = Field(default=None, description="Azure API version (Azure OpenAI-specific). Optional.")
    organization: Optional[str] = Field(default=None, description="Organization for OpenAI or Azure OpenAI. Optional.")
    project: Optional[str] = Field(default=None, description="Project for OpenAI or Azure OpenAI. Optional.")
    azure_ad_token: Optional[str] = Field(default=None, description="Azure AD token for authentication (Azure OpenAI-specific). Optional.")
    azure_client_id: Optional[str] = Field(default=None, description="Client ID for Managed Identity authentication (Azure OpenAI-specific). Optional.")
    timeout: Union[int, float, Dict[str, Any]] = Field(default=1500, description="Timeout for requests. Defaults to 1500 seconds.")
    
    def model_post_init(self, __context: Any) -> None:
        """
        Initializes private attributes for provider, api, config, and client after validation.
        """
        self._provider = "openai"
        self._api = "audio"

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
                azure_deployment=self.azure_deployment,
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

    def create_speech(self, request: Union[AudioSpeechRequest, Dict[str, Any]], file_name: Optional[str] = None) -> Union[bytes, None]:
        """
        Generate speech audio from text and optionally save it to a file.

        Args:
            request (Union[AudioSpeechRequest, Dict[str, Any]]): The request parameters for speech generation.
            file_name (Optional[str]): Optional file name to save the generated audio.

        Returns:
            Union[bytes, None]: The generated audio content as bytes if no file_name is provided, otherwise None.
        """
        # Transform dictionary to Pydantic object if needed
        validated_request: AudioSpeechRequest = validate_request(request, AudioSpeechRequest)

        logger.info(f"Using model '{validated_request.model}' for speech generation.")

        input_text = validated_request.input
        
        max_chunk_size = 4096

        if len(input_text) > max_chunk_size:
            logger.info(f"Input exceeds {max_chunk_size} characters. Splitting into smaller chunks.")

        # Split input text into manageable chunks
        def split_text(text, max_size):
            chunks = []
            while len(text) > max_size:
                split_index = text.rfind(". ", 0, max_size) + 1 or max_size
                chunks.append(text[:split_index].strip())
                text = text[split_index:].strip()
            chunks.append(text)
            return chunks

        text_chunks = split_text(input_text, max_chunk_size)

        audio_chunks = []

        try:
            for chunk in text_chunks:
                validated_request.input = chunk
                with self.client.with_streaming_response.audio.speech.create(**validated_request.model_dump()) as response:
                    if file_name:
                        # Write each chunk incrementally to the file
                        logger.info(f"Saving audio chunk to file: {file_name}")
                        with open(file_name, "ab") as audio_file:
                            for chunk in response.iter_bytes():
                                audio_file.write(chunk)
                    else:
                        # Collect all chunks in memory for combining
                        audio_chunks.extend(response.iter_bytes())

            if file_name:
                return None
            else:
                # Combine all chunks into one bytes object
                return b"".join(audio_chunks)

        except Exception as e:
            logger.error(f"Failed to create or save speech: {e}")
            raise ValueError(f"An error occurred during speech generation: {e}")

    def create_transcription(self, request: Union[AudioTranscriptionRequest, Dict[str, Any]]) -> AudioTranscriptionResponse:
        """
        Transcribe audio to text.

        Args:
            request (Union[AudioTranscriptionRequest, Dict[str, Any]]): The request parameters for transcription.

        Returns:
            AudioTranscriptionResponse: The transcription result.
        """
        validated_request: AudioTranscriptionRequest = validate_request(request, AudioTranscriptionRequest)

        logger.info(f"Using model '{validated_request.model}' for transcription.")

        response = self.client.audio.transcriptions.create(
            file=validated_request.file,
            **validated_request.model_dump(exclude={"file"})
        )
        return response

    def create_translation(self, request: Union[AudioTranslationRequest, Dict[str, Any]]) -> AudioTranslationResponse:
        """
        Translate audio to English.

        Args:
            request (Union[AudioTranslationRequest, Dict[str, Any]]): The request parameters for translation.

        Returns:
            AudioTranslationResponse: The translation result.
        """
        validated_request: AudioTranslationRequest = validate_request(request, AudioTranslationRequest)

        logger.info(f"Using model '{validated_request.model}' for translation.")

        response = self.client.audio.translations.create(
            file=validated_request.file,
            **validated_request.model_dump(exclude={"file"})
        )
        return response