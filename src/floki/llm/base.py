from floki.prompt.base import PromptTemplateBase
from floki.prompt.prompty import Prompty
from typing import Union, Dict, Any, Optional
from pydantic import BaseModel, Field, PrivateAttr
from abc import ABC, abstractmethod
from pathlib import Path

class LLMClientBase(BaseModel, ABC):
    """
    Abstract base class for LLM models.
    """
    model: str = Field(default=None, description="Model name to use, e.g., 'gpt-4o'")
    prompty: Optional[Prompty] = Field(default=None, description="Instance of the Prompty object (optional).")
    prompt_template: Optional[PromptTemplateBase] = Field(default=None, description="Prompt template for rendering (optional).")

    # Private attributes for provider and api
    _provider: str = PrivateAttr()
    _api: str = PrivateAttr()

    # Private attributes for config and client
    _config: Any = PrivateAttr()
    _client: Any = PrivateAttr()
    
    @property
    def provider(self) -> str:
        return self._provider

    @property
    def api(self) -> str:
        return self._api
    
    @property
    def config(self) -> Any:
        return self._config

    @property
    def client(self) -> Any:
        return self._client

    @abstractmethod
    def get_client(self) -> Any:
        """Abstract method to get the client for the LLM model."""
        pass

    @abstractmethod
    def get_config(self) -> Any:
        """Abstract method to get the configuration for the LLM model."""
        pass

    def refresh_client(self) -> None:
        """
        Public method to refresh the client by regenerating the config and client.
        """
        # Refresh config and client using the current state
        self._config = self.get_config()
        self._client = self.get_client()
    
    @classmethod
    @abstractmethod
    def from_prompty(cls, prompty_source: Union[str, Path], timeout: Union[int, float, Dict[str, Any]] = 1500) -> 'LLMClientBase':
        """
        Abstract method to load a Prompty source and configure the LLM client. The Prompty source can be 
        a file path or inline Prompty content.

        Args:
            prompty_source (Union[str, Path]): The source of the Prompty, which can be a path to a file or
                inline Prompty content as a string.
            timeout (Union[int, float, Dict[str, Any]], optional): Timeout for requests, defaults to 1500 seconds.

        Returns:
            LLMClientBase: An instance of the LLM client initialized with the model settings from the Prompty source.
        """
        pass                                                            