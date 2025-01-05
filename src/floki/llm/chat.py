from floki.prompt.base import PromptTemplateBase
from floki.prompt.prompty import Prompty
from typing import Union, Dict, Any, Optional
from pydantic import BaseModel, Field
from abc import ABC, abstractmethod
from pathlib import Path

class ChatClientBase(BaseModel, ABC):
    """
    Base class for chat-specific functionality.
    Handles Prompty integration and provides abstract methods for chat client configuration.
    """
    prompty: Optional[Prompty] = Field(default=None, description="Instance of the Prompty object (optional).")
    prompt_template: Optional[PromptTemplateBase] = Field(default=None, description="Prompt template for rendering (optional).")

    @classmethod
    @abstractmethod
    def from_prompty(cls, prompty_source: Union[str, Path], timeout: Union[int, float, Dict[str, Any]] = 1500) -> 'ChatClientBase':
        """
        Abstract method to load a Prompty source and configure the chat client.

        Args:
            prompty_source (Union[str, Path]): Source of the Prompty, either a file path or inline Prompty content.
            timeout (Union[int, float, Dict[str, Any]]): Timeout for requests.

        Returns:
            ChatClientBase: Configured chat client instance.
        """
        pass