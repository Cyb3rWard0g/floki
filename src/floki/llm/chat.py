from floki.prompt.base import PromptTemplateBase
from floki.prompt.prompty import Prompty
from floki.llm.base import LLMClientBase
from typing import Union, Dict, Any, Optional
from pydantic import Field
from abc import abstractmethod
from pathlib import Path

class ChatClientBase(LLMClientBase):
    """
    Base class for chat-based LLM clients.
    Adds chat-specific functionality like Prompty support.
    """
    prompty: Optional[Prompty] = Field(default=None, description="Instance of the Prompty object (optional).")
    prompt_template: Optional[PromptTemplateBase] = Field(default=None, description="Prompt template for rendering (optional).")

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