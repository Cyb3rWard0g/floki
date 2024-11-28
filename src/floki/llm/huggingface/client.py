from typing import Union, Optional, Dict, Any
from huggingface_hub import InferenceClient
from floki.types.llm import HFInferenceClientConfig
from floki.llm.utils import HTTPHelper
import logging

logger = logging.getLogger(__name__)

class HFHubInferenceClient:
    """
    Client for interfacing with Hugging Face's language models using the Inference API.
    Handles API communication and processes requests and responses.
    """

    def __init__(self, 
            model: Optional[str] = None,
            api_key: Optional[Union[str, bool]] = None,  # Alias for token, default to None or False
            token: Optional[Union[str, bool]] = None,    # Hugging Face token for authentication
            base_url: Optional[str] = None, 
            headers: Optional[Dict[str, str]] = None, 
            cookies: Optional[Dict[str, str]] = None, 
            proxies: Optional[Any] = None,
            timeout: Union[int, float, Dict[str, Any]] = 1500,
        ):
        """
        Initializes the Hugging Face client with model, API key or token, base URL, timeout, headers, cookies, and proxies.

        Args:
            model: The model ID or URL for the Hugging Face API. Cannot be used with base_url.
            api_key: API key or token for accessing Hugging Face's services.
            token: Hugging Face token for authentication. If provided, `api_key` should not be set.
            base_url: The base URL for Hugging Face API. Cannot be used with model.
            headers: Optional custom headers to send with the request.
            cookies: Optional custom cookies to send with the request.
            proxies: Optional proxies for the request.
            timeout: Request timeout in seconds or as a configuration dictionary.
        """
        # Handle mutually exclusive model and base_url
        if model and base_url:
            raise ValueError("Cannot provide both model and base_url. They are mutually exclusive.")
        
        # Handle mutually exclusive api_key and token
        if api_key and token:
            raise ValueError("Cannot provide both api_key and token. They are mutually exclusive.")
        
        # Initialize client parameters
        self.model = model
        self.api_key = api_key or token
        self.base_url = base_url
        self.headers = headers
        self.cookies = cookies
        self.proxies = proxies
        self.timeout = HTTPHelper.configure_timeout(timeout)

    def get_client(self) -> InferenceClient:
        """
        Returns the initialized Hugging Face Inference client.

        Returns:
            InferenceClient: The Hugging Face Inference client.
        """
        return InferenceClient(
            model=self.model,
            api_key=self.api_key,
            base_url=self.base_url if not self.model else None,  # Use base_url if model isn't provided
            headers=self.headers,
            cookies=self.cookies,
            proxies=self.proxies,
            timeout=self.timeout
        )

    @classmethod
    def from_config(cls, client_options: HFInferenceClientConfig, timeout: Union[int, float, dict] = 1500):
        """
        Initializes the HFHubInferenceClient using HFInferenceClientConfig.

        Args:
            client_options: The configuration options for the client.
            timeout: Timeout for requests (default is 1500 seconds).

        Returns:
            HFHubInferenceClient: The initialized client instance.
        """
        return cls(
            model=client_options.model,
            api_key=client_options.api_key,
            token=client_options.api_key,  # Alias for token
            base_url=client_options.base_url,
            headers=client_options.headers,
            cookies=client_options.cookies,
            proxies=client_options.proxies,
            timeout=timeout,
        )