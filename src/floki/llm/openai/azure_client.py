from azure.identity import DefaultAzureCredential, ManagedIdentityCredential, get_bearer_token_provider
from floki.types.llm import AzureOpenAIClientConfig
from floki.llm.utils import HTTPHelper
from openai import AzureOpenAI
from typing import Union, Optional
import logging
import os

logger = logging.getLogger(__name__)

class AzureOpenAIClient:
    """
    Client for Azure OpenAI language models, handling API communication and authentication.
    """

    def __init__(
        self, 
        api_key: Optional[str] = None,
        azure_ad_token: Optional[str] = None,
        organization: Optional[str] = None,
        project: Optional[str] = None,
        api_version: Optional[str] = None,
        azure_endpoint: Optional[str] = None, 
        azure_deployment: Optional[str] = None,
        azure_client_id: Optional[str] = None, 
        timeout: Union[int, float, dict] = 1500
    ):
        """
        Initializes the client with API key or Azure AD credentials.

        Args:
            api_key: Azure OpenAI API key (inferred from env variable if not provided).
            azure_ad_token: Azure AD token (inferred from env variable if not provided).
            organization: Organization name (optional).
            project: Project name (optional).
            api_version: API version (inferred from env variable if not provided).
            azure_endpoint: Azure endpoint (inferred from env variable if not provided).
            azure_deployment: Deployment name (inferred from env variable if not provided).
            azure_client_id: Managed Identity client ID (optional).
            timeout: Request timeout in seconds (default: 1500).
        """
        self.api_key = api_key # or inferred from AZURE_OPENAI_API_KEY env variable.
        self.azure_ad_token = azure_ad_token # or inferred from AZURE_OPENAI_AD_TOKEN env variable.
        self.organization = organization # or inferred from OPENAI_ORG_ID env variable.
        self.project = project # or inferred from OPENAI_PROJECT_ID env veriable.
        self.api_version = api_version or os.getenv("AZURE_OPENAI_API_VERSION")
        self.azure_endpoint = azure_endpoint # or inferred from AZURE_OPENAI_ENDPOINT env variable.
        self.azure_deployment = azure_deployment or os.getenv("AZURE_OPENAI_DEPLOYMENT")

        if not self.azure_endpoint or not self.azure_deployment:
            raise ValueError("Azure OpenAI endpoint and deployment must be provided, either via arguments or environment variables.")

        self.timeout = HTTPHelper.configure_timeout(timeout)

        # Authentication: API Key, Azure AD Token, or Azure Identity
        # The api_key, azure_ad_token, and azure_ad_token_provider arguments are mutually exclusive.
        if self.api_key:
            logger.info("Using API key for authentication.")
            self.client = AzureOpenAI(
                api_key=self.api_key,
                organization=self.organization,
                project=self.project,
                api_version=self.api_version,
                azure_endpoint=self.azure_endpoint,
                azure_deployment=self.azure_deployment
            ) 
        elif self.azure_ad_token:
            logger.info("Using Azure AD token for authentication.")
            self.client = AzureOpenAI(
                azure_ad_token=self.azure_ad_token,
                organization=self.organization,
                project=self.project,
                api_version=self.api_version,
                azure_endpoint=self.azure_endpoint,
                azure_deployment=self.azure_deployment
            )
        else:
            logger.info("No API key or Azure AD token provided, attempting to use Azure Identity credentials.")
            if azure_client_id:
                logger.info(f"Using Managed Identity with client ID: {azure_client_id}.")
                credential = ManagedIdentityCredential(client_id=azure_client_id)
            else:
                logger.info("Using DefaultAzureCredential.")
                credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
            
            # Get bearer token provider using Azure Identity credentials
            self.azure_ad_token_provider = get_bearer_token_provider(
                credential, "https://cognitiveservices.azure.com/.default"
            )
    
    def get_client(self) -> AzureOpenAI:
        """
        Returns the Azure OpenAI client.

        Returns:
            AzureOpenAI: The initialized Azure OpenAI client.
        """
        return AzureOpenAI(
            azure_ad_token_provider=self.azure_ad_token_provider,
            organization=self.organization,
            project=self.project,
            api_version=self.api_version,
            azure_endpoint=self.azure_endpoint,
            azure_deployment=self.azure_deployment
        )
    
    @classmethod
    def from_config(
        cls, 
        client_options: AzureOpenAIClientConfig, 
        azure_client_id: Optional[str] = None, 
        timeout: Union[int, float, dict] = 1500
    ):
        """
        Initialize AzureOpenAIClient using AzureOpenAIClientOptions.

        Args:
            client_options: An instance of AzureOpenAIClientOptions containing configuration details.
            azure_client_id: Optional Azure client ID for Managed Identity authentication.
            timeout: Optional timeout value for requests (default is 1500 seconds).

        Returns:
            AzureOpenAIClient: An initialized instance of AzureOpenAIClient.
        """
        return cls(
            api_key=client_options.api_key,
            azure_ad_token=client_options.azure_ad_token,
            organization=client_options.organization,
            project=client_options.project,
            api_version=client_options.api_version,
            azure_endpoint=client_options.azure_endpoint,
            azure_deployment=client_options.azure_deployment,
            azure_client_id=azure_client_id,
            timeout=timeout
        )