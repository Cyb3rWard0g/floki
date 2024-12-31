from dapr.conf import settings as dapr_settings
from pydantic import BaseModel, Field, ConfigDict
from typing import Optional, Any
import os

class DaprStoreBase(BaseModel):
    """
    Pydantic-based Dapr store base model with configuration options for store name, address, host, and port.
    """

    store_name: str = Field(..., description="The name of the Dapr store.")
    address: Optional[str] = Field(default=None,description="The full address of the Dapr sidecar (host:port). If not provided, constructed from host and port.")
    host: Optional[str] = Field(default=os.getenv('DAPR_RUNTIME_HOST'),description="The host of the Dapr sidecar, defaults to environment variable or '127.0.0.1'.")
    port: Optional[str] = Field(default=os.getenv('DAPR_GRPC_PORT'),description="The port of the Dapr sidecar, defaults to environment variable or '50001'.")

    model_config = ConfigDict(arbitrary_types_allowed=True)

    def model_post_init(self, __context: Any) -> None:
        """
        Post-initialization to set Dapr settings based on provided or environment values for host and port.
        """
        # Check for environment variables
        env_host = os.getenv('DAPR_RUNTIME_HOST')
        env_port = os.getenv('DAPR_GRPC_PORT')

        # Set Dapr settings only if explicit values are provided or fall back to environment values
        self.host = self.host or env_host or dapr_settings.DAPR_RUNTIME_HOST
        self.port = self.port or env_port or dapr_settings.DAPR_GRPC_PORT

        # Determine the address, prioritizing provided address or constructing from host and port
        self.address = self.address or f'{self.host}:{self.port}'

        # Complete post-initialization
        super().model_post_init(__context)