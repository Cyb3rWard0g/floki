from azure.identity import DefaultAzureCredential, ManagedIdentityCredential, get_bearer_token_provider
from floki.types.executor import ExecutionRequest, ExecutionResult
from floki.executors import CodeExecutorBase
from typing import List, Any, Optional, Union, Dict, Literal
from pydantic import Field
from uuid import uuid4
import logging
import requests
import urllib.parse

logger = logging.getLogger(__name__)

SupportedAPIVersions = Literal["2024-02-02-preview", "2024-08-02-preview", "2024-10-02-preview"]

# API version paths mapping
API_VERSION_PATHS: Dict[SupportedAPIVersions, Dict[str, str]] = {
    "2024-08-02-preview": {
        "execute": "code/execute",
        "upload": "upload",
        "download": "files/content/{filename}",
        "list_files": "files",
        "delete_file": "files/{filename}",
    },
    "2024-10-02-preview": {
        "execute": "executions",
        "upload": "files",
        "download": "files/{filename}/content",
        "list_files": "files",
        "delete_file": "files/{filename}",
    },
}

# Alias "2024-02-02-preview" to "2024-08-02-preview"
API_VERSION_PATHS["2024-02-02-preview"] = API_VERSION_PATHS["2024-08-02-preview"]


class AzContainerAppsCodeExecutor(CodeExecutorBase):
    """
    Executes Python code securely inside an Azure Container Apps Dynamic Sessions environment.

    - Supports **synchronous** and **asynchronous** execution.
    - Handles **multiple API versions dynamically**.
    - Uses **Microsoft Entra ID authentication**.
    - Dynamically creates or retrieves a **session pool** when needed.
    """

    # User may provide pool management endpoint directly or provide details for session pool resolution
    pool_management_endpoint: Optional[str] = Field(None, description="Azure API endpoint for Code Interpreter.")
    azure_client_id: Optional[str] = Field(default=None, description="Optional Managed Identity client ID.")
    execution_timeout: int = Field(default=60, description="Max execution time in seconds.")
    execution_type: Literal["synchronous", "asynchronous"] = Field(default="synchronous", description="Execution type.")
    session_id: str = Field(default_factory=lambda: str(uuid4()), description="Unique execution session identifier.")
    api_version: SupportedAPIVersions = Field(default="2024-10-02-preview", description="Azure API version.")

    credential: Optional[Any] = Field(default=None, init=False, description="Azure authentication credential.")
    token_provider: Optional[Any] = Field(default=None, init=False, description="Function for retrieving authentication tokens.")
    api_definition: Dict[str, str] = Field(default=None, init=False, description="API paths based on selected version.")

    def model_post_init(self, __context: Any) -> None:
        """
        Initializes authentication mechanisms, API version mapping, and session pool resolution.
        """
        try:
            # Authentication setup
            if self.azure_client_id:
                self.credential = ManagedIdentityCredential(client_id=self.azure_client_id)
                logger.info("Using Managed Identity authentication with provided client ID.")
            else:
                self.credential = DefaultAzureCredential(exclude_shared_token_cache_credential=True)
                logger.info("Using DefaultAzureCredential for authentication.")

            self.token_provider = get_bearer_token_provider(self.credential, "https://dynamicsessions.io/.default")

            # Validate and assign API definitions
            if self.api_version in API_VERSION_PATHS:
                self.api_definition = API_VERSION_PATHS[self.api_version]
                logger.info(f"Using API version: {self.api_version}")
            else:
                raise ValueError(f"Unsupported API version: {self.api_version}")

        except Exception as e:
            logger.error(f"Failed to initialize Azure Identity credentials or session pool: {str(e)}")
            raise ValueError("Unable to authenticate or initialize session pool. Check your setup.") from e

        super().model_post_init(__context)

    def _build_url(self, action: str, filename: Optional[str] = None) -> str:
        """
        Constructs the full API request URL dynamically based on API version.

        Args:
            action (str): API action (e.g., "execute", "upload", "list_files").
            filename (Optional[str]): Filename (if applicable) for file-related actions.

        Returns:
            str: Fully constructed API request URL.

        Raises:
            ValueError: If the action is unsupported.
        """
        if not self.pool_management_endpoint.endswith("/"):
            self.pool_management_endpoint += "/"

        endpoint_path = self.api_definition.get(action, "")
        if not endpoint_path:
            raise ValueError(f"Unsupported action '{action}' for API version {self.api_version}")

        if filename:
            endpoint_path = endpoint_path.format(filename=urllib.parse.quote(filename))

        encoded_session_id = urllib.parse.quote(self.session_id)
        return f"{self.pool_management_endpoint}{endpoint_path}?api-version={self.api_version}&identifier={encoded_session_id}"

    def _get_auth_headers(self) -> dict:
        """
        Retrieves authentication headers dynamically using the Azure token provider.

        Returns:
            dict: Authorization headers.
        """
        token = self.token_provider()
        return {"Authorization": f"Bearer {token}", "Content-Type": "application/json"}

    async def execute(self, request: Union[ExecutionRequest, dict]) -> List[ExecutionResult]:
        """
        Executes Python code in the Azure Code Interpreter session.

        Args:
            request (Union[ExecutionRequest, dict]): Execution request containing code snippets.

        Returns:
            List[ExecutionResult]: A list of execution results, including status and output.

        Raises:
            RuntimeError: If execution fails due to request errors.
        """
        if isinstance(request, dict):
            request = ExecutionRequest(**request)

        self.validate_snippets(request.snippets)
        results = []

        try:
            for snippet in request.snippets:
                if snippet.language != "python":
                    logger.error(f"Unsupported language: {snippet.language}. Only Python is supported.")
                    results.append(ExecutionResult(status="error", output="Unsupported language", exit_code=1))
                    continue

                # Build payload structure based on API version
                base_payload = {
                    "identifier": self.session_id,
                    "codeInputType": "inline",
                    "executionType": self.execution_type,
                    "code": snippet.code,
                    "timeoutInSeconds": self.execution_timeout,
                }

                # For older API versions, wrap inside "properties"
                if self.api_version in ["2024-02-02-preview", "2024-08-02-preview"]:
                    payload = {"properties": base_payload}
                else:  # 2024-10-02-preview and onwards
                    payload = base_payload

                api_url = self._build_url("execute")
                headers = self._get_auth_headers()

                response = requests.post(api_url, headers=headers, json=payload, timeout=self.execution_timeout)
                response.raise_for_status()

                # Extract data safely
                response_data = response.json()
                logger.debug(f"Execution Response: {response_data}")

                # Handle old API versions (properties-based)
                if "properties" in response_data:
                    data = response_data["properties"]
                    stdout_output = data.get("stdout", "").strip()
                    stderr_output = data.get("stderr", "").strip()
                    result_output = data.get("result", "").strip()
                else:  # Handle new API version (result is a dictionary)
                    result_data = response_data.get("result", {})
                    stdout_output = result_data.get("stdout", "").strip()
                    stderr_output = result_data.get("stderr", "").strip()
                    result_output = result_data.get("executionResult", "").strip()
                
                # Determine the best output to return
                output = result_output if result_output else stdout_output
                if stderr_output:
                    output += f"\n[stderr]: {stderr_output}"
                
                # Determine exit code
                status = response_data.get("status", "Unknown").strip().lower()
                exit_code = 0 if status == "succeeded" else 1

                results.append(ExecutionResult(status="success" if exit_code == 0 else "error", output=output, exit_code=exit_code))

        except requests.exceptions.RequestException as e:
            logger.error(f"Execution error: {str(e)}")
            results.append(ExecutionResult(status="error", output=str(e), exit_code=1))

        return results

    async def restart_session(self) -> None:
        """
        Restarts the execution session by generating a new session identifier.
        """
        self.session_id = str(uuid4())
        logger.info(f"Azure session restarted with new ID: {self.session_id}")