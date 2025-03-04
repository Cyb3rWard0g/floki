from azure.identity import DefaultAzureCredential
from pydantic import BaseModel, Field
from typing import Optional, Any, Literal
import time
import logging
import uuid
import base64
import json

logger = logging.getLogger(__name__)
logging.getLogger("azure").setLevel(logging.WARNING)

ROLE_ID = "0fb8eba5-a2bb-4abe-b1c1-49dfad359bb0"  # Azure ContainerApps Session Executor

class AzContainerAppsSessionPools(BaseModel):
    """
    Manages Azure Container Apps Code Interpreter session pools.
    
    - Initializes Azure SDK clients.
    - Ensures the **Microsoft.App** resource provider is registered.
    - Retrieves or creates a session pool **only when explicitly requested**.
    - Creates the Azure **resource group** if needed.
    - Assigns required role permissions dynamically.
    """

    subscription_id: str = Field(..., description="Azure Subscription ID.")
    resource_group: str = Field(..., description="Azure Resource Group.")
    session_pool_name: str = Field(..., description="Name of the session pool.")
    location: Optional[str] = Field("westus2", description="Azure region for session pool creation.")
    max_sessions: int = Field(100, description="Maximum concurrent sessions (default: 100, max: 600).")
    cooldown_period: int = Field(300, description="Idle timeout before session termination (default: 300, range: 300-3600).")
    network_status: Literal["EgressDisabled", "EgressEnabled"] = Field("EgressDisabled", description="Network status.")
    container_type: Literal["PythonLTS", "CustomContainer"] = Field("PythonLTS", description="Type of code interpreter.")
    pool_management_type: Literal["Dynamic", "Manual"] = Field("Dynamic", description="Management type of session pool.")
    api_endpoint: Optional[str] = Field(None, description="Session pool API endpoint.")
    
    session_pool_id: Optional[str] = Field(None, init=False, description="Session pool resource ID.")
    containerapp_client: Optional[Any] = Field(default=None, init=False, description="Container app client instance.")
    resource_client: Optional[Any] = Field(default=None, init=False, description="Azure resource management client.")
    auth_client: Optional[Any] = Field(default=None, init=False, description="Azure Authorization client instance.")
    credential: Optional[Any] = Field(default=None, init=False, description="Azure authentication credential.")

    def model_post_init(self, __context__):
        """
        Initializes Azure SDK clients and ensures the required resource provider is registered.
        """
        try:
            from azure.mgmt.appcontainers import ContainerAppsAPIClient
            from azure.mgmt.resource import ResourceManagementClient
            from azure.mgmt.authorization import AuthorizationManagementClient
        except ImportError as e:
            raise ImportError(
                "Missing required Azure SDK dependencies. "
                "Install them with: pip install azure-mgmt-appcontainers==3.2.0b1 azure-mgmt-resource azure-mgmt-authorization"
            ) from e

        self.credential = DefaultAzureCredential()

        # Initialize clients
        self.containerapp_client = ContainerAppsAPIClient(credential=self.credential, subscription_id=self.subscription_id)
        self.resource_client = ResourceManagementClient(credential=self.credential, subscription_id=self.subscription_id)
        self.auth_client = AuthorizationManagementClient(credential=self.credential, subscription_id=self.subscription_id)

        # Ensure the Microsoft.App provider is registered
        self.ensure_provider_registration("Microsoft.App")

    def ensure_provider_registration(self, provider_namespace: str, max_wait_time: int = 300, poll_interval: int = 30) -> None:
        """
        Ensures the Azure resource provider is registered. If it's not registered, it will register it and wait until completed.

        Args:
            provider_namespace (str): The Azure resource provider namespace (e.g., "Microsoft.App").
            max_wait_time (int): Maximum time in seconds to wait for registration to complete (default: 300).
            poll_interval (int): Time interval (in seconds) between status checks (default: 30).
        """
        try:
            provider = self.resource_client.providers.get(provider_namespace)
            registration_state = provider.registration_state.lower()
            logger.info(f"Current registration state of '{provider_namespace}': {registration_state}")

            if registration_state == "registered":
                return  # Already registered

            # If not registered, start registration
            logger.info(f"Registering resource provider '{provider_namespace}'...")
            self.resource_client.providers.register(provider_namespace)

            # Poll until registration is complete
            elapsed_time = 0
            while elapsed_time < max_wait_time:
                time.sleep(poll_interval)
                elapsed_time += poll_interval

                provider = self.resource_client.providers.get(provider_namespace)
                registration_state = provider.registration_state.lower()
                logger.info(f"Checking registration status: {registration_state}")

                if registration_state == "registered":
                    logger.info(f"'{provider_namespace}' is now registered.")
                    return

            raise TimeoutError(f"Timed out waiting for '{provider_namespace}' to register.")

        except Exception as e:
            logger.error(f"Error checking/registering provider '{provider_namespace}': {str(e)}")
            raise RuntimeError(f"Failed to ensure provider registration for '{provider_namespace}': {str(e)}")
    
    def ensure_resource_group(self) -> None:
        """
        Ensures the Azure resource group exists, creating it if necessary.
        """
        try:
            logger.info(f"Checking if resource group '{self.resource_group}' exists...")
            self.resource_client.resource_groups.get(self.resource_group)
            logger.info(f"Resource group '{self.resource_group}' already exists.")
        except Exception:
            logger.info(f"Resource group '{self.resource_group}' does not exist. Creating...")
            self.resource_client.resource_groups.create_or_update(
                self.resource_group, {"location": self.location}
            )
            logger.info(f"Resource group '{self.resource_group}' created successfully.")
    
    def get_current_user_info(self) -> dict:
        """
        Retrieves the current authenticated user's details, including Object ID, username, and full name.

        Returns:
            dict: A dictionary containing:
                - `oid` (str): Object ID.
                - `preferred_username` (str, optional): User's UPN or email.
                - `name` (str, optional): Full name of the user.
        """
        try:
            # Get the authentication token
            token = self.credential.get_token("https://management.azure.com/.default")
            base64_meta_data = token.token.split(".")[1]

            # Decode JWT token (handle padding issues)
            padding_needed = -len(base64_meta_data) % 4
            if padding_needed:
                base64_meta_data += "=" * padding_needed

            json_bytes = base64.urlsafe_b64decode(base64_meta_data)
            json_string = json_bytes.decode("utf-8")
            json_dict = json.loads(json_string)

            # Extract details
            user_info = {
                "oid": json_dict.get("oid"),  # Object ID
                "preferred_username": json_dict.get("preferred_username"),  # Email/UPN
                "name": json_dict.get("name")  # Full Name
            }

            return user_info
        except Exception as e:
            logger.error(f"Failed to retrieve current user details: {str(e)}")
            raise RuntimeError(f"Unable to retrieve user details: {str(e)}")

    def check_existing_role_assignment(self, user_object_id: str) -> bool:
        """
        Checks if the given user already has the required role assignment.

        Args:
            user_object_id (str): Object ID of the user.

        Returns:
            bool: True if the role assignment exists, False otherwise.
        """
        try:
            role_assignments = self.auth_client.role_assignments.list_for_resource_group(
                resource_group_name=self.resource_group,
                filter=f"assignedTo('{user_object_id}')",
            )

            if role_assignments:
                for role_assignment in role_assignments:
                    if role_assignment.role_definition_id.lower().endswith(ROLE_ID):
                        return True
            
            return False  # Role not found

        except Exception as e:
            logger.error(f"Failed to check existing role assignments: {str(e)}")
            return False

    def assign_role_to_user(self):
        """
        Assigns the 'Azure ContainerApps Session Executor' role to the authenticated user or service principal.
        """
        user_info = self.get_current_user_info()
        user_object_id = user_info["oid"]
        user_name = user_info.get("name", "Unknown User")
        role_definition_id = f"/subscriptions/{self.subscription_id}/providers/Microsoft.Authorization/roleDefinitions/{ROLE_ID}"

        scope = self.session_pool_id  # Use the session pool resource ID
        role_assignment_name = str(uuid.uuid4())  # Unique role assignment ID

        logger.info(f"Checking if user {user_name} already has the required role...")
        if self.check_existing_role_assignment(user_object_id):
            logger.info(f"User {user_name} already has the required role.")
            return

        try:
            logger.info(f"Assigning 'Azure ContainerApps Session Executor' role to user {user_name}...")
            logger.debug(f"User Object ID: {user_object_id}.")
            logger.debug(f"Scope: {scope}.")

            self.auth_client.role_assignments.create(
                scope=scope,
                role_assignment_name=role_assignment_name,
                parameters={
                    "properties": {
                        "principalId": user_object_id,
                        "principalType": "User",
                        "roleDefinitionId": role_definition_id,
                    }
                },
            )
            logger.info("Role assigned successfully.")

        except Exception as e:
            logger.error(f"Failed to assign role: {str(e)}")
            raise RuntimeError(f"Failed to assign role: {str(e)}")

    def get_session_pool(self) -> Optional[Any]:
        """
        Retrieves full session pool details, including its resource ID.

        Returns:
            dict: Full session pool details.
        """
        try:
            response = self.containerapp_client.container_apps_session_pools.get(
                resource_group_name=self.resource_group,
                session_pool_name=self.session_pool_name
            )
            self.session_pool_id = response.id  # Save resource ID
            return response
        except Exception as e:
            if "ResourceNotFound" in str(e):
                logger.info(f"Session pool '{self.session_pool_name}' not found.")
                return None  # Gracefully return None if not found
            else:
                logger.error(f"Unexpected error retrieving session pool: {str(e)}")
                raise  # Re-raise unexpected errors
    
    def create_session_pool(self) -> Optional[Any]:
        """
        Creates a new Azure Container Apps session pool if it does not exist.

        Returns:
            Optional[Any]: The newly created session pool object.
        """
        try:

            logger.info(f"Creating session pool '{self.session_pool_name}' in '{self.resource_group}'...")

            properties = {
                "containerType": self.container_type,
                "scaleConfiguration": {"maxConcurrentSessions": self.max_sessions},
                "sessionNetworkConfiguration": {"status": self.network_status},
                "poolManagementType": self.pool_management_type,
            }

            if self.pool_management_type == "Dynamic":
                properties["dynamicPoolConfiguration"] = {"cooldownPeriodInSeconds": self.cooldown_period, "executionType": "Timed"}

            response = self.containerapp_client.container_apps_session_pools.begin_create_or_update(
                resource_group_name=self.resource_group,
                session_pool_name=self.session_pool_name,
                session_pool_envelope={
                    "location": self.location,
                    "properties": properties,
                }
            ).result()

            logger.info(f"Session pool '{self.session_pool_name}' created successfully.")
            return response
        except Exception as e:
            logger.error(f"Failed to create session pool: {str(e)}")
            raise RuntimeError(f"Failed to create session pool: {str(e)}")

    def get_or_create_session_pool(self) -> str:
        """
        Retrieves an existing session pool or creates a new one if it does not exist.
        Ensures the current user has the required role before returning the session pool.

        Returns:
            str: The session pool management API endpoint.
        """
        self.ensure_resource_group()

        logger.info(f"Checking session pool '{self.session_pool_name}' in resource group '{self.resource_group}'...")
        session_pool = self.get_session_pool()

        if not session_pool:
            logger.info(f"Session pool '{self.session_pool_name}' not found. Creating new one...")
            session_pool = self.create_session_pool()

        if not session_pool:
            raise ValueError("Failed to retrieve or create session pool.")

        logger.info(f"Session pool '{session_pool.name}' already exists.")
        self.api_endpoint = session_pool.pool_management_endpoint
        self.session_pool_id = session_pool.id

        # Ensure the user has the necessary permissions before returning
        self.assign_role_to_user()

        return session_pool