from floki.agent.services.base import AgentServiceBase, message_router
from floki.types.agent import AgentActorMessage
from floki.types.message import UserMessage
from cloudevents.http.conversion import from_http
from cloudevents.http.event import CloudEvent
from fastapi import Request, Response, status
import json
import logging

logger = logging.getLogger(__name__)

class AgentService(AgentServiceBase):
    """
    A Pydantic-based class for managing services and exposing FastAPI routes with Dapr pub/sub and actor support.
    """
    
    @message_router(message_type="TriggerActionMessage")
    async def process_agent_message(self, request: Request) -> Response:
        """
        Processes messages sent directly to the agent's topic.
        Handles various message types (e.g., TriggerAction) and broadcasts responses if necessary.

        Args:
            request (Request): The incoming pub/sub request containing a task.

        Returns:
            Response: The agent's response after processing the task.
        """
        try:
            # Parse the incoming CloudEvent
            body = await request.body()
            headers = request.headers
            event: CloudEvent = from_http(dict(headers), body)

            message_type = event.get("type")
            message_data: dict = event.data
            source = event.get("source")

            logger.info(f"{self.agent.name} received {message_type} message from {source}.")

            # Extract workflow_instance_id from headers if available
            workflow_instance_id = headers.get("workflow_instance_id")

            # Handle TriggerAction type messages
            if message_type == "TriggerActionMessage":
                task = message_data.get("task")

                # Execute the task, defaulting to the agent's internal memory if no task is provided
                if not task:
                    logger.info(f"{self.agent.name} running a task from memory.")

                # Execute the task or no input
                response = await self.invoke_task(task)

                # Prepare and broadcast the result as AgentActionResultMessage
                response_message = UserMessage(
                    name=self.agent.name,
                    content=response.body.decode()
                )

                # Broadcast results to all agents
                await self.publish_message_to_all(
                    message_type="ActionResponseMessage",
                    message=response_message.model_dump()
                )

                # Send results to task results topic
                additional_metadata = {'ttlInSeconds': '120'}
                if workflow_instance_id:
                    additional_metadata["event_name"] = "AgentCompletedTask"
                    additional_metadata["workflow_instance_id"] = workflow_instance_id
                
                await self.publish_results_message(
                    message_type="ActionResponseMessage",
                    message=response_message.model_dump(),
                    **additional_metadata
                )

                return response

            else:
                # Log unsupported message types
                logger.warning(f"Unsupported message type '{message_type}' received by {self.agent.name}.")
                return Response(content=json.dumps({"error": f"Unsupported message type: {message_type}"}), status_code=status.HTTP_400_BAD_REQUEST)

        except Exception as e:
            logger.error(f"Error processing agent message: {e}", exc_info=True)
            return Response(content=json.dumps({"error": f"Error processing message: {str(e)}"}), status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)
    
    @message_router(message_type=["StartMessage","ActionResponseMessage"], broadcast=True)
    async def process_broadcast_message(self, request: Request) -> Response:
        """
        Processes a message from the broadcast topic.
        Ensures the agent does not process messages sent by itself,
        and adds user messages to both the agent's memory and the actor's state.

        Args:
            request (Request): The incoming broadcast request.

        Returns:
            Response: Acknowledgment of the broadcast processing.
        """
        try:
            # Parse the CloudEvent from the request
            body = await request.body()
            headers = request.headers
            event: CloudEvent = from_http(dict(headers), body)
            
            message_type = event.get("type")
            broadcast_message: dict = event.data
            source = event.get("source")
            message_content = broadcast_message.get("content")
            
            if not message_content:
                logger.warning(f"Broadcast message missing 'content': {broadcast_message}")
                return Response(content="Invalid broadcast message: 'content' is required.", status_code=status.HTTP_400_BAD_REQUEST)

            # Ignore messages sent by this agent (based on CloudEvent source)
            if source == self.agent.name:
                logger.info(f"{self.agent.name} ignored its own broadcast message of type {message_type}.")
                return Response(status_code=status.HTTP_204_NO_CONTENT)

            # Log and process the valid broadcast message
            logger.info(f"{self.agent.name} is processing broadcast message {message_type} from '{source}'")
            logger.debug(f"Message: {message_content}")

            # Add the message to the agent's memory
            self.agent.memory.add_message(broadcast_message)

            # Add the message to the actor's state
            actor_message = AgentActorMessage(**broadcast_message)
            await self.add_message(actor_message)

            return Response(content="Broadcast message added to memory and actor state.", status_code=status.HTTP_200_OK)

        except Exception as e:
            logger.error(f"Error processing broadcast message: {e}", exc_info=True)
            return Response(content=f"Error processing message: {str(e)}", status_code=status.HTTP_500_INTERNAL_SERVER_ERROR)