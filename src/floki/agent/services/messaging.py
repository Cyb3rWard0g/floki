from typing import Optional, Any, Callable, get_type_hints, Tuple, Type
from fastapi import HTTPException, Request
from cloudevents.http import from_http
from cloudevents.http.event import CloudEvent
from pydantic import BaseModel, ValidationError
from copy import deepcopy
import logging

logger = logging.getLogger(__name__)

async def parse_cloudevent(request: Request, model: Optional[Type[BaseModel]] = None) -> Tuple[BaseModel, dict, str]:
    """
    Parses and validates a CloudEvent request. Returns the validated message, metadata, and message_type.
    """
    try:
        # Parse the CloudEvent
        logger.debug("Parsing CloudEvent request...")
        body = await request.body()
        headers = request.headers
        event: CloudEvent = from_http(dict(headers), body)
    except Exception as e:
        logger.error(f"Failed to parse CloudEvent: {e}")
        raise HTTPException(status_code=400, detail=f"Invalid CloudEvent: {str(e)}")

    # Extract metadata
    cloud_event_metadata = {
        "id": event.get("id"),
        "source": event.get("source"),
        "type": event.get("type"),
        "topic": event.get("topic"),
        "pubsubname": event.get("pubsubname"),
        "time": event.get("time"),
        "headers": dict(headers),
    }
    logger.debug(f"Extracted CloudEvent metadata: {cloud_event_metadata}")

    # Validate and parse message payload
    if model:
        try:
            logger.debug(f"Validating payload with model '{model.__name__}'...")
            message = model(**event.data)
            message_type = model.__name__
        except ValidationError as ve:
            logger.error(f"Message validation failed for model '{model.__name__}': {ve}")
            raise HTTPException(status_code=422, detail=f"Message validation failed: {ve}")
    else:
        logger.error("No Pydantic model provided for message validation.")
        raise HTTPException(status_code=500, detail="Message validation failed: No Pydantic model provided.")

    # Return the validated message, metadata, and message_type
    logger.debug(f"Message successfully parsed and validated: {message}")
    return message, cloud_event_metadata, message_type

def message_router(
    func: Optional[Callable[..., Any]] = None,
    *,
    pubsub: Optional[str] = None,
    topic: Optional[str] = None,
    route: Optional[str] = None,
    dead_letter_topic: Optional[str] = None,
    broadcast: bool = False,
) -> Callable[[Callable[..., Any]], Callable[..., Any]]:
    """
    Decorator for dynamically registering message handlers for pub/sub topics.

    Args:
        func (Optional[Callable]): The function to decorate.
        pubsub (Optional[str]): The pubsub name.
        topic (Optional[str]): The topic name.
        route (Optional[str]): The custom route.
        dead_letter_topic (Optional[str]): Dead letter topic name.
        broadcast (bool): Whether the message is broadcast.

    Returns:
        Callable: The decorated function.
    """
    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        # Extract type hints to identify the Pydantic model
        type_hints = get_type_hints(f)
        message_model = None
        for param_name, param_type in type_hints.items():
            if isinstance(param_type, type) and issubclass(param_type, BaseModel):
                message_model = param_type
                logger.debug(f"Parameter '{param_name}' is identified as the message model of type '{message_model.__name__}'.")
                break

        if not message_model:
            raise ValueError(f"No Pydantic model found in parameters for handler '{f.__name__}'.")

        # Attach metadata to the function for registration
        f._is_message_handler = True
        f._message_router_data = deepcopy({
            "pubsub": pubsub,
            "topic": topic,
            "route": route,
            "dead_letter_topic": dead_letter_topic,
            "is_broadcast": broadcast,
            "message_model": message_model,
        })

        return f

    return decorator(func) if func else decorator