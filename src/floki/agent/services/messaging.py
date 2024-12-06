from typing import Optional, Any, Callable, get_type_hints, Tuple, Type, Dict
from floki.types.message import EventMessageMetadata
from pydantic import BaseModel, ValidationError
from inspect import signature, Parameter
from cloudevents.http.event import CloudEvent
from cloudevents.http import from_http
from fastapi import HTTPException, Request
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
    metadata = EventMessageMetadata(
        id=event.get("id"),
        datacontenttype=event.get("datacontenttype"),
        pubsubname=event.get("pubsubname"),
        source=event.get("source"),
        specversion=event.get("specversion"),
        time=event.get("time"),
        topic=event.get("topic"),
        traceid=event.get("traceid"),
        traceparent=event.get("traceparent"),
        type=event.get("type"),
        tracestate=event.get("tracestate"),
        headers=dict(headers),
    )

    logger.debug(f"Extracted CloudEvent metadata: {metadata}")

    # Validate and parse message payload
    if model:
        try:
            logger.debug(f"Validating payload with model '{model.__name__}'...")
            message = model(**event.data)
        except ValidationError as ve:
            logger.error(f"Message validation failed for model '{model.__name__}': {ve}")
            raise HTTPException(status_code=422, detail=f"Message validation failed: {ve}")
    else:
        logger.error("No Pydantic model provided for message validation.")
        raise HTTPException(status_code=500, detail="Message validation failed: No Pydantic model provided.")

    # Return the validated message, and metadata
    logger.debug(f"Message successfully parsed and validated: {message}")
    return message, metadata

def filter_parameters(func: Callable[..., Any]) -> Dict[str, Parameter]:
    """
    Filter parameters of a callable, excluding instance (`self`) or class (`cls`) references
    when the callable is in a class context.

    Args:
        func (Callable): The callable to inspect.

    Returns:
        Dict[str, Parameter]: Filtered parameters.
    """
    sig = signature(func)
    parameters = sig.parameters

    if hasattr(func, "__self__") or getattr(func, "__qualname__", "").split(".")[0] != func.__name__:
        # Class context: skip the first parameter (assumed to be 'self' or 'cls')
        return {
            name: param
            for idx, (name, param) in enumerate(parameters.items())
            if idx != 0
        }
    else:
        # Regular functions: include all parameters
        return parameters

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

    This decorator validates the signature of a handler function to ensure it conforms
    to the expected parameters (`message` and optionally `metadata`). It also attaches
    routing metadata to the handler for pub/sub integration.

    Args:
        func (Optional[Callable]): The function or method to decorate.
        pubsub (Optional[str]): The name of the pub/sub component to use.
        topic (Optional[str]): The topic name for the handler.
        route (Optional[str]): The custom route for this handler (optional).
        dead_letter_topic (Optional[str]): Name of the dead letter topic for failed messages (optional).
        broadcast (bool): Indicates if the message should be broadcast to multiple subscribers.

    Returns:
        Callable: The decorated handler with additional metadata.

    Raises:
        ValueError: If the handler's signature does not meet the requirements.
    """
    def decorator(f: Callable[..., Any]) -> Callable[..., Any]:
        # Filter parameters of the function/method
        filtered_params = filter_parameters(f)

        # Extract type hints
        type_hints = get_type_hints(f)

        # Ensure the 'message' parameter is defined and is a Pydantic BaseModel
        if "message" not in filtered_params:
            raise ValueError(f"The handler '{f.__name__}' must have a 'message' parameter of type BaseModel.")
        if not isinstance(type_hints.get("message"), type) or not issubclass(type_hints["message"], BaseModel):
            raise ValueError(f"The 'message' parameter in handler '{f.__name__}' must be a Pydantic BaseModel.")

        # Validate the optional 'metadata' parameter
        if "metadata" in filtered_params:
            if type_hints.get("metadata") != EventMessageMetadata:
                raise ValueError(f"If 'metadata' is defined in handler '{f.__name__}', it must be of type EventMessageMetadata.")

        # Check for unsupported parameters
        allowed_params = {"message", "metadata"}
        extra_params = set(filtered_params) - allowed_params
        if extra_params:
            raise ValueError(f"The handler '{f.__name__}' has unsupported parameters: {extra_params}.")

        # Attach routing metadata to the function for registration
        f._is_message_handler = True
        f._message_router_data = deepcopy({
            "pubsub": pubsub,
            "topic": topic,
            "route": route,
            "dead_letter_topic": dead_letter_topic,
            "is_broadcast": broadcast,
            "message_model": type_hints["message"],
        })

        return f

    # If the decorator is applied directly, return the decorated function
    return decorator(func) if func else decorator