from typing import Any
import datetime

LIST_LIMIT = 100  # Maximum number of elements in a list to be processed

def value_sanitize(data: Any) -> Any:
    """
    Sanitizes the input data (dictionary or list) for use in a language model context.
    This function filters out large lists and simplifies nested structures to improve the
    efficiency of language model processing.

    Args:
        data (Any): The data to sanitize, which can be a dictionary, list, or other types.

    Returns:
        Any: The sanitized data, or None if a list exceeds the predefined size limit.
    """
    if isinstance(data, dict):
        return {key: sanitized for key, value in data.items() if (sanitized := value_sanitize(value)) is not None}
    elif isinstance(data, list):
        if len(data) > LIST_LIMIT:
            return None
        return [sanitized for item in data if (sanitized := value_sanitize(item)) is not None]
    else:
        return data

def get_current_time():
        """Get current time in UTC for creation and modification of nodes and relationships"""
        return datetime.datetime.now(datetime.timezone.utc).isoformat().replace('+00:00', 'Z')