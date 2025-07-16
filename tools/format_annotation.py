import json
from langchain_core.tools import tool


@tool
def format_annotation(log_entry: str, annotation: str) -> str:
    """
    Formats a log entry and a corresponding annotation into a JSON string.

    Args:
        log_entry: The logged data string (e.g., user request, plan).
        annotation: The human-provided annotation string for the log entry.

    Returns:
        A JSON string containing the log entry and its annotation.
        Example: '{"log_entry": "User request: Find cat pictures", "annotation": "NLU: intent=find_image, entity=cat"}'
    """
    if not isinstance(log_entry, str) or not isinstance(annotation, str):
        # This check might be redundant if LangChain's Pydantic validation catches it first,
        # but provides an explicit error if used directly.
        raise ValueError("Both log_entry and annotation must be strings.")
    output_dict = {"log_entry": log_entry, "annotation": annotation}
    return json.dumps(output_dict)
