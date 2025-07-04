from langchain_core.tools import tool
from typing import Any


@tool
def data_transformer(data: Any, transformation_type: str) -> Any:
    """Transforms input data based on the specified transformation type.

    Args:
        data (Any): The input data to be transformed.
        transformation_type (str): The type of transformation to apply.
                                   Supported types: "to_string", "uppercase", "lowercase", "reverse_string".

    Returns:
        Any: The transformed data.

    Raises:
        ValueError: If an unsupported transformation_type is provided.
        TypeError: If the data type is incompatible with the transformation.
    """
    if transformation_type == "to_string":
        return str(data)
    elif transformation_type == "uppercase":
        if not isinstance(data, str):
            raise TypeError("Data must be a string for 'uppercase' transformation.")
        return data.upper()
    elif transformation_type == "lowercase":
        if not isinstance(data, str):
            raise TypeError("Data must be a string for 'lowercase' transformation.")
        return data.lower()
    elif transformation_type == "reverse_string":
        if not isinstance(data, str):
            raise TypeError(
                "Data must be a string for 'reverse_string' transformation."
            )
        return data[::-1]
    else:
        raise ValueError(
            f"Unsupported transformation type: {transformation_type}. "
            "Supported types: 'to_string', 'uppercase', 'lowercase', 'reverse_string'."
        )
