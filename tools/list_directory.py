import os
from langchain_core.tools import tool
from pydantic import BaseModel, Field
from typing import List


class ListDirectoryInput(BaseModel):
    directory_path: str = Field(description="The path to the directory to list.")


@tool(args_schema=ListDirectoryInput)
def list_directory(directory_path: str) -> List[str]:
    """Lists the contents (files and subdirectories) of a specified directory."""
    if not os.path.isdir(directory_path):
        raise ValueError(f"Error: Directory not found at path: {directory_path}")
    try:
        return os.listdir(directory_path)
    except Exception as e:
        raise ValueError(f"Error listing directory {directory_path}: {e}")
