import os
import time  # Import time
from langchain_core.tools import tool
import logging


@tool
def write_to_file(file: str, file_contents: str) -> str:
    """
    Write the given contents to a file.

    This tool will create the necessary directories if they don't exist.
    It will overwrite the file if it already exists. Use with caution.

    Args:
        file (str): The relative path to the file.
        file_contents (str): The contents to write to the file.

    Returns:
        str: A message indicating success or failure.
    """
    try:
        # Ensure the directory exists
        dir_path = os.path.dirname(file)
        if dir_path:  # Check if there is a directory path
            os.makedirs(dir_path, exist_ok=True)
            logging.info(f"Ensured directory exists: {dir_path}")

            # Ensure __init__.py exists in the target directory
            init_path = os.path.join(dir_path, "__init__.py")
            if not os.path.exists(init_path):
                try:
                    with open(init_path, "w", encoding="utf-8") as init_f:
                        init_f.write("")  # Create empty __init__.py
                    logging.info(f"Created missing __init__.py in {dir_path}")
                except OSError as init_e:
                    # Log a warning but proceed with writing the main file
                    logging.warning(
                        f"Could not create __init__.py in {dir_path}: {init_e}"
                    )

        # Write the file (overwrite if exists)
        print(f"Writing to file: {file}")  # Keep user informed

        # Add retry mechanism for file writing
        max_retries = 3
        retry_delay = 0.1
        for attempt in range(max_retries):
            try:
                with open(file, "w", encoding="utf-8") as f:  # Specify encoding
                    f.write(file_contents)

                # Add a small delay to allow filesystem synchronization
                time.sleep(retry_delay)

                # Verify write by checking file existence and content
                if os.path.exists(file):
                    with open(file, "r", encoding="utf-8") as f:
                        written_content = f.read()
                    if written_content == file_contents:
                        logging.info(f"Successfully verified write to {file}")
                        return f"File {file} written successfully."
                    else:
                        if attempt < max_retries - 1:
                            logging.warning(
                                f"Content mismatch on attempt {attempt + 1}, retrying..."
                            )
                            time.sleep(retry_delay)
                            continue
                        else:
                            logging.error(
                                f"File {file} write verification failed after {max_retries} attempts. Content mismatch."
                            )
                            return f"Error: File {file} was not written correctly (content mismatch)."
                else:
                    if attempt < max_retries - 1:
                        logging.warning(
                            f"File not found on attempt {attempt + 1}, retrying..."
                        )
                        time.sleep(retry_delay)
                        continue
                    else:
                        logging.error(
                            f"File {file} write verification failed after {max_retries} attempts. File does not exist."
                        )
                        return f"Error: File {file} was not written correctly (file not found)."

            except Exception as write_e:
                if attempt < max_retries - 1:
                    logging.warning(
                        f"Write error on attempt {attempt + 1}: {write_e}, retrying..."
                    )
                    time.sleep(retry_delay)
                    continue
                else:
                    logging.error(
                        f"Error writing file {file} after {max_retries} attempts: {write_e}",
                        exc_info=True,
                    )
                    return f"Error writing file {file}: {write_e}"

    except OSError as e:
        logging.error(
            f"Error writing file {file}: {e}", exc_info=True
        )  # Log detailed error
        return f"Error writing file {file}: {e}"
    except Exception as e:  # Catch any other unexpected errors
        logging.error(
            f"An unexpected error occurred while writing file {file}: {e}",
            exc_info=True,
        )
        return f"An unexpected error occurred while writing file {file}: {e}"
