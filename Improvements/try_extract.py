import zipfile
import os
import shutil


def try_extract_archive(archive_path, extract_to):
    """Try different extraction methods"""

    # Create extraction directory
    os.makedirs(extract_to, exist_ok=True)

    # Try as ZIP first
    try:
        with zipfile.ZipFile(archive_path, "r") as zip_ref:
            zip_ref.extractall(extract_to)
            print(f"Successfully extracted as ZIP to {extract_to}")
            return True
    except Exception as e:
        print(f"ZIP extraction failed: {e}")

    # If ZIP fails, try to read as binary and check header
    try:
        with open(archive_path, "rb") as f:
            header = f.read(10)
            print(f"File header (hex): {header.hex()}")
            print(f"File header (ascii): {header}")
    except Exception as e:
        print(f"Could not read file header: {e}")

    return False


def list_directory_contents(path):
    """List contents of extracted directory"""
    if os.path.exists(path):
        print(f"\nContents of {path}:")
        for root, dirs, files in os.walk(path):
            level = root.replace(path, "").count(os.sep)
            indent = " " * 2 * level
            print(f"{indent}{os.path.basename(root)}/")
            subindent = " " * 2 * (level + 1)
            for file in files:
                print(f"{subindent}{file}")
    else:
        print(f"Directory {path} does not exist")


if __name__ == "__main__":
    archive_path = "/workspace/user_input_files/Inferra V.rar"
    extract_to = "/workspace/inferra_project"

    success = try_extract_archive(archive_path, extract_to)

    if success:
        list_directory_contents(extract_to)
    else:
        print(
            "All extraction methods failed. Please provide the files in ZIP format or as individual files."
        )
