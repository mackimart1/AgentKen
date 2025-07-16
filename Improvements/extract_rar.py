import rarfile
import os


def extract_rar(rar_path, extract_to):
    """Extract RAR file to specified directory"""
    try:
        with rarfile.RarFile(rar_path) as rf:
            rf.extractall(extract_to)
            print(f"Successfully extracted {rar_path} to {extract_to}")

            # List extracted contents
            print("\nExtracted contents:")
            for root, dirs, files in os.walk(extract_to):
                level = root.replace(extract_to, "").count(os.sep)
                indent = " " * 2 * level
                print(f"{indent}{os.path.basename(root)}/")
                subindent = " " * 2 * (level + 1)
                for file in files:
                    print(f"{subindent}{file}")
    except Exception as e:
        print(f"Error extracting RAR file: {e}")


if __name__ == "__main__":
    rar_path = "/workspace/user_input_files/Inferra V.rar"
    extract_to = "/workspace/inferra_project"

    # Create extraction directory
    os.makedirs(extract_to, exist_ok=True)

    extract_rar(rar_path, extract_to)
