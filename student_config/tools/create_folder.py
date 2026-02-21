import os

def create_folder(path: str) -> str:
    try:
        os.makedirs(path, exist_ok=True)
        abs_path = os.path.abspath(path)
        return f"Successfully created folder: {abs_path}"
    except Exception as e:
        return f"Error creating folder '{path}': {str(e)}"