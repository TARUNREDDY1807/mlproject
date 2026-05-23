import os
import pickle


def save_object(file_path: str, obj) -> None:
    """Save a Python object to disk using pickle."""
    dir_name = os.path.dirname(file_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)

    with open(file_path, "wb") as f:
        pickle.dump(obj, f)

