import pickle
import numpy as np


def unpickle(file_path) -> dict:
    """Simple unpickle func. Takes in file path and returns the unpickled data."""
    with open(file_path, "rb") as f:
        dict = pickle.load(f, encoding="latin1")
    return dict


def np_save(target_filename, arr) -> int:
    """Takes target filename and array and saves to that file."""
    np.save(target_filename, arr)
    print(f"{target_filename} saved successfully.")
    return 0


def cifar_loader(filepath, arr) -> int:
    """Uses the np_save func to load the cifar data to specified files."""
    np_save(filepath, arr)
    return 0
