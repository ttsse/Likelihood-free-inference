# Added feature
# Usage: from TTSSE_Project.features.dataset_storage import save_to_binary_file, load_from_binary_file

# Used to save and load datasets for sciope inference

import pickle
import os
from TTSSE_Project.utilities.housekeeping import sciope_logger as ml


def save_to_binary_file(data, filename, path=None, use_logger=False):
    """Saves the given data into a binary file."""
    try:
        if path:
            full_path = os.path.join(path, filename)
        else:
            full_path = filename

        with open(full_path, 'wb') as f:
            pickle.dump(data, f)

        if use_logger:
            logger = ml.SciopeLogger().get_logger()
            logger.info(f"Data saved under the file name '{filename}' and the location of the file is: {full_path}")
        else:
            print(f"Data saved under the file name '{filename}' and the location of the file is: {full_path}")

    except Exception as e:
        if use_logger:
            logger = ml.SciopeLogger().get_logger()
            logger.error(f"Could not save data to {full_path}: {e}")
        else:
            print(f"Could not save data to {full_path}: {e}")


def load_from_binary_file(filename, path=None, use_logger=False):
    """Loads data from a given binary file."""
    try:
        if path:
            full_path = os.path.join(path, filename)
        else:
            full_path = filename

        with open(full_path, 'rb') as f:
            data = pickle.load(f)

        if use_logger:
            logger = ml.SciopeLogger().get_logger()
            logger.info(f"Data loaded from the file name '{filename}' and the location of the file is: {full_path}")
        else:
            print(f"Data loaded from the file name '{filename}' and the location of the file is: {full_path}")

        return data

    except Exception as e:
        if use_logger:
            logger = ml.SciopeLogger().get_logger()
            logger.error(f"Could not load data from {full_path}: {e}")
        else:
            print(f"Could not load data from {full_path}: {e}")



# Example usage in some other module
if __name__ == "__main__":
    from TTSSE_Project.features.dataset_storage import save_to_binary_file, load_from_binary_file

    my_data = {'key': 'value'}
    save_to_binary_file(my_data, 'data.pkl', use_logger=True)
    loaded_data = load_from_binary_file('data.pkl', use_logger=True)
