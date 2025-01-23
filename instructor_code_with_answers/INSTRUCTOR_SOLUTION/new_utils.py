# Utils constructed by the students
from numpy.typing import NDArray
from collections import defaultdict
import numpy as np

def check_labels(y):
    """
    Check if labels are integers.
    Parameters:
        y: Labels
    Returns:
        Boolean indicating whether labels are integers
    """
    # Test that the 1D y array are all integers
    return np.issubdtype(y.dtype, np.int32)

def scale_data(X):
    """
    Scale the data to be between 0 and 1.
    Parameters:
        X: Training data matrix
    Returns:
        Scaled data matrix
    Notes:
        The data is rescaled so that the max is one and the min is zero
    """
    # scale the data to lie between 0 and 1
    # Updating self.X would be side effect. Bad for testing
    X = (X - X.min()) / (X.max() - X.min())
    return X

def unique_elements(y: NDArray) -> defaultdict[int, int]:
    """
    Count the number of each class element in array argument.

    Args:
        y: The input label array

    Returns:
        A dictionary of each element value (key) and the number of elements of this value (value)
    """
    # Check that each class has at least 1 element in y
    classes: defaultdict[int, int] = defaultdict(int)
    for y_val in y:
        classes[y_val] += 1

    return classes

def print_dataset_info(X, y):
    """
    Print information about the dataset.
    Parameters:
        X: Data matrix
        y: Labels
    Returns: min(X), max(X), min(y), max(y), shape(X), shape(y)
    """
    return np.min(X), np.max(X), np.min(y), np.max(y), np.shape(X), np.shape(y)

def remove_nines_convert_to_01(X, y, frac):
    """
    frac: fraction of 9s to remove (in [0,1])
    """
    # Count the number of 9s in the array
    num_nines = np.sum(y == 9)

    # Calculate the number of 9s to remove (90% of the total number of 9s)
    num_nines_to_remove = int(frac * num_nines)

    # Identifying indices of 9s in y
    indices_of_nines = np.where(y == 9)[0]

    # Randomly selecting 30% of these indices
    num_nines_to_remove = int(np.ceil(len(indices_of_nines) * frac))
    indices_to_remove = np.random.choice(
        indices_of_nines, size=num_nines_to_remove, replace=False
    )

    # Removing the selected indices from X and y
    X = np.delete(X, indices_to_remove, axis=0)
    y = np.delete(y, indices_to_remove)

    y[y == 7] = 0
    y[y == 9] = 1
    return X, y
