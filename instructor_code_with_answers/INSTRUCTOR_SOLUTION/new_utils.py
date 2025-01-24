"""aUtils constructed by the students."""

from collections import defaultdict

import numpy as np
from numpy.typing import NDArray


def check_labels(y: NDArray) -> np.bool:
    """Check if labels array contains only integers.

    Args:
        y : NDArray
            Array of labels to check

    Returns:
        bool
            True if labels are integers, False otherwise

    """
    # Test that the 1D y array are all integers
    return np.issubdtype(y.dtype, np.int32)


def scale_data(x: NDArray[np.floating]) -> NDArray[np.floating]:
    """Scale the data to be between 0 and 1.

    Parameters
    ----------
    x : NDArray[np.floating]
        Training data matrix to be scaled

    Returns
    -------
    NDArray[np.floating]
        Scaled data matrix with values between 0 and 1

    Notes
    -----
    The data is rescaled using min-max normalization so that the maximum value
    becomes 1 and the minimum value becomes 0. The input array `x` is not modified.

    """
    # scale the data to lie between 0 and 1
    # Updating self.X would be side effect. Bad for testing
    return (x - x.min()) / (x.max() - x.min())


def unique_elements(y: NDArray) -> defaultdict[int, int]:
    """Count the number of each class element in array argument.

    Parameters
    ----------
    y : NDArray
        The input label array

    Returns
    -------
    defaultdict[int, int]
        A dictionary mapping each unique element value (key) to its count (value)

    Notes
    -----
    Uses a defaultdict to automatically initialize counts to 0 for new elements.

    """
    # Check that each class has at least 1 element in y
    classes: defaultdict[int, int] = defaultdict(int)
    for y_val in y:
        classes[y_val] += 1

    return classes


def print_dataset_info(
    x: NDArray[np.floating],
    y: NDArray[np.int32],
) -> tuple:
    """Print information about the dataset.

    Parameters
    ----------
    x : NDArray[np.floating]
        Data matrix containing features
    y : NDArray[np.int32]
        Array of labels

    Returns
    -------
    tuple
        A tuple containing:
        - min(x) : Minimum value in data matrix
        - max(x) : Maximum value in data matrix
        - min(y) : Minimum label value
        - max(y) : Maximum label value
        - shape(x) : Shape of data matrix
        - shape(y) : Shape of labels array

    Notes
    -----
    This function returns key statistics about both the feature matrix x
    and the labels array y to help understand the dataset structure.

    """
    return np.min(x), np.max(x), np.min(y), np.max(y), np.shape(x), np.shape(y)


def remove_nines_convert_to_01(
    x: NDArray[np.floating],
    y: NDArray[np.int32],
    frac: float,
) -> tuple[NDArray[np.floating], NDArray[np.int32]]:
    """Remove a fraction of 9s from the dataset and convert labels to binary.

    Parameters
    ----------
    x : NDArray[np.floating]
        Data matrix containing features
    y : NDArray[np.int32]
        Array of labels containing 7s and 9s
    frac : float
        Fraction of 9s to remove, must be in range [0,1]

    Returns
    -------
    tuple[NDArray[np.floating], NDArray[np.int32]]
        x : Data matrix with fraction of 9s removed
        y : Binary labels with 7->0 and remaining 9->1

    Notes
    -----
    This function randomly removes a specified fraction of samples labeled as 9,
    then converts the remaining labels to binary (0 for 7, 1 for 9).

    """
    # Count the number of 9s in the array
    num_nines = np.sum(y == 9)

    # Calculate the number of 9s to remove (90% of the total number of 9s)
    num_nines_to_remove = int(frac * num_nines)

    # Identifying indices of 9s in y
    indices_of_nines = np.where(y == 9)[0]

    # Randomly selecting 30% of these indices
    num_nines_to_remove = int(np.ceil(len(indices_of_nines) * frac))
    rng = np.random.default_rng()
    indices_to_remove = rng.choice(
        indices_of_nines,
        size=num_nines_to_remove,
        replace=False,
    )

    # Removing the selected indices from X and y
    x = np.delete(X, indices_to_remove, axis=0)
    y = np.delete(y, indices_to_remove)

    y[y == 7] = 0
    y[y == 9] = 1
    return x, y
