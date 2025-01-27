"""Utility functions for the homework."""

import pickle
from enum import Enum
from pathlib import Path

import numpy as np
import utils as u
from numpy.typing import NDArray
from sklearn import datasets
from sklearn.base import BaseEstimator
from sklearn.model_selection import (
    KFold,
    cross_validate,
)
from sklearn.tree import DecisionTreeClassifier


class Normalization(Enum):
    """Enum class to control data normalization.

    This enum provides boolean flags to indicate whether normalization
    should be applied to data or skipped.

    Attributes
    ----------
    APPLY_NORMALIZATION : bool
        Flag indicating normalization should be applied (True)
    SKIP_NORMALIZATION : bool
        Flag indicating normalization should be skipped (False)

    """

    APPLY_NORMALIZATION = True
    SKIP_NORMALIZATION = False


class PrintResults(Enum):
    """Enum class to control the printing of results.

    This enum provides options to determine whether results should be printed
    or skipped during the execution of the program.

    Attributes
    ----------
    PRINT_RESULTS : bool
        Flag indicating that results should be printed (True).
    SKIP_PRINT_RESULTS : bool
        Flag indicating that results should be skipped (False).

    """

    PRINT_RESULTS = True
    SKIP_PRINT_RESULTS = False


def load_mnist_dataset(
    nb_samples: int | None = None,
) -> tuple[NDArray[np.floating], NDArray[np.int32]]:
    """Load the MNIST dataset.

    Parameters
    ----------
    nb_samples : int, optional
        Number of samples to load. If None, loads full dataset.
        Useful for code testing.

    Returns
    -------
    x : NDArray[np.floating]
        Array of flattened images, shape (n_samples, 784)
    y : NDArray[np.int32]
        Array of labels, shape (n_samples,)

    """
    try:
        # Are the datasets already loaded?
        print("... Is MNIST dataset local?")
        x: NDArray[np.floating] = np.load("mnist_X.npy")
        y: NDArray[np.int32] = np.load("mnist_y.npy", allow_pickle=True)
    except FileNotFoundError:
        # Download the datasets
        print("... download MNIST dataset")
        bunch = datasets.fetch_openml(
            "mnist_784",
            version=1,
            return_X_y=True,
            as_frame=False,
        )
        x = np.array(bunch[0], dtype=np.float32)
        y = np.array(bunch[1], dtype=np.int32)

    if nb_samples is not None and nb_samples < x.shape[0]:
        x = x[0:nb_samples, :]
        y = y[0:nb_samples]

    print("x.shape: ", x.shape)
    print("y.shape: ", y.shape)
    np.save("mnist_X.npy", x)
    np.save("mnist_y.npy", y)
    return x, y


def prepare_data(
    num_train: int = 60000,
    num_test: int = 10000,
    normalize: bool = True,
    frac_train: float = 0.8,
) -> tuple[
    NDArray[np.float32],
    NDArray[np.int32],
    NDArray[np.float32],
    NDArray[np.int32],
]:
    """Prepare the data.

    Parameters
    ----------
    num_train : int, optional
        Number of training samples. Default is 60000.
    num_test : int, optional
        Number of testing samples. Default is 10000.
    normalize : bool, optional
        Whether to normalize the data. Default is True.
    frac_train : float, optional
        Fraction of training samples. Default is 0.8.

    Returns
    -------
    x_train : NDArray[np.float32]
        Training data matrix
    y_train : NDArray[np.int32]
        Training labels array
    x_test : NDArray[np.float32]
        Testing data matrix
    y_test : NDArray[np.int32]
        Testing labels array

    Notes
    -----
    The function loads the MNIST dataset, optionally normalizes it, and splits it into
    training and testing sets based on the specified sizes.

    """
    # Check in case the data is already on the computer.
    x, y = load_mnist_dataset()

    # won't work well unless X is greater or equal to zero
    if normalize:
        x = x / x.max()

    y = y.astype(np.int32)
    x_train, x_test = x[:num_train], x[num_train : num_train + num_test]
    y_train, y_test = y[:num_train], y[num_train : num_train + num_test]
    return x_train, y_train, x_test, y_test


def create_data(
    n_rows: int,
    n_features: int,
    frac_train: float = 0.8,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.int32],
    NDArray[np.floating],
    NDArray[np.int32],
]:
    """Create synthetic data for testing and training.

    Parameters
    ----------
    n_rows : int
        Number of total samples to generate
    n_features : int
        Number of features per sample
    frac_train : float, optional
        Fraction of training samples. Default is 0.8.

    Returns
    -------
    tuple[NDArray[np.floating], NDArray[np.int32], NDArray[np.floating], NDArray[np.int32]]
        x_train : Training data matrix
        y_train : Training labels array
        x_test : Testing data matrix
        y_test : Testing labels array

    Notes
    -----
    Creates synthetic data by generating random features and computing binary labels
    based on the sum of the first 5 features. Data is split into training and test
    sets according to frac_train.
    """
    # Create random data
    rng = np.random.default_rng(42)
    x_full = rng.random((n_rows, n_features))
    # Create labels (sum first 5 columns)
    y_full = (x_full[:, :5].sum(axis=1) > 2.5).astype(int)

    # Split into train/test
    n_train = int(frac_train * n_rows)  # Use 80% for training
    x_train = x_full[:n_train, :]
    x_test = x_full[n_train:, :]
    y_train = y_full[:n_train]
    y_test = y_full[n_train:]

    return x_train, y_train, x_test, y_test


def filter_out_7_9s(
    x: NDArray[np.floating],
    y: NDArray[np.int32],
) -> tuple[NDArray[np.floating], NDArray[np.int32]]:
    """Filter the dataset to include only the digits 7 and 9.

    Parameters
    ----------
    x : NDArray[np.floating]
        Data matrix containing digit images
    y : NDArray[np.int32]
        Labels array containing digit classes

    Returns
    -------
    tuple[NDArray[np.floating], NDArray[np.int32]]
        x_binary : Filtered data matrix containing only 7 and 9 digits
        y_binary : Filtered labels containing only 7 and 9 classes

    Notes
    -----
    The function filters the input data to keep only samples labeled as 7 or 9,
    returning both the filtered features and their corresponding labels.

    """
    seven_nine_idx = (y == 7) | (y == 9)
    x_binary = x[seven_nine_idx, :]
    y_binary = y[seven_nine_idx]
    return x_binary, y_binary


def train_simple_classifier_with_cv(
    x_train: NDArray[np.floating],
    y_train: NDArray[np.int32],
    clf: BaseEstimator,
    n_splits: int = 5,
    cv_class: type[KFold] = KFold,
) -> dict[str, NDArray[np.float32]]:
    """Train a simple classifier using k-fold cross-validation.

    Parameters
    ----------
    x_train : NDArray[np.floating]
        Features dataset
    y_train : NDArray[np.int32]
        Labels array
    clf : BaseEstimator
        The classifier to train and validate
    n_splits : int, default=5
        Number of splits for cross-validation
    cv_class : type[KFold], default=KFold
        The cross-validation class to use

    Returns
    -------
    dict[str, NDArray[np.float32]]
        Dictionary containing cross-validation results with keys:
        'fit_time', 'score_time', 'test_score'

    """
    cv = cv_class(n_splits=n_splits)
    return cross_validate(clf, x_train, y_train, cv=cv)


def print_cv_result_dict(
    cv_dict: dict[str, NDArray[np.float32]],
) -> None:
    """Print mean and standard deviation for each metric in cross-validation results.

    Parameters
    ----------
    cv_dict : dict
        Dictionary containing cross-validation results with metrics as keys
        and numpy arrays of scores as values

    Returns
    -------
    None
        Prints formatted results to standard output

    """
    for key, array in cv_dict.items():
        print(f"mean_{key}: {array.mean()}, std_{key}: {array.std()}")


def starter_code() -> int:
    """Run a basic machine learning pipeline on MNIST data.

    This function:
    1. Loads and prepares MNIST data
    2. Filters to keep only digits 7 and 9
    3. Trains a decision tree classifier using cross-validation
    4. Prints the cross-validation results
    5. Return 100 if there is no error

    Parameters
    ----------
    None

    Returns
    -------
    None
        Prints cross-validation results to standard output

    """
    x_train, y_train, x_test, y_test = prepare_data()
    x_train, y_train = u.filter_out_7_9s(x_train, y_train)
    x_test, y_test = u.filter_out_7_9s(x_test, y_test)
    out_dict = u.train_simple_classifier_with_cv(
        x_train,
        y_train,
        DecisionTreeClassifier(),
    )
    print("running cross validation...")
    u.print_cv_result_dict(out_dict)
    return 100


def save_dict(filenm: str, dct: dict) -> None:
    """Save a dictionary to a pickle file.

    Parameters
    ----------
    filenm : str
        The filename to save the dictionary to
    dct : dict
        The dictionary to save

    Returns
    -------
    None
        Saves the dictionary to a pickle file

    """
    with Path(filenm).open("wb") as file:
        pickle.dump(dct, file)


# Loading from a pickle file
def load_dict(filenm: str) -> dict:
    """Load a dictionary from a pickle file.

    Parameters
    ----------
    filenm : str
        The filename to load the dictionary from

    Returns
    -------
    dict
        The loaded dictionary from the pickle file

    """
    with Path(filenm).open("rb") as file:
        return pickle.load(file)


if __name__ == "__main__":
    starter_code()
