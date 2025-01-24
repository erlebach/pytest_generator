"""Part 3 of the assignment."""
# Inspired by GPT4

# Information on type hints
# https://peps.python.org/pep-0585/

# GPT on testing functions, mock functions, testing number of calls, and argument values
# https://chat.openai.com/share/b3fd7739-b691-48f2-bb5e-0d170be4428c

from typing import Any

import new_utils as nu

# ! from collections import defaultdict
# ! import logistic regresssion module
# ! from sklearn.linear_model import LogisticRegression
import numpy as np
from numpy.typing import NDArray

# ! from part_1_template_solution import Section1 as Part1
# ! from part_2_template_solution import Section2 as Part2
# ! from questions_part1_noclass import partC as section1_part_c
# ! from questions_part2_noclass import partC as section2_part_c
# ! from sklearn.base import accuracy_score
from sklearn.metrics import confusion_matrix, top_k_accuracy_score

# Fill in the appropriate import statements from sklearn to solve the homework
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from utils import Normalization

# ! import matplotlib.pyplot as plt
# ! import warnings

"""
In the first two set of tasks, we will narrowly focus on accuracy -
what fraction of our predictions were correct. However, there are several
popular evaluation metrics. You will learn how (and when) to use evaluation metrics.
"""

normalize = Normalization.APPLY_NORMALIZATION
seed = 42
frac_train = 0.8


def remove_nines_convert_to_01(
    x: NDArray[np.floating],
    y: NDArray[np.int32],
    frac: float,
) -> tuple[
    NDArray[np.floating],
    NDArray[np.int32],
]:
    """Remove a specified fraction of the 9s from the dataset and convert the labels.

    Parameters
    ----------
    x : NDArray[np.floating]
        The feature matrix from which to remove 9s.
    y : NDArray[np.int32]
        The labels corresponding to the feature matrix.
    frac : float
        The fraction of 9s to remove from the dataset (between 0 and 1).

    Returns
    -------
    tuple: A tuple containing the modified feature matrix X and the updated labels y.

    """
    # Count the number of 9s in the array
    num_nines = np.sum(y == 9)

    # Calculate the number of 9s to remove (90% of the total number of 9s)
    num_nines_to_remove = int(frac * num_nines)

    # Identifying indices of 9s in y
    indices_of_nines = np.where(y == 9)[0]

    # Randomly selecting 30% of these indices
    num_nines_to_remove = int(np.ceil(len(indices_of_nines) * frac))
    rng = np.random.default_rng(seed)
    indices_to_remove = rng.choice(
        a=indices_of_nines,
        size=num_nines_to_remove,
        replace=False,
    )

    # Removing the selected indices from X and y
    x = np.delete(x, indices_to_remove, axis=0)
    y = np.delete(y, indices_to_remove)

    y[y == 7] = 0
    y[y == 9] = 1
    return x, y


def analyze_class_distribution(
    y: NDArray[np.int32],
) -> dict[str, Any]:
    """Analyzes and prints the class distribution in the dataset.

    Parameters
    ----------
    y : NDArray[np.int32]
        The labels dataset to analyze for class distribution.

    Returns
    -------
    - dict: A dictionary containing the count of elements in each class and the total
        number of classes.

    """
    # Your code here to analyze class distribution
    # Hint: Consider using collections.Counter or numpy.unique for counting

    uniq, counts = np.unique(y, return_counts=True)
    print(f"{uniq=}")
    print(f"{counts=}")
    print(f"{np.sum(counts)=}")

    return {
        "class_counts": {},  # Replace with actual class counts
        "num_classes": 0,  # Replace with the actual number of classes
    }


# --------------------------------------------------------------------------


def partA(
    x: NDArray[np.floating],
    y: NDArray[np.int32],
    x_test: NDArray[np.floating],
    y_test: NDArray[np.int32],
) -> dict[str, Any]:
    """Check the integrity of the labels and analyze the class distribution.

    Parameters
    ----------
    x : NDArray[np.floating]
        The feature matrix for the dataset.
    y : NDArray[np.int32]
        The labels dataset to be checked and analyzed.
    x_test : NDArray[np.int32]
        The feature matrix for the test dataset.
    y_test : NDArray[np.int32]
        The labels for the test dataset.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the analyzed class distribution and integrity
            check results.

    A. Using the same classifier and hyperparameters as the one used at the end
        of part 2.B.
    Get the accuracies of the training/test set scores using the top_k_accuracy score
        for k=1,2,3,4,5.
    Make a plot of k vs. score and comment on the rate of accuracy change.
    Do you think this metric is useful for this dataset?

    """
    is_int = nu.check_labels(y)
    dist_dict = analyze_class_distribution(y)

    # top-k accuracy score
    # ! y_pred is not defined
    # ! score = top_k_accuracy_score(y, y_pred)

    answers = {}
    # MUST FIX THIS
    x_test = x.copy()
    y_test = y.copy()
    answers["x"] = x
    answers["y"] = y
    answers["x_test"] = x_test
    answers["y_test"] = y_test
    return answers  # ! , X, y, Xtest, ytest

    # --------------------------------------------------------------------------


# How to make sure the seed propagates. Perhaps specify in the class constructor.
def partB(
    x: NDArray[np.floating],
    y: NDArray[np.int32],
    x_test: NDArray[np.floating],
    y_test: NDArray[np.int32],
) -> dict[str, Any]:
    """Prepare an imbalanced dataset; convert 7s to 0s and 9s to 1s.

    This function filters the input dataset to retain only the classes 7 and 9,
    removes a specified fraction of the 9s, and converts the labels accordingly.
    It also prepares the test dataset in the same manner.

    Parameters
    ----------
    x : NDArray[np.floating]
        The feature matrix for the training dataset.
    y : NDArray[np.int32]
        The labels for the training dataset.
    x_test : NDArray[np.floating]
        The feature matrix for the test dataset.
    y_test : NDArray[np.int32]
        The labels for the test dataset.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the processed training and test datasets.

    B. Repeat part 1.B but return an imbalanced dataset consisting of 99% of all 9s
        removed.  Also convert the 7s to 0s and 9s to 1s.

    """
    # Only Keep 7 and 9's
    seven_nine_idx = (y == 7) | (y == 9)
    x = x[seven_nine_idx, :]
    y = y[seven_nine_idx]
    frac_to_remove = 0.8
    x, y = remove_nines_convert_to_01(x, y, frac=frac_to_remove)
    x_test, ytest = remove_nines_convert_to_01(x_test, y_test, frac=frac_to_remove)

    answers = {}
    answers["x"] = x
    answers["y"] = y
    answers["x_test"] = x_test
    answers["y_test"] = y_test
    return answers


# --------------------------------------------------------------------------


def partC(
    x: NDArray[np.floating],
    y: NDArray[np.int32],
    x_test: NDArray[np.floating],
    y_test: NDArray[np.int32],
) -> dict[str, Any]:
    """Repeat part 1.C for this dataset with a support vector machine (SVC in sklearn).

    This function implements a support vector machine classifier using a
        stratified cross-validation strategy.
    It evaluates the model's performance by calculating the mean and standard deviation
        of various metrics, including accuracy, F1 score, precision, and recall. The
        function also determines whether precision or recall is higher and provides an
        explanation for the observed results. Finally, the classifier is trained on the
        entire training dataset, and the confusion matrix is generated to assess the
        classification performance.

    Parameters
    ----------
    x : NDArray[np.floating]
        The feature matrix for the training dataset.
    y : NDArray[np.int32]
        The labels for the training dataset.
    x_test : NDArray[np.floating]
        The feature matrix for the test dataset.
    y_test : NDArray[np.int32]
        The labels for the test dataset.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the following keys:
        - "cv": The cross-validation object used for stratified k-fold.
        - "clf": The trained SVC classifier.
        - "mean_F1": The mean F1 score from cross-validation.
        - "mean_recall": The mean recall from cross-validation.
        - "mean_accuracy": The mean accuracy from cross-validation.
        - "mean_precision": The mean precision from cross-validation.
        - "std_F1": The standard deviation of the F1 score from cross-validation.
        - "std_recall": The standard deviation of the recall from cross-validation.
        - "std_accuracy": The standard deviation of the accuracy from cross-validation.
        - "std_precision": The standard deviation of the precision from
            cross-validation.
        - "is_precision_higher_than_recall": A boolean indicating if precision is
            higher than recall.
        - "is_precision_higher_than_recall_explain": A string explanation of the
            precision-recall comparison.
        - "confusion_matrix": The confusion matrix for the classifier's predictions
            on the test dataset.

    C. Repeat part 1.C for this dataset with a support vector machine (SVC in sklearn).
        Make sure to use a stratified cross-validation strategy. In addition to regular
        accuracy also print out the mean/std of the F1 score, precision, and recall.
        Is precision or recall higher? Explain. Finally, train the classifier on all
        the training data and plot the confusion matrix.

    """
    n_splits = 5
    clf = SVC(random_state=seed)
    # Shuffling is fine because of seed
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    score = ["accuracy", "recall", "precision", "f1"]
    scores = cross_validate(clf, x, y, scoring=score, cv=cv)

    # Train on all the data
    clf.fit(x, y)

    # rows: actual, columns: predicted
    # cols: actual, columns: predicted
    # Return confusion matrix (DO NOT USE plot_confusion_matrix, which is deprecated)
    #   (CHECK via test that it is not used)
    # Return confusion matrix (no need to plot it)
    y_pred = clf.predict(x_test)
    conf_mat = confusion_matrix(y_test, y_pred)

    print(f"{scores=}")

    answers = {}
    answers["cv"] = cv
    answers["clf"] = clf
    answers["mean_F1"] = 0
    answers["mean_recall"] = 0
    answers["mean_accuracy"] = 0
    answers["mean_precision"] = 0
    answers["std_F1"] = 0
    answers["std_recall"] = 0
    answers["std_accuracy"] = 0
    answers["std_precision"] = 0
    answers["is_precision_higher_than_recall"] = None  # True/False
    answers["is_precision_higher_than_recall_explain"] = None  # String
    answers["confusion_matrix"] = conf_mat  # 2 x 2 matrix

    ## For testing, I can check the arguments of functions
    return answers


# --------------------------------------------------------------------------


def partD(
    x_train: NDArray[np.floating],
    y_train: NDArray[np.int32],
    x_test: NDArray[np.floating],
    y_test: NDArray[np.int32],
) -> dict[str, Any]:
    """Train and evaluate a Support Vector Classifier (SVC) with class weights.

    Parameters
    ----------
    x_train : NDArray[np.floating]
        The feature matrix for the training data.
    y_train : NDArray[np.int32]
        The labels corresponding to the training data.
    x_test : NDArray[np.floating]
        The feature matrix for the testing data.
    y_test : NDArray[np.int32]
        The labels corresponding to the testing data.

    Returns
    -------
    dict[str, Any]
        A dictionary containing:
        - cv: The cross-validation object used.
        - clf: The trained classifier.
        - mean_F1: The mean F1 score from cross-validation.
        - mean_recall: The mean recall from cross-validation.
        - mean_accuracy: The mean accuracy from cross-validation.
        - mean_precision: The mean precision from cross-validation.
        - std_F1: The standard deviation of the F1 score.
        - std_recall: The standard deviation of the recall.
        - std_accuracy: The standard deviation of the accuracy.
        - std_precision: The standard deviation of the precision.
        - is_precision_higher_than_recall: Boolean indicating if precision is higher
            than recall.
        - is_precision_higher_than_recall_explain: Explanation of the precision vs
            recall comparison.
        - performance_difference_explain: Explanation of the performance difference
            due to class weights.
        - confusion_matrix: The confusion matrix for the predictions on the test set.
        - weight_dict: A dictionary of class weights used in training.

    D. Repeat the same steps as part 3.C but apply a weighted loss function (see the
        class_weights parameter).  Print out the class weights, and comment on the
        performance difference. Use compute_class_weight to compute the class weights.

    """
    n_splits = 5
    # ! clf = SVC(random_state=self.seed, class_weight="balanced")
    clf = SVC(random_state=seed)
    # Shuffling is fine because of seed
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    score = ["accuracy", "recall", "precision", "f1"]
    # ! scores = cross_validate(clf, x_train, y_train, scoring=score, cv=cv)

    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    weight_dict = dict(
        zip(np.unique(y_train), class_weights, strict=True),
    )

    print("class_weights = ", class_weights)  # Dimension 2
    print("weight_dict = ", weight_dict)  # Dimension 2

    # Train on all the data
    clf.fit(x_train, y_train)

    y_pred = clf.predict(x_test)
    conf_mat = confusion_matrix(y_test, y_pred)
    print(f"{conf_mat=}")

    answers = {}
    answers["cv"] = cv
    answers["clf"] = clf
    answers["mean_F1"] = 0
    answers["mean_recall"] = 0
    answers["mean_accuracy"] = 0
    answers["mean_precision"] = 0
    answers["std_F1"] = 0
    answers["std_recall"] = 0
    answers["std_accuracy"] = 0
    answers["std_precision"] = 0
    answers["is_precision_higher_than_recall"] = None  # True/False
    answers["is_precision_higher_than_recall_explain"] = None  # String
    answers["performance_difference_explain"] = None
    answers["conf_mat"] = conf_mat
    answers["weight_dict"] = {}
    return answers
