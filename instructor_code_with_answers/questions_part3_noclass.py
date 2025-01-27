"""Part 3 of the assignment."""
# Inspired by GPT4

# Information on type hints
# https://peps.python.org/pep-0585/

# GPT on testing functions, mock functions, testing number of calls, and argument values
# https://chat.openai.com/share/b3fd7739-b691-48f2-bb5e-0d170be4428c

from typing import Any

import matplotlib.pyplot as plt
import new_utils as nu

# ! from collections import defaultdict
# ! import logistic regresssion module
# ! from sklearn.linear_model import LogisticRegression
import numpy as np
import utils as u
from numpy.typing import NDArray

# ! from part_1_template_solution import Section1 as Part1
# ! from part_2_template_solution import Section2 as Part2
# ! from questions_part1_noclass import part_c as section1_part_c
# ! from questions_part2_noclass import part_c as section2_part_c
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

    Convert remaining 9s to 1, and all 7s to 0.

    Parameters
    ----------
    x : NDArray[np.floating]
        The feature matrix from which to remove 9s.
    y : NDArray[np.int32]
        The labels corresponding to the feature matrix.
        y contains only 7s and 9s.
    frac : float
        The fraction of 9s to remove from the dataset (between 0 and 1).

    Returns
    -------
    tuple: A tuple containing the modified feature matrix x and the updated labels y.

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
    class_counts: dict[np.int32, np.int32] = dict(zip(uniq, counts, strict=True))
    num_classes = len(class_counts)
    print(f"{uniq=}")
    print(f"{counts=}")
    print(f"{class_counts=}")
    print(f"{num_classes=}")
    print(f"{np.sum(counts)=}")

    return {
        "class_counts": class_counts,  # Replace with actual class counts
        "num_classes": num_classes,  # Replace with the actual number of classes
    }


# --------------------------------------------------------------------------


def part_3a(
    x_train_: NDArray[np.floating] | None = None,
    y_train_: NDArray[np.int32] | None = None,
    x_test_: NDArray[np.floating] | None = None,
    y_test_: NDArray[np.int32] | None = None,
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

    Task
    ----
    A. Using the same classifier and hyperparameters as the one used at the end
        of part 2.B, get the accuracies of the training/test set scores using
        the top_k_accuracy score for k=1,2,3,4,5.
    Make a plot of k vs. score and comment on the rate of accuracy change.
    Do you think this metric is useful for this dataset?

    """
    global x_train, y_train, x_test, y_test  # noqa: PLW0603

    if x_train_ is not None:
        x_train = x_train_
    if y_train_ is not None:
        y_train = y_train_
    if x_test_ is not None:
        x_test = x_test_
    if y_test_ is not None:
        y_test = y_test_
    # ==========================================
    # DO NOT CHANGE THE FUNCTION ABOVE THIS LINE
    # ==========================================

    # The data should have 10 classes

    # Read grid search results from part 2b
    grid_search_results = u.load_dict("part_2b_grid_search.pkl")
    print(dir(grid_search_results))
    print(f"{grid_search_results=}")
    print(f"{list(grid_search_results.__dict__.keys())=}")

    best_estimator = grid_search_results.__dict__["best_estimator_"]
    best_estimator.fit(x_train, y_train)
    best_estimator.fit(x_test, y_test)

    train_probs = best_estimator.predict_proba(x_train)
    test_probs = best_estimator.predict_proba(x_test)

    # Prepare for plotting
    k_list = [1, 2, 3, 4, 5]
    test_accuracy = []
    train_accuracy = []

    topk_dict = {}
    for k in k_list:
        train_score = top_k_accuracy_score(y_train, train_probs, k=k)
        test_score = top_k_accuracy_score(y_test, test_probs, k=k)
        train_accuracy.append(train_score)
        test_accuracy.append(test_score)
        topk_dict[k] = [train_score, test_score]
        print(f"{k=}, {train_score=}, {test_score=}")

    plt.plot(k_list, train_accuracy, label="Train Accuracy")
    plt.plot(k_list, test_accuracy, label="Test Accuracy")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Top-k Accuracy for k=1,2,3,4,5")
    plt.legend()
    plt.savefig("top_k_accuracy.png")

    # Count the number of classes in the training and test sets
    count_train = np.unique(y_train, return_counts=True)
    count_test = np.unique(y_test, return_counts=True)
    print(f"{count_train=}")
    print(f"{count_test=}")

    # Check if the labels are integers
    is_int = nu.check_labels(y)

    # Analyze the class distribution
    dist_dict = analyze_class_distribution(y)

    # top-k accuracy score
    # ! y_pred is not defined
    # ! score = top_k_accuracy_score(y, y_pred)

    answers = {}

    # Answer type: dict[int, list[float, float]]
    # The two floats are the train and test accuracy for the top-k accuracy score
    # Remember: order of the elements in a list matters
    answers["top_k_accuracy"] = topk_dict
    print("exit part 3a")

    return answers  # ! , X, y, Xtest, ytest

    # --------------------------------------------------------------------------


# How to make sure the seed propagates. Perhaps specify in the class constructor.
def part_3b(
    x_train_: NDArray[np.floating] | None = None,
    y_train_: NDArray[np.int32] | None = None,
    x_test_: NDArray[np.floating] | None = None,
    y_test_: NDArray[np.int32] | None = None,
) -> dict[str, Any]:
    """Prepare an imbalanced dataset; convert 7s to 0s and 9s to 1s.

    This function filters the input dataset to retain only the classes 7 and 9,
    removes a specified fraction of the 9s, and convert the labels accordingly.
    It also prepares the test dataset in the same manner.

    Parameters
    ----------
    x_train_ : NDArray[np.floating]
        The feature matrix for the training dataset.
    y_train_ : NDArray[np.int32]
        The labels for the training dataset.
    x_test_ : NDArray[np.floating]
        The feature matrix for the test dataset.
    y_test : NDArray[np.int32]
        The labels for the test dataset.

    Returns
    -------
    dict[str, Any]
        A dictionary containing the processed training and test datasets.

    Task
    ----
    B. Repeat part 1.B but return an imbalanced dataset consisting of 99% of all 9s
        removed.  Also convert the 7s to 0s and 9s to 1s.

    """
    global x_train, y_train, x_test, y_test  # noqa: PLW0603

    if x_train_ is not None:
        x_train = x_train_
    if y_train_ is not None:
        y_train = y_train_
    if x_test_ is not None:
        x_test = x_test_
    if y_test_ is not None:
        y_test = y_test_

    # ==========================================
    # DO NOT CHANGE THE FUNCTION ABOVE THIS LINE
    # ==========================================

    # Full dataset
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)

    print(f"{x.shape=}, {y.shape=}")
    # Only keep 7s and 9s
    seven_nine_idx = (y == 7) | (y == 9)  # empty
    print(f"{seven_nine_idx.shape=}")
    x = x[seven_nine_idx, :]
    y = y[seven_nine_idx]
    print(f"{x.shape=}")
    print(f"{y.shape=}")

    # Remove 90% of 9s. Convert 7s to 0s and 9s to 1s.
    frac_to_remove = 0.90
    x, y = remove_nines_convert_to_01(
        x,
        y,
        frac=frac_to_remove,
    )

    # Compute number of 0s and 1s in y
    num_0s = np.sum(y == 0)
    num_1s = np.sum(y == 1)
    print(f"{num_0s=}, {num_1s=}")
    print(f"{x.shape=}, {y.shape=}")

    num_train = int(0.8 * len(x))
    num_test = len(x) - num_train
    x_train = x[:num_train, :]
    y_train = y[:num_train]
    x_test = x[num_train : num_train + num_test, :]
    y_test = y[num_train : num_train + num_test]

    # Check the number of 0s and 1s in the training and test sets
    num_0s_train = np.sum(y_train == 0)
    num_1s_train = np.sum(y_train == 1)
    num_0s_test = np.sum(y_test == 0)
    num_1s_test = np.sum(y_test == 1)
    print(f"{num_0s_train=}, {num_1s_train=}, {num_0s_train/num_1s_train=}")
    print(f"{num_0s_test=}, {num_1s_test=}, {num_0s_test/num_1s_test=}")
    print(f"{y.shape=}, {y_train.shape=}, {y_test.shape=}")

    # Both the test set and the training set are imbalanced.

    answers = {}
    return answers


# --------------------------------------------------------------------------


def part_3c(
    x_train_: NDArray[np.floating] | None = None,
    y_train_: NDArray[np.int32] | None = None,
    x_test_: NDArray[np.floating] | None = None,
    y_test_: NDArray[np.int32] | None = None,
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
    x_train_ : NDArray[np.floating]
        The feature matrix for the training dataset.
    y_train_ : NDArray[np.int32]
        The labels for the training dataset.
    x_test_ : NDArray[np.floating]
        The feature matrix for the test dataset.
    y_test_ : NDArray[np.int32]
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

    Task
    ----
    C. Repeat part 1.C for this dataset with a support vector machine (SVC in sklearn).
        Make sure to use a stratified cross-validation strategy. In addition to regular
        accuracy also print out the mean/std of the F1 score, precision, and recall.
        Is precision or recall higher? Explain. Finally, train the classifier on all
        the training data and plot the confusion matrix.

    """
    global x_train, y_train, x_test, y_test  # noqa: PLW0603

    if x_train_ is not None:
        x_train = x_train_
    if y_train_ is not None:
        y_train = y_train_
    if x_test_ is not None:
        x_test = x_test_
    if y_test_ is not None:
        y_test = y_test_

    # ==========================================
    # DO NOT CHANGE THE FUNCTION ABOVE THIS LINE
    # ==========================================
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    weight_dict = dict(
        zip(np.unique(y_train), class_weights, strict=True),
    )
    print(f"{class_weights=}")
    print(f"{weight_dict=}")
    n_splits = 5
    clf = SVC(random_state=seed)
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_validate(
        clf,
        x_train,
        y_train,
        cv=cv,
        scoring=["accuracy", "recall", "precision", "f1"],
        return_train_score=True,  # What does this do?
    )

    mean_accuracy = scores["test_accuracy"].mean()
    std_accuracy = scores["test_accuracy"].std()
    mean_precision = scores["test_precision"].mean()
    std_precision = scores["test_precision"].std()
    mean_recall = scores["test_recall"].mean()
    std_recall = scores["test_recall"].std()
    mean_f1 = scores["test_f1"].mean()
    std_f1 = scores["test_f1"].std()

    print("scores= ", scores)

    # Train on all the data
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    clf.fit(x, y)
    y_pred = clf.predict(x)
    conf_mat = confusion_matrix(y, y_pred)
    print(f"{conf_mat=}")

    answers = {}
    answers["cv"] = cv
    answers["clf"] = clf
    answers["mean_F1"] = mean_f1
    answers["mean_recall"] = mean_recall
    answers["mean_accuracy"] = mean_accuracy
    answers["mean_precision"] = mean_precision
    answers["std_F1"] = std_f1
    answers["std_recall"] = std_recall
    answers["std_accuracy"] = std_accuracy
    answers["std_precision"] = std_precision

    # Type: bool
    answers["is_precision_higher_than_recall"] = mean_precision > mean_recall
    answers["is_precision_higher_than_recall_explain"] = None  # String

    # Type: 2x2 NDArray (np.array)
    answers["confusion_matrix"] = conf_mat  # 2 x 2 matrix
    answers["weight_dict"] = weight_dict

    ## For testing, I can check the arguments of functions
    return answers


# --------------------------------------------------------------------------


def part_3d(
    x_train_: NDArray[np.floating] | None = None,
    y_train_: NDArray[np.int32] | None = None,
    x_test_: NDArray[np.floating] | None = None,
    y_test_: NDArray[np.int32] | None = None,
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

    Task
    ----
    D. Repeat the same steps as part 3.C but apply a weighted loss function (see the
        class_weights parameter).  Print out the class weights, and comment on the
        performance difference. Use compute_class_weight to compute the class weights.

    """
    global x_train, y_train, x_test, y_test  # noqa: PLW0603

    if x_train_ is not None:
        x_train = x_train_
    if y_train_ is not None:
        y_train = y_train_
    if x_test_ is not None:
        x_test = x_test_
    if y_test_ is not None:
        y_test = y_test_

    # ==========================================
    # DO NOT CHANGE THE FUNCTION ABOVE THIS LINE
    # ==========================================
    class_weights = compute_class_weight(
        class_weight="balanced",
        classes=np.unique(y_train),
        y=y_train,
    )
    weight_dict = dict(
        zip(np.unique(y_train), class_weights, strict=True),
    )
    print(f"{class_weights=}")
    print(f"{weight_dict=}")
    n_splits = 5
    clf = SVC(random_state=seed, class_weight=weight_dict)
    # Shuffling is fine because of seed
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)

    scores = cross_validate(
        clf,
        x_train,
        y_train,
        cv=cv,
        scoring=["accuracy", "recall", "precision", "f1"],
        return_train_score=True,  # What does this do?
    )

    print("class_weights = ", class_weights)  # Dimension 2
    print("weight_dict = ", weight_dict)  # Dimension 2

    # Train on all the data
    x = np.concatenate((x_train, x_test), axis=0)
    y = np.concatenate((y_train, y_test), axis=0)
    clf.fit(x, y)

    y_pred = clf.predict(x)
    conf_mat = confusion_matrix(y, y_pred)
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

    # Type: bool
    answers["is_precision_higher_than_recall"] = None  # True/False
    answers["is_precision_higher_than_recall_explain"] = None  # String

    # Type: 2x2 NDArray (np.array)
    answers["conf_mat"] = conf_mat
    answers["weight_dict"] = weight_dict
    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    """
    Run your code and produce all your results for your report. We will spot check the
    reports, and grade your code with automatic tools.
    """

    ################################################
    # In real code, read MNIST files and define Xtrain and xtest appropriately
    rng = np.random.default_rng(seed)
    n_images = 1200
    n_features = 784
    x = rng.random((n_images, n_features))  # 100 samples, 100 features
    # Fill labels with 0 and 1 (mimic 7 and 9s)
    y = (x[:, :5].sum(axis=1) > 2.5).astype(int)
    n_train = 100
    x_train = x[0:n_train, :]
    x_test = x[n_train:, :]
    y_train = y[0:n_train]
    y_test = y[n_train:]

    (
        x_train,
        y_train,
        x_test,
        y_test,
    ) = u.create_data(
        n_rows=1200,
        n_features=784,
        frac_train=0.8,
    )

    # Replace 0s with 7s and 1s with 9s
    y_train[y_train == 0] = 7
    y_train[y_train == 1] = 9
    y_test[y_test == 0] = 7
    y_test[y_test == 1] = 9

    x = x_train
    y = y_train

    x_train, y_train, x_test, y_test = u.prepare_data()

    print("\nbefore part_3a")
    print(f"{x_train.shape=}, {y_train.shape=}")
    print(f"{x_test.shape=}, {y_test.shape=}")
    print(f"{x.shape=}, {y.shape=}")

    all_answers = {}
    all_answers["part_3a"] = part_3a()
    print()
    print(f"after part_3a, {x_train.shape=}, {y_train.shape=}")
    print(f"after part_3a, {x_test.shape=}, {y_test.shape=}")
    print(f"after part_3a, {x.shape=}, {y.shape=}")

    # The data is the full MNIST dataset
    (
        x_train,
        y_train,
        x_test,
        y_test,
    ) = u.prepare_data()

    print("before part_3b")
    print(f"before part_3b, {x_train.shape=}, {y_train.shape=}")
    print(f"before part_3b, {x_test.shape=}, {y_test.shape=}")
    print("============================================")
    all_answers["part_3b"] = part_3b()
    print()
    print(f"after part_3b, {x_train.shape=}, {y_train.shape=}")
    print(f"after part_3b, {x_test.shape=}, {y_test.shape=}")

    print("============================================")
    all_answers["part_3c"] = part_3c()
    print()
    print(f"after part_3c, {x_train.shape=}, {y_train.shape=}")
    print(f"after part_3c, {x_test.shape=}, {y_test.shape=}")
    print(f"after part_3c, {x.shape=}, {y.shape=}")

    print("============================================")
    all_answers["part_3d"] = part_3d()
    print()
    print(f"after part_3d, {x_train.shape=}, {y_train.shape=}")
    print(f"after part_3d, {x_test.shape=}, {y_test.shape=}")
    print(f"after part_3d, {x.shape=}, {y.shape=}")

    u.save_dict("section3.pkl", dct=all_answers)
