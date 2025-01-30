"""Part 3 of the assignment."""
# Inspired by GPT4

# Information on type hints
# https://peps.python.org/pep-0585/

# GPT on testing functions, mock functions, testing number of calls, and argument values
# https://chat.openai.com/share/b3fd7739-b691-48f2-bb5e-0d170be4428c

from pprint import pprint
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
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.utils.class_weight import compute_class_weight
from utils import Normalization

# The following global definitions guarantee that the variables are
#   global and exist even if the code is not run interactively.

x_train = None
y_train = None
x_test = None
y_test = None
ntrain = None
ntest = None

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
    Use all 10 classes for this part of the problem.

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
    # le = LabelEncoder()
    # y_train_encoded = le.fit_transform(y_train)
    # y_test_encoded = le.fit_transform(y_test)

    # Prepare for plotting
    k_list = [1, 2, 3, 4, 5]
    test_accuracy = []
    train_accuracy = []

    class_order = best_estimator.classes_
    print(f"{class_order=}")

    # y_train, y_test have values 7 and 9

    topk_dict = {}
    for k in k_list:
        # print("==>, topk_dict, k= ", k)
        # print(f"{y_train.shape=}")
        # print(f"{train_probs.shape=}")
        # print(f"{y_train=}")
        # print(f"{y_train_encoded=}")
        # print(f"{train_probs=}")
        train_score = top_k_accuracy_score(
            y_train,
            train_probs,
            k=k,
        )
        print(f"{train_score=}")
        test_score = top_k_accuracy_score(
            y_test,
            test_probs,
            k=k,
        )
        print(f"{test_score=}")
        train_accuracy.append(train_score)
        test_accuracy.append(test_score)
        topk_dict[k] = [float(train_score), float(test_score)]
        print(f"{k=}, {train_score=}, {test_score=}")

    plt.plot(k_list, train_accuracy, label="Train Accuracy")
    plt.plot(k_list, test_accuracy, label="Test Accuracy")
    plt.xlabel("k")
    plt.ylabel("Accuracy")
    plt.title("Top-k Accuracy for k=1,2,3,4,5")
    plt.legend()
    plt.savefig("top_k_accuracy_part_3a.png")

    # Count the number of classes in the training and test sets
    count_train = np.unique(y_train, return_counts=True)
    count_test = np.unique(y_test, return_counts=True)
    print(f"{count_train=}")
    print(f"{count_test=}")

    # Check if the labels are integers
    is_int = nu.check_labels(y_train)

    # Analyze the class distribution
    dist_dict = analyze_class_distribution(y_train)

    # top-k accuracy score
    # ! y_pred is not defined
    # ! score = top_k_accuracy_score(y, y_pred)

    answers = {}

    # Answer type: dict[int, list[float, float]]
    # The two floats are the train and test accuracy for the top-k accuracy score
    # Remember: order of the elements in a list matters
    answers["top_k_accuracy"] = topk_dict

    print("==> part_3a, answers")
    pprint(answers)
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

    (other using 10 classes)?

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
    B. Repeat part 1.B but return an imbalanced dataset consisting of 90% of all 9s
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
    print("\n\n PART 3B")
    print(f"{x_train.shape=}, {y_train.shape=}, {x_test.shape=}, {y_test.shape=}")
    x_train, y_train, x_test, y_test = u.prepare_data()
    print("ABOUT to call u.prepare_and_filter_data")
    (
        x_train,
        y_train,
        x_test,
        y_test,
    ) = u.prepare_and_filter_data(
        x_train,
        y_train,
        x_test,
        y_test,
    )
    print("EXIT u.prepare_and_filter_data")

    # Check the number of 0s and 1s in the training and test sets
    num_0s_train = np.sum(y_train == 0)
    num_1s_train = np.sum(y_train == 1)
    num_0s_test = np.sum(y_test == 0)
    num_1s_test = np.sum(y_test == 1)

    answers: dict[str, Any] = {}

    dct1: dict[str, int] = {}
    dct1["length_x_train"] = len(x_train)
    dct1["length_x_test"] = len(x_test)
    dct1["length_y_train"] = len(y_train)
    dct1["length_y_test"] = len(y_test)
    answers["number_of_samples"] = dct1
    print("\n==> answers['number_of_samples']")
    pprint(dct1)

    dct2: dict[str, float] = {}
    dct2["max_x_train"] = float(np.max(x_train))
    dct2["max_x_test"] = float(np.max(x_test))
    answers["data_bounds"] = dct2

    print(f"{num_0s_train=}, {num_1s_train=}, {num_0s_train/num_1s_train=}")
    print(f"{num_0s_test=}, {num_1s_test=}, {num_0s_test/num_1s_test=}")
    print(f"{y.shape=}, {y_train.shape=}, {y_test.shape=}")

    dct3: dict[str, int] = {}
    dct3["num_0s_train"] = num_0s_train
    dct3["num_1s_train"] = num_1s_train
    dct3["num_0s_test"] = num_0s_test
    dct3["num_1s_test"] = num_1s_test
    answers["class_counts"] = dct3

    print("return from part_3b")

    # Both the test set and the training set are imbalanced.

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
    print("INSIDE partD")
    print(f"{x_train.shape=}, {y_train.shape=}, {x_test.shape=}, {y_test.shape=}")
    print(f"{y_test=}")

    (
        x_train,
        y_train,
        x_test,
        y_test,
    ) = u.prepare_and_filter_data(
        x_train,
        y_train,
        x_test,
        y_test,
    )
    print(f"part_3c: {x_train.shape=}, {y_train.shape=}, {x_test.shape=}, {y_test.shape=}")

    # ----
    print(f"{class_weights=}")
    print(f"{weight_dict=}")
    n_splits = 2  # with less splits, less chance of a NaN recall, orig: 5

    # =====================================================================

    # =====================================================================

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

    # Cross-validation scores
    mean_scores: dict[str, float] = {}
    mean_scores["mean_F1"] = mean_f1
    mean_scores["mean_recall"] = mean_recall
    mean_scores["mean_accuracy"] = mean_accuracy
    mean_scores["mean_precision"] = mean_precision
    answers["mean_metrics"] = mean_scores

    std_scores: dict[str, float] = {}
    std_scores["std_F1"] = std_f1
    std_scores["std_recall"] = std_recall
    std_scores["std_accuracy"] = std_accuracy
    std_scores["std_precision"] = std_precision
    answers["std_metrics"] = std_scores

    # Type: bool
    answers["is_precision_higher_than_recall"] = mean_precision > mean_recall
    answers["is_precision_higher_than_recall_explain"] = None  # String

    # Type: 2x2 NDArray (np.array)
    answers["confusion_matrix"] = conf_mat  # 2 x 2 matrix
    answers["weight_dict"] = weight_dict

    print(f"\n==> 3c: {mean_precision=}, {mean_recall=}")

    print("\n==>return from part_3c")
    pprint(answers)

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

    mean_accuracy = scores["test_accuracy"].mean()
    std_accuracy = scores["test_accuracy"].std()
    mean_precision = scores["test_precision"].mean()
    std_precision = scores["test_precision"].std()
    mean_recall = scores["test_recall"].mean()
    std_recall = scores["test_recall"].std()
    mean_f1 = scores["test_f1"].mean()
    std_f1 = scores["test_f1"].std()

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

    # Cross-validation scores
    mean_scores: dict[str, float] = {}
    mean_scores["mean_F1"] = mean_f1
    mean_scores["mean_recall"] = mean_recall
    mean_scores["mean_accuracy"] = mean_accuracy
    mean_scores["mean_precision"] = mean_precision
    answers["mean_metrics"] = mean_scores

    std_scores: dict[str, float] = {}
    std_scores["std_F1"] = std_f1
    std_scores["std_recall"] = std_recall
    std_scores["std_accuracy"] = std_accuracy
    std_scores["std_precision"] = std_precision
    answers["std_metrics"] = std_scores

    # Type: bool
    print(f"==> {mean_precision=}, {mean_recall=}")
    answers["is_precision_higher_than_recall"] = mean_precision > mean_recall
    answers["is_precision_higher_than_recall_explain"] = None  # String

    # Type: 2x2 NDArray (np.array)
    answers["confusion_matrix"] = conf_mat  # 2 x 2 matrix
    answers["weight_dict"] = weight_dict

    print("\n==>return from part_3d")
    pprint(answers)

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

    # synthetic data with classes 0/1.
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
    quit()

    print("============================================")
    all_answers["part_3d"] = part_3d()
    print()
    print(f"after part_3d, {x_train.shape=}, {y_train.shape=}")
    print(f"after part_3d, {x_test.shape=}, {y_test.shape=}")
    print(f"after part_3d, {x.shape=}, {y.shape=}")

    u.save_dict("section3.pkl", dct=all_answers)
