"""Inspired by GPT4.

Information on type hints
https://peps.python.org/pep-0585/

GPT on testing functions, mock functions, testing number of calls, and argument values
https://chat.openai.com/share/b3fd7739-b691-48f2-bb5e-0d170be4428c

from sklearn.base import ClassifierMixin, RegressorMixin

# ==============================================================
Fill in the appropriate import statements from sklearn to solve the homework
from email.policy import default

IMPORTANT: do not communicate between functions in the class.
In other words: do not define intermediary variables using self.var = xxx
Doing so will make certain tests fail. Class methods should be independent
of each other and be able to execute in any order!

"""

from collections import defaultdict
from typing import Any

import new_utils as nu
import numpy as np
import utils as u

# ! from numpy.linalg import norm
from numpy.typing import NDArray

# For code reuse. Ideally functions used in multiple classes should be put in
# a utils file
# ! from part_1_template_solution import Section1 as Part1
from questions_part1_noclass import partC as section1_part_c
from questions_part1_noclass import partD as section1_part_d

# ! from questions_part1_noclass import partE as section1_part_e
# ! from questions_part1_noclass import partF as section1_part_f
# !  from sklearn.base import BaseEstimator
# ! from sklearn import datasets
# ! from sklearn.base import BaseEstimator
# ! from sklearn.ensemble import RandomForestClassifier
# import logistic regresssion module
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (
    # GridSearchCV,
    # KFold,
    ShuffleSplit,
    cross_validate,
)

# import svm module
# ! from sklearn.svm import SVC  # , LinearSVC
# ! from sklearn.tree import DecisionTreeClassifier
from utils import Normalization

# ======================================================================

# I could make Section 2 a subclass of Section 1, which would facilitate code reuse.
# However, both classes have the same function names. Better to pass Section 1 instance
# as an argument to Section 2 class constructor.

normalize = Normalization.APPLY_NORMALIZATION
seed = 42
frac_train = 0.8


# ---------------------------------------------------------

"""
C. Train your first classifier using k-fold cross validation (see
train_simple_classifier_with_cv function). Use 5 splits and a Decision tree
classifier. Print the mean and standard deviation for the accuracy scores in each
validation set in cross validation. Also print the mean and std of the fit
(or training) time.
"""

# ----------------------------------------------------------------


def print_cv_result_dict(cv_dict: dict, msg: str | None = None) -> None:
    """Print the mean and standard deviation of cross-validation results.

    Parameters
    ----------
    cv_dict : dict
        A dictionary where keys are metric names and values are arrays of scores.
    msg : str | None, optional
        An optional message to print before the results.

    If msg is provided, it will be printed as a header before the results.
    The function iterates through the cv_dict and prints the mean and standard
    deviation for each metric.

    """
    if msg is not None:
        print(f"\n{msg}")
    for key, array in cv_dict.items():
        print(f"mean_{key}: {array.mean()}, std_{key}: {array.std()}")


def partA() -> dict[str, Any]:
    """Prepare the dataset for training and testing.

    This method performs the following steps:
    1. Loads the training and testing data using the `prepare_data` function.
    2. Scales the training and testing data using the `scale_data` function.
    3. Counts the number of elements in each class for both training and
        testing datasets.
    4. Asserts that there are 10 classes in both datasets.
    5. Prints the number of classes in the training and testing datasets.
    6. Returns a dictionary containing the lengths and maximum values of the training
        and testing datasets, along with the scaled training and testing data.

    Returns
    -------
    tuple
        A tuple containing:
        - A dictionary with the lengths and maximum values of the training and testing
            datasets.
        - Scaled training data (Xtrain).
        - Scaled training labels (ytrain).
        - Scaled testing data (Xtest).
        - Scaled testing labels (ytest).

    A. Repeat part 1.B but make sure that your data matrix (and labels) consists of
    all classes by also printing out the number of elements in each class y and
    print out the number of classes for both training and testing datasets.

    """
    (
        x_train,
        y_train,
        x_test,
        y_test,
    ) = u.prepare_data()
    x_train = nu.scale_data(x_train)
    x_test = nu.scale_data(x_test)

    # Check that labels contain all classes
    # count number of elements of each class in train and test sets
    nb_each_class = defaultdict(int)
    for label in y_train:
        nb_each_class[label] += 1
    nb_classes_train = len(nb_each_class)

    for label in y_test:
        nb_each_class[label] += 1
    nb_classes_test = len(nb_each_class)

    if nb_classes_train != 10:
        print(f"Expected 10 classes in training set, got {nb_classes_train}")
        raise ValueError(nb_classes_train)

    if nb_classes_test != 10:
        print(f"Expected 10 classes in testing set, got {nb_classes_test}")
        raise ValueError(nb_classes_test)

    print(f"{nb_classes_train=}")
    print(f"{nb_classes_test=}")

    answers = {}
    answers["length_Xtrain"] = len(x_train)
    answers["length_Xtest"] = len(x_test)
    answers["length_ytrain"] = len(y_train)
    answers["length_ytest"] = len(y_test)
    answers["max_Xtrain"] = np.max(x_train)
    answers["max_Xtest"] = np.max(x_test)
    answers["Xtrain"] = x_train
    answers["ytrain"] = y_train
    answers["Xtest"] = x_test
    answers["ytest"] = y_test
    return answers


def part_b(
    x: NDArray[np.floating],
    y: NDArray[np.int32],
    x_test: NDArray[np.floating],
    y_test: NDArray[np.int32],
    ntrain_list: list[int] | None = None,
) -> dict[Any, Any]:
    """Perform multiple experiments on the dataset using logistic regression.

    This method executes the following steps:
    1. Prepares the training and testing datasets based on the provided sizes.
    2. Repeats parts C, D, and F of the previous analysis for each subset of the data.
    3. Uses logistic regression to train and evaluate the model.
    4. Collects and returns the results, including training and testing scores,
        confusion matrices, and class counts.

    Parameters
    ----------
    x : NDArray[np.floating]
        The feature matrix for training.
    y : NDArray[np.int32]
        The labels for training.
    x_test : NDArray[np.floating]
        The feature matrix for testing.
    y_test : NDArray[np.int32]
        The labels for testing.
    ntrain_list : list[int], optional
        A list of training sizes to evaluate, by default an empty list.

    Returns
    -------
    dict
        A dictionary containing results for each training size, including
            scores and confusion matrices.

    B. Repeat part 1.C, 1.D, and 1.F, for the multiclass problem.
    Use the Logistic Regression for part F with 300 iterations.
    Explain how multi-class logistic regression works (inherent,
    one-vs-one, one-vs-the-rest, etc.).  Repeat the experiment
    for N = 1000, 5000, 20000. Choose
            ntrain = 0.8 * N
            ntestn = 0.2 * N
    Comment on the results. Is the accuracy higher for the
    training or testing set?

    For the final classifier you trained in 2.B (partF),
    plot a confusion matrix for the test predictions.
    Earlier we stated that 7s and 9s were a challenging pair to
    distinguish. Do your results support this statement? Why or why not?

    """
    ntrain_list = [] if ntrain_list is None else ntrain_list
    ntest_list = [i // 4 for i in ntrain_list]
    print(f"{ntrain_list=}")
    print(f"{ntest_list=}")

    def part_b_sub(
        x: NDArray[np.floating],
        y: NDArray[np.int32],
        x_test: NDArray[np.float32],
        y_test: NDArray[np.int32],
    ) -> dict[Any, Any]:
        # ------
        print("===> repeat part C")
        print(f"{x.shape=}, {y.shape=}, {x_test.shape=}, {y_test.shape=}")
        # ! answer = section1_part_c(X, y)
        return_part_c = section1_part_c()
        answer_part_c = {
            "scores_C": return_part_c["scores"],
            "clf": return_part_c["clf"],
            "cv": return_part_c["cv"],
        }
        # ------
        print("===> repeat part D")
        # ! answer_d = self.part1.partD(X, y)
        return_part_d = section1_part_d()
        answer_part_d = {
            "scores_D": return_part_d["scores"],
            "clf": return_part_d["clf"],
            "cv": return_part_d["cv"],
        }

        # ------
        print("===> repeat part F")
        # Repeat part 1F
        # Use logistic regressor with default arguments.
        # Make sure you set the random state argument.
        # ! return_part_f = section_1_part_f()  # self.part1.partF(X, y)

        cv = ShuffleSplit(n_splits=5, random_state=seed)
        clf = LogisticRegression(
            random_state=seed,
            multi_class="multinomial",
            max_iter=500,
        )
        # Use logistic regressor
        scores = cross_validate(
            clf,
            x,
            y,
            cv=cv,
            return_train_score=True,
        )
        clf.fit(x, y)
        scores_train_f = clf.score(x, y)  # scalar
        scores_test_f = clf.score(x_test, y_test)  # scalar
        mean_cv_accuracy_f = scores["test_score"].mean()

        y_pred = clf.predict(x)
        ytest_pred = clf.predict(x_test)

        conf_mat_train = confusion_matrix(y_pred, y)
        conf_mat_test = confusion_matrix(ytest_pred, y_test)

        # Using entire dataset
        answer_part_f = {
            "scores_train_F": scores_train_f,
            "scores_test_F": scores_test_f,
            "mean_cv_accuracy_F": mean_cv_accuracy_f,
            "clf": clf,
            "cv": cv,
            "conf_mat_train": conf_mat_train,
            "conf_mat_test": conf_mat_test,
        }

        # -----------------------------------------------
        answers = {}
        answers["partC"] = answer_part_c
        answers["partD"] = answer_part_d
        answers["partF"] = answer_part_f
        answers["ntrain"] = len(y)
        answers["ntest"] = len(y_test)
        answers["class_count_train"] = np.unique(y, return_counts=True)[1]
        answers["class_count_test"] = np.unique(y_test, return_counts=True)[1]
        return answers
        # ----------------

    answer = {}

    for ntr, nte in zip(ntrain_list, ntest_list, strict=True):
        x_r = x[0:ntr, :]
        y_r = y[0:ntr]
        x_test_r = x_test[0:nte, :]
        y_test_r = y_test[0:nte]
        print(f"{ntr=}, {nte=}")
        print(f"{x_r.shape=}, {y_r.shape=}, {x_test_r.shape=}, {y_test_r.shape=}")
        answer[ntr] = part_b_sub(x_r, y_r, x_test_r, y_test_r)

    return answer
