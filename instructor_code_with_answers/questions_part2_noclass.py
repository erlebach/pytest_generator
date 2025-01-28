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

import pickle
import time
from collections import defaultdict
from pprint import pprint
from typing import Any

import new_utils as nu
import numpy as np
import utils as u

# ! from numpy.linalg import norm
from numpy.typing import NDArray
from questions_part1_noclass import part_1c as section1_part_c
from questions_part1_noclass import part_1d as section1_part_d
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix, top_k_accuracy_score

# For code reuse. Ideally functions used in multiple classes should be put in
# a utils file
# ! from part_1_template_solution import Section1 as Part1
from sklearn.model_selection import (
    GridSearchCV,
    # GridSearchCV,
    # KFold,
    ShuffleSplit,
    cross_validate,
)

# import svm module
# ! from sklearn.svm import SVC  # , LinearSVC
# ! from sklearn.tree import DecisionTreeClassifier
from utils import Normalization

# These definitions guarantee that the variables are global and exist even if
# the code is not run interactively.
x_train = None
y_train = None
x_test = None
y_test = None
ntrain = None
ntest = None
ntrain_list = None

# ======================================================================

# I could make Section 2 a subclass of Section 1, which would facilitate code reuse.
# However, both classes have the same function names. Better to pass Section 1 instance
# as an argument to Section 2 class constructor.

normalize = Normalization.APPLY_NORMALIZATION
seed = 42
frac_train = 0.8


# ----------------------------------------------------------------
def part_b_sub(
    x_train: NDArray[np.floating],
    y_train: NDArray[np.int32],
    x_test: NDArray[np.floating],
    y_test: NDArray[np.int32],
) -> dict[Any, Any]:
    """Perform cross-validation and evaluation using logistic regression.

    This function repeats parts C, D, F, and G from the previous analysis on
    the given data.  On part F, perform cross-validation using logistic regression
    and evaluates performance on both training and test sets.

    Parameters
    ----------
    x_train : NDArray[np.floating]
        Training feature matrix
    y_train : NDArray[np.int32]
        Training labels
    x_test : NDArray[np.floating]
        Test feature matrix
    y_test : NDArray[np.int32]
        Test labels

    Returns
    -------
    dict[Any, Any]
        Dictionary containing:
        - Results from part C (cross-validation scores and classifier)
        - Results from part D (cross-validation scores and classifier)
        - Results from part F (training/test scores and confusion matrices)
        - Results from part G (training/test scores and confusion matrices)

    """
    x = x_train
    y = y_train

    previous_answers = {}

    print("===> repeat part C")
    print(f"{x.shape=}, {y.shape=}, {x_test.shape=}, {y_test.shape=}")
    print("globals: ", list(globals().keys()))

    # Repeat part_c
    return_part_c = section1_part_c(x, y, x_test, y_test)

    #  list(return_part_c.keys())=['clf', 'cv', 'scores']
    print(f"\n\n{list(return_part_c.keys())=}")
    answer_part_c = {
        "scores_c": return_part_c["scores"],
        "clf_c": return_part_c["clf"],
        "cv_c": return_part_c["cv"],
    }

    previous_answers["part_c"] = answer_part_c
    # ------
    print("===> repeat part D")
    # Repeat part_1d
    return_part_d = section1_part_d(x, y, x_test, y_test)
    print(f"\n\n{list(return_part_d.keys())=}")
    answer_part_d = {
        "scores_d": return_part_d["scores"],
        "clf_d": return_part_d["clf"],
        "cv_d": return_part_d["cv"],
    }
    previous_answers["part_d"] = answer_part_d

    # ------
    print("===> repeat part F")
    # Repeat part_1f
    # Use logistic regressor with default arguments.
    # Make sure you set the random state argument.
    # ! return_part_f = section_1_part_f()  # self.part1.partF(X, y)

    cv = ShuffleSplit(n_splits=5, random_state=seed)
    # Was RandomForestClassifier in part_1f, now is LogisticRegression as requested.
    clf = LogisticRegression(
        random_state=seed,
        # ! multi_class="multinomial",  # Deprecated.
        # ! Auto-selected since version 1.5 of Scikit-Learn
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
    # ! print(f"part_b_sub: {x.shape=}, {y.shape=}, {x_test.shape=}, {y_test.shape=}")
    print("part_b_sub, partF: ", x.shape, y.shape, x_test.shape, y_test.shape)
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
        "accuracy_train_F": scores_train_f,  # float, accuracy
        "accuracy_test_F": scores_test_f,
        "mean_cv_accuracy_F": mean_cv_accuracy_f,
        "clf": clf,
        "cv": cv,
        "conf_mat_train": conf_mat_train,
        "conf_mat_test": conf_mat_test,
    }

    previous_answers["part_f"] = answer_part_f

    # ----------------------------------------------------------------
    # Repeat part G, part_1g, with a LogisticRegressor
    clf = RandomForestClassifier(random_state=seed)
    clf.fit(x_train, y_train)

    print("**** part_b_sub, part G: ")
    print("**** x_train.shape: ", x_train.shape)
    print("**** x_test.shape: ", x_test.shape)

    # Look at documentation (parameters used above in LogisticRegression)
    default_parameters = {
        "fit_intercept": True,
        "C": 1.0,
        "penalty": ["l2"],
    }

    # 12 (=2*3*2) runs: 2 criteria, 3 estimators, 2 depths
    grid_parameters: dict[str, Any] = {
        "fit_intercept": [True, False],
        "C": [0.5, 1.0, 1.5],
        "penalty": ["elasticnet", "l1", "l2"],
    }

    # LogisticRegression uses 'lbfgs' solver by default which stops when:
    # 1. max_iter iterations are reached, or
    # 2. norm of gradient is below tol (default 1e-4), indicating convergence
    clf = LogisticRegression(random_state=seed, max_iter=500, tol=1e-3)
    n_splits = 5
    # Uses stratified cross-validator by default
    # ! cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # Perform Grid search
    grid_search = GridSearchCV(
        clf,
        param_grid=grid_parameters,
        refit=True,  # ?
        cv=5,
        return_train_score=True,
    )
    # Performs grid search with cv, then fits the training data
    grid_search.fit(x, y)
    # `best_estimator` is availalble because refit=True
    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    print(f"**** part_1g, {y_train.shape=}, {x_train.shape=}")
    # quit()
    clf.fit(x_train, y_train)
    best_estimator.fit(x_train, y_train)

    # predictions using the best estimator
    y_train_pred_best = best_estimator.predict(x_train)
    # original predictions
    y_train_pred_orig = clf.predict(x_train)

    # Store the best estimator in a pkl file
    # Alternatively, create a function in utils.py to save and retrieve the
    #   best estimator. This can be done with a class that can save and retreive
    #   objects. But this will not work across invocations of different code.
    # Use grid_search.best_estimator_ for best results when applying the model
    with open("part_2b_grid_search.pkl", "wb") as f:
        pickle.dump(grid_search, f)

    print("part_2b, x_test.shape: ", x_test.shape)
    print("part_2b, type(x_test): ", type(x_test))
    # original predictions
    y_test_pred_orig = clf.predict(x_test)
    # predictions using the best estimator
    y_test_pred_best = best_estimator.predict(x_test)

    # The `cm` prefix refers to `confusion_matrix`
    print("part_2b, y_train_pred_orig.shape: ", y_train_pred_orig.shape)  # 100
    print("part_2b, y_train_pred_best.shape: ", y_train_pred_best.shape)  # 100
    print("part_2b, y_train.shape: ", y_train.shape)  # 96
    print("part_2b, y_test_pred_orig.shape: ", y_test_pred_orig.shape)  # 24
    print("part_2b, y_test_pred_best.shape: ", y_test_pred_best.shape)  # 24
    print("part_2b, y_test.shape: ", y_test.shape)  #  24
    cm_train_pred_orig = confusion_matrix(y_train, y_train_pred_orig)
    cm_train_pred_best = confusion_matrix(y_train, y_train_pred_best)
    cm_test_pred_orig = confusion_matrix(y_test, y_test_pred_orig)
    cm_test_pred_best = confusion_matrix(y_test, y_test_pred_best)

    def accuracy(cm: NDArray[np.int32]) -> float:
        return (cm[0, 0] + cm[1, 1]) / cm.sum()

    def precision(cm: NDArray[np.int32]) -> float:
        return cm[0, 0] / (cm[0, 0] + cm[1, 0])

    # Compute the accuracy from the confusion matrix
    accuracy_train_pred_orig = accuracy(cm_train_pred_orig)
    accuracy_train_pred_best = accuracy(cm_train_pred_best)
    accuracy_test_pred_orig = accuracy(cm_test_pred_orig)
    accuracy_test_pred_best = accuracy(cm_test_pred_best)

    # Compute the precision from the confusion matrix
    precision_train_pred_orig = precision(cm_train_pred_orig)
    precision_train_pred_best = precision(cm_train_pred_best)
    precision_test_pred_orig = precision(cm_test_pred_orig)
    precision_test_pred_best = precision(cm_test_pred_best)

    # Confusion matrix is improved with the best estimator
    print("confusion matrix on original test predictor")
    print(cm_test_pred_orig)
    print()
    print("confusion matrix on best test predictor")
    print(cm_test_pred_best)
    print()
    print("confusion matrix on original train predictor")
    print(cm_train_pred_orig)
    print()
    print("confusion matrix on best train predictor")
    print(cm_train_pred_best)
    print()

    answers_part_g: dict[str, Any] = {}

    # The type is an instance of RandomForestClassifier
    answers_part_g["clf"] = clf

    # The type is an instance of RandomForestClassifier
    answers_part_g["best_estimator"] = best_estimator

    # The type is an instance of GridSearchCV
    answers_part_g["grid_search"] = grid_search

    # The type is a dict[str, Any]
    answers_part_g["default_parameters"] = default_parameters

    # The type is a float
    # answers_part_g["mean_accuracy_cv"] = None  # ! FIX!

    # The answer type is a numpy.ndarray
    # Return the 2x2 confusion matrix computed from the predictions
    answers_part_g["confusion_matrix_train_orig"] = cm_train_pred_orig
    answers_part_g["confusion_matrix_train_best"] = cm_train_pred_best
    answers_part_g["confusion_matrix_test_orig"] = cm_test_pred_orig
    answers_part_g["confusion_matrix_test_best"] = cm_test_pred_best

    # compute: C11 + C22 / |C|_1  (accuracy based on confusion)
    # The answer type is a float
    # Return the accuracy computed from the confusion matrix
    # Full training refers to the full training set (as opposed to the training
    #     subset used in Cross-validation)
    answers_part_g["accuracy_orig_full_training"] = accuracy_train_pred_orig
    answers_part_g["accuracy_best_full_training"] = accuracy_train_pred_best
    answers_part_g["accuracy_orig_full_testing"] = accuracy_test_pred_orig
    answers_part_g["accuracy_best_full_testing"] = accuracy_test_pred_best

    # The answer type is a float
    # Return the precision computed from the confusion matrix
    answers_part_g["precision_orig_full_training"] = precision_train_pred_orig
    answers_part_g["precision_best_full_training"] = precision_train_pred_best
    answers_part_g["precision_orig_full_testing"] = precision_test_pred_orig
    answers_part_g["precision_best_full_testing"] = precision_test_pred_best

    previous_answers["part_g"] = answers_part_g

    # End Part 1G with LogisticRegressor
    # ----------------------------------------------------------------

    # Save the model for use in part 3 into a pkl file
    # part3 will check that the model is the correct one
    with open("part2f_model.pkl", "wb") as f:
        pickle.dump(clf, f)

    # -----------------------------------------------
    previous_answers["ntrain"] = len(x_train)
    previous_answers["ntest"] = len(x_test)
    previous_answers["class_count_train"] = np.unique(y_train, return_counts=True)[1]
    previous_answers["class_count_test"] = np.unique(y_test, return_counts=True)[1]
    return previous_answers


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


def part_2a() -> dict[str, Any]:
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
    global x_train, y_train, x_test, y_test  # noqa: PLW0603

    # ==========================================
    # DO NOT CHANGE THE FUNCTION ABOVE THIS LINE
    # ==========================================

    (
        x_train,
        y_train,
        x_test,
        y_test,
    ) = u.prepare_data()
    x_train = nu.scale_data(x_train)
    x_test = nu.scale_data(x_test)
    print(f"{x_train.shape=}, {y_train.shape=}, {x_test.shape=}, {y_test.shape=}")

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
    nb_classes_dict: dict[str, int] = {}
    nb_classes_dict["nb_classes_train"] = len(np.unique(y_train))
    nb_classes_dict["nb_classes_test"] = len(np.unique(y_test))
    answers["nb_classes"] = nb_classes_dict

    classes_train, counts_train = np.unique(y_train, return_counts=True)
    classes_test, counts_test = np.unique(y_test, return_counts=True)
    class_count_dict: dict[str, int] = {}
    class_count_dict["class_count_train"] = counts_train
    class_count_dict["class_count_test"] = counts_test
    answers["class_count"] = class_count_dict

    class_lengths_dict: dict[str, int] = {}
    class_lengths_dict["nb_samples_x_train"] = len(x_train)
    class_lengths_dict["nb_samples_x_test"] = len(x_test)
    class_lengths_dict["nb_samples_y_train"] = len(y_train)
    class_lengths_dict["nb_samples_y_test"] = len(y_test)
    answers["nb_samples_data"] = class_lengths_dict

    class_max_dict: dict[str, int] = {}
    class_max_dict["max_x_train"] = np.max(x_train)
    class_max_dict["max_x_test"] = np.max(x_test)
    answers["max_data"] = class_max_dict

    print("\n==>about to exit part_2a")
    pprint(answers)
    # answers is a dictionary of dictionaries
    return answers


def part_2b(
    x_train_: NDArray[np.floating] | None = None,
    y_train_: NDArray[np.int32] | None = None,
    x_test_: NDArray[np.floating] | None = None,
    y_test_: NDArray[np.int32] | None = None,
    ntrain_list_: list[int] | None = None,
) -> dict[Any, Any]:
    """Perform multiple experiments on the dataset using logistic regression.

    This method executes the following steps:
    1. Prepares the training and testing datasets based on the provided sizes.
    2. Repeats parts C, D, and F of the previous analysis for each subset of the data.
    3. Uses logistic regression to train and evaluate the model (part F only).
    4. Collects and returns the results, including training and testing scores,
        confusion matrices, and class counts.

    Parameters
    ----------
    x_train_ : NDArray[np.floating]
        The feature matrix for training.
    y_train_ : NDArray[np.int32]
        The labels for training.
    x_test_ : NDArray[np.floating]
        The feature matrix for testing.
    y_test_ : NDArray[np.int32]
        The labels for testing.
    ntrain_list_ : list[int], optional
        A list of training sizes to evaluate, by default an empty list.

    Returns
    -------
    dict
        A dictionary containing results for each training size, including
            scores and confusion matrices.

    Task
    ----
    B. Repeat part 1.C, 1.D, and 1.F, for the multiclass problem.
    Use the Logistic Regression for part F with 500 iterations.
    Explain how multi-class logistic regression works (inherent,  # ! TODO
    one-vs-one, one-vs-the-rest, etc.).
    Use the entire MNIST training set to train the model.
    Comment on the results. Is the accuracy higher for the training or testing set?

    Notes
    -----
    Use try/except clauses to handle any errors and to prevent the code from failing.
    For example:

    def sumb(a: float, b: float) -> float:
        try:
            ret = a + b / 0.0
        except ZeroDivisionError:
            print("ZeroDivisionError")
            return 0.
        return ret

    `sumb(3., 4.)` will return 0. because of the exception, but will not fail.
    Therefore, you can submit the code to Gradescope without Gradescope failing.

    For the final classifier you trained in 2.B (partF),
    plot a confusion matrix for the test predictions.
    Earlier we stated that 7s and 9s were a challenging pair to
    distinguish. Do your results support this statement? Why or why not?

    """
    global ntrain_list, x_train, y_train, x_test, y_test  # noqa: PLW0603

    if ntrain_list_ is not None:
        ntrain_list = ntrain_list_
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

    ntrain = len(x_train)
    ntest = len(x_test)

    # Create a test sets that are 20% of each training set
    print("**** part_2b, ntrain_list: ", ntrain_list)

    answers = {}

    print("**** calling part_b_sub")
    print("*** Shapes should be 60000, 784")
    print(f"*** {x_train.shape=}")
    print(f"*** {y_train.shape=}")
    print(f"*** {x_test.shape=}")
    print(f"*** {y_test.shape=}")

    # For each training size, we will create an optimization using GridSearchCV

    ntr = ntrain
    nte = ntest
    answer_dict = {}
    print(f"==> {ntr=}, {nte=}")
    x_r = x_train[0:ntr, :]
    y_r = y_train[0:ntr]
    x_test_r = x_test[0:nte, :]
    y_test_r = y_test[0:nte]
    print(f"{ntr=}, {nte=}")
    print(f"**** {x_r.shape=}, {y_r.shape=}, {x_test_r.shape=}, {y_test_r.shape=}")
    start = time.time()
    previous_answers = part_b_sub(x_r, y_r, x_test_r, y_test_r)
    print(f"TIME(part_b_sub), {ntr=}, {nte=}, {time.time() - start=} sec")

    # Looking at the confusion matrix of the test set, identify the top five pairs of
    # numbers that are most difficult to distinguish. For example, the pair (3,4)
    # error rate c # can be estimated as (C[4, 5] + C[5,4]) / 2 (recall that the count
    # starts at 0).

    # Answer type: list[tuple(int)].  For example: [(3,6), (5,6), (6,7)]
    # The smaller digit is always listed first. Thus: (6,3) will be rejected.
    # There should be exactly three pairs of digits. I expect the pairs to be the
    # result of a calculation based on the confusion matrix. Otherwise, the grading
    # will be incorrect. On do the calculation for N=20000.

    answers: dict[str, Any] = {}

    # answers["part_c"] = previous_answers["part_c"]
    # answers["part_f"] = previous_answers["part_f"]
    # answers["part_g"] = previous_answers["part_g"]

    print(f"\n\n==> KEYS: {list(previous_answers['part_c'].keys())=}")
    answers["scores_1c"] = previous_answers["part_c"]["scores_c"]
    answers["clf_1c"] = previous_answers["part_c"]["clf_c"]
    answers["cv_1c"] = previous_answers["part_c"]["cv_c"]

    answers["scores_1d"] = previous_answers["part_d"]["scores_d"]
    answers["clf_1d"] = previous_answers["part_d"]["clf_d"]
    answers["cv_1d"] = previous_answers["part_d"]["cv_d"]

    answers["accuracy_train_1f"] = previous_answers["part_f"]["accuracy_train_F"]
    answers["accuracy_test_1f"] = previous_answers["part_f"]["accuracy_test_F"]
    answers["mean_cv_accuracy_1f"] = previous_answers["part_f"]["mean_cv_accuracy_F"]
    answers["clf_1f"] = previous_answers["part_f"]["clf"]
    answers["cv_1f"] = previous_answers["part_f"]["cv"]
    confusion_1f: dict[str, NDArray] = {}
    confusion_1f["conf_mat_train_1f"] = previous_answers["part_f"]["conf_mat_train"]
    confusion_1f["conf_mat_test_1f"] = previous_answers["part_f"]["conf_mat_test"]
    answers["confusion_matrix_1f"] = confusion_1f

    pag = previous_answers["part_g"]
    confusion_matrix = {}
    confusion_matrix["confusion_matrix_train_orig"] = pag["confusion_matrix_train_orig"]
    confusion_matrix["confusion_matrix_train_best"] = pag["confusion_matrix_train_best"]
    confusion_matrix["confusion_matrix_test_orig"] = pag["confusion_matrix_test_orig"]
    confusion_matrix["confusion_matrix_test_best"] = pag["confusion_matrix_test_best"]

    accuracy = {}
    accuracy["accuracy_orig_full_training"] = pag["accuracy_orig_full_training"]
    accuracy["accuracy_best_full_training"] = pag["accuracy_best_full_training"]
    accuracy["accuracy_orig_full_testing"] = pag["accuracy_orig_full_testing"]
    accuracy["accuracy_best_full_testing"] = pag["accuracy_orig_full_testing"]

    precision = {}
    precision["precision_orig_full_training"] = pag["precision_orig_full_training"]
    precision["precision_best_full_training"] = pag["precision_best_full_training"]
    precision["precision_orig_full_testing"] = pag["precision_orig_full_testing"]
    precision["precision_best_full_testing"] = pag["precision_orig_full_testing"]

    answers["clf_1g"] = pag["clf"]
    answers["best_estimator_1g"] = pag["best_estimator"]
    answers["grid_search_1g"] = pag["grid_search"]
    answers["default_parameters_1g"] = pag["default_parameters"]

    answers["confusion_matrix_1g"] = confusion_matrix
    answers["accuracy_1g"] = accuracy
    answers["precision_1g"] = precision

    class_count: dict[str, NDArray[np.int32]] = {}
    class_count["class_count_train"] = previous_answers["class_count_train"]
    class_count["class_count_test"] = previous_answers["class_count_test"]
    answers["class_count_1g"] = class_count

    answers["hard_to_distinguish_pairs"] = [(3, 5), (7, 9), (4, 9), (5, 8), (3, 8)]

    print("\n\nAnswers part_2b")
    pprint(answers)
    print("\n\n")

    print("\n==>about to exit part_2b")
    pprint(answers)  # noqa: T203
    return answers


# ----------------------------------------------------------------
'''
def part_2c(
    x_train_: NDArray[np.floating] | None = None,
    y_train_: NDArray[np.int32] | None = None,
    x_test_: NDArray[np.floating] | None = None,
    y_test_: NDArray[np.int32] | None = None,
) -> dict[str, Any]:
    """Train a decision tree classifier using k-fold cross validation.

    This function trains a decision tree classifier using k-fold cross validation
    and returns the mean and standard deviation of the accuracy scores in each
    validation set.

    Parameters
    ----------
    x_train_ : NDArray[np.floating] | None, optional
        Training feature matrix, by default None
    y_train_ : NDArray[np.int32] | None, optional
        Training labels, by default None
    x_test_ : NDArray[np.floating] | None, optional
        Test feature matrix, by default None
    y_test_ : NDArray[np.int32] | None, optional
        Test labels, by default None

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - Mean accuracy scores from cross validation
        - Standard deviation of accuracy scores
        - Trained classifier
        - Cross validation object used

    Task
    ----
    C. For the final classifier you trained in 2.B. Plot a confusion matrix for the
        test predictions. Earlier we stated that 7s and 9s were a challenging pair
        to distinguish. Do your results support this statement? Why or why not?

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

    answers: dict[str, Any] = {}

    # Load the model from the pkl file
    with open("part2f_model.pkl", "rb") as f:
        clf = pickle.load(f)

    y_train_pred = clf.predict(x_train)
    y_test_pred = clf.predict(x_test)
    cm_test = confusion_matrix(y_test, y_test_pred)
    cm_train = confusion_matrix(y_train, y_train_pred)

    print("\ncm_train")
    print(cm_train)
    print("\ncm_test")
    print(cm_test)

    answers["cm_train"] = cm_train
    answers["cm_test"] = cm_test

    # Compute top_k accuracy scores
    scores_train: dict[int, float] = {}
    scores_test: dict[int, float] = {}

    # Get probability predictions
    train_probs = clf.predict_proba(x_train)
    test_probs = clf.predict_proba(x_test)

    # All 10 classes, complete MNIST dataset
    # ! print("y_test: ", y_test.shape)

    for k in [1, 2, 3, 4, 5]:
        top_k_accuracy_train = top_k_accuracy_score(y_train, train_probs, k=k)
        top_k_accuracy_test = top_k_accuracy_score(y_test, test_probs, k=k)
        scores_train[k] = top_k_accuracy_train
        scores_test[k] = top_k_accuracy_test

    # Answer type: dict[int, float]
    answers["top_k_accuracy_train"] = scores_train
    answers["top_k_accuracy_test"] = scores_test
    print("\n==>about to exit part_2b")
    pprint(answers)  # noqa: T203

    # Compare the accuracies of the training and testing sets. Which is larger.
    # Key: "higher_accuracy", value: "training" or "testing"
    # What was your expectation?
    # Key: "expectation", value: "training" or "testing"
    answers["Which accuracies are larger?"] = None
    answers["expectation?"] = None

    # Please justify your expectation.
    # Key: "explain_expectation", value: str
    answers["explain_expectation"] = None

    return answers
'''

# ----------------------------------------------------------------
if __name__ == "__main__":
    """
    Run your code and produce all your results for your report. We will spot check the
    reports, and grade your code with automatic tools.
    """

    # ------------------------------------------------------------
    # # In real code, read MNIST files and define Xtrain and xtest appropriately
    # rng = np.random.default_rng(seed)
    # x = rng.random((120, 120))  # 100 samples, 100 features
    # # Fill labels with 0 and 1 (mimic 7 and 9s)
    # y: NDArray[np.int32] = (x[:, :5].sum(axis=1) > 2.5).astype(int)
    # n_train = 1000
    # x_train = x[0:n_train, :]
    # x_test = x[n_train:, :]
    # y_train = y[0:n_train]
    # y_test = y[n_train:]
    # ntrain_list = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    # ntrain_list = [1000, 5000]  # max is 60000

    # Restrict the size for faster training
    ntrain_list = [1000, 5000, 10000, 20000]  # max is 60000

    x_train, y_train, x_test, y_test = u.prepare_data()

    print("before part_a")
    print(f"{x_train.shape=}, {y_train.shape=}")
    print(f"{x_test.shape=}, {y_test.shape=}")
    # ------------------------------------------------------------

    all_answers = {}
    all_answers["part_2a"] = part_2a()

    print()
    print("before part_2b")
    print(f"{x_train.shape=}, {y_train.shape=}")
    print(f"{x_test.shape=}, {y_test.shape=}")
    all_answers["part_2b"] = part_2b()
    print("after part_2b")
    print(f"{x_train.shape=}, {y_train.shape=}")
    print(f"{x_test.shape=}, {y_test.shape=}")

    # print()
    # print("before part_2c")
    # print(f"{x_train.shape=}, {y_train.shape=}")
    # print(f"{x_test.shape=}, {y_test.shape=}")
    # all_answers["part_2c"] = part_2c()

    # print("after part_2c")
    # print(f"{x_train.shape=}, {y_train.shape=}")
    # print(f"{x_test.shape=}, {y_test.shape=}")

    u.save_dict("section2.pkl", dct=all_answers)
