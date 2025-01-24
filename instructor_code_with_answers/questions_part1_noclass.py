"""Questions for part1 of Assignment 1.

Class: Introduction to Data Mining
Spring 2025

Students should be able to run this file.
"""
# Inspired by GPT4

# Information on type hints
# https://peps.python.org/pep-0585/

# GPT on testing functions, mock functions, testing number of calls, and argument values
# https://chat.openai.com/share/b3fd7739-b691-48f2-bb5e-0d170be4428c

# Fill in the appropriate import statements from sklearn to solve the homework
from enum import Enum
from typing import Any

import new_utils as nu
import numpy as np
import utils as u
from numpy.typing import NDArray
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier

# import logistic regresssion module
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import (
    BaseCrossValidator,
    BaseShuffleSplit,
    GridSearchCV,
    KFold,
    ShuffleSplit,
    cross_validate,
)

# import svm module
# ! from sklearn.svm import SVC  # , LinearSVC
from sklearn.tree import DecisionTreeClassifier
from utils import Normalization, PrintResults

# ======================================================================
seed = 42
frac_train = 0.2


class Section1:
    def __init__(
        self,
        normalize: Normalization = Normalization.APPLY_NORMALIZATION,
        seed: int | None = None,
        frac_train: float = 0.2,
    ) -> None:
        """Initialize an instance of Section1.

        Parameters
        ----------
        normalize : bool, optional
            Whether to normalize the data, by default True
        seed : int or None, optional
            Random seed for reproducibility. If None, results will be randomized.
            If int, results will be reproducible, by default None
        frac_train : float, optional
            Fraction of data to use for training, by default 0.2

        Notes
        -----
        This class implements various machine learning experiments from Section 1
        of the assignment, including cross-validation and classifier comparisons.

        """
        self.normalize = normalize
        self.frac_train = frac_train
        self.seed = seed


# ---------------------------------------------------------


def train_simple_classifier_with_cv(
    # self,
    x: NDArray[np.floating],
    y: NDArray[np.int32],
    cv: BaseCrossValidator | BaseShuffleSplit,  # : BaseCrossValidator (class instance)
    # estimator_class: Type[BaseEstimator],  # a class
    clf: BaseEstimator,  #: class instance of the estimator
    n_splits: int = 5,
    print_results: PrintResults = PrintResults.SKIP_PRINT_RESULTS,
    seed: int = 42,
) -> dict[str, float]:
    """Train a simple classifier using k-fold cross-validation.

    Parameters
    ----------
    x : NDArray[np.floating]
        Features dataset
    y : NDArray[np.int32]
        Labels
    cv : float
        Cross-validation splitter instance
    clf : BaseEstimator
        The classifier instance to use for training
    n_splits : int, optional
        Number of splits for cross-validation, by default 5
    print_results : bool, optional
        Whether to print the results, by default False
    seed : int, optional
        Random seed for reproducibility, by default 42

    Returns
    -------
    dict[str, float]
        Dictionary containing:
        - mean_accuracy: Mean accuracy across CV splits
        - std_accuracy: Standard deviation of accuracy
        - mean_fit_time: Mean training time
        - std_fit_time: Standard deviation of training time

    """
    clf1 = DecisionTreeClassifier(random_state=62)
    clf1.fit(x, y)

    cv_results: dict[str, NDArray[np.floating]] = cross_validate(
        estimator=clf,
        X=x,
        y=y,
        cv=cv,
        return_train_score=True,
    )
    mean_accuracy: float = cv_results["test_score"].mean()
    std_accuracy: float = cv_results["test_score"].std()
    mean_fit_time: float = cv_results["fit_time"].mean()
    std_fit_time: float = cv_results["fit_time"].std()

    scores = {
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "mean_fit_time": mean_fit_time,
        "std_fit_time": std_fit_time,
    }

    if print_results:
        print("Mean Accuracy:", mean_accuracy)
        print("Std Accuracy:", std_accuracy)
        print("Mean Fit Time:", mean_fit_time)
        print("Std Fit Time:", std_fit_time)

    return scores


# ----------------------------------------------------------------


def print_cv_result_dict(cv_dict: dict, msg: str | None = None) -> None:
    """Print mean and standard deviation of values in cross validation.

    Parameters
    ----------
    cv_dict : dict
        Dictionary containing cross validation results with array values
    msg : str | None, optional
        Optional message to print before results, by default None

    Returns
    -------
    None
        Prints formatted mean and std for each key-value pair in dictionary

    """
    if msg is not None:
        print(f"\n{msg}")
    for key, array in cv_dict.items():
        print(f"mean_{key}: {array.mean()}, std_{key}: {array.std()}")

    # ----------------------------------------------------------------------
    """
    A. We will start by ensuring that your python environment is configured correctly
       and that you have all the required packages installed. For information about
       setting up Python please consult the following link:
       https://www.anaconda.com/products/individual.
       To test that your environment is set up correctly, simply import and print the
       mnist_assignment_starter.py module. You can also run the file as a script.
    """


def partA():
    """Import and print the mnist_assignment_starter.py module."""
    answers = {}
    answers["starter_code"] = u.starter_code()
    return answers


# ----------------------------------------------------------------------


def partB() -> dict[Any, Any]:
    """Load and prepare MNIST dataset, filtering for digits 7 and 9.

    Loads MNIST data, filters for digits 7 and 9, scales values between 0-1,
    and returns statistics about the filtered datasets.

    Returns
    -------
    dict[Any, Any]
        Dictionary containing:
        - length_x_train: Length of filtered training data
        - length_x_test: Length of filtered test data
        - length_y_train: Length of filtered training labels
        - length_ytest: Length of filtered test labels
        - max_x_train: Maximum value in training data
        - max_x_test: Maximum value in test data
        - x_train: Filtered and scaled training data
        - y_train: Filtered training labels
        - x_test: Filtered and scaled test data
        - y_test: Filtered test labels

    B. Load and prepare the mnist dataset, i.e., call the prepare_data and
       `filter_out_7_9s` functions in utils.py, to obtain a data matrix X consisting of
       only the digits 7 and 9. Make sure that every element in the data matrix is a
       floating point number and scaled between 0 and 1 (write a function to
       achieve this. Checking is not sufficient.)
       Also check that the labels are integers. Print out the length of the filtered
       `x` and `y`, and the maximum value of `x` for both training and test sets. Use
       the routines provided in utils.

    """
    x, y, x_test, y_test = u.prepare_data()
    x_train, y_train = u.filter_out_7_9s(x, y)
    x_test, y_test = u.filter_out_7_9s(x_test, y_test)
    x_train = nu.scale_data(x_train)
    x_test = nu.scale_data(x_test)

    answers = {}

    # Type type is int
    answers["length_x_train"] = len(x_train)
    answers["length_x_test"] = len(x_test)
    answers["length_y_train"] = len(y_train)
    answers["length_y_test"] = len(y_test)

    # The type is np.float32
    answers["max_x_train"] = np.max(x_train)
    answers["max_x_test"] = np.max(x_test)

    # The type should be NDArray[np.float32]
    answers["x_train"] = x_train
    answers["y_train"] = y_train
    answers["x_test"] = x_test
    answers["y_test"] = y_test
    print("EXIT partB")
    return answers


# ----------------------------------------------------------------------
def partC(
    # x: NDArray[np.floating],
    # y: NDArray[np.int32],
) -> dict[Any, Any]:
    """Train a Decision Tree classifier using k-fold cross validation.

    Parameters
    ----------
    x : NDArray[np.floating]
        Training data matrix with floating point values
    y : NDArray[np.int32]
        Integer class labels

    Returns
    -------
    dict
        Dictionary containing:
        - clf: The Decision Tree classifier instance
        - cv: The KFold cross-validator instance
        - scores: Dictionary with mean/std of accuracy and fit times
            - mean_accuracy: Mean accuracy across folds
            - std_accuracy: Standard deviation of accuracy
            - mean_fit_time: Mean training time
            - std_fit_time: Standard deviation of training time

    C. Train your first classifier using k-fold cross validation (see
    train_simple_classifier_with_cv function). Use 5 splits and a Decision tree
    classifier. Print the mean and standard deviation for the accuracy scores
    in each validation set in cross validation. Also print the mean and std
    of the fit (or training) time.  (Be more specific about the output format)

    """
    print(f"partC, {x.shape=}, {y.shape=}")
    n_splits = 5
    clf = DecisionTreeClassifier(random_state=seed)
    cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)

    scores = cross_validate(clf, x, y, cv=cv, return_train_score=True)
    mean_accuracy = scores["test_score"].mean()
    std_accuracy = scores["test_score"].std()
    mean_fit_time = scores["fit_time"].mean()
    std_fit_time = scores["fit_time"].std()

    scores_dict = {
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "mean_fit_time": mean_fit_time,
        "std_fit_time": std_fit_time,
    }

    answers = {}

    # The type is an instance of DecisionTreeClassifier
    answers["clf"] = clf

    # The type is an instance of KFold
    answers["cv"] = cv

    # The type is a dict[str, float] with keys:
    #   "mean_accuracy", "std_accuracy", "mean_fit_time", "std_fit_time"
    answers["scores"] = scores_dict
    print("EXIT partC")
    return answers

    # ---------------------------------------------------------


def partD(
    # x: NDArray[np.floating],
    # y: NDArray[np.int32],
) -> dict[Any, Any]:
    """Train a decision tree classifier using ShuffleSplit cross-validation.

    Parameters
    ----------
    x : NDArray[np.floating]
        Training data matrix
    y : NDArray[np.int32]
        Integer class labels

    Returns
    -------
    dict
        Dictionary containing:
        - clf: The Decision Tree classifier instance
        - cv: The ShuffleSplit cross-validator instance
        - scores: Dictionary with mean/std of accuracy and fit times
            - mean_accuracy: Mean accuracy across folds
            - std_accuracy: Standard deviation of accuracy
            - mean_fit_time: Mean training time
            - std_fit_time: Standard deviation of training time

    D. Repeat Part C with a random permutation (Shuffle-Split) k-fold cross-validator.

    """
    n_splits = 5
    clf = DecisionTreeClassifier(random_state=seed)
    # Check that the student does not use KFold again with Shuffle=True
    cv = ShuffleSplit(n_splits=n_splits, random_state=seed)

    scores = cross_validate(clf, x, y, cv=cv, return_train_score=True)
    mean_accuracy = scores["test_score"].mean()
    std_accuracy = scores["test_score"].std()
    mean_fit_time = scores["fit_time"].mean()
    std_fit_time = scores["fit_time"].std()

    scores_dict = {
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "mean_fit_time": mean_fit_time,
        "std_fit_time": std_fit_time,
    }

    answers = {}
    # The type is an instance of DecisionTreeClassifier
    answers["clf"] = clf
    # The type is an instance of ShuffleSplit
    answers["cv"] = cv
    # The type is a dict[str, float] with keys:
    #   "mean_accuracy", "std_accuracy", "mean_fit_time", "std_fit_time"
    answers["scores"] = scores_dict
    print("EXIT partD")
    return answers

    # ----------------------------------------------------------------------


def partE(
    # x: NDArray[np.floating],
    # y: NDArray[np.int32],
) -> dict[str, Any]:
    """Perform cross-validation using a Decision Tree classifier.

    Parameters
    ----------
    x : NDArray[np.floating]
        Data matrix containing features
    y : NDArray[np.int32]
        Integer class labels

    Returns
    -------
    dict
        Dictionary with keys being the number of splits (2, 5, 8, 16).
        For each split k, the value is a dictionary containing:
        - 'scores': dict
            - 'mean_accuracy': float, Mean accuracy across folds
            - 'std_accuracy': float, Standard deviation of accuracy
            - 'mean_fit_time': float, Mean training time
            - 'std_fit_time': float, Standard deviation of training time
        - 'cv': ShuffleSplit instance used for cross-validation
        - 'clf': DecisionTreeClassifier instance used for training

    E. Repeat part D for `k=2,5,8,16`, but do not print the training time.
    Note that this may take a long time (2-5 mins) to run. Do you notice
    anything about the mean and/or standard deviation of the scores for each `k`?

    """
    n_splits = [2, 5, 8, 16]
    answers = {}

    for n_split in n_splits:
        cv = ShuffleSplit(n_splits=n_split, random_state=seed)
        clf = DecisionTreeClassifier(random_state=seed)
        scores = train_simple_classifier_with_cv(
            x,
            y,
            cv=cv,
            clf=clf,
        )

        scores = cross_validate(clf, x, y, cv=cv, return_train_score=True)
        mean_accuracy = scores["test_score"].mean()
        std_accuracy = scores["test_score"].std()
        mean_fit_time = scores["fit_time"].mean()
        std_fit_time = scores["fit_time"].std()

        scores_dict = {
            "mean_accuracy": mean_accuracy,
            "std_accuracy": std_accuracy,
            "mean_fit_time": mean_fit_time,
            "std_fit_time": std_fit_time,
        }

        answers = {}
        # The type is a dict[str, float] with keys:
        #   "mean_accuracy", "std_accuracy", "mean_fit_time", "std_fit_time"
        answers["scores"] = scores_dict

        # The type is an instance of ShuffleSplit
        answers["cv"] = cv

        # The type is an instance of DecisionTreeClassifier
        answers["clf"] = clf

    print("exit partE")
    return answers

    # ----------------------------------------------------------------------


def partF(
    # x: NDArray[np.floating],
    # y: NDArray[np.int32],
) -> dict[str, Any]:
    """Return a dictionary with data for Random Forest and Decision Tree classifiers.

    Parameters
    ----------
    x : NDArray[np.floating]
        Data matrix with shape (n_samples, n_features)
    y : NDArray[np.int32]
        Labels with shape (n_samples,)

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - clf_RF: Random Forest classifier instance
        - cv_RF: ShuffleSplit cross-validator for RF
        - scores_RF: Dict with RF scores (mean/std accuracy and fit time)
        - clf_DT: Decision Tree classifier instance
        - cv_DT: ShuffleSplit cross-validator for DT
        - scores_DT: Dict with DT scores (mean/std accuracy and fit time)
        - model_highest_accuracy: String indicating model with highest accuracy
        - model_lowest_variance: String indicating model with lowest variance
        - model_fastest: String indicating fastest model

    Notes
    -----
    - The suffix _RF and _DT are used to distinguish between the Random Forest
        and Decision Tree models.

    F. Repeat part D with a Random Forest classifier with default parameters.
    Make sure the train test splits are the same for both models when performing
    cross-validation. Use ShuffleSplit for cross-validation. Which model has
    the highest accuracy on average?
    Which model has the lowest variance on average? Which model is faster
    to train? (compare results of part D and part F)

    Make sure your answers are calculated and not copy/pasted. Otherwise, the
    automatic grading will generate the wrong answers.

    Use a Random Forest classifier (an ensemble of DecisionTrees).

    """
    results_dict = {}

    # Logistic Regression
    clf = RandomForestClassifier(random_state=seed)
    # ! clf = LogisticRegression(random_state=self.seed, max_iter=max_iter)
    cv = ShuffleSplit(n_splits=5, random_state=seed)

    scores = cross_validate(
        clf,
        x,
        y,
        cv=cv,
        return_train_score=True,
    )
    mean_accuracy = scores["test_score"].mean()
    std_accuracy = scores["test_score"].std()
    mean_fit_time = scores["fit_time"].mean()
    std_fit_time = scores["fit_time"].std()

    scores_dict = {
        "mean_accuracy": mean_accuracy,
        "std_accuracy": std_accuracy,
        "mean_fit_time": mean_fit_time,
        "std_fit_time": std_fit_time,
    }

    answers = {}

    # The type is an instance of RandomForestClassifier
    answers["clf_RF"] = clf

    # The type is an instance of ShuffleSplit
    answers["cv_RF"] = cv

    # The type is a dict[str, float] with keys:
    #   "mean_accuracy", "std_accuracy", "mean_fit_time", "std_fit_time"
    answers["scores_RF"] = scores_dict

    answer_dt = partD()  # x, y)

    # The type is an instance of DecisionTreeClassifier
    answers["clf_DT"] = answer_dt["clf"]

    # The type is an instance of ShuffleSplit
    answers["cv_DT"] = answer_dt["cv"]

    # The type is a dict[str, float] with keys:
    #   "mean_accuracy", "std_accuracy", "mean_fit_time", "std_fit_time"
    answers["scores_DT"] = answer_dt["scores"]

    score_dt = answers["scores_DT"]["mean_accuracy"]
    score_rf = answers["scores_RF"]["mean_accuracy"]

    fit_time_dt = answers["scores_DT"]["mean_fit_time"]
    fit_time_rf = answers["scores_RF"]["mean_fit_time"]

    # We assume that the square of the mean standard deviation is the average variance.
    # This is not quite true, but is probably good enough.
    variance_dt = answers["scores_DT"]["std_accuracy"] ** 2
    variance_rf = answers["scores_RF"]["std_accuracy"] ** 2

    # The type is a string, one of "decision-tree" or "random-forest"
    answers["model_highest_accuracy"] = (
        "decision-tree" if score_dt > score_rf else "random-forest",
    )

    # The type is a string, one of "decision-tree" or "random-forest"
    answers["model_lowest_variance"] = (
        "decision-tree" if variance_dt < variance_rf else "random-forest",
    )

    # The type is a string, one of "decision-tree" or "random-forest"
    answers["model_fastest"] = "decision-tree" if fit_time_dt < fit_time_rf else "random-forest"

    return answers

    # ----------------------------------------------------------------------


def partG(
    # x: NDArray[np.floating],
    # y: NDArray[np.int32],
    # x_test: NDArray[np.floating],
    # y_test: NDArray[np.int32],
) -> dict[str, Any]:
    """Train a Random Forest classifier using grid search.

    Evaluate performance on training and test sets and compare with cross-validation
    results. Estimate best hyperparameters using grid search.

    Parameters
    ----------
    x : NDArray[np.floating]
        Training data features
    y : NDArray[np.int32]
        Training data labels
    x_test : NDArray[np.floating]
        Test data features
    y_test : NDArray[np.int32]
        Test data labels

    Returns
    -------
    dict[str, Any]
        Dictionary containing:
        - best hyperparameters
        - training accuracy
        - test accuracy
        - cross validation mean accuracy

    G. For the Random Forest classifier trained in part F, manually (or systematically,
    i.e., using grid search), modify hyperparameters, and see if you can get
    a higher mean accuracy.  Finally train the classifier on all the training
    data and get an accuracy score on the test set.  Print out the training
    and testing accuracy and comment on how it relates to the mean accuracy
    when performing cross validation. Is it higher, lower or about the same?

    Choose among the following hyperparameters:
        1) criterion,
        2) max_depth,
        3) min_samples_split,
        4) min_samples_leaf,
        5) max_features

    """
    # Notice: no seed since I can't predict how
    # the student will use the grid search
    # Ask student to use at least two parameters per
    #  parameters for three parameters,  minimum of 8 tests.
    # (SVC can be found in the documention. So uses another search).
    # ! clf = RandomForestClassifier(random_state=self.seed)

    # Test: What are the possible parameters to vary for LogisticRegression
    # or SVC
    # Possibly use RandomForest.
    # standard

    # refit=True: fit with the best parameters when complete
    # A test should look at best_index_, best_score_ and best_params_
    """
         1) criterion,
         2) max_depth,
         3) min_samples_split,
         4) min_samples_leaf,
         5) max_features
         5) n_estimators
    """

    clf = RandomForestClassifier(random_state=seed)
    clf.fit(x, y)
    print(f"{list(clf.__dict__.keys())=}")

    # Look at documentation
    default_parameters = {
        "criterion": "gini",
        "max_features": 100,
        "n_estimators": 100,
    }

    parameters = {
        "criterion": ["entropy", "gini", "log_loss"],
        "max_features": [50],  # 5  (with low values, training is not improved  )
        "n_estimators": [200],  # 20
    }

    clf = RandomForestClassifier(random_state=seed)
    n_splits = 5
    # Uses stratified cross-validator by default
    # ! cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    # THERE IS SOME UNCONTROLLED RANDOMESS!
    # return_train_score is False by default
    grid_search = GridSearchCV(
        clf,
        param_grid=parameters,
        refit=True,
        cv=5,
        return_train_score=True,
    )
    # Performs grid search with cv, then fits the training data
    grid_search.fit(x, y)
    best_estimator = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    clf.fit(x, y)
    best_estimator.fit(x, y)

    y_test_pred_best = best_estimator.predict(x_test)
    y_test_pred_orig = clf.predict(x_test)

    # Confusion matrix is improved with the best estimator
    print(confusion_matrix(y_test, y_test_pred_best))

    y_train_pred_best = best_estimator.predict(x)
    y_train_pred_orig = clf.predict(x)

    answers = {}

    # The type is an instance of RandomForestClassifier
    answers["clf"] = clf

    # The type is an instance of RandomForestClassifier
    answers["best_estimator"] = best_estimator

    # The type is an instance of GridSearchCV
    answers["grid_search"] = grid_search

    # The type is a dict[str, Any]
    answers["default_parameters"] = default_parameters

    # The type is a float
    answers["mean_accuracy_cv"] = None

    cm_train_orig = confusion_matrix(y_train_pred_orig, y)
    cm_train_best = confusion_matrix(y_train_pred_best, y)
    cm_test_orig = confusion_matrix(y_test_pred_orig, y_test)
    cm_test_best = confusion_matrix(y_test_pred_best, y_test)

    score_train_orig = (cm_train_orig[0, 0] + cm_train_orig[1, 1]) / y.size
    score_train_best = (cm_train_orig[0, 0] + cm_train_best[1, 1]) / y.size
    score_test_orig = (cm_test_orig[0, 0] + cm_test_orig[1, 1]) / y_test.size
    score_test_best = (cm_test_orig[0, 0] + cm_test_best[1, 1]) / y_test.size

    # The answer type is a numpy.ndarray
    answers["confusion_matrix_train_orig"] = cm_train_orig
    answers["confusion_matrix_train_best"] = cm_train_best
    answers["confusion_matrix_test_orig"] = cm_test_orig
    answers["confusion_matrix_test_best"] = cm_test_best

    # compute: C11 + C22 / |C|_1  (accuracy based on confusion)
    # The answer type is a float
    answers["accuracy_orig_full_training"] = score_train_orig
    answers["accuracy_best_full_training"] = score_train_best
    answers["accuracy_orig_full_testing"] = score_test_orig
    answers["accuracy_best_full_testing"] = score_test_best

    # Train score is 1.0: (return_train_score=1). Overfitting?

    # Questions to answer: Did confusion matrix improve? On both classes?
    # Test: check the confusion matrices myself to confirm answer
    # Question: is there overfitting? Why? How do you know?
    # How would you fix overfitting?

    print("EXIT partG")
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
    x = rng.random((120, 120))  # 100 samples, 100 features
    # Fill labels with 0 and 1 (mimic 7 and 9s)
    y = (x[:, :5].sum(axis=1) > 2.5).astype(int)
    n_train = 100
    x_train = x[0:n_train, :]
    x_test = x[n_train:, :]
    y_train = y[0:n_train]
    y_test = y[n_train:]
    x = x_train
    y = y_train

    x_train, y_train, x_test, y_test = u.create_data(
        n_rows=120,
        n_features=120,
        frac_train=0.8,
    )

    print(f"{x_train.shape=}, {y_train.shape=}")
    print(f"{x_test.shape=}, {y_test.shape=}")
    ##############################################

    # Attention: the seed should never be changed. If it is, automatic grading
    # of the assignment could very well fail, and you'd lose points.
    # Make sure that all sklearn functions you use that require a seed have this
    # seed specified in the argument list, namely: `random_state=self.seed` if
    # you are inside the solution class.

    answer1A = partA()

    # x and Y are Mnist datasets
    answer1B = partB()
    answer1C = partC()  # x, y)
    answer1C = partC()  # x, y)
    answer1D = partD()  # x, y)
    answer1E = partE()  # x, y)
    answer1F = partF()  # x, y)

    answer1G = partG()  # x, y, x_test, y_test)

    answer = {}
    answer["1A"] = answer1A
    answer["1B"] = answer1B
    answer["1C"] = answer1C
    answer["1D"] = answer1D
    answer["1E"] = answer1E
    answer["1F"] = answer1F
    answer["1G"] = answer1G

    u.save_dict("section1.pkl", answer)
    """
    Run your code and produce all your results for your report. We will spot check the
    reports, and grade your code with automatic tools.
    """
