# Inspired by GPT4

# Information on type hints
# https://peps.python.org/pep-0585/

# GPT on testing functions, mock functions, testing number of calls, and argument values
# https://chat.openai.com/share/b3fd7739-b691-48f2-bb5e-0d170be4428c

# from sklearn.base import ClassifierMixin, RegressorMixin

# ==============================================================
# Fill in the appropriate import statements from sklearn to solve the homework
# from email.policy import default

# IMPORTANT: do not communicate between functions in the class.
# In other words: do not define intermediary variables using self.var = xxx
# Doing so will make certain tests fail. Class methods should be independent
# of each other and be able to execute in any order!


from sklearn.metrics import confusion_matrix
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    KFold,
)

from numpy.linalg import norm

# from sklearn.base import BaseEstimator
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier

# import svm module
from sklearn.svm import SVC  # , LinearSVC

from collections import defaultdict

# import logistic regresssion module
from sklearn.linear_model import LogisticRegression
import numpy as np

from typing import Type, Any
from numpy.typing import NDArray

from sklearn.model_selection import GridSearchCV

# For code reuse. Ideally functions used in multiple classes should be put in
# a utils file
from part_1_template_solution import Section1 as Part1
import utils as u
import new_utils as nu

# ======================================================================

# I could make Section 2 a subclass of Section 1, which would facilitate code reuse.
# However, both classes have the same function names. Better to pass Section 1 instance
# as an argument to Section 2 class constructor.


class Section2:
    def __init__(
        self,
        normalize: bool = True,
        seed: int | None = None,
        frac_train: float = 0.2,
        # section1: Section1 | None = None,
    ):
        """
        Initializes an instance of MyClass.

        Args:
            normalize (bool, optional): Indicates whether to normalize the data. Defaults to True.
            seed (int, optional): The seed value for randomization. If None, each call will be randomized.
                If an integer is provided, calls will be repeatable.

        Returns:
            None
        """
        self.normalize = normalize
        self.seed = seed
        self.frac_train = frac_train
        self.part1 = Part1(seed=seed, frac_train=frac_train)

    # ---------------------------------------------------------
    def prepare_data(
        self, X: NDArray[np.floating], y: NDArray[np.int32], frac_train: float = 0.2
    ):
        """
        Prepare the data.
        Parameters:
            X: A data matrix
            frac_train: Fraction of the data used for training in [0,1]
        Returns:
            Prepared data matrix
        Side effect: update of self.Xtrain, self.ytrain, self.Xtest, self.ytest
        """
        num_train = int(frac_train * X.shape[0])
        self.Xtrain = X[:num_train]
        self.Xtest = X[num_train:]
        self.ytrain = y[:num_train]
        self.ytest = y[num_train:]
        # print("prepare data: X: ", X)
        return self.Xtrain, self.ytrain, self.Xtest, self.ytest

    # ---------------------------------------------------------
    def filter_out_7_9s(
        self, X: NDArray[np.floating], y: NDArray[np.int32]
    ) -> tuple[NDArray[np.floating], NDArray[np.int32]]:
        """
        Filter the dataset to include only the digits 7 and 9.
        Parameters:
            X: Data matrix
            y: Labels
        Returns:
            Filtered data matrix and labels
        Notes:
            np.int32 is a type with a range based on 32-bit ints
            np.int has no bound; it can hold arbitrarily long numbers
        """
        seven_nine_idx = (y == 7) | (y == 9)
        X_binary = X[seven_nine_idx, :]
        y_binary = y[seven_nine_idx]

        return X_binary, y_binary

    # ---------------------------------------------------------

    """
    C. Train your first classifier using 𝑘-fold cross validation (see train_simple_classifier_with_cv function). Use 5 splits and a Decision tree classifier. Print the mean and standard deviation for the accuracy scores in each validation set in cross validation. Also print the mean and std of the fit (or training) time.
    """

    # ======================================================================
    def train_simple_classifier_with_cv(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        # cv_class,  # : Type[BaseCrossValidator],  # a class
        cv,  # : BaseCrossValidator,  # a class instance (ask students why this choice)
        # estimator_class: Type[BaseEstimator],  # a class
        clf,  #: Type[BaseEstimator],  # a class instansce of the estimator
        n_splits: int = 5,
        print_results: bool = False,
        seed: int = 42,
    ) -> dict[str, float]:
        """
        Train a simple classifier using cross-validation.

        Parameters:
            - X: Features dataset.
            - y: Labels.
            - cv_class: The cross-validation class to use.
            - estimator_class: The training classifier class to use.
            - n_splits: Number of splits for cross-validation.
            - print_results: Whether to print the results.

        Returns:
            - A dictionary with mean and std of accuracy and fit time.
        """

        # clf = estimator_class(random_state=seed)
        # Not all cross validators have a shuffle parameter. Therefore, passing
        # the class as an argument does not always work. Passing the estimator,
        # where the class was instantiated in the calling function will be more flebile.
        # cv = cv_class(n_splits=n_splits, shuffle=True, random_state=seed)

        # FOR DEBUGGING
        clf1 = DecisionTreeClassifier(random_state=62)
        clf1.fit(X, y)

        cv_results: dict[str, NDArray[np.floating]] = cross_validate(
            estimator=clf, X=X, y=y, cv=cv, return_train_score=True
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

    def print_cv_result_dict(self, cv_dict: dict, msg: str | None = None):
        if msg is not None:
            print(f"\n{msg}")
        for key, array in cv_dict.items():
            print(f"mean_{key}: {array.mean()}, std_{key}: {array.std()}")

    """
    A. Repeat part 1.B but make sure that your data matrix (and labels) consists of
        all classes by also printing out the number of elements in each class y and 
        print out the number of classes for both training and testing datasets. 
"""

    def partA(self):
        Xtrain, ytrain, Xtest, ytest = u.prepare_data()
        Xtrain = nu.scale_data(Xtrain)
        Xtest = nu.scale_data(Xtest)

        # Check that labels contain all classes
        # count number of elements of each class in train and test sets
        nb_each_class = defaultdict(int)
        for label in ytrain:
            nb_each_class[label] += 1
        nb_classes_train = len(nb_each_class)

        for label in ytest:
            nb_each_class[label] += 1
        nb_classes_test = len(nb_each_class)

        assert nb_classes_train == 10
        assert nb_classes_test == 10

        print(f"{nb_classes_train=}")
        print(f"{nb_classes_test=}")

        answer = {}
        answer["length_Xtrain"] = len(Xtrain)
        answer["length_Xtest"] = len(Xtest)
        answer["length_ytrain"] = len(ytrain)
        answer["length_ytest"] = len(ytest)
        answer["max_Xtrain"] = np.max(Xtrain)
        answer["max_Xtest"] = np.max(Xtest)
        return answer, Xtrain, ytrain, Xtest, ytest

        nb_classes_train_dict = {}
        nb_classes_test_dict = {}
        is_train_label_int = True
        is_test_label_int = True
        is_train_data_float = True
        is_test_data_float = True
        is_train_data_in_zero_one = True
        is_test_data_in_zero_one = True
        # Check that labels are integers
        # is_int: True if labels are integers, False if not
        answer = {}
        answer["nb_classes_train_dict"] = nb_classes_train_dict
        answer["nb_classes_test_dict"] = nb_classes_test_dict
        answer["is_train_label_int"] = None  # bool
        answer["is_test_label_int"] = None  # bool
        answer["is_train_data_float"] = None  # bool
        answer["is_test_data_float"] = None  # bool
        answer["is_train_data_in_zero_one"] = None  # bool
        answer["is_test_data_in_zero_one"] = None  # bool
        print("part_A answer: ", answer)
        return answer

    # B. Repeat part 1.C, 1.D, and 1.F, for the multiclass problem.
    #    Use the Logistic Regression for part F with 300 iterations.
    #    Explain how multi-class logistic regression works (inherent,
    #    one-vs-one, one-vs-the-rest, etc.).  Repeat the experiment
    #    for N = 1000, 5000, 20000. Choose
    #            ntrain = 0.8 * N
    #            ntestn = 0.2 * N
    #    Comment on the results. Is the accuracy higher for the
    #    training or testing set?
    #
    #    For the final classifier you trained in 2.B (partF),
    #     plot a confusion matrix for the test predictions.
    #     Earlier we stated that 7s and 9s were a challenging pair to
    #    distinguish. Do your results support this statement? Why or why not?

    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
        ntrain_list: list[int] = [],
    ):
        ntest_list = [i // 4 for i in ntrain_list]
        print(f"{ntrain_list=}")
        print(f"{ntest_list=}")

        def partB_sub(X, y, Xtest, ytest):
            # ------
            print("===> repeat part C")
            print(f"{X.shape=}, {y.shape=}, {Xtest.shape=}, {ytest.shape=}")
            answer = self.part1.partC(X, y)
            scores_C = answer["scores"]
            partC = {"scores_C": scores_C, "clf": answer["clf"], "cv": answer["cv"]}
            # ------
            print("===> repeat part D")
            answer_D = self.part1.partD(X, y)
            scores_D = answer_D["scores"]
            partD = {"scores_D": scores_D, "clf": answer_D["clf"], "cv": answer_D["cv"]}

            # ------
            print("===> repeat part F")
            # Repeat part 1F
            # Use logistic regressor with default arguments.
            # Make sure you set the random state argument.
            # answer = self.part1.partF(X, y)

            cv = ShuffleSplit(n_splits=5, random_state=self.seed)
            clf = LogisticRegression(
                random_state=self.seed, multi_class="multinomial", max_iter=500
            )
            # Use logistic regressor
            scores = cross_validate(clf, X, y, cv=cv, return_train_score=True)
            clf.fit(X, y)
            scores_train_F = clf.score(X, y)  # scalar
            scores_test_F = clf.score(Xtest, ytest)  # scalar
            mean_cv_accuracy_F = scores["test_score"].mean()

            y_pred = clf.predict(X)
            ytest_pred = clf.predict(Xtest)

            conf_mat_train = confusion_matrix(y_pred, y)
            conf_mat_test = confusion_matrix(ytest_pred, ytest)

            # Using entire dataset
            partF = {
                "scores_train_F": scores_train_F,
                "scores_test_F": scores_test_F,
                "mean_cv_accuracy_F": mean_cv_accuracy_F,
                "clf": clf,
                "cv": cv,
                "conf_mat_train": conf_mat_train,
                "conf_mat_test": conf_mat_test,
            }

            # -----------------------------------------------
            answer = {}
            answer["partC"] = partC
            answer["partD"] = partD
            answer["partF"] = partF
            answer["ntrain"] = len(y)
            answer["ntest"] = len(ytest)
            answer["class_count_train"] = np.unique(y, return_counts=True)[1]
            answer["class_count_test"] = np.unique(ytest, return_counts=True)[1]
            return answer
            # ----------------

        answer = {}

        for ntr, nte in zip(ntrain_list, ntest_list):
            X_r = X[0:ntr, :]
            y_r = y[0:ntr]
            Xtest_r = Xtest[0:nte, :]
            ytest_r = ytest[0:nte]
            print(f"{ntr=}, {nte=}")
            print(f"{X_r.shape=}, {y_r.shape=}, {Xtest_r.shape=}, {ytest_r.shape=}")
            answer[ntr] = partB_sub(X_r, y_r, Xtest_r, ytest_r)

        return answer
