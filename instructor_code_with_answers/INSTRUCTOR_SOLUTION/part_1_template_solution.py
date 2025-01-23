# Inspired by GPT4

# Information on type hints
# https://peps.python.org/pep-0585/

# GPT on testing functions, mock functions, testing number of calls, and argument values
# https://chat.openai.com/share/b3fd7739-b691-48f2-bb5e-0d170be4428c

# from sklearn.base import ClassifierMixin, RegressorMixin

# Fill in the appropriate import statements from sklearn to solve the homework
# from email.policy import default
from sklearn.base import BaseEstimator
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import (
    ShuffleSplit,
    cross_validate,
    KFold,
    StratifiedKFold,
)

from numpy.linalg import norm

# from sklearn.base import BaseEstimator
from sklearn import datasets
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix

# import svm module
from sklearn.svm import SVC  # , LinearSVC

from collections import defaultdict

# import logistic regresssion module
from sklearn.linear_model import LogisticRegression
import numpy as np

from typing import Type, Any
from numpy.typing import NDArray

from sklearn.model_selection import GridSearchCV
import utils as u
import new_utils as nu


# ======================================================================
class Section1:
    def __init__(
        self,
        normalize: bool = True,
        seed: int | None = None,
        frac_train: float = 0.2,
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
        self.frac_train = frac_train
        self.seed = seed

    # ---------------------------------------------------------
    """
    C. Train your first classifier using 𝑘-fold cross validation 
       (see train_simple_classifier_with_cv function).  Use 5 splits 
       and a Decision tree classifier. Print the mean and standard 
       deviation for the accuracy scores in each validation set in 
       cross validation. Also print the mean and std of the fit 
       (or training) time.
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
        Train a simple classifier using k-vold cross-validation.

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
        # cv_results: dict[str, NDArray[np.floating]] = cross_validate(
        # clf, X, y, cv=cv,
        # )
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
    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        n_splits: int,
    ):
        print("before scores, partC")
        n_splits = 5
        clf = DecisionTreeClassifier(random_state=self.seed)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=seed)
        scores = self.train_simple_classifier_with_cv(X, y, cv=cv, clf=clf)
        ## For testing, I can check the arguments of functions
        return scores
    """
    # ----------------------------------------------------------------------
    """
    A. We will start by ensuring that your python environment is configured correctly and 
       that you have all the required packages installed. For information about setting up 
       Python please consult the following link: https://www.anaconda.com/products/individual. 
       To test that your environment is set up correctly, simply import and print the 
       mnist_assignment_starter.py module. You may also want to run the file as a script.
    """

    def partA(self):
        u.starter_code()

    # ----------------------------------------------------------------------
    """
    B. Load and prepare the mnist dataset, i.e., call the prepare_data and filter_out_7_9s 
       functions in utils.py, to obtain a data matrix X consisting of only the digits 7 and 9. Make sure that 
       every element in the data matrix is a floating point number and scaled between 0 and 1 (write
       a function to achieve this. Checking is not sufficient.) 
       Also check that the labels are integers. Print out the length of the filtered 𝑋 and 𝑦, 
       and the maximum value of 𝑋 for both training and test sets. Use the routines provided in utils.
    """

    def partB(
        self,
    ):
        X, y, Xtest, ytest = u.prepare_data()
        Xtrain, ytrain = u.filter_out_7_9s(X, y)
        Xtest, ytest = u.filter_out_7_9s(Xtest, ytest)
        Xtrain = nu.scale_data(Xtrain)
        Xtest = nu.scale_data(Xtest)

        answer = {}
        answer["length_Xtrain"] = len(Xtrain)
        answer["length_Xtest"] = len(Xtest)
        answer["length_ytrain"] = len(ytrain)
        answer["length_ytest"] = len(ytest)
        answer["max_Xtrain"] = np.max(Xtrain)
        answer["max_Xtest"] = np.max(Xtest)
        return answer, Xtrain, ytrain, Xtest, ytest

    """
    C. Train your first classifier using 𝑘-fold cross validation (see train_simple_classifier_with_cv 
       function). Use 5 splits and a Decision tree classifier. Print the mean and standard deviation 
       for the accuracy scores in each validation set in cross validation. Also print the mean and std 
       of the fit (or training) time.  (Be more specific about the output format)
    """

    # ----------------------------------------------------------------------
    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        n_splits = 5
        clf = DecisionTreeClassifier(random_state=self.seed)
        cv = KFold(n_splits=n_splits, shuffle=True, random_state=self.seed)

        scores = cross_validate(clf, X, y, cv=cv, return_train_score=True)
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

        answer = {}
        answer["clf"] = clf
        answer["cv"] = cv
        answer["scores"] = scores_dict
        return answer

    # ---------------------------------------------------------
    """
    D. Repeat Part C with a random permutation (Shuffle-Split) 𝑘-fold cross-validator.
    """

    def partD(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        n_splits = 5
        clf = DecisionTreeClassifier(random_state=self.seed)
        # Check that the student does not use KFold again with Shuffle=True
        cv = ShuffleSplit(n_splits=n_splits, random_state=self.seed)

        scores = cross_validate(clf, X, y, cv=cv, return_train_score=True)
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

        answer = {}
        answer["clf"] = clf
        answer["cv"] = cv
        answer["scores"] = scores_dict
        return answer

    # ----------------------------------------------------------------------
    """
    E. Repeat part D for 𝑘=2,5,8,16, but do not print the training time. 
       Note that this may take a long time (2–5 mins) to run. Do you notice 
       anything about the mean and/or standard deviation of the scores for each k?
    """

    def partE(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ):
        """
        Return a dictionary with data for all the splits.
        Main key: n_splits (2, 5, 8, and 16)
        Secondary keys: 'scores', 'cv', 'clf'
        """
        n_splits = [2, 5, 8, 16]
        answer = {}

        for n_split in n_splits:
            cv = ShuffleSplit(n_splits=n_split, random_state=self.seed)
            clf = DecisionTreeClassifier(random_state=self.seed)
            scores = self.train_simple_classifier_with_cv(
                X,
                y,
                cv=cv,
                clf=clf,
            )

            scores = cross_validate(clf, X, y, cv=cv, return_train_score=True)
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

            all_data = {}
            all_data["scores"] = scores_dict
            all_data["cv"] = cv
            all_data["clf"] = clf
            answer[n_split] = all_data

        return answer

    # ----------------------------------------------------------------------
    """
    F. Repeat part D with a Random Forest classifier with default parameters. 
       Make sure the train test splits are the same for both models when performing cross-validation. 
       Use ShuffleSplit for cross-validation. Which model has the highest accuracy on average? 
       Which model has the lowest variance on average? Which model is faster 
       to train? (compare results of part D and part F)

       Make sure your answers are calculated and not copy/pasted. Otherwise, the automatic grading 
       will generate the wrong answers. 
       
       Use a Random Forest classifier (an ensemble of DecisionTrees). 
    """

    def partF(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
    ) -> dict[str, Any]:
        results_dict = {}

        """
        """
        ## Logistic Regression
        clf = RandomForestClassifier(random_state=self.seed)
        # clf = LogisticRegression(random_state=self.seed, max_iter=max_iter)
        cv = ShuffleSplit(n_splits=5, random_state=self.seed)

        scores = cross_validate(clf, X, y, cv=cv, return_train_score=True)
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

        answer = {}
        answer["clf_RF"] = clf
        answer["cv_RF"] = cv
        answer["scores_RF"] = scores_dict

        answer_DT = self.partD(X, y)
        answer["clf_DT"] = answer_DT["clf"]
        answer["cv_DT"] = answer_DT["cv"]
        answer["scores_DT"] = answer_DT["scores"]

        score_DT = answer["scores_DT"]["mean_accuracy"]
        score_RF = answer["scores_RF"]["mean_accuracy"]

        fit_time_DT = answer["scores_DT"]["mean_fit_time"]
        fit_time_RF = answer["scores_RF"]["mean_fit_time"]

        # We assume that the square of the mean standard deviation is the average variance.
        # This is not quite true, but is probably good enough.
        variance_DT = answer["scores_DT"]["std_accuracy"] ** 2
        variance_RF = answer["scores_RF"]["std_accuracy"] ** 2

        answer["model_highest_accuracy"] = (
            "decision-tree" if score_DT > score_RF else "random-forest"
        )
        answer["model_lowest_variance"] = (
            "decision-tree" if variance_DT < variance_RF else "random-forest"
        )
        answer["model_fastest"] = (
            "decision-tree" if fit_time_DT < fit_time_RF else "random-forest"
        )

        return answer

        # ---------------------------------------------------------
        """
        Not necessarily the best way to write code, but good habits are developed to allow
        for comprehensive testing
        """

    # ----------------------------------------------------------------------
    """
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

    def partG(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> dict[str, Any]:
        """
        Perform classification using the given classifier and cross validator.

        Parameters:
        - clf: The classifier instance to use for classification.
        - cv: The cross validator instance to use for cross validation.
        - X: The test data.
        - y: The test labels.
        - n_splits: The number of splits for cross validation. Default is 5.

        Returns:
        - y_pred: The predicted labels for the test data.

        Note:
        This function is not fully implemented yet.
        """

        # Notice: no seed since I can't predict how
        # the student will use the grid search
        # Ask student to use at least two parameters per
        #  parameters for three parameters,  minimum of 8 tests.
        # (SVC can be found in the documention. So uses another search).
        # clf = RandomForestClassifier(random_state=self.seed)

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

        # default parameters
        # criterion: "gini"
        # max_features: sqrt(n_features)
        # n_estimators: 100
        clf = RandomForestClassifier(random_state=self.seed)
        clf.fit(X, y)
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

        clf = RandomForestClassifier(random_state=self.seed)
        n_splits = 5
        # Uses stratified cross-validator by default
        # cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

        # THERE IS SOME UNCONTROLLED RANDOMESS!
        # return_train_score is False by default
        grid_search = GridSearchCV(
            clf, param_grid=parameters, refit=True, cv=5, return_train_score=True
        )
        # Performs grid search with cv, then fits the training data
        grid_search.fit(X, y)
        best_estimator = grid_search.best_estimator_
        best_params = grid_search.best_params_
        best_score = grid_search.best_score_
        results = grid_search.cv_results_

        # print()
        # print(f"{best_estimator=}")
        # print(f"{best_params=}")
        # print(f"{best_score=}")
        # print(f"Gridsearch cv_results: {results}")

        clf.fit(X, y)
        best_estimator.fit(X, y)
        # print(f" {clf.score(X, y)=}")
        # print(f"{best_estimator.score(X, y)=}")

        ytest_pred_best = best_estimator.predict(Xtest)
        ytest_pred_orig = clf.predict(Xtest)

        # Confusion matrix is improved with the best estimator
        # print("test confusion matrix orig")
        # print(confusion_matrix(ytest, ytest_pred_orig))
        # print()
        # print("test confusion matrix best")
        print(confusion_matrix(ytest, ytest_pred_best))

        ytrain_pred_best = best_estimator.predict(X)
        ytrain_pred_orig = clf.predict(X)

        # Training is perfect
        # print("train confusion matrix orig")
        # print(confusion_matrix(y, ytrain_pred_orig))
        # print()
        # print("train confusion matrix best")
        # print(confusion_matrix(y, ytrain_pred_best))

        answer = {}
        answer["clf"] = clf
        answer["best_estimator"] = best_estimator
        answer["grid_search"] = grid_search
        answer["default_parameters"] = default_parameters
        answer["mean_accuracy_cv"] = None

        cm_train_orig = confusion_matrix(ytrain_pred_orig, y)
        cm_train_best = confusion_matrix(ytrain_pred_best, y)
        cm_test_orig = confusion_matrix(ytest_pred_orig, ytest)
        cm_test_best = confusion_matrix(ytest_pred_best, ytest)

        score_train_orig = (cm_train_orig[0, 0] + cm_train_orig[1, 1]) / y.size
        score_train_best = (cm_train_orig[0, 0] + cm_train_best[1, 1]) / y.size
        score_test_orig = (cm_test_orig[0, 0] + cm_test_orig[1, 1]) / y.size
        score_test_best = (cm_test_orig[0, 0] + cm_test_best[1, 1]) / y.size

        answer["confusion_matrix_train_orig"] = cm_train_orig
        answer["confusion_matrix_train_best"] = cm_train_best
        answer["confusion_matrix_test_orig"] = cm_test_orig
        answer["confusion_matrix_test_best"] = cm_test_best

        # compute: C11 + C22 / |C|_1  (accuracy based on confusion)
        answer["accuracy_orig_full_training"] = score_train_orig
        answer["accuracy_best_full_training"] = score_train_best
        answer["accuracy_orig_full_testing"] = score_test_orig
        answer["accuracy_best_full_testing"] = score_test_best

        # Train score is 1.0: (return_train_score=1). Overfitting?

        # Questions to answer: Did confusion matrix improve? On both classes?
        # Test: check the confusion matrices myself to confirm answer
        # Question: is there overfitting? Why? How do you know?
        # How would you fix overfitting?

        return answer
