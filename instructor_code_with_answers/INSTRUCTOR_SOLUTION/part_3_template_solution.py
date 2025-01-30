# Inspired by GPT4

# Information on type hints
# https://peps.python.org/pep-0585/

# GPT on testing functions, mock functions, testing number of calls, and argument values
# https://chat.openai.com/share/b3fd7739-b691-48f2-bb5e-0d170be4428c

from sklearn.base import accuracy_score
from part_1_template_solution import Section1 as Part1
from part_2_template_solution import Section2 as Part2

from sklearn.metrics import top_k_accuracy_score

# Fill in the appropriate import statements from sklearn to solve the homework
from sklearn.model_selection import cross_validate
from sklearn.metrics import confusion_matrix
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from sklearn.svm import SVC

# from collections import defaultdict

# import logistic regresssion module
# from sklearn.linear_model import LogisticRegression
import numpy as np
from numpy.typing import NDArray
from typing import Any

import new_utils as nu

# import matplotlib.pyplot as plt
# import warnings

"""
   In the first two set of tasks, we will narrowly focus on accuracy - 
   what fraction of our predictions were correct. However, there are several 
   popular evaluation metrics. You will learn how (and when) to use these evaluation metrics.
"""


# ======================================================================
class Section3:
    def __init__(
        self,
        frac_train=0.2,
        seed=42,
    ):
        self.seed = seed
        self.part1 = Part1(seed=seed, frac_train=frac_train)
        self.part2 = Part2(seed=seed, frac_train=frac_train)

    def remove_nines_convert_to_01(self, X, y, frac):
        """
        frac: fraction of 9s to remove
        """
        # Count the number of 9s in the array
        num_nines = np.sum(y == 9)

        # Calculate the number of 9s to remove (90% of the total number of 9s)
        num_nines_to_remove = int(frac * num_nines)

        # Identifying indices of 9s in y
        indices_of_nines = np.where(y == 9)[0]

        # Randomly selecting 30% of these indices
        num_nines_to_remove = int(np.ceil(len(indices_of_nines) * frac))
        indices_to_remove = np.random.choice(
            indices_of_nines, size=num_nines_to_remove, replace=False
        )

        # Removing the selected indices from X and y
        X = np.delete(X, indices_to_remove, axis=0)
        y = np.delete(y, indices_to_remove)

        y[y == 7] = 0
        y[y == 9] = 1
        return X, y

    def analyze_class_distribution(self, y: NDArray[np.int32]) -> dict[str, Any]:
        """
        Analyzes and prints the class distribution in the dataset.

        Parameters:
        - y (array-like): Labels dataset.

        Returns:
        - dict: A dictionary containing the count of elements in each class and the total number of classes.
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
    """
    A. Using the same classifier and hyperparameters as the one used at the end of part 2.B. 
       Get the accuracies of the training/test set scores using the top_k_accuracy score for k=1,2,3,4,5. 
       Make a plot of k vs. score and comment on the rate of accuracy change. 
       Do you think this metric is useful for this dataset?
    """

    def partA(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        Xtest: NDArray[np.int32],
        ytest: NDArray[np.int32],
    ):
        """ """
        self.is_int = nu.check_labels(y)
        self.is_int = nu.check_labels(y)
        self.dist_dict = self.analyze_class_distribution(y)

        # top-k accuracy score
        score = top_k_accuracy_score(y_y, y_pred)

        answer = {}
        # MUST FIX THIS
        Xtest = X.copy()
        ytest = y.copy()
        return answer, X, y, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    B. Repeat part 1.B but return an imbalanced dataset consisting of 99% of all 9s removed.  Also convert the 7s to 0s and 9s to 1s.
    """

    # How to make sure the seed propagates. Perhaps specify in the class constructor.
    def partB(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.floating],
        Xtest: NDArray[np.int32],
        ytest: NDArray[np.int32],
    ):
        # Only Keep 7 and 9's
        seven_nine_idx = (y == 7) | (y == 9)
        X = X[seven_nine_idx, :]
        y = y[seven_nine_idx]
        frac_to_remove = 0.8
        X, y = self.remove_nines_convert_to_01(X, y, frac=frac_to_remove)
        Xtest, ytest = self.remove_nines_convert_to_01(
            Xtest, ytest, frac=frac_to_remove
        )
        answer = {}
        return answer, X, y, Xtest, ytest

    # --------------------------------------------------------------------------
    """
    C. Repeat part 1.C for this dataset but use a support vector machine (SVC in sklearn). 
        Make sure to use a stratified cross-validation strategy. In addition to regular accuracy 
        also print out the mean/std of the F1 score, precision, and recall. Is precision or 
        recall higher? Explain. Finally, train the classifier on all the training data and plot the confusion matrix.
    """

    def partC(
        self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ):
        n_splits = 5
        clf = SVC(random_state=self.seed)
        # Shuffling is fine because of seed
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        score = ["accuracy", "recall", "precision", "f1"]
        scores = cross_validate(clf, X, y, scoring=score, cv=cv)

        # Train on all the data
        clf.fit(X, y)

        # rows: actual, columns: predicted
        # cols: actual, columns: predicted
        # Return confusion matrix (DO NOT USE plot_confusion_matrix, which is deprecated) (CHECK via test that it is not used)
        # Return confusion matrix (no need to plot it)
        ypred = clf.predict(Xtest)
        conf_mat = confusion_matrix(ytest, ypred)

        print(f"{scores=}")

        answer = {}
        answer["cv"] = cv
        answer["clf"] = clf
        answer["mean_F1"] = 0
        answer["mean_recall"] = 0
        answer["mean_accuracy"] = 0
        answer["mean_precision"] = 0
        answer["std_F1"] = 0
        answer["std_recall"] = 0
        answer["std_accuracy"] = 0
        answer["std_precision"] = 0
        answer["is_precision_higher_than_recall"] = None  # True/False
        answer["is_precision_higher_than_recall_explain"] = None  # String
        answer["confusion_matrix"] = conf_mat  # 2 x 2 matrix

        ## For testing, I can check the arguments of functions
        return answer

    # --------------------------------------------------------------------------
    """
    D. Repeat the same steps as part 3.C but apply a weighted loss function (see the class_weights parameter).  Print out the class weights, and comment on the performance difference. Use compute_class_weight to compute the class weights. 
    """

    def partD(
        self,
        Xtrain: NDArray[np.floating],
        ytrain: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ):
        n_splits = 5
        # clf = SVC(random_state=self.seed, class_weight="balanced")
        clf = SVC(random_state=self.seed)
        # Shuffling is fine because of seed
        cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=self.seed)
        score = ["accuracy", "recall", "precision", "f1"]
        scores = cross_validate(clf, Xtrain, ytrain, scoring=score, cv=cv)

        class_weights = compute_class_weight(
            class_weight="balanced", classes=np.unique(ytrain), y=ytrain
        )
        weight_dict = {
            label: weight for label, weight in zip(np.unique(ytrain), class_weights)
        }

        print("class_weights = ", class_weights)  # Dimension 2
        print("weight_dict = ", weight_dict)  # Dimension 2

        # Train on all the data
        clf.fit(Xtrain, ytrain)

        ypred = clf.predict(Xtest)
        conf_mat = confusion_matrix(ytest, ypred)
        print(f"{conf_mat=}")

        answer = {}
        answer["cv"] = cv
        answer["clf"] = clf
        answer["mean_F1"] = 0
        answer["mean_recall"] = 0
        answer["mean_accuracy"] = 0
        answer["mean_precision"] = 0
        answer["std_F1"] = 0
        answer["std_recall"] = 0
        answer["std_accuracy"] = 0
        answer["std_precision"] = 0
        answer["is_precision_higher_than_recall"] = None  # True/False
        answer["is_precision_higher_than_recall_explain"] = None  # String
        answer["performance_difference_explain"] = None
        answer["conf_mat"] = conf_mat
        answer["weight_dict"] = {}
        return answer
