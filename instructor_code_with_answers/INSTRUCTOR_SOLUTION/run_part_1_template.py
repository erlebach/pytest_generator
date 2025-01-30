# Run this file from the command line
#   python run_part1.py
# Will not run properly from VSCode, and I don't understand

"""
# General instructions for all parts in all assignment sections: 
- Follow instructions very carefully. I tried to balance structure 
  with flexbility.  Giving you instructions that are too precise eases 
  grading but does not present a challenge. On the other hand, lack of 
  structure complexifies grading. 
- Create a github repository that clones the assignment template located 
  on [Github](https://github.com/fsu-sc/CAP-5777-s24-assignment-1) by 
  clicking the "Use this template" button, which creates a new repository 
  for your assignment. Make sure you include your name in the name of your 
  repository. 
- Register your repository on the Gradescope website. 

   All calls to functions should use only keyword argument. For example, 
      clf.fit(X, y)  should be replaced by  clf.fit(X=Xdata, y=ydata) where
      Xdata and ydata are your variable names. 
   Violation of this rule will lose you points, and possibly prevent the automatic. 
   grader from proper functioning
"""


def part_A():
    pass

def part_B():
    pass

def part_C(self, X: NDArray[np.floating], y: NDArray[np.int32]):
    """ Self is the class in which yuo implement the solution """
    clf = None #  Estimator class instance
    cv = None # cross-validator class instance
    scores = self.partC(X, y)
    return scores, clf, cv

def part_D(self, X: NDArray[np.floating], y: NDArray[np.int32]):
    clf = None
    cv = None
    scores = self.partD(X, y)
    return scores, clf, cv

def part_E(self, X: NDArray[np.floating], y: NDArray[np.int32]):
    # Return dictionary (keys: n_split). Value: dict with keys 'scores', 'clf', 'cv'
    # structure of all_data: 
    #   all_data[k] is a dict D where k is the number of cross-validator (cv) splits 
    #   D has keys: ['scores', 'clf', 'cv'] 
    # D['scores'] has the structure: 
    #   scores = {
    #       "mean_accuracy": mean_accuracy,
    #       "std_accuracy": std_accuracy,
    #       "mean_fit_time": mean_fit_time,
    #       "std_fit_time": std_fit_time,
    #   }
    # D['clf'] refers to the base estimator used (a classifier, regressor, random forest, SVM, etc.). 
    # D['cv'] refers to the cross-validator used (KFold, ShuffleSplit, StratifiedKFold, etc.)
    all_data = self.partE(X, y)
    return all_data

"""
F. Repeat part D with both logistic regression and support vector machine). Make sure the train test splits are the same for both models when performing cross-validation. Use ShuffleSplit for cross-validation. Which model has the highest accuracy on average? Which model has the lowest variance on average? Which model is faster to train?
"""
def part_F(self, X: NDArray[np.floating], y: NDArray[np.int32]):
    # Return dictionary (keys: n_split). Value: dict with keys 'scores', 'clf', 'cv'
    n_splits = 5
    all_data = self.partF(X, y, n_splits=n_splits)
    return all_data

"""
G. For the SVM classifier trained in part F, manually (or systematically, i.e., using grid search), modify hyperparameters, and see if you can get a higher mean accuracy. Finally train the classifier on all the training data and get an accuracy score on the test set. Print out the training and testing accuracy and comment on how it relates to the mean accuracy when performing cross validation. Is it higher, lower or about the same?
"""
def part_G(self,
        X: NDArray[np.floating],
        y: NDArray[np.int32],
        Xtest: NDArray[np.floating],
        ytest: NDArray[np.int32],
    ) -> NDArray[np.int32]:
        """
        Perform classification using the given classifier and cross validator.

        Parameters:
        - X: The test data.
        - y: The test labels.
        - n_splits: The number of splits for cross validation. Default is 5.

        Returns:
        - y_pred: The predicted labels for the test data.

        Note:
        This function is not fully implemented yet.
        """
        # Accuracy score on the test set. HOW TO DO THIS?
        # Count how many match between y_pred and y
        clf, clf_gs = self.partG(X, y, Xtest, ytest)
        return clf, clf_gs

#----------------------------------------------------------------------------------------------------
if __name__ == '__main__':
    # Read MNIST files and define Xtrain and xtest appropriately
    # X and Y are Mnist datasets
    # Define X, y (which are Xtrain and ytrain)
    # Define Xtest, Ytest

    # Attention: the seed should never be changed. If it is, automatic grading
    # of the assignment could very well fail, and you'd lose points. 
    # Make sure that all sklearn functions you use that require a seed have this
    # seed specified in the argument list, namely: `random_state=self.seed` if 
    # you are inside the solution class. 

    part1 = Part1(seed=42, frac_train=0.2)

    # The various parts of section 1 must be solved by calling the functions below

    part_A()
    part_B()
    part_C(part1, X , y)
    part_D(part1, X , y)
    part_E(part1, X , y)
    part_F(part1, X , y)
    part_G(part1, X, y, Xtest, ytest)

