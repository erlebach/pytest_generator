import matplotlib
matplotlib.use('Agg')  # must be before any pyplot import

import numpy as np
import pytest
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # registers 3d projection


# ---------------------------------------------------------------------------
# numpy fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_1d_float_array():
    return np.array([1.0, 2.0, 3.0])


@pytest.fixture
def sample_2d_float_array():
    return np.random.rand(4, 3)


@pytest.fixture
def sample_1d_int_array():
    return np.array([1, 2, 3], dtype=int)


# ---------------------------------------------------------------------------
# matplotlib fixtures (session-scoped)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def line2d_fixture():
    fig, ax = plt.subplots()
    lines = ax.plot([1, 2, 3], [4, 5, 6])
    yield lines[0]
    plt.close(fig)


@pytest.fixture(scope="session")
def scatter2d_fixture():
    fig, ax = plt.subplots()
    col = ax.scatter([1, 2, 3], [4, 5, 6])
    yield col
    plt.close(fig)


@pytest.fixture(scope="session")
def scatter3d_fixture():
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    col = ax.scatter3D([1], [2], [3])
    yield col
    plt.close(fig)


# ---------------------------------------------------------------------------
# sklearn fixtures (session-scoped)
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def fitted_kfold():
    try:
        from sklearn.model_selection import KFold
        return KFold(n_splits=5)
    except ImportError:
        pytest.skip("scikit-learn not available")


@pytest.fixture(scope="session")
def fitted_stratifiedkfold():
    try:
        from sklearn.model_selection import StratifiedKFold
        return StratifiedKFold(n_splits=5)
    except ImportError:
        pytest.skip("scikit-learn not available")


@pytest.fixture(scope="session")
def fitted_shufflesplit():
    try:
        from sklearn.model_selection import ShuffleSplit
        return ShuffleSplit(n_splits=5, test_size=0.2)
    except ImportError:
        pytest.skip("scikit-learn not available")


@pytest.fixture(scope="session")
def fitted_svc():
    try:
        from sklearn.svm import SVC
        model = SVC()
        model.fit([[0, 0], [1, 1]], [0, 1])
        return model
    except ImportError:
        pytest.skip("scikit-learn not available")


@pytest.fixture(scope="session")
def fitted_gridsearchcv():
    try:
        from sklearn.svm import SVC
        from sklearn.model_selection import GridSearchCV
        model = GridSearchCV(SVC(), {'C': [1, 10]})
        model.fit([[0, 0], [1, 1], [2, 2], [3, 3]], [0, 1, 0, 1])
        return model
    except ImportError:
        pytest.skip("scikit-learn not available")


@pytest.fixture(scope="session")
def fitted_randomforestclassifier():
    try:
        from sklearn.ensemble import RandomForestClassifier
        model = RandomForestClassifier(n_estimators=10, random_state=42)
        model.fit([[0, 0], [1, 1], [2, 2], [3, 3]], [0, 1, 0, 1])
        return model
    except ImportError:
        pytest.skip("scikit-learn not available")


@pytest.fixture(scope="session")
def fitted_decisiontreeclassifier():
    try:
        from sklearn.tree import DecisionTreeClassifier
        model = DecisionTreeClassifier()
        model.fit([[0, 0], [1, 1], [2, 2], [3, 3]], [0, 1, 0, 1])
        return model
    except ImportError:
        pytest.skip("scikit-learn not available")


@pytest.fixture(scope="session")
def fitted_logisticregression():
    try:
        from sklearn.linear_model import LogisticRegression
        model = LogisticRegression()
        model.fit([[0, 0], [1, 1], [2, 2], [3, 3]], [0, 1, 0, 1])
        return model
    except ImportError:
        pytest.skip("scikit-learn not available")
