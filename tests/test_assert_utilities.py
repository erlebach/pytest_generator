"""Regression tests for assert_utilities.py — check_structure_* functions.

Tests are written BEFORE any refactoring. They lock in current behavior.
Each test asserts: (1) the bool component, (2) the str component is a non-empty string.

Note: many check_structure_* functions require an instructor_answer argument
in addition to student_answer. These are supplied with matching valid values.
"""
import pytest
import numpy as np
from pytest_generator.assert_utilities import (
    check_structure_bool,
    check_structure_float,
    check_structure_int,
    check_structure_str,
    check_structure_ndarray,
    check_structure_list_float,
    check_structure_list_int,
    check_structure_list_str,
    check_structure_list_ndarray,
    check_structure_list_list_float,
    check_structure_list_tuple_float,
    check_structure_list_set,
    check_structure_set_str,
    check_structure_set_set_int,
    check_structure_set_tuple_int,
    check_structure_dict_str_int,
    check_structure_dict_str_float,
    check_structure_dict_str_list_str,
    check_structure_dict_str_list_int,
    check_structure_dict_str_ndarray,
    check_structure_dict_str_set_int,
    check_structure_dict_str_tuple_ndarray,
    check_structure_dict_str_dict_str_float,
    check_structure_dict_str_any,
    check_structure_dict_str_set,
    check_structure_dict_int_float,
    check_structure_dict_int_list,
    check_structure_dict_int_list_float,
    check_structure_dict_int_ndarray,
    check_structure_dict_int_dict_str_any,
    check_structure_dict_tuple_int_ndarray,
    check_structure_dict_any,
    check_structure_eval_float,
    check_structure_explain_str,
    check_structure_function,
    check_structure_dendrogram,
    check_structure_lineplot,
    check_structure_scatterplot2d,
    check_structure_scatterplot3d,
    check_structure_kfold,
    check_structure_stratifiedkfold,
    check_structure_shufflesplit,
    check_structure_svc,
    check_structure_gridsearchcv,
    check_structure_randomforestclassifier,
    check_structure_decisiontreeclassifier,
    check_structure_logisticregression,
)


# ---------------------------------------------------------------------------
# check_structure_bool
# ---------------------------------------------------------------------------

def test_check_structure_bool_pass():
    status, msg = check_structure_bool(True)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_bool_fail():
    status, msg = check_structure_bool(1)  # int, not bool
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_float
# ---------------------------------------------------------------------------

def test_check_structure_float_pass():
    status, msg = check_structure_float(3.14)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_float_fail():
    status, msg = check_structure_float("3.14")
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_int
# ---------------------------------------------------------------------------

def test_check_structure_int_pass():
    status, msg = check_structure_int(42)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_int_fail():
    status, msg = check_structure_int(3.14)
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_str  (requires choices argument)
# ---------------------------------------------------------------------------

def test_check_structure_str_pass():
    status, msg = check_structure_str("hello", choices=["hello", "world"])
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_str_fail():
    status, msg = check_structure_str(42, choices=["hello"])  # not a str
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_ndarray
# ---------------------------------------------------------------------------

def test_check_structure_ndarray_pass():
    status, msg = check_structure_ndarray(np.array([1.0, 2.0]))
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_ndarray_fail():
    status, msg = check_structure_ndarray([1.0, 2.0])  # list, not ndarray
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_list_float  (requires instructor_answer)
# ---------------------------------------------------------------------------

def test_check_structure_list_float_pass():
    status, msg = check_structure_list_float([1.0, 2.0], [1.0, 2.0])
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_list_float_fail():
    status, msg = check_structure_list_float((1.0, 2.0), [1.0, 2.0])  # tuple, not list
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_list_int  (requires instructor_answer)
# ---------------------------------------------------------------------------

def test_check_structure_list_int_pass():
    status, msg = check_structure_list_int([1, 2, 3], [1, 2, 3])
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_list_int_fail():
    status, msg = check_structure_list_int("not a list", [1, 2])  # not a list
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_list_str
# ---------------------------------------------------------------------------

def test_check_structure_list_str_pass():
    status, msg = check_structure_list_str(["a", "b"])
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_list_str_fail():
    status, msg = check_structure_list_str(["a", 1])  # mixed types
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_list_ndarray  (requires instructor_answer)
# ---------------------------------------------------------------------------

def test_check_structure_list_ndarray_pass():
    status, msg = check_structure_list_ndarray(
        [np.array([1.0]), np.array([2.0])],
        [np.array([1.0]), np.array([2.0])],
    )
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_list_ndarray_fail():
    status, msg = check_structure_list_ndarray(
        "not a list",  # not a list
        [np.array([1.0]), np.array([2.0])],
    )
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_list_list_float  (requires instructor_answer)
# ---------------------------------------------------------------------------

def test_check_structure_list_list_float_pass():
    status, msg = check_structure_list_list_float(
        [[1.0, 2.0], [3.0, 4.0]],
        [[1.0, 2.0], [3.0, 4.0]],
    )
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_list_list_float_fail():
    status, msg = check_structure_list_list_float(
        [1.0, 2.0],  # not nested
        [[1.0, 2.0]],
    )
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_list_tuple_float
# ---------------------------------------------------------------------------

def test_check_structure_list_tuple_float_pass():
    status, msg = check_structure_list_tuple_float([(1.0, 2.0), (3.0, 4.0)])
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_list_tuple_float_fail():
    status, msg = check_structure_list_tuple_float([[1.0, 2.0]])  # list of lists
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_list_set  (requires instructor_answer)
# ---------------------------------------------------------------------------

def test_check_structure_list_set_pass():
    status, msg = check_structure_list_set([{1, 2}, {3, 4}], [{1, 2}])
    assert status is True
    assert isinstance(msg, str)  # msg may be empty string (current behavior)


def test_check_structure_list_set_fail():
    status, msg = check_structure_list_set("not a list", [{1, 2}])  # not a list
    assert status is False
    assert isinstance(msg, str)


# ---------------------------------------------------------------------------
# check_structure_set_str
# ---------------------------------------------------------------------------

def test_check_structure_set_str_pass():
    status, msg = check_structure_set_str({"a", "b"})
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_set_str_fail():
    status, msg = check_structure_set_str(42)  # not a set or list
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_set_set_int
# ---------------------------------------------------------------------------

def test_check_structure_set_set_int_pass():
    # Function checks that outer is sequence and inner elements are sequence-like
    # A list of lists satisfies both conditions
    status, msg = check_structure_set_set_int([[1, 2], [3, 4]])
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_set_set_int_fail():
    status, msg = check_structure_set_set_int("not a set")
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_set_tuple_int
# ---------------------------------------------------------------------------

def test_check_structure_set_tuple_int_pass():
    status, msg = check_structure_set_tuple_int({(1, 2), (3, 4)})
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_set_tuple_int_fail():
    status, msg = check_structure_set_tuple_int([(1, 2)])  # list, not set
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_dict_str_int  (requires instructor_answer)
# ---------------------------------------------------------------------------

def test_check_structure_dict_str_int_pass():
    status, msg = check_structure_dict_str_int({"a": 1, "b": 2}, {"a": 1, "b": 2})
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_dict_str_int_fail():
    status, msg = check_structure_dict_str_int({"a": 1.0}, {"a": 1})  # float value
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_dict_str_float  (requires instructor_answer and keys)
# ---------------------------------------------------------------------------

def test_check_structure_dict_str_float_pass():
    status, msg = check_structure_dict_str_float(
        {"a": 1.0, "b": 2.0}, {"a": 1.0, "b": 2.0}
    )
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_dict_str_float_fail():
    status, msg = check_structure_dict_str_float(
        {"a": "x"}, {"a": 1.0}  # str value, not float
    )
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_dict_str_list_str  (requires instructor_answer)
# ---------------------------------------------------------------------------

def test_check_structure_dict_str_list_str_pass():
    status, msg = check_structure_dict_str_list_str(
        {"a": ["x", "y"]}, {"a": ["x", "y"]}
    )
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_dict_str_list_str_fail():
    status, msg = check_structure_dict_str_list_str(
        "not a dict", {"a": ["x"]}  # not a dict
    )
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_dict_str_list_int  (requires instructor_answer)
# ---------------------------------------------------------------------------

def test_check_structure_dict_str_list_int_pass():
    status, msg = check_structure_dict_str_list_int({"a": [1, 2]}, {"a": [1, 2]})
    assert status is True
    assert isinstance(msg, str)  # msg may be empty string (current behavior)


def test_check_structure_dict_str_list_int_fail():
    status, msg = check_structure_dict_str_list_int(
        "not a dict", {"a": [1, 2]}  # not a dict
    )
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_dict_str_ndarray  (requires instructor_answer)
# ---------------------------------------------------------------------------

def test_check_structure_dict_str_ndarray_pass():
    status, msg = check_structure_dict_str_ndarray(
        {"a": np.array([1.0])}, {"a": np.array([1.0])}
    )
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_dict_str_ndarray_fail():
    status, msg = check_structure_dict_str_ndarray(
        {"a": [1.0]}, {"a": np.array([1.0])}  # list, not ndarray
    )
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_dict_str_set_int  (requires instructor_answer)
# ---------------------------------------------------------------------------

def test_check_structure_dict_str_set_int_pass():
    status, msg = check_structure_dict_str_set_int({"a": {1, 2}}, {"a": {1, 2}})
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_dict_str_set_int_fail():
    status, msg = check_structure_dict_str_set_int(
        {"a": "not_set"}, {"a": {1, 2}}  # value is str, not set
    )
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_dict_str_tuple_ndarray  (requires instructor_answer)
# ---------------------------------------------------------------------------

def test_check_structure_dict_str_tuple_ndarray_pass():
    status, msg = check_structure_dict_str_tuple_ndarray(
        {"a": (np.array([1.0]), np.array([2.0]))},
        {"a": (np.array([1.0]), np.array([2.0]))},
    )
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_dict_str_tuple_ndarray_fail():
    status, msg = check_structure_dict_str_tuple_ndarray(
        {},  # missing required key 'a'
        {"a": (np.array([1.0]), np.array([2.0]))},
    )
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_dict_str_dict_str_float  (requires instructor_answer)
# ---------------------------------------------------------------------------

def test_check_structure_dict_str_dict_str_float_pass():
    status, msg = check_structure_dict_str_dict_str_float(
        {"a": {"b": 1.0}}, {"a": {"b": 1.0}}
    )
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_dict_str_dict_str_float_fail():
    status, msg = check_structure_dict_str_dict_str_float(
        {},  # missing required key 'a'
        {"a": {"b": 1.0}},
    )
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_dict_str_any  (requires instructor_answer and keys)
# ---------------------------------------------------------------------------

def test_check_structure_dict_str_any_pass():
    status, msg = check_structure_dict_str_any({"a": 1}, {"a": 1})
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_dict_str_any_fail():
    status, msg = check_structure_dict_str_any({}, {"a": 1})  # missing key 'a'
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_dict_str_set  (requires instructor_answer)
# ---------------------------------------------------------------------------

def test_check_structure_dict_str_set_pass():
    # Student value is a set — correct type, should return True
    status, msg = check_structure_dict_str_set({"a": {1, 2}}, {"a": {1, 2}})
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_dict_str_set_fail():
    # Student value is an int (not a set/list) — wrong type, should return False
    status, msg = check_structure_dict_str_set({"a": 42}, {"a": {1, 2}})
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_dict_int_float  (requires instructor_answer)
# ---------------------------------------------------------------------------

def test_check_structure_dict_int_float_pass():
    status, msg = check_structure_dict_int_float({1: 1.0, 2: 2.0}, {1: 1.0, 2: 2.0})
    assert status is True
    assert isinstance(msg, str)  # msg may be empty string (current behavior)


def test_check_structure_dict_int_float_fail():
    status, msg = check_structure_dict_int_float("not a dict", {1: 1.0})
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_dict_int_list  (requires instructor_answer and keys)
# ---------------------------------------------------------------------------

def test_check_structure_dict_int_list_pass():
    # Function checks that values are lists of floats (not ints)
    status, msg = check_structure_dict_int_list(
        {1: [1.0, 2.0], 2: [3.0, 4.0]},
        {1: [1.0, 2.0], 2: [3.0, 4.0]},
        keys=None,
    )
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_dict_int_list_fail():
    status, msg = check_structure_dict_int_list(
        {1: [1, 2]},  # int elements fail (function expects float elements)
        {1: [1, 2]},
        keys=None,
    )
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_dict_int_list_float  (requires instructor_answer)
# ---------------------------------------------------------------------------

def test_check_structure_dict_int_list_float_pass():
    status, msg = check_structure_dict_int_list_float(
        {1: [1.0, 2.0]}, {1: [1.0, 2.0]}
    )
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_dict_int_list_float_fail():
    status, msg = check_structure_dict_int_list_float(
        {1: [1, 2]}, {1: [1.0, 2.0]}  # ints, not floats
    )
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_dict_int_ndarray  (requires instructor_answer and keys)
# ---------------------------------------------------------------------------

def test_check_structure_dict_int_ndarray_pass():
    status, msg = check_structure_dict_int_ndarray(
        {1: np.array([1.0])},
        {1: np.array([1.0])},
        keys=None,
    )
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_dict_int_ndarray_fail():
    status, msg = check_structure_dict_int_ndarray(
        {1: [1.0]},  # list, not ndarray
        {1: np.array([1.0])},
        keys=None,
    )
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_dict_int_dict_str_any  (requires instructor_answer and keys)
# ---------------------------------------------------------------------------

def test_check_structure_dict_int_dict_str_any_pass():
    status, msg = check_structure_dict_int_dict_str_any(
        {1: {"a": 1}}, {1: {"a": 1}}
    )
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_dict_int_dict_str_any_fail():
    status, msg = check_structure_dict_int_dict_str_any(
        {1: "x"}, {1: {"a": 1}}
    )
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_dict_tuple_int_ndarray  (requires instructor_answer)
# ---------------------------------------------------------------------------

def test_check_structure_dict_tuple_int_ndarray_pass():
    status, msg = check_structure_dict_tuple_int_ndarray(
        {(0, 1): np.array([1.0])},
        {(0, 1): np.array([1.0])},
    )
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_dict_tuple_int_ndarray_fail():
    status, msg = check_structure_dict_tuple_int_ndarray(
        {(0, 1): [1.0]},  # list, not ndarray
        {(0, 1): np.array([1.0])},
    )
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_dict_any
# ---------------------------------------------------------------------------

def test_check_structure_dict_any_pass():
    status, msg = check_structure_dict_any({"a": 1}, {"a": 1})
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_dict_any_fail():
    status, msg = check_structure_dict_any({}, {"a": 1})  # missing key
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_eval_float
# ---------------------------------------------------------------------------

def test_check_structure_eval_float_pass():
    status, msg = check_structure_eval_float("3.14 + 1.0")  # valid Python expression
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_eval_float_fail():
    status, msg = check_structure_eval_float(42)  # not a string
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_explain_str
# ---------------------------------------------------------------------------

def test_check_structure_explain_str_pass():
    # Function requires more than 5 words
    status, msg = check_structure_explain_str(
        "this is a longer explanation with more than ten words total"
    )
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_explain_str_fail():
    status, msg = check_structure_explain_str(42)
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_function
# ---------------------------------------------------------------------------

def test_check_structure_function_pass():
    status, msg = check_structure_function(lambda x: x)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_function_fail():
    status, msg = check_structure_function("not a function")
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_dendrogram
# ---------------------------------------------------------------------------

def test_check_structure_dendrogram_pass():
    dendro = {
        'icoord': [[0, 0, 1, 1]],
        'dcoord': [[0, 1, 1, 0]],
        'ivl': ['a', 'b'],
        'color_list': ['b'],
        'leaves': [0, 1],
    }
    status, msg = check_structure_dendrogram(dendro)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_dendrogram_fail():
    status, msg = check_structure_dendrogram({})  # empty dict, missing all required keys
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_lineplot (fixture-based)
# ---------------------------------------------------------------------------

def test_check_structure_lineplot_pass():
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    # Function requires x/y labels and title on the plot
    fig, ax = plt.subplots()
    lines = ax.plot([1, 2, 3], [4, 5, 6])
    ax.set_xlabel("x label")
    ax.set_ylabel("y label")
    ax.set_title("title")
    line = lines[0]
    status, msg = check_structure_lineplot(line)
    plt.close(fig)
    assert status is True
    assert isinstance(msg, str)  # may be empty (current behavior)


def test_check_structure_lineplot_fail():
    status, msg = check_structure_lineplot("not a line")
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_scatterplot2d (fixture-based)
# ---------------------------------------------------------------------------

def test_check_structure_scatterplot2d_pass(scatter2d_fixture):
    status, msg = check_structure_scatterplot2d(scatter2d_fixture)
    assert status is True
    assert isinstance(msg, str)  # may be empty (current behavior)


def test_check_structure_scatterplot2d_fail():
    # Function has a bug: raises AttributeError for non-PathCollection input
    # (does not return early after type check). This test documents that behavior.
    import pytest
    with pytest.raises(AttributeError):
        check_structure_scatterplot2d("not a scatter")


# ---------------------------------------------------------------------------
# check_structure_scatterplot3d (fixture-based)
# ---------------------------------------------------------------------------

def test_check_structure_scatterplot3d_pass(scatter3d_fixture):
    status, msg = check_structure_scatterplot3d(scatter3d_fixture)
    assert status is True
    assert isinstance(msg, str)  # may be empty (current behavior)


def test_check_structure_scatterplot3d_fail():
    # Function has same bug as scatterplot2d: raises AttributeError for wrong type
    import pytest
    with pytest.raises((AttributeError, TypeError)):
        check_structure_scatterplot3d("not a scatter")


# ---------------------------------------------------------------------------
# check_structure_kfold (fixture-based)
# ---------------------------------------------------------------------------

def test_check_structure_kfold_pass(fitted_kfold):
    status, msg = check_structure_kfold(fitted_kfold)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_kfold_fail():
    status, msg = check_structure_kfold("not kfold")
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_stratifiedkfold (fixture-based)
# ---------------------------------------------------------------------------

def test_check_structure_stratifiedkfold_pass(fitted_stratifiedkfold):
    status, msg = check_structure_stratifiedkfold(fitted_stratifiedkfold)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_stratifiedkfold_fail():
    status, msg = check_structure_stratifiedkfold("not stratifiedkfold")
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_shufflesplit (fixture-based)
# ---------------------------------------------------------------------------

def test_check_structure_shufflesplit_pass(fitted_shufflesplit):
    status, msg = check_structure_shufflesplit(fitted_shufflesplit)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_shufflesplit_fail():
    status, msg = check_structure_shufflesplit("not shufflesplit")
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_svc (fixture-based)
# ---------------------------------------------------------------------------

def test_check_structure_svc_pass(fitted_svc):
    status, msg = check_structure_svc(fitted_svc)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_svc_fail():
    status, msg = check_structure_svc("not svc")
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_gridsearchcv (fixture-based)
# ---------------------------------------------------------------------------

def test_check_structure_gridsearchcv_pass(fitted_gridsearchcv):
    status, msg = check_structure_gridsearchcv(fitted_gridsearchcv)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_gridsearchcv_fail():
    status, msg = check_structure_gridsearchcv("not gridsearchcv")
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_randomforestclassifier (fixture-based)
# ---------------------------------------------------------------------------

def test_check_structure_randomforestclassifier_pass(fitted_randomforestclassifier):
    status, msg = check_structure_randomforestclassifier(fitted_randomforestclassifier)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_randomforestclassifier_fail():
    status, msg = check_structure_randomforestclassifier("not rfc")
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_decisiontreeclassifier (fixture-based)
# ---------------------------------------------------------------------------

def test_check_structure_decisiontreeclassifier_pass(fitted_decisiontreeclassifier):
    status, msg = check_structure_decisiontreeclassifier(fitted_decisiontreeclassifier)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_decisiontreeclassifier_fail():
    status, msg = check_structure_decisiontreeclassifier("not dtc")
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_structure_logisticregression (fixture-based)
# ---------------------------------------------------------------------------

def test_check_structure_logisticregression_pass(fitted_logisticregression):
    status, msg = check_structure_logisticregression(fitted_logisticregression)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_logisticregression_fail():
    status, msg = check_structure_logisticregression("not lr")
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ===========================================================================
# check_answer_* tests (88 functions = 44 functions x 2 tests each)
# ===========================================================================

from pytest_generator.assert_utilities import (
    check_answer_bool,
    check_answer_float,
    check_answer_int,
    check_answer_str,
    check_answer_ndarray,
    check_answer_list_float,
    check_answer_list_int,
    check_answer_list_str,
    check_answer_list_ndarray,
    check_answer_list_list_float,
    check_answer_list_tuple_float,
    check_answer_list_set,
    check_answer_set_str,
    check_answer_set_set_int,
    check_answer_set_tuple_int,
    check_answer_dict_str_int,
    check_answer_dict_str_float,
    check_answer_dict_str_list_str,
    check_answer_dict_str_list_int,
    check_answer_dict_str_ndarray,
    check_answer_dict_str_set_int,
    check_answer_dict_str_tuple_ndarray,
    check_answer_dict_str_dict_str_float,
    check_answer_dict_str_any,
    check_answer_dict_int_float,
    check_answer_dict_int_list_float,
    check_answer_dict_int_ndarray,
    check_answer_dict_int_dict_str_any,
    check_answer_dict_tuple_int_ndarray,
    check_answer_eval_float,
    check_answer_explain_str,
    check_answer_function,
    check_answer_dendrogram,
    check_answer_lineplot,
    check_answer_scatterplot2d,
    check_answer_scatterplot3d,
    check_answer_kfold,
    check_answer_stratifiedkfold,
    check_answer_shufflesplit,
    check_answer_svc,
    check_answer_gridsearchcv,
    check_answer_randomforestclassifier,
    check_answer_decisiontreeclassifier,
    check_answer_logisticregression,
)


# ---------------------------------------------------------------------------
# check_answer_bool
# ---------------------------------------------------------------------------

def test_check_answer_bool_pass():
    status, msg = check_answer_bool(True, True)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_bool_fail():
    status, msg = check_answer_bool(True, False)
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_float
# ---------------------------------------------------------------------------

def test_check_answer_float_pass():
    status, msg = check_answer_float(1.0, 1.0, 1e-5, 1e-8)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_float_fail():
    status, msg = check_answer_float(999.0, 1.0, 1e-5, 1e-8)
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_int
# ---------------------------------------------------------------------------

def test_check_answer_int_pass():
    status, msg = check_answer_int(42, 42)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_int_fail():
    status, msg = check_answer_int(1, 42)
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_str
# ---------------------------------------------------------------------------

def test_check_answer_str_pass():
    status, msg = check_answer_str("hello", "hello", ["hello", "world"], False)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_str_fail():
    # student answer "zzz" is not in choices ["hello", "world"] -> fail
    status, msg = check_answer_str("zzz", "world", ["hello", "world"], False)
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_ndarray
# ---------------------------------------------------------------------------

def test_check_answer_ndarray_pass():
    status, msg = check_answer_ndarray(np.array([1.0, 2.0]), np.array([1.0, 2.0]), 1e-5)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_ndarray_fail():
    status, msg = check_answer_ndarray(np.array([9.0, 9.0]), np.array([1.0, 2.0]), 1e-5)
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_list_float
# ---------------------------------------------------------------------------

def test_check_answer_list_float_pass():
    status, msg = check_answer_list_float([1.0, 2.0], [1.0, 2.0], 1e-5, partial_score_frac=[0.0])
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_list_float_fail():
    status, msg = check_answer_list_float([9.0, 9.0], [1.0, 2.0], 1e-5, partial_score_frac=[0.0])
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_list_int
# ---------------------------------------------------------------------------

def test_check_answer_list_int_pass():
    status, msg = check_answer_list_int([1, 2], [1, 2], [0.0])
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_list_int_fail():
    status, msg = check_answer_list_int([9, 9], [1, 2], [0.0])
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_list_str
# ---------------------------------------------------------------------------

def test_check_answer_list_str_pass():
    status, msg = check_answer_list_str(["a"], ["a"], None, None, [0.0])
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_list_str_fail():
    status, msg = check_answer_list_str(["x"], ["a"], None, None, [0.0])
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_list_ndarray
# ---------------------------------------------------------------------------

def test_check_answer_list_ndarray_pass():
    status, msg = check_answer_list_ndarray(
        [np.array([1.0])], [np.array([1.0])], 1e-5, [0.0]
    )
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_list_ndarray_fail():
    status, msg = check_answer_list_ndarray(
        [np.array([9.0])], [np.array([1.0])], 1e-5, [0.0]
    )
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_list_list_float
# ---------------------------------------------------------------------------

def test_check_answer_list_list_float_pass():
    status, msg = check_answer_list_list_float([[1.0, 2.0]], [[1.0, 2.0]], 1e-5, [0.0])
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_list_list_float_fail():
    status, msg = check_answer_list_list_float([[9.0, 0.0]], [[1.0, 2.0]], 1e-5, [0.0])
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_list_tuple_float
# ---------------------------------------------------------------------------

def test_check_answer_list_tuple_float_pass():
    status, msg = check_answer_list_tuple_float([(1.0, 2.0)], [(1.0, 2.0)], 1e-5)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_list_tuple_float_fail():
    status, msg = check_answer_list_tuple_float([(9.0,)], [(1.0, 2.0)], 1e-5)
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_list_set
# ---------------------------------------------------------------------------

def test_check_answer_list_set_pass():
    status, msg = check_answer_list_set([{1, 2}], [{1, 2}])
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_list_set_fail():
    status, msg = check_answer_list_set([{9}], [{1, 2}])
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_set_str
# ---------------------------------------------------------------------------

def test_check_answer_set_str_pass():
    status, msg = check_answer_set_str({"a"}, {"a"})
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_set_str_fail():
    status, msg = check_answer_set_str({"x"}, {"a"})
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_set_set_int
# ---------------------------------------------------------------------------

def test_check_answer_set_set_int_pass():
    status, msg = check_answer_set_set_int({frozenset({1, 2})}, {frozenset({1, 2})})
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_set_set_int_fail():
    status, msg = check_answer_set_set_int({frozenset({9})}, {frozenset({1, 2})})
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_set_tuple_int
# ---------------------------------------------------------------------------

def test_check_answer_set_tuple_int_pass():
    status, msg = check_answer_set_tuple_int({(1, 2)}, {(1, 2)})
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_set_tuple_int_fail():
    status, msg = check_answer_set_tuple_int({(9,)}, {(1, 2)})
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_dict_str_int
# ---------------------------------------------------------------------------

def test_check_answer_dict_str_int_pass():
    status, msg = check_answer_dict_str_int({"a": 1}, {"a": 1})
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_dict_str_int_fail():
    # NOTE: check_answer_dict_str_int has a pre-existing bug (keys logic always
    # results in empty key list), so it always returns True. Documented as
    # regression baseline — do not change this assertion without fixing the source.
    status, msg = check_answer_dict_str_int({"a": 9}, {"a": 1})
    assert status is True  # current behavior: always passes due to bug
    assert isinstance(msg, str)


# ---------------------------------------------------------------------------
# check_answer_dict_str_float
# ---------------------------------------------------------------------------

def test_check_answer_dict_str_float_pass():
    status, msg = check_answer_dict_str_float({"a": 1.0}, {"a": 1.0}, 1e-5)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_dict_str_float_fail():
    status, msg = check_answer_dict_str_float({"a": 9.0}, {"a": 1.0}, 1e-5)
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_dict_str_list_str
# ---------------------------------------------------------------------------

def test_check_answer_dict_str_list_str_pass():
    status, msg = check_answer_dict_str_list_str({"a": ["x"]}, {"a": ["x"]})
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_dict_str_list_str_fail():
    status, msg = check_answer_dict_str_list_str({"a": ["z"]}, {"a": ["x"]})
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_dict_str_list_int
# ---------------------------------------------------------------------------

def test_check_answer_dict_str_list_int_pass():
    status, msg = check_answer_dict_str_list_int({"a": [1, 2]}, {"a": [1, 2]}, [0.0])
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_dict_str_list_int_fail():
    status, msg = check_answer_dict_str_list_int({"a": [9, 8]}, {"a": [1, 2]}, [0.0])
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_dict_str_ndarray
# ---------------------------------------------------------------------------

def test_check_answer_dict_str_ndarray_pass():
    status, msg = check_answer_dict_str_ndarray(
        {"a": np.array([1.0])}, {"a": np.array([1.0])}, 1e-5
    )
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_dict_str_ndarray_fail():
    status, msg = check_answer_dict_str_ndarray(
        {"a": np.array([9.0])}, {"a": np.array([1.0])}, 1e-5
    )
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_dict_str_set_int
# ---------------------------------------------------------------------------

def test_check_answer_dict_str_set_int_pass():
    status, msg = check_answer_dict_str_set_int({"a": {1, 2}}, {"a": {1, 2}})
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_dict_str_set_int_fail():
    status, msg = check_answer_dict_str_set_int({"a": {9}}, {"a": {1, 2}})
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_dict_str_tuple_ndarray
# ---------------------------------------------------------------------------

def test_check_answer_dict_str_tuple_ndarray_pass():
    status, msg = check_answer_dict_str_tuple_ndarray(
        {"a": (np.array([1.0]),)}, {"a": (np.array([1.0]),)}, 1e-5, [0.0]
    )
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_dict_str_tuple_ndarray_fail():
    status, msg = check_answer_dict_str_tuple_ndarray(
        {"a": (np.array([9.0]),)}, {"a": (np.array([1.0]),)}, 1e-5, [0.0]
    )
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_dict_str_dict_str_float
# ---------------------------------------------------------------------------

def test_check_answer_dict_str_dict_str_float_pass():
    # NOTE: check_answer_dict_str_dict_str_float has a pre-existing ZeroDivisionError bug
    # when the ps_dict nb_total remains 0. Document as regression baseline.
    with pytest.raises(ZeroDivisionError):
        check_answer_dict_str_dict_str_float(
            {"a": {"b": 1.0}}, {"a": {"b": 1.0}}, 1e-5, {}, [0.0]
        )


def test_check_answer_dict_str_dict_str_float_fail():
    # Same bug — raises ZeroDivisionError
    with pytest.raises(ZeroDivisionError):
        check_answer_dict_str_dict_str_float(
            {"a": {"b": 9.0}}, {"a": {"b": 1.0}}, 1e-5, {}, [0.0]
        )


# ---------------------------------------------------------------------------
# check_answer_dict_str_any
# ---------------------------------------------------------------------------

def test_check_answer_dict_str_any_pass():
    status, msg = check_answer_dict_str_any({"a": 1}, {"a": 1})
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_dict_str_any_fail():
    status, msg = check_answer_dict_str_any({"a": 9}, {"a": 1})
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_dict_int_float
# ---------------------------------------------------------------------------

def test_check_answer_dict_int_float_pass():
    status, msg = check_answer_dict_int_float({1: 1.0}, {1: 1.0}, 1e-5, [0.0])
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_dict_int_float_fail():
    status, msg = check_answer_dict_int_float({1: 9.0}, {1: 1.0}, 1e-5, [0.0])
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_dict_int_list_float
# ---------------------------------------------------------------------------

def test_check_answer_dict_int_list_float_pass():
    status, msg = check_answer_dict_int_list_float({1: [1.0]}, {1: [1.0]}, None, 1e-5, [0.0])
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_dict_int_list_float_fail():
    status, msg = check_answer_dict_int_list_float({1: [9.0]}, {1: [1.0]}, None, 1e-5, [0.0])
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_dict_int_ndarray
# ---------------------------------------------------------------------------

def test_check_answer_dict_int_ndarray_pass():
    status, msg = check_answer_dict_int_ndarray({1: np.array([1.0])}, {1: np.array([1.0])}, 1e-5, None)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_dict_int_ndarray_fail():
    status, msg = check_answer_dict_int_ndarray({1: np.array([9.0])}, {1: np.array([1.0])}, 1e-5, None)
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_dict_int_dict_str_any
# ---------------------------------------------------------------------------

def test_check_answer_dict_int_dict_str_any_pass():
    status, msg = check_answer_dict_int_dict_str_any({1: {"a": 1}}, {1: {"a": 1}})
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_dict_int_dict_str_any_fail():
    status, msg = check_answer_dict_int_dict_str_any({1: {"a": 9}}, {1: {"a": 1}})
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_dict_tuple_int_ndarray
# ---------------------------------------------------------------------------

def test_check_answer_dict_tuple_int_ndarray_pass():
    # NOTE: pre-existing bug — ps_dict not initialized, causes ZeroDivisionError
    with pytest.raises((ZeroDivisionError, NameError)):
        check_answer_dict_tuple_int_ndarray(
            {(0, 1): np.array([1.0])}, {(0, 1): np.array([1.0])}, 1e-5
        )


def test_check_answer_dict_tuple_int_ndarray_fail():
    # Same pre-existing bug
    with pytest.raises((ZeroDivisionError, NameError)):
        check_answer_dict_tuple_int_ndarray(
            {(0, 1): np.array([9.0])}, {(0, 1): np.array([1.0])}, 1e-5
        )


# ---------------------------------------------------------------------------
# check_answer_eval_float
# ---------------------------------------------------------------------------

def test_check_answer_eval_float_pass():
    # Both expressions evaluate to the same value for any x in [1.0, 2.0]
    status, msg = check_answer_eval_float(
        "x * 2.0", "x * 2.0", {"x": (1.0, 2.0)}, 1e-5
    )
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_eval_float_fail():
    # Different expressions — student returns x+999, instructor x*2
    status, msg = check_answer_eval_float(
        "x + 999.0", "x * 2.0", {"x": (1.0, 2.0)}, 1e-5
    )
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_explain_str  (always returns True — no grading)
# ---------------------------------------------------------------------------

def test_check_answer_explain_str_pass():
    status, msg = check_answer_explain_str("reasonable answer", "reasonable answer")
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_explain_str_fail():
    # Function always returns True (no grading); document as regression baseline
    status, msg = check_answer_explain_str("", "some expected text")
    assert status is True  # current behavior: always passes
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_function  (always returns True — not graded)
# ---------------------------------------------------------------------------

def test_check_answer_function_pass():
    status, msg = check_answer_function(lambda x: x, lambda x: x)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_function_fail():
    # Function always returns True (not graded); document as regression baseline
    status, msg = check_answer_function(lambda x: x + 1, lambda x: x)
    assert status is True  # current behavior: always passes
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_dendrogram
# ---------------------------------------------------------------------------

def _make_dendro(icoord, dcoord, leaves):
    return {
        'icoord': icoord,
        'dcoord': dcoord,
        'ivl': ['a', 'b'],
        'color_list': ['b'],
        'leaves': leaves,
    }


def test_check_answer_dendrogram_pass():
    dendro = _make_dendro([[0, 0, 1, 1]], [[0, 1, 1, 0]], [0, 1])
    status, msg = check_answer_dendrogram(dendro, dendro, 1e-5)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_dendrogram_fail():
    student = _make_dendro([[0, 0, 1, 1]], [[0, 1, 1, 0]], [0, 1])
    instructor = _make_dendro([[5, 5, 9, 9]], [[0, 3, 3, 0]], [1, 0])
    status, msg = check_answer_dendrogram(student, instructor, 1e-5)
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_lineplot  (fixture-based)
# ---------------------------------------------------------------------------

def test_check_answer_lineplot_pass(line2d_fixture):
    # check_answer_lineplot always returns True (no actual comparison implemented)
    status, msg = check_answer_lineplot(line2d_fixture, line2d_fixture, 1e-5)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_lineplot_fail(line2d_fixture):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    other_lines = ax.plot([10, 20, 30], [40, 50, 60])
    other_line = other_lines[0]
    # Function always returns True (no comparison implemented); document baseline
    status, msg = check_answer_lineplot(line2d_fixture, other_line, 1e-5)
    assert status is True  # current behavior: always passes
    assert isinstance(msg, str) and len(msg) > 0
    plt.close(fig)


# ---------------------------------------------------------------------------
# check_answer_scatterplot2d  (fixture-based)
# ---------------------------------------------------------------------------

def test_check_answer_scatterplot2d_pass(scatter2d_fixture):
    # NOTE: check_answer_scatterplot2d has a pre-existing NameError (mcolors not imported)
    # Document as regression baseline
    with pytest.raises(NameError):
        check_answer_scatterplot2d(scatter2d_fixture, scatter2d_fixture, {}, [])


def test_check_answer_scatterplot2d_fail(scatter2d_fixture):
    # Same pre-existing NameError
    with pytest.raises(NameError):
        check_answer_scatterplot2d(scatter2d_fixture, scatter2d_fixture, {}, [])


# ---------------------------------------------------------------------------
# check_answer_scatterplot3d  (fixture-based)
# ---------------------------------------------------------------------------

def test_check_answer_scatterplot3d_pass(scatter3d_fixture):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("title")
    col = ax.scatter3D([1, 2], [3, 4], [5, 6])
    status, msg = check_answer_scatterplot3d(col, col, {}, [])
    assert isinstance(status, bool)
    assert isinstance(msg, str)
    plt.close(fig)


def test_check_answer_scatterplot3d_fail(scatter3d_fixture):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from mpl_toolkits.mplot3d import Axes3D
    # Create a 3d scatter without labels/title
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    col = ax.scatter3D([1, 2], [3, 4], [5, 6])
    status, msg = check_answer_scatterplot3d(col, col, {}, [])
    assert isinstance(status, bool)
    assert isinstance(msg, str)
    plt.close(fig)


# ---------------------------------------------------------------------------
# check_answer_kfold  (fixture-based)
# ---------------------------------------------------------------------------

def test_check_answer_kfold_pass(fitted_kfold):
    status, msg = check_answer_kfold(fitted_kfold, fitted_kfold)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_kfold_fail(fitted_kfold):
    from sklearn.model_selection import KFold
    other = KFold(n_splits=3)
    status, msg = check_answer_kfold(fitted_kfold, other)
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_stratifiedkfold  (fixture-based)
# ---------------------------------------------------------------------------

def test_check_answer_stratifiedkfold_pass(fitted_stratifiedkfold):
    status, msg = check_answer_stratifiedkfold(fitted_stratifiedkfold, fitted_stratifiedkfold)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_stratifiedkfold_fail(fitted_stratifiedkfold):
    from sklearn.model_selection import StratifiedKFold
    other = StratifiedKFold(n_splits=3)
    status, msg = check_answer_stratifiedkfold(fitted_stratifiedkfold, other)
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_shufflesplit  (fixture-based)
# ---------------------------------------------------------------------------

def test_check_answer_shufflesplit_pass(fitted_shufflesplit):
    status, msg = check_answer_shufflesplit(fitted_shufflesplit, fitted_shufflesplit)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_shufflesplit_fail(fitted_shufflesplit):
    from sklearn.model_selection import ShuffleSplit
    other = ShuffleSplit(n_splits=3)
    status, msg = check_answer_shufflesplit(fitted_shufflesplit, other)
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_svc  (fixture-based)
# ---------------------------------------------------------------------------

def test_check_answer_svc_pass(fitted_svc):
    status, msg = check_answer_svc(fitted_svc, fitted_svc)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_svc_fail(fitted_svc):
    from sklearn.svm import SVC
    other = SVC(C=999)
    other.fit([[0, 0], [1, 1]], [0, 1])
    status, msg = check_answer_svc(fitted_svc, other)
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_gridsearchcv  (fixture-based)
# ---------------------------------------------------------------------------

def test_check_answer_gridsearchcv_pass(fitted_gridsearchcv):
    status, msg = check_answer_gridsearchcv(fitted_gridsearchcv, fitted_gridsearchcv)
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_gridsearchcv_fail(fitted_gridsearchcv):
    from sklearn.svm import SVC
    from sklearn.model_selection import GridSearchCV
    # Use different param_grid to force mismatch
    other = GridSearchCV(SVC(), {'C': [1, 100]}, cv=2)
    other.fit([[0, 0], [1, 1], [2, 2], [3, 3]], [0, 1, 0, 1])
    status, msg = check_answer_gridsearchcv(fitted_gridsearchcv, other)
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_randomforestclassifier  (fixture-based)
# ---------------------------------------------------------------------------

def test_check_answer_randomforestclassifier_pass(fitted_randomforestclassifier):
    status, msg = check_answer_randomforestclassifier(
        fitted_randomforestclassifier, fitted_randomforestclassifier
    )
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_randomforestclassifier_fail(fitted_randomforestclassifier):
    from sklearn.ensemble import RandomForestClassifier
    other = RandomForestClassifier(n_estimators=100, random_state=0)
    other.fit([[0, 0], [1, 1], [2, 2], [3, 3]], [0, 1, 0, 1])
    status, msg = check_answer_randomforestclassifier(fitted_randomforestclassifier, other)
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_decisiontreeclassifier  (fixture-based)
# ---------------------------------------------------------------------------

def test_check_answer_decisiontreeclassifier_pass(fitted_decisiontreeclassifier):
    status, msg = check_answer_decisiontreeclassifier(
        fitted_decisiontreeclassifier, fitted_decisiontreeclassifier
    )
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_decisiontreeclassifier_fail(fitted_decisiontreeclassifier):
    from sklearn.tree import DecisionTreeClassifier
    other = DecisionTreeClassifier(max_depth=1)
    other.fit([[0, 0], [1, 1], [2, 2], [3, 3]], [0, 1, 0, 1])
    status, msg = check_answer_decisiontreeclassifier(fitted_decisiontreeclassifier, other)
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ---------------------------------------------------------------------------
# check_answer_logisticregression  (fixture-based)
# ---------------------------------------------------------------------------

def test_check_answer_logisticregression_pass(fitted_logisticregression):
    status, msg = check_answer_logisticregression(
        fitted_logisticregression, fitted_logisticregression
    )
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_answer_logisticregression_fail(fitted_logisticregression):
    from sklearn.linear_model import LogisticRegression
    other = LogisticRegression(C=999)
    other.fit([[0, 0], [1, 1], [2, 2], [3, 3]], [0, 1, 0, 1])
    status, msg = check_answer_logisticregression(fitted_logisticregression, other)
    assert status is False
    assert isinstance(msg, str) and len(msg) > 0


# ===========================================================================
# AST-based inventory completeness test
# ===========================================================================

import ast
import pathlib
import sys


def test_inventory_completeness():
    src_path = pathlib.Path("src/pytest_generator/assert_utilities.py")
    src = src_path.read_text()
    tree = ast.parse(src)

    defined = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef):
            n = node.name
            if n.startswith("check_structure_") or n.startswith("check_answer_"):
                defined.add(n)

    # Collect tested names from this module's test function names
    this_module = sys.modules[__name__]
    test_names = [name for name in dir(this_module) if name.startswith("test_check_")]
    tested = set()
    for tname in test_names:
        # test_check_structure_float_pass -> check_structure_float
        # test_check_answer_float_pass -> check_answer_float
        for prefix in ("check_structure_", "check_answer_"):
            if tname.startswith("test_" + prefix):
                # strip test_ prefix and _pass/_fail suffix
                body = tname[len("test_"):]  # e.g. check_structure_float_pass
                # strip trailing _pass or _fail
                if body.endswith("_pass") or body.endswith("_fail"):
                    body = body.rsplit("_", 1)[0]
                tested.add(body)

    missing = defined - tested
    assert not missing, (
        f"The following checker functions have no regression test: {sorted(missing)}\n"
        f"Add tests for each function to maintain full coverage."
    )
