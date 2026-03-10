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
    # NOTE: this function has inverted logic; it passes when value is NOT a set/list
    # Testing actual current behavior (regression baseline)
    status, msg = check_structure_dict_str_set({"a": 42}, {"a": {1, 2}})
    assert status is True
    assert isinstance(msg, str) and len(msg) > 0


def test_check_structure_dict_str_set_fail():
    # Function incorrectly fails when value IS a set (inverted logic bug)
    status, msg = check_structure_dict_str_set(
        {}, {"a": {1, 2}}  # missing key triggers False
    )
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
