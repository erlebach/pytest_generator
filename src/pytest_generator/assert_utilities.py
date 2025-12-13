"""Utilities for generating and running pytest assertions.

This module provides functions and utilities for generating pytest assertions
and validating student answers against instructor solutions. It includes tools
for checking numerical equality, structural matching of plots and figures,
and partial credit scoring.

Author: Gordon Erlebacher
Year: 2024-2025

Use Python 3.10 or above
"""

import ast
import inspect
import math
import random
import re
from collections.abc import Callable
from pathlib import Path
from pprint import pprint
from typing import Any, cast

# ! import matplotlib.pyplot as plt
import numpy as np
import yaml

# Used for typechecking arguments
from matplotlib.lines import Line2D
from matplotlib.figure import Figure

# ! from matplotlib.collections import PathCollection
from numpy.typing import NDArray

"""
# from pprint import pprint
# import matplotlib.pyplot as plt
"""

FLOAT_TOL = 1.0e-5


def init_partial_score_dict() -> dict[str, float | int]:
    """Initialize a partial score dictionary with default values.

    Returns:
        dict[str, float | int]: A dictionary with keys for mismatches, total
            items, and partial score fraction.

    """
    return {"nb_mismatches": 0, "nb_total": 0, "partial_score_frac": 0}


# ----------------------------------------------------------------------


def check_missing_keys(
    missing_keys: list[str],
    msg_list: list[str],
) -> tuple[bool, list[str]]:
    """Check if there are any missing keys and update message list accordingly.

    Args:
        missing_keys (list[str]): List of keys that are missing
        msg_list (list[str]): List of messages to append status to

    Returns:
        tuple[bool, list[str]]: A tuple containing:
            - bool: True if no missing keys, False otherwise
            - list[str]: Updated message list with status of missing keys check

    """
    if len(missing_keys) > 0:
        status = False
        msg_list.append(f"- Missing keys: {[repr(k) for k in missing_keys]}.")
    else:
        status = True
        msg_list.append("- No missing keys")
    return status, msg_list


# ----------------------------------------------------------------------
def check_float(
    i_el: float,
    s_el: float,
    rel_tol: float = 1.0e-2,
    abs_tol: float = 1.0e-5,
) -> tuple[bool, str]:
    """Check if the student float is within the instructor float's tolerance.

    Args:
        i_el (float): Instructor float
        s_el (float): Student float
        rel_tol (float): Relative tolerance (default: 1.0e-2)
        abs_tol (float): Absolute tolerance (default: 1.0e-5)

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if within tolerance, False otherwise
            - str: Message detailing the result of the check

    """
    status = True
    msg = ""

    print("===> check_float <====")
    # print(f"{i_el=}, {s_el=}")
    # print(f"{rel_tol=}, {abs_tol=}")
    if math.fabs(i_el) <= abs_tol:
        abs_err = math.fabs(i_el - s_el)
        print(f"{abs_err=}")
        status = abs_err < FLOAT_TOL
    elif math.fabs((i_el - s_el) / i_el) < rel_tol:
        status = True
    else:
        status = False
        msg = f"Student element {s_el} has rel error > {100 * rel_tol}% "
        msg += f"relative to instructor element {i_el}"
    return status, msg


# ----------------------------------------------------------------------


def check_int(i_el: int, s_el: int) -> tuple[bool, str]:
    """Check if the student integer matches the instructor integer.

    Args:
        i_el (int): Instructor integer
        s_el (int): Student integer

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if the integers match, False otherwise
            - str: Message detailing the result of the check

    """
    status = True
    msg = ""
    if i_el != s_el:
        status = False
        msg = f"Student element {s_el} != instructor element {i_el}"
    return status, msg


# ----------------------------------------------------------------------


def check_list_float(
    i_arr: list[float],
    s_arr: list[float],
    rel_tol: float,
    abs_tol: float,
    ps_dict: dict[str, float | int],
) -> tuple[bool, str]:
    """Check if the student list of floats matches the instructor list of floats.

    Args:
        i_arr (list[float]): Instructor list of floats
        s_arr (list[float]): Student list of floats
        rel_tol (float): Relative tolerance (default: 1.0e-2)
        abs_tol (float): Absolute tolerance (default: 1.0e-5)
        ps_dict (dict[str, float | int]): Partial score dictionary

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if the lists match, False otherwise
            - str: Message detailing the result of the check

    """
    msg_list = []
    status = True
    ps_dict["nb_total"] += len(i_arr)

    for i_el, s_el in zip(i_arr, s_arr, strict=True):
        status_, msg_ = check_float(i_el, s_el, rel_tol=rel_tol, abs_tol=abs_tol)
        if status_ is False:
            status = False
            msg_list.append(msg_)
            ps_dict["nb_mismatches"] += 1
    return status, "\n".join(msg_list)


# ----------------------------------------------------------------------


def check_list_int(
    i_arr: list[int],
    s_arr: list[int],
    ps_dict: dict[str, float | int],
) -> tuple[bool, str]:
    """Check if the student list of integers matches the instructor list of integers.

    Args:
        i_arr (list[int]): Instructor list of integers
        s_arr (list[int]): Student list of integers
        ps_dict (dict[str, float | int]): Partial score dictionary

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if the lists match, False otherwise
            - str: Message detailing the result of the check

    """
    msg_list = []
    status = True
    ps_dict["nb_total"] += len(i_arr)
    print(f"==> inside check_list_int, {ps_dict=}")

    for i_el, s_el in zip(i_arr, s_arr, strict=True):
        status_, msg_ = check_int(i_el, s_el)
        if status_ is False:
            status = False
            msg_list.append(msg_)
            ps_dict["nb_mismatches"] += 1
    return status, "\n".join(msg_list)


# ----------------------------------------------------------------------


def check_set_int(
    i_set: set[int],
    s_set: set[int],
    ps_dict: dict[str, float | int],
) -> tuple[bool, str]:
    """Check if the student set of integers matches the instructor set of integers.

    Args:
        i_set (set[int]): Instructor set of integers
        s_set (set[int]): Student set of integers
        ps_dict (dict[str, float | int]): Partial score dictionary

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if the sets match, False otherwise
            - str: Message detailing the result of the check

    """
    msg_list: list[str] = []
    ps_dict["nb_total"] += len(i_set)

    # Elements in the instructor set not in the student set
    el_unique_to_i = i_set - s_set
    el_unique_to_s = s_set - i_set

    # Partial credits for sets
    # If the student has less elements than the teacher, subtrace the difference
    nb_deficit = len(el_unique_to_i)  # will be zero if the sets are equal

    # Number of student elements not in the instructor set are errors.
    nb_s_errors = len(el_unique_to_s)

    ps_dict["nb_mismatches"] += nb_deficit + nb_s_errors
    ps_dict["partial_score_frac"] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]

    if nb_deficit + nb_s_errors > 0:
        status = False
        msg_list.append(f"Student has {nb_deficit} fewer elements than the instructor")
        msg_list.append(f"Student has {nb_s_errors} elements not in the instructor set")
    else:
        status = True
        msg_list.append("Student set is correct")

    return status, "\n".join(msg_list)


# ----------------------------------------------------------------------
def check_str(
    i_str: str,
    s_str: str,
    str_choices: list[str] | None = None,
    remove_spaces: bool = False,  # noqa: FBT001, FBT002
) -> tuple[bool, str]:
    """Check math of student string matches the instructor string.

    Args:
        i_str (str): The instructor's string to compare against.
        s_str (str): The student's string to check.
        str_choices (list[str] | None): Optional list of valid string choices for
            the student.
        remove_spaces (bool | None): Optional flag to indicate if spaces should be
            removed from the strings.

    Returns:
        tuple[bool, str]: A tuple containing a boolean status indicating if the strings
        match and a message detailing any mismatch.

    """
    status = True
    msg = ""
    # Allow argument to be int (maybe temporary)
    if isinstance(s_str, int):
        s_str = str(s_str)
    # Allow argument to be int (maybe temporary)
    if isinstance(i_str, int):
        i_str = str(i_str)
    if str_choices is None:
        str_choices = []
    str_choices = [clean_str_answer(s) for s in str_choices]
    i_str = clean_str_answer(i_str)
    s_str = clean_str_answer(s_str)

    if remove_spaces is True:
        i_str = re.sub(r"\s+", "", i_str)
        s_str = re.sub(r"\s+", "", s_str)
        str_choices = [re.sub(r"\s+", "", el) for el in str_choices]

    if s_str in str_choices:
        s_str = i_str

    if i_str != s_str:
        status = False
        msg = f"String element mismatch. Instructor: {i_str}, Student: {s_str}"

    return status, msg


# ----------------------------------------------------------------------
def check_list_str(
    i_list: list[str],
    s_list: list[str],
    ps_dict: dict[str, float | int],
) -> tuple[bool, list[str]]:
    """Check if two lists of strings match element by element.

    Args:
        i_list: List of instructor strings to compare against
        s_list: List of student strings to check
        ps_dict: Dictionary tracking partial scoring statistics

    Returns:
        Status indicating if lists match and list of error messages

    """
    msg_list = []
    status = True
    ps_dict["nb_total"] += len(i_list)

    for i_el, s_el in zip(i_list, s_list, strict=True):
        status_, msg_ = check_str(i_el, s_el)
        if status_ is False:
            status = False
            msg_list.append(msg_)
            ps_dict["nb_mismatches"] += 1
    return status, msg_list


# ----------------------------------------------------------------------
def check_dict_str_str(
    i_dict: dict[str, str],
    s_dict: dict[str, str],
    ps_dict: dict[str, float | int],
) -> tuple[bool, list[str]]:
    """Check if the student dictionary of strings matches that of the instructor.

    Args:
        i_dict (dict[str, str]): Instructor dictionary of strings
        s_dict (dict[str, str]): Student dictionary of strings
        ps_dict (dict[str, float | int]): Partial score dictionary

    Returns:
        tuple[bool, list[str]]: A tuple containing:
            - bool: True if the dictionaries match, False otherwise
            - list[str]: Message detailing the result of the check

    """
    status = True
    msg_list = []
    ps_dict["nb_total"] += len(i_dict)
    for i_key, i_value in i_dict.items():
        status_, msg_ = check_str(i_value, s_dict[i_key])
        if status_ is False:
            status = False
            msg_list.append(msg_)
            ps_dict["nb_mismatches"] += 1
    update_score(ps_dict)
    return status, msg_list


# ----------------------------------------------------------------------
def update_score(ps_dict: dict[str, float | int]) -> None:
    """Update the partial score dictionary.

    Args:
        ps_dict (dict[str, float | int]): Partial score dictionary

    """
    ps_dict["partial_frac_score"] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]


# ----------------------------------------------------------------------


def check_dict_str_float(
    keys: list[str],
    i_dict: dict[str, float],
    s_dict: dict[str, float],
    rel_tol: float,
    abs_tol: float,
    ps_dict: dict[str, float | int],
) -> tuple[bool, list[str]]:
    """Check if the student dictionary of strings matches that of the instructor.

    Args:
        keys (list[str]): List of keys to check
        i_dict (dict[str, float]): Instructor dictionary of floats
        s_dict (dict[str, float]): Student dictionary of floats
        rel_tol (float): Relative tolerance for float comparisons
        abs_tol (float): Absolute tolerance for float comparisons
        ps_dict (dict[str, float | int]): Partial score dictionary

    Returns:
        tuple[bool, list[str]]: A tuple containing:
            - bool: True if the dictionaries match, False otherwise
            - list[str]: Message detailing the result of the check

    """
    msg_list = []
    status = True

    for i_key in keys:
        i_el = i_dict.get(i_key)
        s_el = s_dict.get(i_key)
        if i_el is None or s_el is None:
            continue
        status_, msg_ = check_float(i_el, s_el, rel_tol=rel_tol, abs_tol=abs_tol)
        if status_ is False:
            msg_list.append(msg_)
            status = False
            ps_dict["nb_mismatches"] += 1

    return status, msg_list


# ----------------------------------------------------------------------


# ......................................................................


def clean_str_answer(answer: str) -> str:
    """Clean the answer string.

    Args:
        answer (str): The answer string to clean

    Returns:
        str: The cleaned answer string

    """
    answer = answer.lower().strip()
    # Transform double spaces to single space
    return re.sub(r"\s+", " ", answer)


# ......................................................................


def load_yaml_file(file_path: str) -> dict:
    """Load a YAML file.

    Args:
        file_path (str): The path to the YAML file

    Returns:
        dict: The contents of the YAML file

    """
    path = Path(file_path)
    with path.open("r") as file:
        return yaml.safe_load(file)


def extract_config_dict() -> dict[str, Any]:
    """Extract the configuration dictionary from the YAML file.

    Returns:
        dict: The configuration dictionary

    """
    dct: dict[str, Any] = {}
    config_dict_ = load_yaml_file("generator_config.yaml")
    test_structure = config_dict_.get("test_structure", None)
    dct["max_nb_words"] = test_structure.get("max_nb_words", 10) if test_structure else 10  # noqa: E501
    types = test_structure.get("types", {})
    eval_float = types.get("eval_float", {})
    dct["local_namespaces"] = eval_float.get("local_namespaces")
    return dct


config_dict = extract_config_dict()


# ----------------------------------------------------------------------
def fmt_ifstr(x: object) -> str:
    """Format the input as a string if it is a string, otherwise convert it to a string.

    Args:
        x (Any | float): The input to format

    Returns:
        str: The formatted string

    """
    # ! return repr(x) if isinstance(x, str) else str(x)
    # ! if isinstance(x, str), str(x) == x
    return str(x)


# ----------------------------------------------------------------------
def return_value(
    status: bool,  # noqa: FBT001
    msg_list: list[str],
    s_answ: object,
    i_answ: object,
) -> tuple[bool, str]:
    """Return the status and message list.

    Args:
        status (bool): The status of the check
        msg_list (list[str]): The message list
        s_answ (Any): The student answer
        i_answ (Any): The instructor answer

    Returns:
        tuple[bool, str]: The status and message list

    """
    # if status:
    #     msg_list.append("Answer is correct")
    # else:
    #     msg_list.append("Answer is incorrect.")
    # msg_list.append(f"Instructor answer: {fmt_ifstr(i_answ)}")
    # msg_list.append(f"Student answer: {fmt_ifstr(s_answ)}")

    print(f"*** Instructor answer: {i_answ}")
    print(f"*** Student answer: {s_answ}")

    # return status, "\n".join(msg_list)

    # For debug messages, just convert everything to strings directly
    # since we're only using them for display purposes
    debug_s = str(s_answ)
    debug_i = str(i_answ)

    # Ensure all elements in msg_list are strings
    msg_list = [str(msg) for msg in msg_list]

    return status, "\n".join(
        [
            *msg_list,
            f"return_value: student_answer={debug_s}",
            f"return_value: instructor_answer={debug_i}",
        ],
    )


# ----------------------------------------------------------------------
def are_sets_equal(
    set1: set[float],
    set2: set[float],
    rtol: float = 1e-5,
    atol: float = 1e-6,
) -> bool:
    """Compare two sets of floats for equality within a relative and absolute tolerance.

    Args:
        set1: The first set of floats.
        set2: The second set of floats.
        rtol: The relative tolerance (default: 1e-5).
        atol: The absolute tolerance (default: 1e-8).

    Returns:
        True if the sets are equal within the specified tolerances, False otherwise.

    """
    if len(set1) != len(set2):
        return False  # Sets must have the same length
    for x, y in zip(set1, set2, strict=True):
        if not np.isclose(x, y, rtol=rtol, atol=atol):
            return False
    return True


# ======================================================================
def check_answer_float(
    student_answer: float,
    instructor_answer: float,
    rel_tol: float,
    abs_tol: float,
) -> tuple[bool, str]:
    """Check answer correctness. Assume the structure is correct.

    Args:
        student_answer (float): The student answer
        instructor_answer (float): The instructor answer
        rel_tol (float): The relative tolerance
        abs_tol (float): The absolute tolerance

    Returns:
        tuple[bool, str]: The status and message

    """
    status, msg = check_float(
        instructor_answer,
        student_answer,
        rel_tol=rel_tol,
        abs_tol=abs_tol,
    )

    if not msg:
        msg = "Float value is as expected."

    return return_value(status, [msg], student_answer, instructor_answer)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_float(student_answer: float) -> tuple[bool, str]:
    """Check the structure of the student answer.

    Args:
        student_answer (float): The student answer

    Returns:
        tuple[bool, str]: The status and message

    """
    if isinstance(student_answer, float | np.floating):
        status = True
        msg_list = ["Answer is of type float as expected."]
    else:
        status = False
        msg_list = [
            "Answer should be of type float. It is of type ",
            f"{type(student_answer).__name__!r}",
        ]
    return status, "\n".join(msg_list)


# ======================================================================
def check_answer_eval_float(
    student_answer: str,
    instructor_answer: str,
    local_vars_dict: dict[str, tuple[float, float]],
    rel_tol: float,
) -> tuple[bool, str]:
    """Check the answer correctness.

    Args:
        student_answer (str): The student answer
        instructor_answer (str): The instructor answer
        local_vars_dict (dict[str, tuple[float, float]]): The local variables dictionary
        rel_tol (float): The relative tolerance

    Returns:
        tuple[bool, str]: The status and message

    Notes:
        - local_vars_dict is a dictionary that maps variable names to their allowed
          value ranges (as tuples of lower and upper bounds). These ranges are used
          to randomly generate test values for variables in both the student's and
          instructor's expressions, allowing the function to compare their results
          across multiple random inputs.

    """
    msg_list = []
    status = True
    s_answ = student_answer
    i_answ = instructor_answer
    random_values = {}
    local_dct = {}

    nb_evals = 3
    for _ in range(nb_evals):
        for var, (lower, upper) in local_vars_dict.items():
            local_dct[var] = random.uniform(lower, upper)  # noqa: S311, Non-deterministic
        # Unsafe to use eval. ast.literal_eval does not allow global and locals.
        # SECURITY RISK: eval is used to evaluate the student and instructor answers.
        s_float = eval(s_answ, config_dict["local_namespaces"], local_dct)
        i_float = eval(i_answ, config_dict["local_namespaces"], local_dct)
        status, msg = check_float(i_float, s_float, rel_tol=rel_tol, abs_tol=1.0e-5)
        msg_list.append(msg)

    if not msg_list:
        msg_list = ["Expression provided generates the correct response."]

    return return_value(status, msg_list, s_answ, i_answ)


# ======================================================================
def check_structure_eval_float(student_answer: str) -> tuple[bool, str]:
    """Check the structure of the student answer.

    Args:
        student_answer (str): The student answer

    Returns:
        tuple[bool, str]: The status and message

    """
    if not isinstance(student_answer, str):
        msg = (
            f"Student_answer is {student_answer}. "
            "Should be string defining a valid Python expression."
        )
        return False, msg

    try:
        ast.parse(student_answer, mode="eval")
        return_value = True, "Valid python expression"
    except SyntaxError:
        return_value = False, "Your valid expression is not valid Python"

    return return_value


# ======================================================================


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_dict_str_list_int(
    student_answer: dict[str, list[int]],
    instructor_answer: dict[str, list[int]],
) -> tuple[bool, str]:
    """Check the structure of the student's answer.

    Args:
        student_answer (dict[str, list[int]]): The student's submitted answer, expected to be a dictionary where each key is a string and each value is a list of integers.

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if the structure is correct, False otherwise
            - str: Message explaining the validation result

    """
    status: bool = True
    msg_list: list[str] = []

    print(f"structure, {student_answer=}")
    print(f"structure, {instructor_answer=}")

    if not isinstance(student_answer, dict):
        return False, "Answer must be a dict"

    for k, v in student_answer.items():
        if not isinstance(v, list):
            msg_list.append(f"Element {k} of your list should be of type 'list'")
            status = False
            continue

    # Check that all list elements are ints
    for v in student_answer.values():
        for e in v:
            if not isinstance(e, int | np.integer):
                msg_list.append(f"Element {e} of your list should be of type 'int'")
                msg_list.append("Check all other list elements")
                status = False
                break

    return status, "\n".join(msg_list)


# ----------------------------------------------------------------------


def check_answer_dict_str_list_int(
    student_answer: dict[str, list[int]],
    instructor_answer: dict[str, list[int]],
    partial_score_frac_l: list[float],
) -> tuple[bool, str]:
    """Check if student answer matches instructor answer for dict[str, list[int]] type.

    Compares student and instructor answers that are dictionaries with string keys and
    lists of integers as values. Checks that list elements match exactly.

    Args:
        student_answer (dict[str, list[int]]): Student's submitted answer
        instructor_answer (dict[str, list[int]]): Instructor's reference answer
        partial_score_frac (list[float]): List to store partial credit score fraction

    Returns:
        tuple[bool, str]: Status indicating if answers match and message detailing
            any mismatches

    """
    print(f"==> inside check_answer_dict_str_list_int, {partial_score_frac_l=}", flush=True)
    msg_list = []
    status = True
    ps_dict = init_partial_score_dict()
    ps_dict["nb_total"] = len(instructor_answer)

    # Check each key in instructor answer
    for key, i_list in instructor_answer.items():
        if key not in student_answer:
            status = False
            msg_list.append(f"Missing key: {key!r}")
            # ps_dict["nb_mismatches"] += 1
            continue

        s_list = student_answer[key]
        status_, msg_ = check_list_int(i_list, s_list, ps_dict)
        print("check_list return, ps_dict= ", ps_dict, "\n", flush=True)
        if not status_:
            status = False
            msg_list.extend([f"For key {key!r}:"] + [msg_])

    try:
        partial_score_frac_l[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]
    except ZeroDivisionError:
        partial_score_frac_l[0] = 1.0
    print(f"===check_answer_dict_str_list_int, {partial_score_frac_l=}")
    if not msg_list:
        msg_list = ["Answer matches expected values."]

    return return_value(status, msg_list, student_answer, instructor_answer)


# ----------


def check_structure_dict_str_list_str(
    student_answer: dict[str, list[str]],
    instructor_answer: dict[str, list[str]],
    key_choices: dict[str, list[str]] | None = None,
) -> tuple[bool, str]:
    """Check if student answer matches expected structure of dict[str, list[str]].

    Verifies that:
    1. Student answer is a dictionary
    2. All dictionary keys are strings
    3. All dictionary values are lists of strings
    4. Student answer contains all required keys from instructor answer or their
       acceptable alternatives

    Args:
        student_answer (dict[str, list[str]]): Student's submitted answer to check
        instructor_answer (dict[str, list[str]]): Instructor's reference answer
            defining expected structure
        key_choices (dict[str, list[str]] | None): Dictionary mapping instructor keys
            to lists of acceptable alternative key spellings

    Returns:
        tuple[bool, str]: Status indicating if structure is valid and message detailing
            any validation errors
    """
    status = False
    msg_list = []

    # Check if answer is a dictionary
    if not isinstance(student_answer, dict):
        return False, "Answer must be a dict"

    if key_choices is None:
        key_choices = {}

    # Create mapping from alternative keys to instructor keys
    key_mapping = {}
    for i_key, alternatives in key_choices.items():
        for alt_key in alternatives:
            key_mapping[clean_str_answer(alt_key)] = i_key

    # Check for missing keys, accounting for alternatives
    missing_keys = set()
    for i_key in instructor_answer.keys():
        found = False
        if i_key in student_answer:
            found = True
        else:
            # Look for alternative spellings
            clean_student_keys = {clean_str_answer(k): k for k in student_answer.keys()}
            for clean_key in clean_student_keys:
                if clean_key in key_mapping and key_mapping[clean_key] == i_key:
                    found = True
                    msg_list.append(
                        f"Accepted alternative key {clean_student_keys[clean_key]!r} for {i_key!r}"
                    )
                    break
        if not found:
            missing_keys.add(i_key)

    if missing_keys:
        msg_list.append(f"- Missing keys: {[repr(k) for k in missing_keys]}.")
    else:
        msg_list.append("- No missing keys")

    # Check that all values are lists of strings
    for s_key, value in student_answer.items():
        if not isinstance(value, list):
            msg_list.append(
                f"- Value for key {s_key!r} must be a list, but is type {type(value).__name__}"
            )
            continue

        # Check that all elements in the list are strings
        for i, elem in enumerate(value):
            if not isinstance(elem, str | int | float):  # student can use string or int or float
                msg_list.append(
                    f"- Element {i} for key {s_key!r} must be a string, but is type {type(elem).__name__}"
                )

    # Set status to True if no errors found (or only alternative key messages exist)
    if not any(msg.startswith("- ") for msg in msg_list) or (
        all(not msg.startswith("- ") or "No missing keys" in msg for msg in msg_list)
    ):
        status = True
        msg_list.append("Type 'dict[str, list[str]]' is correct")

    return status, "\n".join(msg_list)


# ----------------------------------------------------------------------


def check_answer_dict_str_list_str(
    student_answer: dict[str, list[str]],
    instructor_answer: dict[str, list[str]],
    key_choices: dict[str, list[str]] | None = None,
    partial_score_frac_l: list[float] = [0.0],
) -> tuple[bool, str]:
    """Check if student answer matches instructor answer for dict[str, list[str]] type.
    Assumes structure validation has already passed.
    """
    msg_list = []
    status = True  # Start with True since structure is valid
    ps_dict = init_partial_score_dict()
    # ps_dict["nb_total"] = len(instructor_answer)
    # gets updated in check_list
    ps_dict["nb_total"] = 0

    if key_choices is None:
        key_choices = {}

    # Create sets of acceptable keys for each instructor key
    i_keys = {
        i_key: set([i_key] + key_choices.get(i_key, [])) for i_key in instructor_answer.keys()
    }

    # Map student keys to instructor keys
    student_to_instructor = {}
    for s_key in student_answer:
        for i_key, acceptable_keys in i_keys.items():
            if s_key in acceptable_keys:
                student_to_instructor[s_key] = i_key
                if s_key != i_key:
                    msg_list.append(f"Accepted alternative key {s_key!r} for {i_key!r}")
                break

    # Check values for each instructor key
    for i_key, i_list in instructor_answer.items():
        # Find the corresponding student key
        s_key = next((k for k, v in student_to_instructor.items() if v == i_key), None)

        if s_key is None:
            # This shouldn't happen if structure check passed
            status = False
            msg_list.append(f"Missing key: {i_key!r}")
            ps_dict["nb_mismatches"] += 1
            continue

        # Check list values
        s_list = student_answer[s_key]
        print(f"\nsend to check_list_str, {i_list=}")
        print(f"send to check_list_str, {s_list=}")
        status_, msg_ = check_list_str(i_list, s_list, ps_dict)
        print("return from check_list_str, ps_dict= ", ps_dict, "\n")
        if not status_:
            # Update status and increment mismatch counter
            status = False
            # ps_dict["nb_mismatches"] += 1

            # Create message prefix for this key
            key_msg = f"For key {i_key!r}:"

            # Format message based on whether msg_ is string or list
            if isinstance(msg_, str):
                message_lines = [key_msg, msg_]
                print(f"if instance, {message_lines=}", flush=True)
            else:
                message_lines = [key_msg] + msg_
                print(f"else, {message_lines=}", flush=True)

            # Add formatted message to msg_list
            msg_list.extend(message_lines)
            msg_list.append(message_lines)

    # Calculate partial credit
    try:
        partial_score_frac_l[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]
    except ZeroDivisionError:
        partial_score_frac_l[0] = 1.0
    print(f"==> inside check_answer_dict_str_list_str, {partial_score_frac_l=}")

    if status and not msg_list:
        msg_list = ["Answer matches expected values."]
    print(f"{msg_list=}")
    print(f"{status=}")
    print(f"{partial_score_frac_l=}")
    return return_value(status, msg_list, student_answer, instructor_answer)


# ----------------------------------------------------------------------


def check_structure_dict_str_list_str(
    student_answer: dict[str, list[str]],
    instructor_answer: dict[str, list[str]],
    key_choices: dict[str, list[str]] | None = None,
) -> tuple[bool, str]:
    """Check if student answer matches expected structure of dict[str, list[str]].

    Args:
        student_answer: Student's submitted answer
        instructor_answer: Instructor's reference answer
        key_choices: Dictionary mapping instructor keys to lists of acceptable alternatives

    Returns:
        tuple[bool, str]: Status indicating if structure is valid and message
    """
    status = True
    msg_list = []

    if not isinstance(student_answer, dict):
        return False, "Answer must be a dict"

    if key_choices is None:
        key_choices = {}

    # Create dictionary of sets of acceptable keys for each instructor key
    i_keys = {
        i_key: set([i_key] + key_choices.get(i_key, [])) for i_key in instructor_answer.keys()
    }

    # Track which instructor keys have been matched
    matched_i_keys = set()

    # Check each student key against acceptable keys
    for s_key in student_answer:
        found_match = False
        for i_key, acceptable_keys in i_keys.items():
            if i_key in matched_i_keys:
                continue
            if s_key in acceptable_keys:
                matched_i_keys.add(i_key)
                found_match = True
                if s_key != i_key:
                    msg_list.append(f"Accepted alternative key {s_key!r} for {i_key!r}")
                break

        if not found_match:
            status = False
            msg_list.append(f"Invalid key {s_key!r} - no matching instructor key")

    # Check for unmatched instructor keys
    unmatched_keys = set(i_keys.keys()) - matched_i_keys
    if unmatched_keys:
        status = False
        msg_list.append(f"Missing keys: {[repr(k) for k in unmatched_keys]}")
    else:
        msg_list.append("All required keys present")

    # Check that values are lists of strings
    if status:
        for s_key, value in student_answer.items():
            if not isinstance(value, list):
                msg_list.append(
                    f"Value for key {s_key!r} must be a list, but is type {type(value).__name__}"
                )
                status = False
                continue

            # Check that all elements in the list are strings
            for i, elem in enumerate(value):
                if not isinstance(elem, (str, int, float)):  # Allow string, int or float
                    msg_list.append(
                        f"Element {i} for key {s_key!r} must be a string, int, or float, but is type {type(elem).__name__}"
                    )
                    status = False

    if status:
        msg_list.append("Type 'dict[str, list[str]]' is correct")

    return status, "\n".join(msg_list)


# ======================================================================
def check_answer_dict_str_dict_str_float(
    student_answer: dict[str, dict[str, float]],
    instructor_answer: dict[str, dict[str, float]],
    rel_tol: float,
    dict_float_choices: dict[str, list[float]],
    partial_score_frac: list[float],
) -> tuple[bool, str]:
    """Check if student answer matches instructor answer.

    Compares student and instructor answers that are nested dictionaries with string
        keys and float values (dict[str,dict[str,float]]).
    Allows for multiple acceptable float values specified in dict_float_choices.

    Args:
        student_answer: Student's submitted answer as dict[str, dict[str, float]]
        instructor_answer: Instructor's reference answer as dict[str, dict[str, float]]
        rel_tol: Relative tolerance for float comparisons
        dict_float_choices: Dictionary mapping keys to lists of acceptable float values
        partial_score_frac: List to store partial credit score fraction

    Returns:
        tuple[bool, str]: Status indicating if answers match and message detailing any
            mismatches
        False: If student answer is not a dictionary

    """
    status = True
    msg_list = []
    ps_dict = init_partial_score_dict()

    # msg_list.append(f"DEBUG: {dict_float_choices=}")
    # msg_list.append(f"DEBUG: {instructor_answer=}")
    # msg_list.append(f"DEBUG: {student_answer=}")

    # Should go in structure check
    if not isinstance(student_answer, dict):
        return False, ""

    for k, v in instructor_answer.items():
        keys = list(instructor_answer.keys())

        # Initialize variables before use
        status_ = True  # Default to True
        msg_list_ = []  # Empty list to start

        if len(dict_float_choices) > 0 and k in dict_float_choices:
            for val in dict_float_choices[k]:
                status_, msg_list_ = check_dict_str_float(
                    keys,
                    v,
                    student_answer[k],
                    rel_tol,
                    1.0e-5,
                    ps_dict,
                )
                if status_ is True:
                    msg_list_.extend(
                        [
                            f"Student answer ({student_answer[k]}) is within rel error ",  # noqa: E501
                            f"of {rel_tol * 100}%% of one of the accepted answers ({val})",  # noqa: E501
                        ],
                    )
                    break
        else:
            status_, msg_list_ = check_dict_str_float(
                keys,
                v,
                student_answer[k],
                rel_tol,
                1.0e-5,
                ps_dict,
            )

        if status_ is False:
            status = status_
            msg_list.extend(msg_list_)

    partial_score_frac[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]

    if not msg_list:
        msg_list = ["Answer matches expected values."]

    return return_value(status, msg_list, student_answer, instructor_answer)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_dict_str_dict_str_float(
    student_answer: dict[str, dict[str, float]],
    instructor_answer: dict[str, dict[str, float]],
) -> tuple[bool, str]:
    """Check if student answer matches expected structure of dict[str, dict[str, float]].

    Verifies that:
    1. Student answer contains all required keys from instructor answer
    2. All values in the outer dictionary are dictionaries
    3. Inner dictionaries have string keys and float values

    Args:
        student_answer: The student's submitted answer to check
        instructor_answer: The instructor's reference answer defining expected structure

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if structure is valid, False otherwise
            - str: Newline-separated list of validation messages

    """
    msg_list = []
    status = True
    i_ans = instructor_answer
    s_ans = student_answer

    # Assumes instructor answer is correct. Ideally, the keys should be listed
    # in the yaml file.
    missing_keys = set(i_ans.keys()) - set(s_ans.keys())
    if len(missing_keys) > 0:
        return False, f"- Missing keys: {[repr(k) for k in missing_keys]}."

    # ! print("answer keys: ", list(student_answer.keys()))
    # ! for k, v in instructor_answer.items():
    # ! print(f"key: {k}, value: {v}")

    for k, v in instructor_answer.items():
        if not isinstance(v, dict):
            msg_list.append(f"- answer[{k!r}] must be of type 'dict'")
            status = False
            continue
        # v is a dict
        for kk, vv in v.items():
            if not (isinstance(kk, str) and isinstance(vv, float | np.floating | np.integer | int)):
                msg = f"- answer[{k!r}] must have keys of type 'str' and values of type 'float'"
                msg_list.append(msg)
                msg = f"-    Instead, key has type {type(kk)} and value has type {type(vv)}" 
                msg_list.append(msg)
                status = False

    if status is True:
        msg_list.append("Type 'dict[str, dict[str, float]' is correct.")

    return status, "\n".join(msg_list)


# ======================================================================


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
def check_key_structure(
    s_dict: dict[Any, Any],
    i_dict: dict[Any, Any],
) -> bool:
    """Recursively checks if two dictionaries have matching key structures.

    Compares the key structure of a student dictionary against an instructor
    dictionary (gold standard) to ensure they have the same nested structure.
    Will check nested dictionaries recursively two levels deep.

    Args:
        s_dict (dict): Student dictionary to check
        i_dict (dict): Instructor dictionary to compare against (gold standard)

    Returns:
        bool: True if dictionaries have matching key structures at all levels,
              False if structures differ

    """
    if not isinstance(s_dict, dict) or not isinstance(i_dict, dict):
        return False

    # Check the top-level keys match
    if set(s_dict.keys()) != set(i_dict.keys()):
        return False

    # Iterate through keys and check structures
    # Both dictionaries have the same keys
    for i_key, i_value in i_dict.items():
        # If both values are dictionaries, compare their key sets
        if isinstance(s_dict.get(i_key), dict) and isinstance(i_value, dict):
            if not check_key_structure(s_dict[i_key], i_value):
                return False
        elif isinstance(s_dict.get(i_key), dict) != isinstance(i_value, dict):
            # One is a dict and the other is not, key structure does not match
            return False

    return True


# ======================================================================


def check_answer_str(
    student_answer: str,
    instructor_answer: str,
    str_choices: list[str],
    remove_spaces: bool,  # noqa: FBT001
) -> tuple[bool, str]:
    """Check if a student's string answer matches the instructor's answer.

    Args:
        student_answer (str): The student's submitted answer
        instructor_answer (str): The instructor's correct answer
        str_choices (list[str]): Optional list of valid answer choices. If provided,
            validates that student_answer is one of these choices
        remove_spaces (bool): Whether to remove spaces when comparing answers

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    # ! print(f"check_answer_str, {remove_spaces=}")
    status, msg = check_str(
        instructor_answer,
        student_answer,
        str_choices,
        remove_spaces=remove_spaces,
    )

    if not msg:
        msg = "Answer matches expected values."

    return return_value(status, [msg], student_answer, instructor_answer)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


# MUST FIX
def check_structure_str(
    student_answer: str,
    choices: list[str],
) -> tuple[bool, str]:
    """Check if a student's string answer matches the expected structure.

    Args:
        student_answer (str): The student's submitted answer
        choices (list[str]): Optional list of valid answer choices. If provided,
            validates that student_answer is one of these choices

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    status = True
    msg_list = []
    choices = [clean_str_answer(c) for c in choices]

    # Ideally, should be done when yaml file is preprocessed
    # All strings should be lowered at that time.
    # ! choices = [clean_str_answer(s) for s in choices]

    if not isinstance(student_answer, str):
        status = False
        msg_list += ["- Answer must be of type 'str'"]
    else:
        msg_list += ["- type 'str' is correct"]
        # clean choices (lower, strip, '  ' -> ' ')
        student_answer = clean_str_answer(student_answer)

    if status and choices != []:
        if student_answer not in choices:
            status = False
            msg_list += [f"- Answer must be one of {choices}"]
        else:
            msg_list += [f"- Answer {student_answer!r} is among the valid choices"]

    return status, "\n".join(msg_list)


# ======================================================================


def check_answer_explain_str(
    student_answer: str,
    instructor_answer: str,
) -> tuple[bool, str]:
    """Check if a student's string answer matches the instructor's answer.

    Args:
        student_answer (str): The student's submitted answer
        instructor_answer (str): The instructor's correct answer

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    msg_list = []
    status = True

    if not msg_list:
        msg_list = ["Explanatory string satisfies requirements. Not evaluated."]

    return return_value(status, msg_list, student_answer, instructor_answer)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_explain_str(student_answer: str) -> tuple[bool, str]:
    """Check if a student's string answer matches the expected structure.

    Args:
        student_answer (str): The student's submitted answer

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    msg_list = []
    status = True
    if isinstance(student_answer, str):
        status = True
        msg_list.append("- Type 'str' is correct")
    else:
        status = False
        msg_list.append("- Type must be of type 'str'")

    # Check the number of words in the string
    max_nb_words = config_dict["max_nb_words"]

    if status:
        is_nb_words = len(student_answer.split()) >= max_nb_words
        if is_nb_words:
            msg_list.append("- Length of answer is sufficient")
        else:
            status = False
            msg_list.append(f"- Length of answer must be > {max_nb_words}")

    return status, "\n".join(msg_list)


# ======================================================================

# IMPLEMENT partial_scoring. Add option in yaml file to turn it off. On by default .
# NOT DONE


# def check_answer_set_str(
#     student_answer: set[str],
#     instructor_answer: set[str],
#     partial_score_frac: list[float],
#     choices: list[set[str]] | None = None,
# ) -> tuple[bool, str]:
#     """Check if a student's set of strings answer matches the instructor's answer.

#     Args:
#         student_answer (set[str]): The student's submitted answer
#         instructor_answer (set[str]): The instructor's correct answer
#         partial_score_frac (list[float]): The partial score fraction
#         choices (list[set[str]] |  None): Optional list of valid answer choices.
#             If provided, validates that student_answer is one of these choices.
#             Use None rather than [] as a default to avoid mutable structures.

#     Returns:
#         tuple[bool, str]: A tuple containing:
#             - bool: True if answers match and validation passes, False otherwise
#             - str: Message explaining the validation result

#     Notes:
#         Set equality is order-independent. Sets {a,b} and {b,a} are equal.
#         All strings are cleaned using clean_str_answer before comparison.
#     """
#     msg_list = []
#     status = False
#     ps_dict = init_partial_score_dict()

#     if choices is None:
#         choices = []

#     nb_total = 0
#     nb_missing = 0

#     # Clean all strings using clean_str_answer
#     s_answ = {clean_str_answer(i) for i in student_answer}
#     i_answ = {clean_str_answer(i) for i in instructor_answer}

#     # Clean choices if provided
#     if choices and isinstance(choices[0], list):
#         choices = [set(clean_str_answer(el) for el in c) for c in choices]
#     elif choices and isinstance(choices[0], set):
#         choices = [set(clean_str_answer(el) for el in c) for c in choices]

#     # Check answer against choices or instructor answer
#     if choices and isinstance(choices[0], set):
#         if s_answ in choices:
#             status = True
#     else:
#         if s_answ == i_answ:
#             nb
#             status = True

#     msg_list.append("All strings are cleaned (lowercased, stripped, spaces normalized)")
#     if choices and isinstance(choices[0], set):
#         msg_list.append(f"Student answer is one of {choices}")

#     if status and not msg_list:
#         msg_list = ["Your set[str] answer matches expected values."]

#     try:
#         partial_score_frac[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]
#     except ZeroDivisionError:
#         partial_score_frac[0] = 1.0

#     return return_value(status, msg_list, s_answ, i_answ)


def check_answer_set_str(
    student_answer: set[str],
    instructor_answer: set[str],
    choices: list[set[str]] | None = None,
    partial_score_frac_l: list[float] = [0.0],
) -> tuple[bool, str]:
    """Check if a student's set of strings answer matches the instructor's answer.

    Args:
        student_answer (set[str]): The student's submitted answer
        instructor_answer (set[str]): The instructor's correct answer
        partial_score_frac (list[float]): The partial score fraction
        choices (list[set[str]] |  None): Optional list of valid answer choices.
            If provided, validates that student_answer is one of these choices.
            Use None rather than [] as a default to avoid mutable structures.

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    Notes:
        Set equality is order-independent. Sets {a,b} and {b,a} are equal.
        All strings are cleaned using clean_str_answer before comparison.

    """
    msg_list = []
    status = False
    ps_dict = init_partial_score_dict()
    ps_dict["nb_total"] = len(instructor_answer)
    print("top ==> choices: ", choices)
    if not choices:
        choices = set()
        print("choices is None -> set()")

    # Clean all strings using clean_str_answer
    s_answ = {clean_str_answer(i) for i in student_answer}
    i_answ = {clean_str_answer(i) for i in instructor_answer}

    all_choices = set()
    # if choices is a list of sets, add all the elements into a single set
    if choices and isinstance(choices, list):
        for s in choices:
            all_choices.update(s)
    all_choices.update(i_answ)

    # clean all the choices
    all_choices = {clean_str_answer(i) for i in all_choices}

    # Loop over the student answer and check if it is in the all_choices
    for s in s_answ:
        if s not in all_choices:
            status = False
            ps_dict["nb_mismatches"] += 1
            msg_list.append(f"Student answer contains {s}, which is not in the valid choices")

    if ps_dict["nb_mismatches"] == 0:
        status = True

    try:
        partial_score_frac_l[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]
    except ZeroDivisionError:
        partial_score_frac_l[0] = 1.0

    print(f"==> inside check_answer_set_str, {partial_score_frac_l=}")

    return return_value(status, msg_list, s_answ, i_answ)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_set_str(student_answer: set[str] | list[str]) -> tuple[bool, str]:
    """Check if a student's set of strings answer matches the expected structure.

    Args:
        student_answer (set[str] | list[str]): The student's submitted answer

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    msg_list = []
    status = True
    are_all_str = None

    if isinstance(student_answer, set | list):
        status = True
        msg_list.append("- Type is either 'list' or 'set' (correct).")
    else:
        status = False
        are_all_str = False
        msg_list.append("- Answer must be of type 'set' or 'list'.")

    if status:
        for s in student_answer:
            print("s= ", s, flush=True)
            if not isinstance(s, str):
                msg = f"- Set element {s!r} must be of type 'str'"
                msg_list.append(msg)
                status = False
                are_all_str = False

    if are_all_str is None:
        msg_list.append("- All elements are 'str', as required")
        status = True

    return status, "\n".join(msg_list)


# ======================================================================


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_dict_str_set(
    student_answer: dict[str, set[str] | list[str]],
    instructor_answer: dict[str, set[str] | list[str]],
) -> tuple[bool, str]:
    """Check if a student's dictionary of strings answer matches the expected structure.

    Args:
        student_answer (dict[str, set[str] | list[str]]): The student's submitted answer
        instructor_answer (dict[str, set[str] | list[str]]): The instructor's correct answer

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    status = True
    msg_list = []

    if not isinstance(student_answer, dict):
        msg_list.append("- Answer should be of type 'dict'.")
    else:
        msg_list.append("- Type 'dict' is correct")

    keys = set(instructor_answer.keys())
    student_keys = set(student_answer.keys())
    missing_keys = keys - student_keys

    if status:
        if len(missing_keys) > 0:
            list_of_missing_keys = [repr(k) for k in missing_keys]
            msg_list.append(f"- Missing keys: {list_of_missing_keys}.")
            status = False
        else:
            msg_list.append("- No missing keys")

    if status:
        is_item_type_list = True
        for k, v in student_answer.items():
            if k in keys and isinstance(v, set | list):
                msg = f"- Answer[{k!r}] must be of type 'set' or 'list'."
                msg_list.append(msg)
                # msg_list.append(f"- Answer[repr(k)r}] must be of type 'set' or 'list'.")
                # The answer is cast to a set when checked for accuracy
                status = False
                is_item_type_list = False

        if is_item_type_list:
            msg_list.append("- All list elements are of type 'list' or 'set' as expected")
            status = True

    return status, "\n".join(msg_list)


# ======================================================================


def check_answer_dict_str_set_int(
    student_answer: dict[str, set[int]],
    instructor_answer: dict[str, set[int]],
    keys: list[str] | None = None,
) -> tuple[bool, str]:
    """Check whether a student and instruct's dictionary of strings match.

    Args:
        student_answer (dict[str, set[int]]): The student's submitted answer
        instructor_answer (dict[str, set[int]]): The instructor's correct answer
        keys (list[str] | None): Optional list of keys to check. If None, all keys are
            checked. If `keys` is provided, only the keys in `keys` are checked.

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    msg_list = []
    status = True
    keys = list(instructor_answer.keys()) if keys is None else keys
    ps_dict = init_partial_score_dict()

    # ! TODO: check that keys are in the instructor answer
    for i_key, i_value in instructor_answer.items():
        status_, msg_ = check_set_int(set(student_answer[i_key]), set(i_value), ps_dict)
        status = False if status_ is False else status
        msg_list += [msg_]

    if not msg_list:
        msg_list = ["Answer matches expected values. "]

    return return_value(status, msg_list, student_answer, instructor_answer)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_dict_str_set_int(
    student_answer: dict[str, set[int] | list[int]],
    instructor_answer: dict[str, set[int] | list[int]],
) -> tuple[bool, str]:
    """Check if a student's dictionary of strings to sets of integers answer matches the expected structure.

    Args:
        student_answer (dict[str, set[int] | list[int]]): The student's submitted answer
        instructor_answer (dict[str, set[int] | list[int]]): The instructor's correct answer

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    status = True
    msg_list = []

    if not isinstance(student_answer, dict):
        msg_list.append("- Answer should be of type 'dict'.")
    else:
        msg_list.append("- Type 'dict' is correct")

    keys = set(instructor_answer.keys())
    student_keys = set(student_answer.keys())
    missing_keys = keys - student_keys

    if status:
        if len(missing_keys) > 0:
            list_of_missing_keys = [repr(k) for k in missing_keys]
            msg_list.append(f"- Missing keys: {list_of_missing_keys}.")
            status = False
        else:
            msg_list.append("- No missing keys")

    if status:
        is_item_type_list = True
        for k, v in student_answer.items():
            if k in keys and not isinstance(v, set | list):
                msg = f"- Answer[{k!r}] must be of type 'set' or 'list'."
                msg_list.append(msg)
                # The answer is cast to a set when checked for accuracy
                status = False
                is_item_type_list = False

        if is_item_type_list:
            msg_list.append("- All list elements are of type 'list' or 'set' as expected")
            status = True

    # Check that all set/list elements are integers
    if status:
        are_all_int = True
        for k, v in student_answer.items():
            if k in keys and isinstance(v, set | list):
                for e in v:
                    if not isinstance(e, int | np.integer):
                        msg = f"- Answer[{k!r}] contains element {e!r} which must be of type 'int'"
                        msg_list.append(msg)
                        status = False
                        are_all_int = False
                        break
                if not are_all_int:
                    break

        if are_all_int:
            msg_list.append("- All set/list elements are of type 'int' as expected")

    return status, "\n".join(msg_list)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


# ======================================================================


def check_answer_dict_str_int(
    student_answer: dict[str, int],
    instructor_answer: dict[str, int],
    keys: list[str] | None = None,
    dict_int_choices: dict[str, int] | None = None,
    partial_score_frac: list[float] = [0.0],
) -> tuple[bool, str]:
    """Check if a student's dictionary of strings answer matches the instructor's answer.

    Args:
        student_answer (dict[str, int]): The student's submitted answer
        instructor_answer (dict[str, int]): The instructor's correct answer
        keys (list[str] | None): Optional list of keys to check. If None, all keys are
            checked. If `keys` is provided, only the keys in `keys` are checked.
        dict_int_choices (dict[str, int] | None): Optional dictionary of float choices
            for each key. If provided, validates that student_answer[k] is one of these choices.
        partial_score_frac (list[float]): The partial score fraction

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    Use the function (NOT DONE):
        def check_dict_str_float(
            keys: list[str],
            i_dict: dict[str, int],
            s_dict: dict[str, int],
            ps_dict: dict[str, int],
        ) -> tuple[bool, list[str]]:

    """
    if dict_int_choices is not None:
        dict_int_choices = {}
        print("dict_int_choices is not implemented in dict[str,int] types")

    # keys not implement
    if keys is not None or keys is None:
        keys = []
        print("keys is not yet implemented for dict[str,int] types")

    msg_list = []

    # msg_list.append(f"DEBUG: {dict_float_choices=}")
    # msg_list.append(f"DEBUG: {instructor_answer=}")
    # msg_list.append(f"DEBUG: {student_answer=}")

    status = True
    keys = list(instructor_answer.keys()) if keys is [] else keys
    ps_dict = init_partial_score_dict()
    ps_dict["nb_total"] = len(keys)
    if dict_int_choices is None:
        dict_int_choices = {}

    print(f"==> {student_answer=}")
    print(f"==> {instructor_answer=}")

    # Need an exception in case the student key is not found
    for k in keys:
        s_int = student_answer[k]
        i_int = instructor_answer[k]

        # if clause not exeucted if dict_int_choices is None or {}
        if len(dict_int_choices) > 0 and k in dict_int_choices:
            for val in dict_int_choices[k]:
                if val == "i":  # use instructor answer
                    val = i_int
                status_, msg_list_ = check_int(s_int, val)
                if status_ is True:
                    break
        else:
            status_, msg_ = check_int(i_int, s_int)

        if status_ is False:
            status = False
            ps_dict["nb_mismatches"] += 1
            msg_list.append(msg_)

    print(f"==> {ps_dict=}")
    try:
        partial_score_frac[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]
    except ZeroDivisionError:
        print("ZeroDivisionError: check_answer_dict_str_int. FIX.")
        partial_score_frac[0] = 1.0

    if not msg_list:
        msg_list = ["Answer matches expected values. "]

    return return_value(status, msg_list, student_answer, instructor_answer)


# ======================================================================


def check_structure_dict_str_int(
    student_answer: dict[str, int],
    instructor_answer: dict[str, int],
    keys: list[str] | None = None,
) -> tuple[bool, str]:
    """Check if a student's dictionary of strings answer matches the instructor's answer.

    Args:
        student_answer (dict[str, int]): The student's submitted answer
        instructor_answer (dict[str, int]): The instructor's correct answer
        keys (list[str] | None): Optional list of keys to check. If None, all keys are
            checked. If `keys` is provided, only the keys in `keys` are checked.

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    status = True
    msg_list = []

    print(f"check_structure_dict_str_int, {student_answer}")
    print(f"check_structure_dict_str_int, {instructor_answer}")

    if status and not isinstance(student_answer, dict):
        msg_list += ["Student answer should be a dict"]
        status = False

    if status:
        keys = keys if keys else list(instructor_answer.keys())
        instructor_keys = set(keys)
        instructor_answer = {k: v for k, v in instructor_answer.items() if k in keys}
        student_keys = set(student_answer.keys())
        missing_keys = list(instructor_keys - student_keys)

        if len(missing_keys) > 0:
            msg_list.append(f"- Missing keys: {[repr(k) for k in missing_keys]}.")
            status = False
        else:
            msg_list.append("- No missing keys.")

    if status:
        # some keys are filtered. Student is allowed to have
        # keys not in the instructor set
        for k in instructor_answer:
            vs = student_answer[k]
            if not isinstance(vs, int | np.integer):
                msg_list.append(f"- answer[{k!r}] should be a float.")
                status = False

        if status:
            msg_list.append("- All elements are of type float as expected.")

    if status:
        msg_list.append("Type 'dict[str, float]' is correct")

    return status, "\n".join(msg_list)


# ======================================================================


# ======================================================================


def check_answer_dict_str_ndarray(
    student_answer: dict[str, NDArray],
    instructor_answer: dict[str, NDArray],
    rel_tol: float,
    keys: list[str] | None = None,
    partial_score_frac: list[float] | None = None,
) -> tuple[bool, str]:
    """Check if a student's dictionary of strings answer matches the instructor's answer.

    Args:
        student_answer (dict[str, NDArray]): The student's submitted answer
        instructor_answer (dict[str, NDArray]): The instructor's correct answer
        rel_tol (float): The relative tolerance for comparing floats
        keys (list[str] | None): Optional list of keys to check. If None, all keys are
            checked. If `keys` is provided, only the keys in `keys` are checked.
        partial_score_frac (list[float]): The partial score fraction

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    if partial_score_frac is None:
        partial_score_frac = [0.0]

    msg_list = []
    status = True
    msg_list.append("Check array norms")

    keys = list(instructor_answer.keys()) if keys is None else keys
    ps_dict = init_partial_score_dict()
    ps_dict["nb_total"] = len(keys)

    # Need an exception in case the student key is not found
    i_norms = cast(dict[str, float], {k: np.linalg.norm(instructor_answer[k]) for k in keys})
    s_norms = cast(dict[str, float], {k: np.linalg.norm(student_answer[k]) for k in keys})

    status, msg_list = check_dict_str_float(keys, i_norms, s_norms, rel_tol, 1.0e-5, ps_dict)
    partial_score_frac[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]

    if not msg_list:
        msg_list = ["Answer matches expected values. "]

    return return_value(status, msg_list, s_norms, i_norms)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_dict_str_ndarray(
    student_answer: dict[str, NDArray],
    instructor_answer: dict[str, NDArray],
    keys: list[str] | None = None,
) -> tuple[bool, str]:
    """Check if a student's dictionary of strings answer matches the instructor's answer.

    Args:
        student_answer (dict[str, NDArray]): The student's submitted answer
        instructor_answer (dict[str, NDArray]): The instructor's correct answer
        keys (list[str] | None): Optional list of keys to check. If None or empty list,
            all keys in instructor_answer are checked. If `keys` is provided and non-empty,
            only the keys in `keys` are checked for structure validation.

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    status = True
    msg_list = []

    if not isinstance(instructor_answer, dict):
        msg_list += ["Instructor answer should be a dict"]
        status = False

    if status and not isinstance(student_answer, dict):
        msg_list += ["Student answer should be a dict"]
        status = False

    if status:
        # If keys is None or empty, check all keys in instructor_answer
        if keys is None or len(keys) == 0:
            keys = list(instructor_answer.keys())
        instructor_keys = set(keys)
        instructor_answer = {k: v for k, v in instructor_answer.items() if k in keys}
        student_keys = set(student_answer.keys())
        missing_keys = list(instructor_keys - student_keys)

        if len(missing_keys) > 0:
            msg_list.append(f"- Missing keys: {[repr(k) for k in missing_keys]}.")
            status = False
        else:
            msg_list.append("- No missing keys.")

    if status:
        # some keys are filtered. Student is allowed to have
        # keys not in the instructor set
        # Only check structure for keys in the keys list
        for s_key in keys:
            if s_key not in student_answer:
                continue  # Skip if key is missing (already checked above)
            s_value = student_answer[s_key]
            vs = s_value
            if not isinstance(vs, np.ndarray):
                msg_list.append(f"- answer[{s_key!r}] should be a numpy array.")
                status = False

        if status:
            msg_list.append("- All elements are of type ndarray as expected.")

    if status:
        msg_list.append("Type 'dict[str, ndarray]' is correct")

    return status, "\n".join(msg_list)


# ======================================================================


def check_answer_dict_tuple_int_ndarray(
    student_answer: dict[tuple[int], NDArray],
    instructor_answer: dict[tuple[int], NDArray],
    rel_tol: float,
    keys: list[tuple[int]] | None = None,
    partial_score_frac: list[float] | None = None,
) -> tuple[bool, str]:
    """Check if a student's dictionary of strings answer matches the instructor's answer.

    Args:
        student_answer (dict[tuple[int], NDArray]): The student's submitted answer
        instructor_answer (dict[tuple[int], NDArray]): The instructor's correct answer
        rel_tol (float): The relative tolerance for comparing floats
        keys (list[str] | None): Optional list of keys to check. If None, all keys are
            checked. If `keys` is provided, only the keys in `keys` are checked.
        partial_score_frac (list[float]): The partial score fraction

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    msg_list = ["Check array norms"]
    if partial_score_frac is None:
        partial_score_frac = [0.0]
    status = True
    i_dict_norm = {}
    s_dict_norm = {}
    keys = list(instructor_answer.keys()) if keys is None else keys
    sub_instructor_answer = {k: instructor_answer[k] for k in keys}

    # ! print("Assert_utilities, type dict_tuple_int_ndarray NOT HANDLED")
    # ! return False, ""

    ps_dict = init_partial_score_dict()
    ps_dict["total_nb"] = len(sub_instructor_answer)

    # Need an exception in case the student key is not found
    for k in sub_instructor_answer.keys():  # noqa: SIM118
        if k not in student_answer:
            status = False
            msg_list.append(f"The key {k} is missing")
            break
        s_arr = student_answer[k]
        i_arr = instructor_answer[k]
        if s_arr.shape != i_arr.shape:
            status = False
            msg_list.extend(
                [
                    f"key: {k}, incorrect shape {s_arr.shape}, ",
                    f"should be {i_arr.shape}.",
                ],
            )
        i_dict_norm[k] = s_norm = cast(float, np.linalg.norm(s_arr))
        s_dict_norm[k] = i_norm = cast(float, np.linalg.norm(i_arr))
        status_, msg = check_float(i_norm, s_norm, rel_tol, abs_tol=1.0e-5)
        if status_ is False:
            status = False
            msg_list.append(msg)

    partial_score_frac[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]

    if not msg_list:
        msg_list = ["Answer matches expected values. "]

    return return_value(status, msg_list, student_answer, instructor_answer)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_dict_tuple_int_ndarray(
    student_answer: dict[tuple[int], NDArray],
    instructor_answer: dict[tuple[int], NDArray],
    keys: list[tuple[int]] | None = None,
) -> tuple[bool, str]:
    """Check if a student's dictionary of strings answer matches the instructor's answer.

    Args:
        student_answer (dict[tuple[int], NDArray]): The student's submitted answer
        instructor_answer (dict[tuple[int], NDArray]): The instructor's correct answer
        keys (list[tuple[int]] | None): Optional list of keys to check. If None, all keys are
            checked. If `keys` is provided, only the keys in `keys` are checked.

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    status = True
    msg_list = []

    # First check that both answers are dictionaries
    if not isinstance(instructor_answer, dict) or not isinstance(student_answer, dict):
        msg_list.append("Both instructor and student answers must be dictionaries")
        return False, "\n".join(msg_list)

    # Validate that each student answer key is a tuple of integers and value is a numpy array
    for s_key, s_value in student_answer.items():
        if not isinstance(s_key, tuple) or any(
            not isinstance(el, int | np.integer) for el in s_key
        ):
            status = False
            msg_list.append(f"Key {s_key} must be a tuple of integers")
        if not isinstance(s_value, np.ndarray):
            msg_list.append(f"- answer[{s_key!r}] should be a numpy array.")
            status = False

    # If basic type validation failed, return early
    if not status:
        return status, "\n".join(msg_list)

    # Get the set of keys to check - either provided keys or all instructor keys
    keys = keys if keys else list(instructor_answer.keys())
    instructor_keys = set(keys)
    # Filter instructor answer to only include specified keys
    instructor_answer = {k: v for k, v in instructor_answer.items() if k in keys}
    student_keys = set(student_answer.keys())
    # Find any required keys missing from student answer
    missing_keys = list(instructor_keys - student_keys)

    # Generate final status messages
    if missing_keys:
        msg_list.append(f"- Missing keys: {[repr(k) for k in missing_keys]}.")
        status = False
    else:
        msg_list.append("- No missing keys.")
        msg_list.append("- All keys are tuples of ints and values are of type ndarray as expected.")
        msg_list.append("Type 'dict[tuple[int], ndarray]' is correct")

    return status, "\n".join(msg_list)


# ----------------------------------------------------------------------


def check_answer_dict_int_ndarray(
    student_answer: dict[int, NDArray],
    instructor_answer: dict[int, NDArray],
    rel_tol: float,
    keys: list[int] | None,
) -> tuple[bool, str]:
    """Check if student answer matches instructor answer for dict[int, ndarray] type.

    Compares student and instructor answers that are dictionaries with integer keys and numpy array values.
    Checks that array shapes match and array norms are within specified tolerance.

    Args:
        student_answer (dict[int, ndarray]): Student's submitted answer
        instructor_answer (dict[int, ndarray]): Instructor's reference answer
        rel_tol (float): Relative tolerance for comparing array norms
        keys (list[int] | None): Keys to check. If None, checks all instructor keys

    Returns:
        tuple[bool, str]: Status indicating if answers match and message detailing
            any mismatches

    """
    err = 1.0e-5
    msg_list = []
    status = True
    i_dict_norm = {}
    s_dict_norm = {}
    keys = list(instructor_answer.keys()) if keys is None else keys

    msg_list.append(f"We are comparing array norms with a {rel_tol} relative accuracy.")

    # Need an exception in case the student key is not found
    for k in keys:
        s_arr = student_answer[k]
        i_arr = instructor_answer[k]
        if s_arr.shape != i_arr.shape:
            status = False
            msg_list.append(f"key: {k}, incorrect shape {s_arr.shape}, should be {i_arr.shape}.")
            continue
        s_norm = np.linalg.norm(s_arr)
        i_norm = np.linalg.norm(i_arr)
        i_dict_norm[k] = i_norm
        s_dict_norm[k] = s_norm
        if i_norm < err:
            abs_err = math.fabs(s_norm - i_norm)
            if abs_err > err:
                status = False
                msg_list.append(f"- key {k} has a norm with absolute error > 1.e-5")
        else:
            rel_err = math.fabs(s_norm - i_norm) / math.fabs(i_norm)
            if rel_err > rel_tol:
                status = False
                msg = f"Answer must be of type 'int'. Your answer is of type {type(student_answer).__name__}."
                msg_list.append(msg)

    if not msg_list:
        msg_list = ["Answer matches expected values. "]

    return return_value(status, msg_list, student_answer, instructor_answer)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_dict_int_ndarray(
    student_answer: dict[int, NDArray],
    instructor_answer: dict[int, NDArray],
    keys: list[int] | None,
) -> tuple[bool, str]:
    """Check if student answer matches expected structure of dict[int, ndarray].

    Verifies that:
    1. Both student and instructor answers are dictionaries
    2. All dictionary keys are integers
    3. All dictionary values are numpy ndarrays
    4. Student answer contains all required keys from instructor answer

    Args:
        student_answer: Student's submitted answer to check
        instructor_answer: Instructor's reference answer defining expected structure
        keys: Optional list of integer keys to check. If None, checks all instructor keys

    Returns:
        tuple[bool, str]: Status indicating if structure is valid and message detailing
            any validation errors

    """
    status = True
    msg_list = []

    if not isinstance(instructor_answer, dict):
        msg_list += ["Instructor answer should be a dict"]
        status = False

    if status and not isinstance(student_answer, dict):
        msg_list += ["Student answer should be a dict"]
        status = False

    for key in student_answer:
        if not isinstance(key, int | np.integer):
            status = False
            msg_list += [f"key {key} should be of type 'int', but is type {type(key).__name__}."]

    if status:
        keys = keys if keys else list(instructor_answer.keys())
        instructor_keys = set(keys)
        instructor_answer = {k: v for k, v in instructor_answer.items() if k in keys}
        student_keys = set(student_answer.keys())
        missing_keys = list(instructor_keys - student_keys)

        if len(missing_keys) > 0:
            msg_list.append(f"- Missing keys: {[repr(k) for k in missing_keys]}.")
            status = False
        else:
            msg_list.append("- No missing keys.")

        # some keys are filtered. Student is allowed to have
        # keys not in the instructor set
        for k in instructor_answer:
            vs = student_answer[k]
            if not isinstance(vs, type(np.zeros(1))):
                msg_list.append(f"- answer[{k!r}] should be a numpy array.")
                status = False

        if status:
            msg_list.append("- All elements are of type ndarray as expected.")

        msg_list.append("Type 'dict[str, ndarray]' is correct")

    return status, "\n".join(msg_list)


# ======================================================================
def check_answer_dict_str_int(
    student_answer: dict[str, int],
    instructor_answer: dict[str, int],
    keys: list[str] | None = None,
    dict_int_choices: dict[str, int] | None = None,
    partial_score_frac: list[float] = [0.0],
) -> tuple[bool, str]:
    """Check if a student's dictionary of strings answer matches the instructor's answer.

    Args:
        student_answer (dict[str, int]): The student's submitted answer
        instructor_answer (dict[str, int]): The instructor's correct answer
        keys (list[str] | None): Optional list of keys to check. If None, all keys are
            checked. If `keys` is provided, only the keys in `keys` are checked.
        dict_int_choices (dict[str, int] | None): Optional dictionary of float choices
            for each key. If provided, validates that student_answer[k] is one of these choices.
        partial_score_frac (list[float]): The partial score fraction

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    Use the function (NOT DONE):
        def check_dict_str_float(
            keys: list[str],
            i_dict: dict[str, int],
            s_dict: dict[str, int],
            ps_dict: dict[str, int],
        ) -> tuple[bool, list[str]]:

    """
    if dict_int_choices is not None:
        dict_int_choices = {}
        print("dict_int_choices is not implemented in dict[str,int] types")

    # keys not implement
    if keys is not None or keys is None:
        keys = []
        print("keys is not yet implemented for dict[str,int] types")

    msg_list = []

    # msg_list.append(f"DEBUG: {dict_float_choices=}")
    # msg_list.append(f"DEBUG: {instructor_answer=}")
    # msg_list.append(f"DEBUG: {student_answer=}")

    status = True
    keys = list(instructor_answer.keys()) if keys is [] else keys
    ps_dict = init_partial_score_dict()
    ps_dict["nb_total"] = len(keys)
    if dict_int_choices is None:
        dict_int_choices = {}

    print(f"==> {student_answer=}")
    print(f"==> {instructor_answer=}")

    # Need an exception in case the student key is not found
    for k in keys:
        s_int = student_answer[k]
        i_int = instructor_answer[k]

        # if clause not exeucted if dict_int_choices is None or {}
        if len(dict_int_choices) > 0 and k in dict_int_choices:
            for val in dict_int_choices[k]:
                if val == "i":  # use instructor answer
                    val = i_int
                status_, msg_list_ = check_int(s_int, val)
                if status_ is True:
                    break
        else:
            status_, msg_ = check_int(i_int, s_int)

        if status_ is False:
            status = False
            ps_dict["nb_mismatches"] += 1
            msg_list.append(msg_)

    print(f"==> {ps_dict=}")
    try:
        partial_score_frac[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]
    except ZeroDivisionError:
        print("ZeroDivisionError: check_answer_dict_str_int. FIX.")
        partial_score_frac[0] = 1.0

    if not msg_list:
        msg_list = ["Answer matches expected values. "]

    return return_value(status, msg_list, student_answer, instructor_answer)


# ======================================================================


def check_structure_dict_int_list(
    student_answer: dict[int, list[float]],
    instructor_answer: dict[int, list[float]],
    keys: list[int] | None,  # NOT USED
) -> tuple[bool, str]:
    """Check if student answer matches expected structure of dict[int, list[float]].

    Verifies that:
    1. Both student and instructor answers are dictionaries
    2. Student answer contains all required keys from instructor answer
    3. Values in student answer are lists containing only float elements

    Args:
        student_answer (dict[int, list[float]]): Student's submitted answer to check
        instructor_answer (dict[int, list[float]]): Instructor's reference answer
            defining expected structure
        keys (list[int] | None): Keys to check. If None, checks all instructor keys

    Returns:
        tuple[bool, str]: Status indicating if structure is valid and message
            detailing any issues

    """
    status = True
    msg_list = []

    if not isinstance(instructor_answer, dict):
        msg_list += ["Instructor answer should be a dict"]
        status = False

    if status and not isinstance(student_answer, dict):
        msg_list += ["Student answer should be a dict"]
        status = False

    # I am not handling the keys argument yet <<<<<<
    # Check the length of the lists (NOT DONE) <<<<<
    # I could convert list to ndarray and call the function with NDARRAY for checking.
    # If the list cannot be converted, it has the wrong format. So use an try/except.

    if status:
        # some keys are filtered. Student is allowed to have
        # keys not in the instructor set
        for k in instructor_answer:
            key = student_answer.get(k)  # default is None
            if key is None:
                status = False
                msg_list.append(f"Key {k} is missing from student answer")
                continue
            vs = student_answer[k]
            if not isinstance(vs, list):
                status = False
                msg = f"student_answer[{k}] is not type 'list'. Cannot proceed with answer check."  # noqa: E501
                msg_list.append(msg)
            for el in vs:
                if not isinstance(el, float | np.floating):
                    status = False
                    msg_list.extend(
                        [
                            f"student_answer[{k}] is a list with at least one ",
                            "non-float element. Cannot proceed with answer check.",
                        ],
                    )
                    break

        if status:
            msg_list.append("- All elements are of type list of float as expected.")

        msg_list.append("Type 'dict[str, list]' is correct")

    return status, "\n".join(msg_list)


# ======================================================================
def check_answer_dict_int_list_float(
    student_answer: dict[int, list[float]],
    instructor_answer: dict[int, list[float]],
    keys: list[int] | None,
    rel_tol: float,
    partial_score_frac: list[float],
) -> tuple[bool, str]:
    """Check if student answer matches instructor answer for dict[int, list[float]] type.

    Compares student and instructor answers that are dictionaries with integer keys and
    lists of float values. Checks that list elements match within specified tolerance.

    Args:
        student_answer (dict[int, list[float]]): Student's submitted answer
        instructor_answer (dict[int, list[float]]): Instructor's reference answer
        keys (list[int] | None): Keys to check. If None, checks all instructor keys
        rel_tol (float): Relative tolerance for comparing float values
        partial_score_frac (list[float]): List to store partial credit score fraction

    Returns:
        tuple[bool, str]: Status indicating if answers match and message detailing
            any mismatches

    """
    msg_list = []
    status = True
    keys = list(instructor_answer.keys()) if keys is None else keys
    ps_dict = init_partial_score_dict()
    ps_dict["nb_total"] = len(keys)

    # Need an exception in case the student key is not found
    for k in keys:
        s_anw = student_answer[k]
        i_anw = instructor_answer[k]
        status_, msg_list_ = check_list_float(
            i_anw,
            s_anw,
            rel_tol=rel_tol,
            abs_tol=1.0e-6,
            ps_dict=ps_dict,
        )
        if status_ is False:
            status = False
            msg_list.extend(msg_list_)

    partial_score_frac[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]

    if not msg_list:
        msg_list = ["Answer matches expected values. "]

    return return_value(status, msg_list, student_answer, instructor_answer)


# ======================================================================


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


# ! Discrepancy: keys is list[int], but student_answer values are list[float]
def check_structure_dict_int_list_float(
    student_answer: dict[int, list[float]],
    instructor_answer: dict[int, list[float]],
    keys: list[int] | None = None,
) -> tuple[bool, str]:
    """Check if student answer matches expected structure of dict[int, list[float]].

    Verifies that:
    1. Student answer is a dictionary
    2. All dictionary keys are integers
    3. All dictionary values are lists of floats
    4. Student answer contains all required keys from instructor answer

    Args:
        student_answer: Student's submitted answer to check
        instructor_answer: Instructor's reference answer defining expected structure
        keys: Optional list of integer keys to check. If None, checks all instructor keys

    Returns:
        tuple[bool, str]: Status indicating if structure is valid and message detailing
            any validation errors
    """
    status = True
    msg_list = []

    # Check student answer is a dict
    if not isinstance(student_answer, dict):
        msg_list.append("Answer must be a dict")
        return False, "\n".join(msg_list)

    # Check all keys are integers
    for key in student_answer:
        if not isinstance(key, int | np.integer):
            status = False
            msg_list.append(f"Key {key} must be of type 'int', but is type {type(key).__name__}")

    # Check all values are lists of floats
    for key, value in student_answer.items():
        if not isinstance(value, list):
            status = False
            msg_list.append(
                f"Value for key {key} must be a list, but is type {type(value).__name__}"
            )
            continue

        for i, elem in enumerate(value):
            if not isinstance(elem, float | np.floating):
                status = False
                msg_list.append(
                    f"Element {i} for key {key} must be a float, but is type {type(elem).__name__}"
                )

    # Check required keys are present
    if status:
        check_keys = keys if keys is not None else list(instructor_answer.keys())
        missing_keys = [k for k in check_keys if k not in student_answer]
        if missing_keys:
            status = False
            msg_list.append(f"Missing required keys: {missing_keys}")

    if status:
        msg_list.append("Answer has correct structure: dict[int, list[float]]")

    return status, "\n".join(msg_list)


# ======================================================================
def check_structure_dict_int_float(
    student_answer: dict[int, float],
    instructor_answer: dict[int, float],
) -> tuple[bool, str]:
    """Check if student answer matches expected structure of dict[int, float].

    Verifies that:
    1. Student answer is a dictionary
    2. All dictionary keys are integers
    3. All dictionary values are floats
    4. Student answer contains all required keys from instructor answer

    Args:
        student_answer (dict[int, float]): The student's submitted answer
        instructor_answer (dict[int, float]): The instructor's correct answer

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answer has correct structure, False otherwise
            - str: Message explaining any validation failures

    """
    status = True
    msg_list = []

    if not isinstance(student_answer, dict):
        msg_list.append("Answer must be a dict")
        return False, "\n".join(msg_list)

    # Check that all instructor keys are present
    for key in instructor_answer:
        if key not in student_answer:
            status = False
            msg_list.append(f"Key {key} is missing from student answer")

    for key, value in student_answer.items():
        if not isinstance(key, int | np.integer):
            msg_list.append(f"Key {key} must be of type 'int', but is type {type(key).__name__}")
        if not isinstance(value, float | np.floating):
            msg_list.append(
                f"Value for key {key} must be a float, but is type {type(value).__name__}"
            )

    return status, "\n".join(msg_list)


def check_answer_dict_int_float(
    student_answer: dict[int, float],
    instructor_answer: dict[int, float],
    rel_tol: float,
    partial_score_frac_l: list[float],
) -> tuple[bool, str]:
    """Check if student answer matches instructor answer for dict[int, float] type.

    Compares student and instructor answers that are dictionaries with integer keys and float values.
    Checks that float values match within the specified relative tolerance.

    Args:
        student_answer (dict[int, float]): Student's submitted answer
        instructor_answer (dict[int, float]): Instructor's reference answer
        rel_tol (float): Relative tolerance for comparing float values
        partial_score_frac_l (list[float]): List to store partial score fraction

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match within tolerance, False otherwise
            - str: Message explaining any mismatches

    """
    status = True
    msg_list = []
    ps_dict = init_partial_score_dict()
    ps_dict["nb_total"] = len(instructor_answer)
    for key, i_val in instructor_answer.items():
        s_val = student_answer.get(key)
        if s_val is None:
            status = False
            msg_list.append(f"Key {key} is missing from student answer")
            continue
        status_, msg = check_float(i_val, s_val, rel_tol, abs_tol=1.0e-4)
        msg_list.append(msg)
        if status_ is False:
            status = False
            ps_dict["nb_mismatches"] += 1
    try:
        print(f"==> {ps_dict=}")
        partial_score_frac_l[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]
    except ZeroDivisionError:
        print("ZeroDivisionError in check_answer_dict_int_float. TO FIX.")
        partial_score_frac_l[0] = 1.0

    if not msg_list:
        msg_list = ["Answer values are as expected. "]

    return return_value(status, msg_list, student_answer, instructor_answer)


# ======================================================================


def check_answer_list_int(
    student_answer: list[int],
    instructor_answer: list[int],
    partial_score_frac: list[float],
) -> tuple[bool, str]:
    """Check that all elements in the list have matching norms.

    Args:
        student_answer (list[int]): The student's submitted answer
        instructor_answer (list[int]): The instructor's correct answer
        partial_score_frac (list[float]): A list to store the partial score fraction

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    msg_list = []
    status = True
    answ_eq_len = len(student_answer) == len(instructor_answer)
    ps_dict = init_partial_score_dict()
    ps_dict["nb_total"] = len(instructor_answer)

    if answ_eq_len:
        status, msg_list_ = check_list_int(student_answer, instructor_answer, ps_dict)
        msg_list.extend(msg_list_)

    if not status:
        msg_list.append("Some elements are incorrect")

    partial_score_frac[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]

    if not msg_list:
        msg_list = ["Answer matches expected values. "]

    return return_value(status, msg_list, student_answer, instructor_answer)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_list_int(
    student_answer: list[int],
    instructor_answer: list[int],
) -> tuple[bool, str]:
    """Check if a student's list of integers matches the instructor's answer.

    Args:
        student_answer (list[int]): The student's submitted answer
        instructor_answer (list[int]): The instructor's correct answer

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    status = True
    msg_list = []

    if not isinstance(student_answer, list):
        status = False
        name_type = type(student_answer).__name__
        msg = f"- The answer should be of type 'list'; your type is {name_type}"
        msg_list.append(msg)
    else:
        msg = "- The answer is type list. Correct."
        msg_list.append(msg)

    # Check length of list
    if status:
        if len(student_answer) != len(instructor_answer):
            status = False
            msg = (
                "- The length of your list is incorrect. Your list length is "
                "{len(student_answer)}. The length should be {len(instructor_answer)}."
            )
            msg_list.append(msg)
        else:
            msg_list.append("- The length of the list is correct.")

    if status:
        for s_arr in instructor_answer:
            if not isinstance(s_arr, int | np.integer):
                status = False
                msg_list.append("- Element {i} of your list should be of type 'int'.")

    if status:
        msg_list.append("- All list elements are type 'int'.")

    return status, "\n".join(msg_list)


# ======================================================================
def check_answer_list_float(
    student_answer: list[float],
    instructor_answer: list[float],
    rel_tol: float,
    monotone_increasing: bool | None = None,
    partial_score_frac: list[float] | None = None,
) -> tuple[bool, str]:
    """Check that all elements in the list have matching norms.

    Args:
        student_answer (list[float]): The student's submitted answer
        instructor_answer (list[float]): The instructor's correct answer
        rel_tol (float): The relative tolerance for comparing float values
        monotone_increasing (bool | None): Whether the list should be monotone increasing
        partial_score_frac (list[float] | None): A list to store the partial score fraction

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    msg_list = []
    status = True
    answ_eq_len = len(student_answer) == len(instructor_answer)  # checked in structure
    ps_dict = init_partial_score_dict()
    ps_dict["nb_total"] = len(instructor_answer)

    if answ_eq_len and (monotone_increasing is None or monotone_increasing is False):
        status, msg_list_ = check_list_float(
            student_answer,
            instructor_answer,
            rel_tol=rel_tol,
            abs_tol=1.0e-6,
            ps_dict=ps_dict,
        )
        msg_list.extend(msg_list_)
    elif monotone_increasing is True:
        # Check whether the list is monotone incrreasing. If not, fail.
        val = student_answer[0]
        for el_val in student_answer[1:]:
            if el_val >= val:
                continue
            status = False
            msg_list.append("The answer is not monotonically increasing")

    if not status:
        msg_list.append("Some elements are incorrect")

    if partial_score_frac is None:
        partial_score_frac = []

    if monotone_increasing:
        partial_score_frac[0] = 1.0
    else:
        partial_score_frac[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]

    if not msg_list:
        msg_list = ["Answer matches expected values. "]

    return return_value(status, msg_list, student_answer, instructor_answer)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_list_float(
    student_answer: list[float],
    instructor_answer: list[float],
) -> tuple[bool, str]:
    """Check if a student's list of floats matches the instructor's answer.

    Args:
        student_answer (list[float]): The student's submitted answer
        instructor_answer (list[float]): The instructor's correct answer

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    status = True
    msg_list = []

    if not isinstance(student_answer, list):
        status = False
        msg = f"- The answer should be of type 'list'; your type is {type(student_answer).__name__}"
        msg_list.append(msg)
    else:
        msg = "- The answer is type list. Correct."
        msg_list.append(msg)

    # Check length of list
    if status:
        if len(student_answer) != len(instructor_answer):
            status = False
            msg = (
                "- The length of your list is incorrect. Your list length is "
                f"{len(student_answer)}.  The length should be {len(instructor_answer)}."  # noqa: E501
            )
            msg_list.append(msg)
        else:
            msg = "- The length of the list is correct."
            msg_list.append(msg)

    if status:
        for s_arr in instructor_answer:
            if not isinstance(s_arr, float | np.floating | int | np.integer):
                status = False
                msg_list.append("- Element {i} of your list should be of type 'float'.")

    if status:
        msg_list.append("- All list elements are type 'float'.")

    return status, "\n".join(msg_list)


# ======================================================================


def check_answer_list_ndarray(
    student_answer: list[np.ndarray],
    instructor_answer: list[np.ndarray],
    rel_tol: float,
    partial_score_frac: list[float],
) -> tuple[bool, str]:
    """Check that all elements in the list have matching norms.

    Args:
        student_answer (list[np.ndarray]): The student's submitted answer
        instructor_answer (list[np.ndarray]): The instructor's correct answer
        rel_tol (float): The relative tolerance for comparing float values
        partial_score_frac (list[float]): A list to store the partial score fraction

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    msg_list = []
    status = True
    answ_eq_len = len(student_answer) == len(instructor_answer)
    i_norm_list = []
    s_norm_list = []
    ps_dict = init_partial_score_dict()
    ps_dict["nb_total"] = len(instructor_answer)

    if answ_eq_len:
        for s_arr, i_arr in zip(student_answer, instructor_answer, strict=True):
            s_norm = cast(float, np.linalg.norm(s_arr))
            i_norm = cast(float, np.linalg.norm(i_arr))
            i_norm_list.append(i_norm)
            s_norm_list.append(s_norm)
            """
            # print(
            #   "IMPROVE: could first create a list of norms, and call check_list_float"
            # )
            """
            status_, msg = check_float(i_norm, s_norm, rel_tol, abs_tol=1.0e-5)
            if status_ is False:
                status = False
                msg_list.append([msg])
                ps_dict["nb_mismatches"] += 1

    if not status:
        msg_list.append("Replace the arrays by their norms")

    partial_score_frac[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]

    if not msg_list:
        msg_list = ["Answer matches expected values. "]

    return return_value(status, msg_list, s_norm_list, i_norm_list)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_list_ndarray(
    student_answer: list[np.ndarray],
    instructor_answer: list[np.ndarray],
) -> tuple[bool, str]:
    """Check that elements in the list are ndarrays.

    Args:
        student_answer (list[np.ndarray]): The student's submitted answer
        instructor_answer (list[np.ndarray]): The instructor's correct answer

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    status = True
    msg_list = []

    if not isinstance(student_answer, list):
        status = False
        msg = f"The answer should be of type 'list'; your type is {type(student_answer).__name__}"  # noqa: E501
        msg_list.append(msg)
    else:
        msg = "The answer is type list. Correct."
        msg_list.append(msg)

    # Check length of list
    if status:
        if len(student_answer) != len(instructor_answer):
            status = False
            msg = (
                "The length of your list is incorrect. Your list length is "
                f"{len(student_answer)}. The length should be {len(instructor_answer)}."
            )
            msg_list.append(msg)
        else:
            msg = "The length of the list is correct."
            msg_list.append(msg)

    if status:
        for s_arr in instructor_answer:
            if not isinstance(s_arr, type(np.zeros(1))):
                status = False
                msg_list.append("Element {i} of your list should be of type 'numpy.array'.")

    if status:
        msg_list.append("All list elements are type ndarray.")

    return status, "\n".join(msg_list)


# ======================================================================


def check_answer_ndarray(
    student_answer: np.ndarray,
    instructor_answer: np.ndarray,
    rel_tol: float,
) -> tuple[bool, str]:
    """Check that all elements in the list have matching norms.

    Args:
        student_answer (np.ndarray): The student's submitted answer
        instructor_answer (np.ndarray): The instructor's correct answer
        rel_tol (float): The relative tolerance for comparing float values

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    msg_list = []
    status = True
    s_norm = cast(float, np.linalg.norm(student_answer))
    i_norm = cast(float, np.linalg.norm(instructor_answer))

    # Can i_norm be zero?
    status, msg_ = check_float(i_norm, s_norm, rel_tol, 1.0e-5)

    if not status:
        msg_list.append(msg_)
        msg_list.append("For comparison, the array was replaced by its norm")
        msg_list.append(f"The norms have relative error > {rel_tol}")

    if not msg_list:
        msg_list = ["Answer matches expected values. "]

    return return_value(status, msg_list, s_norm, i_norm)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_ndarray(student_answer: np.ndarray) -> tuple[bool, str]:
    """Check that all elements in the list have matching norms.

    Args:
        student_answer (np.ndarray): The student's submitted answer

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    if not isinstance(student_answer, type(np.zeros([1]))):
        return (
            False,
            f"- Answer should be a numpy array rather than {type(student_answer)}",
        )
    return True, "Type 'ndarray' is correct."


# ======================================================================


def check_answer_function(
    student_answer: Callable,
    instructor_answer: Callable,
) -> tuple[bool, str]:
    """Check that the student's function matches the instructor's function.

    Args:
        student_answer (Callable): The student's submitted answer
        instructor_answer (Callable): The instructor's correct answer

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    s_source = inspect.getsource(student_answer)
    i_source = inspect.getsource(instructor_answer)

    msg_list = []
    msg_list.append("Functions are not graded, unless not present.")
    msg_list.append("Instructor function Source")
    msg_list.append(s_source)

    status = True

    if not msg_list:
        msg_list = ["Answer matches expected values."]

    return return_value(status, msg_list, s_source, i_source)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_function(student_answer: Callable) -> tuple[bool, str]:
    """Check that the student's function matches the instructor's function.

    Args:
        student_answer (Callable): The student's submitted answer

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    if not isinstance(student_answer, type(lambda: None)):
        return False, "- Answer should be a Python function."
    return True, "Type 'function' is correct."


# ======================================================================


def check_answer_list_list_float(
    student_answer: list[list[float]],
    instructor_answer: list[list[float]],
    rel_tol: float,
    partial_score_frac: list[float],
) -> tuple[bool, str]:
    """Check two lists of lists of floats with each other.

    Args:
        student_answer (list[list[float]]): The student's submitted answer
        instructor_answer (list[list[float]]): The instructor's correct answer
        rel_tol (float): The relative tolerance for comparing float values
        partial_score_frac (list[float]): A list to store the partial score fraction

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    status = True
    msg_list = []
    ps_dict = init_partial_score_dict()

    for s_lst, i_lst in zip(student_answer, instructor_answer, strict=True):
        status_, msg_list_ = check_list_float(i_lst, s_lst, rel_tol, 1.0e-6, ps_dict)
        msg_list.extend(msg_list_)
        if status is True:
            status = status_

    partial_score_frac[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]
    msg_list.append(f"Answer correct if relative error < {rel_tol * 100} percent")

    if not msg_list:
        msg_list = ["Answer matches expected values."]

    return return_value(status, msg_list, student_answer, instructor_answer)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_list_list_float(
    student_answer: list[list[float]],
    instructor_answer: list[list[float]],
) -> tuple[bool, str]:
    """Check structure of student_answer.

    Args:
        student_answer (list[list[float]]): The student's submitted answer
        instructor_answer (list[list[float]]): The instructor's correct answer

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    msg_list = []
    status = True

    if not isinstance(student_answer, list):
        msg_list.append("- answer should be a list.")
        status = False
        return status, "\n".join(msg_list)

    if len(student_answer) != len(instructor_answer):
        msg_list.append("- Number of elements in the answer is incorrect.")
        status = False

    for i, s_list in enumerate(student_answer):
        if not isinstance(s_list, list):
            msg_list.append(f"- answer[{i}] is not a list. Recheck all elements.")
            status = False
            continue

        for j, el in enumerate(s_list):
            if not isinstance(el, float | np.floating | int | np.integer):
                msg = (
                    f"- answer[{i}][{j}]={el} cannot be cast to a float. All elements "
                    "must be castable to float."
                )
                msg_list.append(msg)
                status = False

    if status:
        msg_list.append("Type 'list[list[float]]' is correct.")

    if not msg_list:
        msg_list = ["Answer matches expected values."]

    return status, "\n".join(msg_list)


# ======================================================================
def check_structure_list_set(
    student_answer: list[set],
    instructor_answer: list[set],
) -> tuple[bool, str]:
    """Check structure of student_answer.

    Args:
        student_answer (list[set]): The student's submitted answer
        instructor_answer (list[set]): The instructor's correct answer

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    msg_list: list[str] = []
    status = True
    if not isinstance(student_answer, list):
        msg_list.append("- The answer should be a list.")
        status = False
    return status, "\n".join(msg_list)


# ======================================================================


def check_answer_list_set(
    student_answer: list[set],
    instructor_answer: list[set],
) -> tuple[bool, str]:
    """Check two lists of sets (although the set is encoded as a list) with each other.

    Args:
        student_answer (list[set]): The student's submitted answer
        instructor_answer (list[set]): The instructor's correct answer

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    msg_list: list[str] = []
    status = True
    for s_set, i_set in zip(student_answer, instructor_answer, strict=True):
        if i_set != s_set:  # works for sets of ints and strings
            status = False
            msg_list.append("- Instruct and student sets are not equal.")
            break

    if not msg_list:
        msg_list = ["Answer matches expected values."]

    return status, "\n".join(msg_list)


# ======================================================================


def convert_to_set_of_sets(input_sequence: set[set[Any]]) -> set[frozenset[Any]]:
    """Convert each inner sequence to a set, then the outer sequence to a set of sets.

    Args:
        input_sequence (set[set[Any]]): The input sequence to convert

    Returns:
        set[frozenset[Any]]: The converted set of sets

    """
    return {frozenset(inner) for inner in input_sequence}


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_answer_set_set_int(
    student_answer: set[set[int]],
    instructor_answer: set[set[int]],
) -> tuple[bool, str]:
    """Check two sets of sets of integers with each other.

    Args:
        student_answer (set[set[int]]): The student's submitted answer
        instructor_answer (set[set[int]]): The instructor's correct answer

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    Notes:
        Both student answer and instructor answer should be a set of sets or a structure
            that can be converted to a set of sets

    """
    status = True
    msg_list = []

    seq_s = student_answer
    seq_i = instructor_answer
    # Convert both sequences to sets of sets
    # One might start with a list of lists [of objects]
    set_of_sets_s = convert_to_set_of_sets(seq_s)
    set_of_sets_i = convert_to_set_of_sets(seq_i)

    """
    # print("set_of_sets_s= ", set_of_sets_s)
    # print("set_of_sets_i= ", set_of_sets_i)
    """

    # Compare the sets of sets
    # What is actually compared?
    status = set_of_sets_s == set_of_sets_i

    if not msg_list:
        msg_list = ["Answer matches expected values. "]

    return return_value(status, msg_list, set_of_sets_s, set_of_sets_i)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .
# Function to check if an object is a sequence but not a string
def is_sequence_but_not_str(obj: list | tuple | set) -> bool:
    """Check if an object is a sequence (list, tuple, or set) but not a string.

    Args:
        obj (Any): The object to check

    Returns:
        bool: True if obj is a list, tuple or set but not a string, False otherwise

    """
    return isinstance(obj, list | tuple | set)


def check_structure_set_set_int(student_answer: set[set[int]]) -> tuple[bool, str]:
    """Check structure of student_answer.

    Args:
        student_answer (set[set[int]]): The student's submitted answer

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    msg_list = []
    status = True
    seq_s = student_answer

    # Check if the outer structures are sequences
    if not is_sequence_but_not_str(seq_s):
        msg_list.append("- The outer structure is not a sequence (list or set or tuple).")
        status = False
    else:
        msg_list.append("- The outer structure is a sequence (list or set or tuple).")

    # If outer structures are sequences, check each inner structure
    if status:
        for _, seq in enumerate(seq_s, start=1):
            if not is_sequence_but_not_str(seq):
                msg_list.append(
                    "Element {i} of the outer set is not compatible with a set and has type {seq}.",  # noqa: E501
                )
                status = False
                continue
        if status:
            msg_list.append(
                "- All elements of the outer set are compatible with a set (which means I can coerce it into a set",  # noqa: E501
            )
            msg_list.append("- Answer has the correct structure")

    return status, "\n".join(msg_list)


# ======================================================================


def check_answer_dict_str_tuple_ndarray(
    student_answer: dict[str, tuple[NDArray]],
    instructor_answer: dict[str, tuple[NDArray]],
    rel_tol: float,
    partial_score_frac: list[float],
) -> tuple[bool, str]:
    """Check two dictionaries with keys:str, values: tuple of ndarrays with each other.

    Args:
        student_answer (dict[str, tuple[NDArray]]): The student's submitted answer
        instructor_answer (dict[str, tuple[NDArray]]): The instructor's correct answer
        rel_tol (float): The relative tolerance for comparing float values
        partial_score_frac (list[float]): A list to store the partial score fraction

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    msg_list = []
    status = True  # Assuming correct until proven otherwise
    ps_dict = init_partial_score_dict()

    # Dictionaries to hold norms for student and instructor answers
    s_norms = {}
    i_norms = {}

    for k, i_value in instructor_answer.items():
        # Initialize norms list for current key in both dicts
        s_norms[k] = []
        i_norms[k] = []

        try:
            s_tuple = student_answer[k]
        except KeyError:
            msg_list.append(f"Error: key {k!r} is missing")
            continue

        i_tuple = i_value
        for s_arr, i_arr in zip(s_tuple, i_tuple, strict=True):
            # Calculate norms
            s_norm = np.linalg.norm(s_arr)
            i_norm = np.linalg.norm(i_arr)

            # Store norms
            s_norms[k].append(s_norm)
            i_norms[k].append(i_norm)

        # ! print(f"{i_norms=}, {s_norms=}")
        status_, msg_ = check_list_float(
            i_norms[k],
            s_norms[k],
            rel_tol,
            abs_tol=1.0e-6,
            ps_dict=ps_dict,
        )

        if status_ is False:
            msg_list.append(msg_)
            status = False

    partial_score_frac[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]
    msg_list.append("Only print the norms of the arrays")

    if not msg_list:
        msg_list = ["Answer matches expected values. "]

    return return_value(status, msg_list, s_norms, i_norms)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_dict_str_tuple_ndarray(
    student_answer: dict[str, tuple[NDArray]],
    instructor_answer: dict[str, tuple[NDArray]],
) -> tuple[bool, str]:
    """Check structure of student_answer.

    Args:
        student_answer (dict[str, tuple[NDArray]]): The student's submitted answer
        instructor_answer (dict[str, tuple[NDArray]]): The instructor's correct answer

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    status = True
    msg_list = []
    for k, v in instructor_answer.items():
        # repr adds additional quotes; str does not.
        if k not in student_answer:
            msg_list.append(f"- Missing key {k!r}")
            status = False
            continue
        if not isinstance(v, tuple | list):
            msg_list.append(f"- dict[{k!r}] is not a tuple")
            status = False
            continue
        for i, v_el in enumerate(v):
            if not isinstance(v_el, type(np.zeros(1))):
                msg_list.append(f"- dict[{k!r}][{i}] is not an numpy array")
                status = False

    if status:
        msg_list.append("Type 'dict[str, tuple(ndarray)]' is correct.")

    return status, "\n".join(msg_list)


# ======================================================================


def check_answer_dendrogram(
    student_dendro: dict[str, Any],
    instructor_dendro: dict[str, Any],
    rel_tol: float,
) -> tuple[bool, str]:
    """Check if the student's dendrogram is equal to the instructor's dendrogram.

    Args:
        student_dendro (dict[str, Any]): The student's dendrogram
        instructor_dendro (dict[str, Any]): The instructor's dendrogram
        rel_tol (float): The relative tolerance for coordinate comparison

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    status = True
    msg_list = []

    dend1 = student_dendro
    dend2 = instructor_dendro

    # Might crash if structures don't match. To fix later.
    # Coordinate comparison
    if not np.allclose(dend1["icoord"], dend2["icoord"], rtol=rel_tol):
        status = False
    if not np.allclose(dend1["dcoord"], dend2["dcoord"], rtol=rel_tol):
        status = False

    # Leaf order comparison
    if not np.array_equal(dend1["leaves"], dend2["leaves"]):
        status = False

    # Optional: Color comparison
    if "color_list" in dend1 and "color_list" in dend2:
        zip_iterator = zip(dend1["color_list"], dend2["color_list"], strict=True)
        if not all(np.array_equal(c1, c2) for c1, c2 in zip_iterator):
            status = False
            msg_list.append("Color list mismatch")

    if not msg_list:
        msg_list = ["Answer matches expected values. "]

    return return_value(status, msg_list, student_dendro, instructor_dendro)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_dendrogram(student_dendro: dict[str, Any]) -> tuple[bool, str]:
    """Check if the structure and types of the student's dendrogram dictionary match.

    Args:
         student_dendro: The dendrogram dictionary from a student.
         instructor_dendro: not used.

    Returns:
         A tuple (bool, str): True and an empty string if the structure matches,
         otherwise False and a message indicating the mismatch.

    """
    # Expected keys in a scipy dendrogram
    expected_keys = {"icoord", "dcoord", "leaves", "ivl", "color_list"}
    student_keys = set(student_dendro.keys())

    # Check for missing keys in student dendrogram
    missing_keys = expected_keys - student_keys
    if missing_keys:
        return False, f"Missing keys in student dendrogram: {missing_keys}."

    # For each key, check if the values are lists and have a consistent structure
    for key in expected_keys:
        if (
            key in student_keys
        ):  # Check only if key is present, though missing keys were already caught
            value = student_dendro[key]

            # Ensure value is a list
            if not isinstance(value, list):
                return False, f"Expected a list for key '{key}', found {type(value)}."

            # Specific checks for 'icoord' and 'dcoord' which should contain lists of lists
            if key in ["icoord", "dcoord"] and not all(isinstance(item, list) for item in value):  # noqa: E501
                return False, f"Expected a list of lists for key '{key}'."

            # 'leaves' and 'ivl' checks could be added here, such as ensuring 'leaves' contains integers,
            # and 'ivl' contains strings, if necessary for the scope of your validation.

    # If we reach here, the structure is as expected
    return True, "Type Dendrogram has correct structure."


# ======================================================================


def check_answer_int(
    student_answer: int,
    instructor_answer: int,
) -> tuple[bool, str]:
    """Check if the student's answer is equal to the instructor's answer.

    Args:
        student_answer (int): The student's submitted answer
        instructor_answer (int): The instructor's correct answer

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    status, msg = check_int(instructor_answer, student_answer)

    if not msg:
        msg = "Answer matches expected value. "

    return return_value(status, [msg], student_answer, instructor_answer)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_int(student_answer: int) -> tuple[bool, str]:
    """Check if the student's answer is an integer.

    Args:
        student_answer (int): The student's submitted answer

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    if not isinstance(student_answer, int | np.integer):
        status = False
        msg = (
            f"Answer must be of type 'int'. Your answer is of type {type(student_answer).__name__}."
        )
        msg_list = [msg]
    else:
        status = True
        msg_list = ["Answer is of type 'int' as expected."]

    return status, "\n".join(msg_list)


# ======================================================================


def check_answer_bool(student_answer: bool, instructor_answer: bool) -> tuple[bool, str]:
    """Check if the student's answer is equal to the instructor's answer.

    Args:
        student_answer (bool): The student's submitted answer
        instructor_answer (bool): The instructor's correct answer

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    msg_list = []
    status = True

    if student_answer != instructor_answer:
        status = False
        msg_list = ["Answer is incorrect."]
    else:
        status = True
        msg_list = ["Answer is correct."]

    if not msg_list:
        msg_list = ["Answer matches expected value. "]

    return return_value(status, msg_list, student_answer, instructor_answer)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_bool(student_answer: bool) -> tuple[bool, str]:
    """Check if the student's answer is a boolean.

    Args:
        student_answer (bool): The student's submitted answer

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    print(f"\n==> inside check_structure_bool, {student_answer=}")
    if not isinstance(student_answer, (bool, np.bool_)):
        status = False
        msg = f"Answer must be of type 'bool'. Your answer is of type {type(student_answer)}."
        msg_list = [msg]
    else:
        status = True
        msg_list = ["Answer is of type 'bool' as expected."]

    return status, "\n".join(msg_list)


# ======================================================================


def check_answer_list_str(
    student_answer: list[str],
    instructor_answer: list[str],
    include_indices: list[int],
    exclude_indices: list[int],
    partial_score_frac: list[float],
) -> tuple[bool, str]:
    """Check if the student's answer is equal to the instructor's answer.

    Args:
        student_answer (list[str]): The student's submitted answer
        instructor_answer (list[str]): The instructor's correct answer
        include_indices (list[int]): The indices to include in the comparison
        exclude_indices (list[int]): The indices to exclude from the comparison
        partial_score_frac (list[float]): A list to store the partial score fraction

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    msg_list = []
    status = True
    mismatched_strings = []
    ps_dict = init_partial_score_dict()

    normalized_s_answ = [clean_str_answer(s) for s in student_answer]
    normalized_i_answ = [clean_str_answer(s) for s in instructor_answer]

    for i, i_a in enumerate(normalized_i_answ):
        if exclude_indices != [] and include_indices == [] and i in exclude_indices:
            continue
        if include_indices != [] and exclude_indices == [] and i not in include_indices:
            continue

        ps_dict["nb_total"] += 1
        s_a = normalized_s_answ[i]

        if s_a != i_a:
            status = False
            ps_dict["nb_mismatches"] += 1
            mismatched_strings.append(s_a)

    try:
        partial_score_frac[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]
        print(f"==> inside check_answer_list_str, {partial_score_frac=}")
    except ZeroDivisionError:
        partial_score_frac[0] = 1.0

    # ! TODO: Explicitly state the indices considered for grading.
    """
    # msg_list += [f"List elements in position()s {exclude_indices} is/are not graded.\n"]
    # msg_list += [f"Only list elements in position()s {include_indices} is/are not graded.\n"]
    """
    msg = f"There is/are {len(mismatched_strings)} mismatched string(s): ({mismatched_strings})."
    msg_list += [msg]

    if not msg_list:
        msg_list = ["Answer matches expected values. "]

    return return_value(status, msg_list, normalized_s_answ, normalized_i_answ)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_list_str(student_answer: list[str]) -> tuple[bool, str]:
    """Check if the student's answer is a list of strings.

    Args:
        student_answer (list[str]): The student's submitted answer

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    msg_list = []

    # Function to check if an object is a list of strings
    def is_list_of_strings(obj: list[str]) -> bool:
        return isinstance(obj, list) and all(isinstance(element, str) for element in obj)

    # Check if both sequences are lists of strings
    if not is_list_of_strings(student_answer):
        msg_list.append("Answer must be a list of strings.")
        status = False
    else:
        msg_list.append("Type 'list[str]' is correct")
        status = True

    return status, "\n".join(msg_list)


# ======================================================================


# The type of student_answer is matplotlib.plot
def check_answer_lineplot(
    student_answer: list[Line2D],
    instructor_answer: list[Line2D],
    rel_tol: float,
) -> tuple[bool, str]:
    """Check if the student's answer is equal to the instructor's answer.

    Args:
        student_answer (matplotlib.plot): The student's submitted answer
        instructor_answer (matplotlib.plot): The instructor's correct answer
        rel_tol (float): The relative tolerance for coordinate comparison

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    Notes:
        Check the following:
            For each line plot:
            Number of points on each plot
            Min and max point values in x and y

    """

    status = True
    msg_list = []

    if not msg_list:
        msg_list = ["Answer matches expected values. "]

    return return_value(status, msg_list, student_answer, instructor_answer)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_lineplot(student_answer: list[Line2D] | Line2D) -> tuple[bool, str]:
    """Check if the student's answer is a lineplot.

    Args:
        student_answer (matplotlib.plot): The student's submitted answer

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    Notes:
        Assume use of Matplotlib
        Lineplots generated by matlab. Check one or multiple lines.
        Check whether subplot is used. This should be stipulated in the assignment.
        Check the number of curves
        Check presence of axis labels and title
        Check presence of grid in x and y

        # I might need intructor answer to check the number of plots. Alternatively,
        # include options to allow specified checking, via kwargs. In that way,
        # I can check the instructor answer AND the student answer structurally

    """
    # from matplotlib.figure import Figure
    # from matplotlib.lines import Line2D

    status = True
    msg_list = []

    """
    # print("==> check structure lineplot")
    # get axis object
    # print(f"{student_answer=}")
    """
    if isinstance(student_answer, list):
        student_answer = student_answer[0]
        msg_list.append("A list of Line2D obects detected. We only consider the first ")
        msg_list.append("one in the structure evaluation.")

    if not isinstance(student_answer, Line2D):
        msg_list.append("Wrong plot type, not created with plt.plot!")
        status = False
        return status, "\n".join(msg_list)

    student_answer.figure = cast(Figure, student_answer.figure)
    s_ax = student_answer.figure.gca()
    """
    # print(f"{dir(s_ax)=}")
    # print(f"{type(s_ax).__name__=}")
    # s_ax = s_ax.gcf().get_axes()
    # i_ax = instructor_answer[0].gcf().get_axes()
    """

    """
    if len(s_ax) != len(i_ax):
        msg_list.append(f"There should only be {len(i_ax)} plot(s)!")
        status = False
    """

    if type(student_answer).__name__ != "Line2D":
        msg_list.append("Wrong plot type, not created with plt.plot!")
        status = False

    if not status:
        return status, "\n".join(msg_list)

    # What happens if the label is not a sring? None if no label
    s_xlabel = clean_str_answer(s_ax.get_xlabel())  # if s_ax.get_xlabel() else None
    # ! i_xlabel = clean_str_answer(i_ax.get_xlabel())  # if s_ax.get_xlabel() else None
    s_ylabel = clean_str_answer(s_ax.get_ylabel())  # if s_ax.get_ylabel() else None
    # ! i_ylabel = clean_str_answer(i_ax.get_ylabel())  # if s_ax.get_ylabel() else None
    s_title = clean_str_answer(s_ax.get_title())  # if s_ax.get_title() else None
    # ! i_title = clean_str_answer(i_ax.get_title())  # if s_ax.get_title() else None

    if not s_xlabel or not s_ylabel:
        msg_list.append("Either x or y label is missing! Must be there to get a grade.")
        status = False

    if not s_title:
        msg_list.append("The title is missing")
        status = False

    """ USE LATER
    s_xgrid_vis = any(line.getvisible() for line in s_ax.xaxis.get_gridlines())
    i_xgrid_vis = any(line.getvisible() for line in i_ax.xaxis.get_gridlines())
    s_ygrid_vis = any(line.getvisible() for line in s_ax.yaxis.get_gridlines())
    i_ygrid_vis = any(line.getvisible() for line in i_ax.yaxis.get_gridlines())
    """

    """
    ax:  Axes3D
    fig:  Figure
    scat3:  Path3DCollection
    scat2:  PathCollection
    plot:  list
    plot[0]:  Line2D
    """

    return status, "\n".join(msg_list)


# ======================================================================


def check_structure_decisiontreeclassifier(student_answer) -> tuple[bool, str]:
    from sklearn.tree import DecisionTreeClassifier

    if not isinstance(student_answer, DecisionTreeClassifier):
        status = False
        msg = (
            f"Answer must be of type 'DecisionTreeClassifier'. Your answer is "
            f"of type {type(student_answer).__name__}."
        )
        msg_list = [msg]
    else:
        status = True
        msg_list = ["Answer is of type 'DecisionTreeClassifier' as expected."]

    return status, "\n".join(msg_list)


def check_answer_decisiontreeclassifier(
    student_answer,
    instructor_answer,
) -> tuple[bool, str]:
    """Check if student's DecisionTreeClassifier matches instructor's.

    Args:
        student_answer: Student's DecisionTreeClassifier object
        instructor_answer: Instructor's DecisionTreeClassifier object

    Returns:
        tuple[bool, str]: Status indicating if answers match and message detailing
            any mismatches
    """
    status = True
    msg_list = []

    # Check core parameters
    params_to_check = ["criterion", "max_depth", "min_samples_split", "random_state"]
    for param in params_to_check:
        student_val = getattr(student_answer, param)
        instructor_val = getattr(instructor_answer, param)
        if student_val != instructor_val:
            status = False
            msg_list.append(
                f"Parameter '{param}' mismatch: expected {instructor_val}, got {student_val}"
            )

    if not msg_list:
        msg_list = ["DecisionTreeClassifier parameters match expected values"]

    return return_value(status, msg_list, student_answer, instructor_answer)


# ======================================================================


def check_structure_logisticregression(student_answer) -> tuple[bool, str]:
    from sklearn.linear_model import LogisticRegression

    if not isinstance(student_answer, LogisticRegression):
        status = False
        msg = (
            f"Answer must be of type 'LogisticRegression'. Your answer is "
            f"of type {type(student_answer).__name__}."
        )
        msg_list = [msg]
    else:
        status = True
        msg_list = ["Answer is of type 'LogisticRegression' as expected."]

    return status, "\n".join(msg_list)


# ======================================================================


def check_structure_svc(student_answer) -> tuple[bool, str]:
    from sklearn.svm import SVC

    if not isinstance(student_answer, SVC):
        status = False
        msg = (
            f"Answer must be of type 'SVC'. Your answer is of type {type(student_answer).__name__}."
        )
        msg_list = [msg]
    else:
        status = True
        msg_list = ["Answer is of type 'SVC' as expected."]

    return status, "\n".join(msg_list)


def check_answer_svc(
    student_answer,
    instructor_answer,
) -> tuple[bool, str]:
    """Check if student's SVC classifier matches instructor's.

    Args:
        student_answer: Student's SVC classifier
        instructor_answer: Instructor's SVC classifier

    Returns:
        tuple[bool, str]: Status indicating if answers match and message detailing any mismatches
    """
    status = True
    msg_list = []

    # Check core parameters
    params_to_check = ["kernel", "C", "random_state"]
    for param in params_to_check:
        student_val = getattr(student_answer, param)
        instructor_val = getattr(instructor_answer, param)
        if student_val != instructor_val:
            status = False
            msg_list.append(
                f"Parameter '{param}' mismatch: expected {instructor_val}, got {student_val}"
            )

    # Check kernel-specific parameters if using non-linear kernel
    # if student_answer.kernel == "rbf":
    #     if student_answer.gamma != instructor_answer.gamma:
    #         status = False
    #         msg_list.append(
    #             f"Parameter 'gamma' mismatch: expected {instructor_answer.gamma}, got {student_answer.gamma}"
    #         )

    if not msg_list:
        msg_list = ["SVC parameters match expected values. "]

    return return_value(status, msg_list, student_answer, instructor_answer)


# ======================================================================


# ======================================================================


def check_answer_shufflesplit(
    student_answer,
    instructor_answer,
) -> tuple[bool, str]:
    """Check if student's ShuffleSplit matches instructor's.

    Args:
        student_answer: Student's ShuffleSplit object
        instructor_answer: Instructor's ShuffleSplit object

    Returns:
        tuple[bool, str]: Status indicating if answers match and message detailing
            any mismatches

    """
    status = True
    msg_list = []

    # Check core parameters
    params_to_check = ["n_splits", "test_size", "train_size", "random_state"]
    for param in params_to_check:
        student_val = getattr(student_answer, param)
        instructor_val = getattr(instructor_answer, param)
        if student_val != instructor_val:
            status = False
            msg_list.append(
                f"Parameter '{param}' mismatch: expected {instructor_val}, got {student_val}"
            )

    return status, "\n".join(
        msg_list
    ) if msg_list else "ShuffleSplit parameters match expected values"


# ======================================================================
"""
def check_structure_kfold(student_answer) -> tuple[bool, str]:
    from sklearn.model_selection import KFold

    if not isinstance(student_answer, KFold):
        status = False
        msg = (
            f"Answer must be of type 'KFold'. Your answer is "
            f"of type {type(student_answer).__name__}."
        )
        msg_list = [msg]
    else:
        status = True
        msg_list = ["Answer is of type 'KFold' as expected."]

    return status, "\n".join(msg_list)
"""


def check_structure_kfold(student_answer) -> tuple[bool, str]:
    from sklearn.model_selection import KFold

    if not isinstance(student_answer, KFold):
        status = False
        msg = (
            f"Answer must be of type 'KFold'. Your answer is "
            f"of type {type(student_answer).__name__}."
        )
        msg_list = [msg]
    else:
        status = True
        msg_list = ["Answer is of type 'KFold' as expected."]

    return status, "\n".join(msg_list)


# ======================================================================


def check_structure_stratifiedkfold(student_answer) -> tuple[bool, str]:
    from sklearn.model_selection import StratifiedKFold

    if not isinstance(student_answer, StratifiedKFold):
        status = False
        msg = (
            f"Answer must be of type 'StratifiedKFold'. Your answer is "
            f"of type {type(student_answer).__name__}."
        )
        msg_list = [msg]
    else:
        status = True
        msg_list = ["Answer is of type 'StratifiedKFold' as expected."]

    return status, "\n".join(msg_list)


def check_answer_stratifiedkfold(
    student_answer,
    instructor_answer,
) -> tuple[bool, str]:
    """Check if student's StratifiedKFold matches instructor's.

    Args:
        student_answer: Student's StratifiedKFold object
        instructor_answer: Instructor's StratifiedKFold object

    Returns:
        tuple[bool, str]: Status indicating if answers match and message detailing
            any mismatches

    """
    status = True
    msg_list = []

    # Check core parameters
    params_to_check = ["n_splits", "shuffle", "random_state"]
    for param in params_to_check:
        student_val = getattr(student_answer, param)
        instructor_val = getattr(instructor_answer, param)
        if student_val != instructor_val:
            status = False
            msg_list.append(
                f"Parameter '{param}' mismatch: expected {instructor_val}, got {student_val}"
            )

    if not msg_list:
        msg_list = ["StratifiedKFold parameters match expected values. "]

    return return_value(status, msg_list, student_answer, instructor_answer)


# ======================================================================


def check_answer_kfold(
    student_answer,
    instructor_answer,
) -> tuple[bool, str]:
    """Check if student's KFold matches instructor's.

    Args:
        student_answer: Student's KFold object
        instructor_answer: Instructor's KFold object

    Returns:
        tuple[bool, str]: Status indicating if answers match and message detailing
            any mismatches
    """
    status = True
    msg_list = []

    # Check core parameters
    params_to_check = ["n_splits", "shuffle", "random_state"]
    for param in params_to_check:
        student_val = getattr(student_answer, param)
        instructor_val = getattr(instructor_answer, param)
        if student_val != instructor_val:
            status = False
            msg_list.append(
                f"Parameter '{param}' mismatch: expected {instructor_val}, got {student_val}"
            )

    if not msg_list:
        msg_list = ["KFold parameters match expected values. "]

    return return_value(status, msg_list, student_answer, instructor_answer)


# ======================================================================
"""
def check_structure_shufflesplit(student_answer) -> tuple[bool, str]:
    from sklearn.model_selection import ShuffleSplit

    if not isinstance(student_answer, ShuffleSplit):
        status = False
        msg = (
            f"Answer must be of type 'ShuffleSplit'. Your answer is "
            f"of type {type(student_answer).__name__}."
        )
        msg_list = [msg]
    else:
        status = True
        msg_list = ["Answer is of type 'ShuffleSplit' as expected."]

    return status, "\n".join(msg_list)
    """


def check_structure_shufflesplit(student_answer) -> tuple[bool, str]:
    from sklearn.model_selection import ShuffleSplit

    if not isinstance(student_answer, ShuffleSplit):
        status = False
        msg = (
            f"Answer must be of type 'ShuffleSplit'. Your answer is "
            f"of type {type(student_answer).__name__}."
        )
        msg_list = [msg]
    else:
        status = True
        msg_list = ["Answer is of type 'ShuffleSplit' as expected."]

    return status, "\n".join(msg_list)


# ======================================================================


def check_structure_gridsearchcv(student_answer) -> tuple[bool, str]:
    from sklearn.model_selection import GridSearchCV

    if not isinstance(student_answer, GridSearchCV):
        status = False
        msg = (
            f"Answer must be of type 'GridSearchCV'. Your answer is "
            f"of type {type(student_answer).__name__}."
        )
        msg_list = [msg]
    else:
        status = True
        msg_list = ["Answer is of type 'GridSearchCV' as expected."]

    return status, "\n".join(msg_list)


def check_answer_gridsearchcv(
    student_answer,
    instructor_answer,
) -> tuple[bool, str]:
    """Check if student's GridSearchCV matches instructor's.

    Args:
        student_answer: Student's GridSearchCV object
        instructor_answer: Instructor's GridSearchCV object

    Returns:
        tuple[bool, str]: Status indicating if answers match and message detailing
            any mismatches
    """
    status = True
    msg_list = []

    # Check simple parameters
    params_to_check = ["cv", "scoring", "refit"]
    for param in params_to_check:
        student_val = getattr(student_answer, param)
        instructor_val = getattr(instructor_answer, param)
        if student_val != instructor_val:
            status = False
            msg_list.append(
                f"Parameter '{param}' mismatch: expected {instructor_val}, got {student_val}"
            )

    # Check estimator type and parameters
    if type(student_answer.estimator) != type(instructor_answer.estimator):
        status = False
        msg_list.append(
            f"Estimator type mismatch: expected {type(instructor_answer.estimator)}, "
            f"got {type(student_answer.estimator)}"
        )
    else:
        # Compare estimator parameters instead of the estimator objects
        student_params = student_answer.estimator.get_params()
        instructor_params = instructor_answer.estimator.get_params()
        if student_params != instructor_params:
            status = False
            msg_list.append(
                f"Estimator parameters mismatch: expected {instructor_params}, got {student_params}"
            )

    # Check param_grid separately
    if student_answer.param_grid != instructor_answer.param_grid:
        status = False
        msg_list.append(
            f"param_grid mismatch: expected {instructor_answer.param_grid}, "
            f"got {student_answer.param_grid}"
        )

    return status, "\n".join(
        msg_list
    ) if msg_list else "GridSearchCV parameters match expected values"


# ======================================================================


def check_structure_randomforestclassifier(student_answer) -> tuple[bool, str]:
    from sklearn.ensemble import RandomForestClassifier

    if not isinstance(student_answer, RandomForestClassifier):
        status = False
        msg = (
            f"Answer must be of type 'RandomForestClassifier'. Your answer is "
            f"of type {type(student_answer).__name__}."
        )
        msg_list = [msg]
    else:
        status = True
        msg_list = ["Answer is of type 'RandomForestClassifier' as expected."]

    return status, "\n".join(msg_list)


'''
def check_answer_randomforestclassifier(
    student_answer,
    instructor_answer,
) -> tuple[bool, str]:
    """Check if student's RandomForestClassifier matches instructor's.

    Args:
        student_answer: Student's RandomForestClassifier object
        instructor_answer: Instructor's RandomForestClassifier object

    Returns:
        tuple[bool, str]: Status indicating if answers match and message detailing
            any mismatches
    """
    status = True
    msg_list = []

    # Check core parameters
    params_to_check = [
        "n_estimators",
        "criterion",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "random_state",
    ]

    for param in params_to_check:
        student_val = getattr(student_answer, param)
        instructor_val = getattr(instructor_answer, param)
        if student_val != instructor_val:
            status = False
            msg_list.append(
                f"Parameter '{param}' mismatch: expected {instructor_val}, got {student_val}"
            )

    return status, "\n".join(
        msg_list
    ) if msg_list else "RandomForestClassifier parameters match expected values"
'''


def check_answer_randomforestclassifier(
    student_answer,
    instructor_answer,
) -> tuple[bool, str]:
    """Check if student's RandomForestClassifier matches instructor's.

    Args:
        student_answer: Student's RandomForestClassifier object
        instructor_answer: Instructor's RandomForestClassifier object

    Returns:
        tuple[bool, str]: Status indicating if answers match and message detailing
            any mismatches
    """
    status = True
    msg_list = []

    # Check core parameters
    params_to_check = [
        "n_estimators",
        "criterion",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "random_state",
    ]

    for param in params_to_check:
        student_val = getattr(student_answer, param)
        instructor_val = getattr(instructor_answer, param)
        if student_val != instructor_val:
            status = False
            msg_list.append(
                f"Parameter '{param}' mismatch: expected {instructor_val}, got {student_val}"
            )

    if not msg_list:
        msg_list = ["RandomForestClassifier parameters match expected values. "]

    return return_value(status, msg_list, student_answer, instructor_answer)


# ======================================================================


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


# ======================================================================
'''
def check_structure_dict_str_float(
    student_answer: dict[str, float],
    instructor_answer: dict[str, float],
    keys: list[str] | None = None,
) -> tuple[bool, str]:
    """Check if a student's dictionary of strings answer matches the instructor's answer.

    Args:
        student_answer (dict[str, float]): The student's submitted answer
        instructor_answer (dict[str, float]): The instructor's correct answer
        keys (list[str] | None): Optional list of keys to check. If None, all keys are
            checked. If `keys` is provided, only the keys in `keys` are checked.

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    """
    status = True
    msg_list = []

    if status and not isinstance(student_answer, dict):
        msg_list += ["Student answer should be a dict"]
        status = False

    if status:
        keys = keys if keys else list(instructor_answer.keys())
        instructor_keys = set(keys)
        instructor_answer = {k: v for k, v in instructor_answer.items() if k in keys}
        student_keys = set(student_answer.keys())
        missing_keys = list(instructor_keys - student_keys)

        if len(missing_keys) > 0:
            msg_list.append(f"- Missing keys: {[repr(k) for k in missing_keys]}.")
            status = False
        else:
            msg_list.append("- No missing keys.")

    if status:
        # some keys are filtered. Student is allowed to have
        # keys not in the instructor set
        for k in instructor_answer:
            vs = student_answer[k]
            if not isinstance(vs, float | np.floating | int | np.integer):
                msg_list.append(f"- answer[{k!r}] should be a float.")
                status = False

        if status:
            msg_list.append("- All elements are of type float as expected.")

    if status:
        msg_list.append("Type 'dict[str, float]' is correct")

    return status, "\n".join(msg_list)
'''


def check_structure_dict_str_float(
    student_answer: dict,
    instructor_answer: dict,
    keys: list[str] | None = None,
) -> tuple[bool, str]:
    """Check if student's dictionary has same structure/types as instructor's.

    Args:
        student_answer: Student's submitted dictionary
        instructor_answer: Instructor's reference dictionary
        keys: Optional list of keys to check. If None, checks all instructor keys

    Returns:
        tuple[bool, str]: Status indicating if structures match and message detailing
            any type mismatches

    """
    msg_list = []
    status = True
    keys = list(instructor_answer.keys()) if keys is None else keys

    # First check if all required keys exist
    for k in keys:
        if k not in student_answer:
            status = False
            msg_list.append(f"Missing key '{k}' in student answer")
            continue

        s_val = student_answer[k]
        i_val = instructor_answer[k]

        # Check types match
        if not isinstance(s_val, float | np.floating):
            status = False
            msg_list.append(
                f"Type mismatch for key '{k}': "
                f"expected {type(i_val).__name__}, "
                f"got {type(s_val).__name__}"
            )

        # For nested dictionaries, recursively check structure
        if isinstance(i_val, dict):
            nested_status, nested_msg = check_structure_dict_str_float(s_val, i_val)
            if not nested_status:
                status = False
                msg_list.append(f"In nested dict '{k}': {nested_msg}")

    return status, "\n".join(
        msg_list
    ) if msg_list else "Dictionary structure matches expected types"


'''
def check_answer_dict_str_float(
    student_answer,
    instructor_answer,
    rel_tol: float,
    keys: list[str] | None = None,
    dict_float_choices: dict[str, float] | None = None,
    partial_score_frac: list[float] = [0.0],
) -> tuple[bool, str]:
    """Check if student answer matches instructor answer for dict[str, float] type.

    Args:
        student_answer: Student's submitted answer
        instructor_answer: Instructor's reference answer
        rel_tol: Relative tolerance for comparing floats
        keys: Optional list of keys to check. If None, checks all instructor keys
        dict_float_choices: Optional dictionary of acceptable float values for each key
        partial_score_frac: List to store partial credit score fraction

    Returns:
        tuple[bool, str]: Status indicating if answers match and message detailing
            any mismatches
    """
    # Initialize dict_float_choices if None
    if dict_float_choices is None:
        dict_float_choices = {}

    msg_list = []
    status = True
    keys = list(instructor_answer.keys()) if keys is None else keys
    ps_dict = init_partial_score_dict()
    ps_dict["nb_total"] = len(keys)

    # Need an exception in case the student key is not found
    for k in keys:
        s_float = student_answer[k]
        i_float = instructor_answer[k]

        if k in dict_float_choices and dict_float_choices[k]:  # Check if key exists and has values
            for val in dict_float_choices[k]:
                if val == "i":  # use instructor answer
                    val = i_float
                status_, msg_list_ = check_float(s_float, val, rel_tol, 1.0e-5)
                if status_ is True:
                    break
        else:
            status_, msg_ = check_float(i_float, s_float, rel_tol=rel_tol, abs_tol=1.0e-6)

        if status_ is False:
            status = False
            ps_dict["nb_mismatches"] += 1
            msg_list.append(msg_)

    partial_score_frac[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]

    if not msg_list:
        msg_list = ["Answer matches expected values. "]

    return return_value(status, msg_list, student_answer, instructor_answer)
'''


def check_answer_dict_str_float(
    student_answer: dict[str, float],
    instructor_answer: dict[str, float],
    rel_tol: float,
    keys: list[str] | None = None,
    exclude_keys: list[str] | None = None,
    dict_float_choices: dict[str, float] | None = None,
    partial_score_frac: list[float] = [0.0],
) -> tuple[bool, str]:
    """Check if a student's dictionary of strings answer matches the instructor's answer.

    Args:
        student_answer (dict[str, float]): The student's submitted answer
        instructor_answer (dict[str, float]): The instructor's correct answer
        rel_tol (float): The relative tolerance for comparing floats
        keys (list[str] | None): Optional list of keys to check. If None, all keys are
            checked. If `keys` is provided, only the keys in `keys` are checked.
            For examples, if the instructor answer is {'a': 1.0, 'b': 2.0, 'c': 3.0},
            then keys = ['a', 'b', 'c'] will check all keys.
            If keys = ['a', 'b'], then only keys 'a' and 'b' are checked.
        dict_float_choices (dict[str, float] | None): Optional dictionary of float choices
            for each key. If provided, validates that student_answer[k] is one of these choices.
        partial_score_frac (list[float]): The partial score fraction

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if answers match and validation passes, False otherwise
            - str: Message explaining the validation result

    Use the function (NOT DONE):
        def check_dict_str_float(
            keys: list[str],
            i_dict: dict[str, float],
            s_dict: dict[str, float],
            rel_tol: float,
            abs_tol: float,
            ps_dict: dict[str, float | int],
        ) -> tuple[bool, list[str]]:

    """
    if dict_float_choices is not None:
        print("dict_float_choices is not implemented in dict[str,float] types")
        dict_float_choices = {}

    msg_list = []
    status = True
    keys = list(instructor_answer.keys()) if keys is None else keys
    ps_dict = init_partial_score_dict()
    ps_dict["nb_total"] = len(keys)
    if exclude_keys is None:
        exclude_keys = []

    if dict_float_choices is None:
        dict_float_choices = {}

    # Need an exception in case the student key is not found
    for k in keys:
        print(f"{exclude_keys=}, key={k}")
        if k in exclude_keys:
            print(f"==> Exclude {k=}")
            continue
        s_float = student_answer[k]
        i_float = instructor_answer[k]

        if len(dict_float_choices) > 0 and k in dict_float_choices:
            val = dict_float_choices[k]
            if val == "i":  # use instructor answer
                val = i_float
            status_, msg_list_ = check_float(s_float, val, rel_tol, 1.0e-5)
            if status_ is True:
                break
        else:
            status_, msg_ = check_float(i_float, s_float, rel_tol=rel_tol, abs_tol=1.0e-6)

        if status_ is False:
            status = False
            ps_dict["nb_mismatches"] += 1
            msg_list.append(msg_)

    partial_score_frac[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]

    if not msg_list:
        msg_list = ["Answer parameters match expected values. "]

    return return_value(status, msg_list, student_answer, instructor_answer)


# ======================================================================


def check_answer_dict_str_any(
    student_answer: dict[str, Any],
    instructor_answer: dict[str, Any],
    rel_tol: float = 1e-04,
    keys: list[str] | None = None,
    partial_score_frac: list[float] = [0.0],
) -> tuple[bool, str]:
    """Check if student's dictionary matches instructor's, handling various value types.

    Args:
        student_answer: Student's submitted dictionary
        instructor_answer: Instructor's reference dictionary
        rel_tol: Relative tolerance for float comparisons
        keys: Optional list of keys to check. If None, checks all instructor keys
        partial_score_frac: List to store partial credit score fraction

    Returns:
        tuple[bool, str]: Status indicating if answers match and message detailing
            any mismatches
    """
    msg_list = []
    status = True
    keys = list(instructor_answer.keys()) if keys is None else keys
    ps_dict = init_partial_score_dict()
    ps_dict["nb_total"] = len(keys)

    for k in keys:
        s_val = student_answer[k]
        i_val = instructor_answer[k]

        # Handle different types of values
        if isinstance(i_val, (int, str)):
            status_ = s_val == i_val
            msg_ = f"Value mismatch for key '{k}': expected {i_val}, got {s_val}"

        elif isinstance(i_val, float):
            status_, msg_ = check_float(i_val, s_val, rel_tol=rel_tol, abs_tol=1.0e-6)

        elif isinstance(i_val, dict):
            status_, msg_ = check_answer_dict_str_any(s_val, i_val, rel_tol=rel_tol)

        elif isinstance(i_val, (list, tuple, np.ndarray)):
            if isinstance(i_val, np.ndarray):
                status_ = np.allclose(i_val, s_val, rtol=rel_tol, atol=1.0e-6)
            else:
                status_ = i_val == s_val
            msg_ = f"Value mismatch for key '{k}': expected {i_val}, got {s_val}"

        else:
            # For other types (like sklearn objects), compare type and parameters if available
            if type(i_val) != type(s_val):
                status_ = False
                msg_ = f"Type mismatch for key '{k}': expected {type(i_val)}, got {type(s_val)}"
            elif hasattr(i_val, "get_params"):
                i_params = i_val.get_params()
                s_params = s_val.get_params()
                status_ = i_params == s_params
                msg_ = f"Parameter mismatch for key '{k}': expected {i_params}, got {s_params}"
            else:
                status_ = i_val == s_val
                msg_ = f"Value mismatch for key '{k}': expected {i_val}, got {s_val}"

        if not status_:
            status = False
            ps_dict["nb_mismatches"] += 1
            msg_list.append(msg_)

    try:
        partial_score_frac[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]
    except ZeroDivisionError:
        partial_score_frac[0] = 1.0

    if not msg_list:
        msg_list = ["Answre parameters match expected values. "]

    return return_value(status, msg_list, student_answer, instructor_answer)


def check_structure_dict_any(
    student_dict: dict[Any, Any],
    instructor_dict: dict[Any, Any],
) -> tuple[bool, str]:
    """Check if student's dictionary has same structure/types as instructor's.

    Args:
        student_dict: Student's submitted dictionary
        instructor_dict: Instructor's reference dictionary

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if structures match, False otherwise
            - str: Message explaining any type mismatches

    """
    msg_list = []
    status = True

    # Check if all instructor keys exist in student dict
    for k, i_val in instructor_dict.items():
        if k not in student_dict:
            status = False
            msg_list.append(f"Missing key '{k}' in student answer")
            continue

        s_val = student_dict[k]

        # Check if types match
        if not isinstance(s_val, type(i_val)):
            status = False
            msg_list.append(
                f"Type mismatch for key '{k}': "
                f"expected {type(i_val).__name__}, "
                f"got {type(s_val).__name__}"
            )

    if not msg_list:
        msg_list = ["Dictionary structure matches expected types"]

    return status, "\n".join(msg_list)


def check_structure_dict_str_any(
    student_answer: dict[str, Any],
    instructor_answer: dict[str, Any],
    keys: list[str] | None = None,
) -> tuple[bool, str]:
    """Check if student's dictionary has same structure/types as instructor's.

    Args:
        student_answer: Student's submitted dictionary
        instructor_answer: Instructor's reference dictionary
        keys: Optional list of keys to check. If None, checks all instructor keys

    Returns:
        tuple[bool, str]: Status indicating if structures match and message detailing
            any type mismatches

    """
    msg_list = []
    status = True
    keys = list(instructor_answer.keys()) if keys is None else keys

    # First check if all required keys exist
    for k in keys:
        if k not in student_answer:
            status = False
            msg_list.append(f"Missing key '{k}' in student answer")
            continue

        s_val = student_answer[k]
        i_val = instructor_answer[k]

        # Check types match
        if not isinstance(s_val, type(i_val)):
            status = False
            msg_list.append(
                f"Type mismatch for key '{k}': "
                f"expected {type(i_val).__name__}, "
                f"got {type(s_val).__name__}"
            )
            continue

        # For nested dictionaries, recursively check structure
        if isinstance(i_val, dict):
            nested_status, nested_msg = check_structure_dict_any(s_val, i_val)
            if not nested_status:
                status = False
                msg_list.append(f"In nested dict '{k}': {nested_msg}")

        # For sklearn objects, check if they have the same parameters
        elif hasattr(i_val, "get_params"):
            i_params = set(i_val.get_params().keys())
            s_params = set(s_val.get_params().keys())
            if i_params != s_params:
                status = False
                missing = i_params - s_params
                extra = s_params - i_params
                if missing:
                    msg_list.append(f"Missing parameters in {k}: {missing}")
                if extra:
                    msg_list.append(f"Extra parameters in {k}: {extra}")

        # For numpy arrays, check shape and dtype
        elif isinstance(i_val, np.ndarray):
            if s_val.shape != i_val.shape:
                status = False
                msg_list.append(
                    f"Shape mismatch for array '{k}': expected {i_val.shape}, got {s_val.shape}"
                )
            if s_val.dtype != i_val.dtype:
                status = False
                msg_list.append(
                    f"Dtype mismatch for array '{k}': expected {i_val.dtype}, got {s_val.dtype}"
                )

    return status, "\n".join(
        msg_list
    ) if msg_list else "Dictionary structure matches expected types"


# ======================================================================


def check_structure_dict_int_dict_str_any(
    student_answer: dict[int, dict[str, Any]],
    instructor_answer: dict[int, dict[str, Any]],
    keys: list[str] | None = None,
) -> tuple[bool, str]:
    """Check if student's nested dictionary structure matches instructor's.

    Checks a dictionary where each value is expected to be another dictionary
    mapping strings to any type.

    Args:
        student_answer: Student's submitted dictionary of dictionaries
        instructor_answer: Instructor's reference dictionary of dictionaries
        keys: Optional list of keys to check. If None, checks all instructor keys

    Returns:
        tuple[bool, str]: Status indicating if structures match and message detailing
            any type mismatches
    """
    msg_list = []
    status = True
    keys = list(instructor_answer.keys()) if keys is None else keys

    ## ARE WE CHECKING THAT THE KEYS ARE int ! TODO

    # Check top-level keys
    for k in keys:
        if k not in student_answer:
            status = False
            msg_list.append(f"Missing top-level key '{k}' in student answer")
            continue

        s_val = student_answer[k]
        i_val = instructor_answer[k]

        # Check if value is a dictionary
        if not isinstance(s_val, dict):
            status = False
            msg_list.append(
                f"Type mismatch for key '{k}': expected dict, got {type(s_val).__name__}"
            )
            continue

        # Check nested dictionary keys
        nested_keys = i_val.keys()
        for nested_k in nested_keys:
            if nested_k not in s_val:
                status = False
                msg_list.append(f"Missing nested key '{nested_k}' in '{k}'")
                continue

            s_nested_val = s_val[nested_k]
            i_nested_val = i_val[nested_k]

            # Check types of nested values
            if not isinstance(s_nested_val, type(i_nested_val)):
                status = False
                msg_list.append(
                    f"Type mismatch for nested key '{k}.{nested_k}': "
                    f"expected {type(i_nested_val).__name__}, "
                    f"got {type(s_nested_val).__name__}"
                )
                continue

            # Special handling for specific types
            if isinstance(i_nested_val, dict):
                nested_status, nested_msg = check_structure_dict_any(s_nested_val, i_nested_val)
                if not nested_status:
                    status = False
                    msg_list.append(f"In nested dict '{k}.{nested_k}': {nested_msg}")

            elif hasattr(i_nested_val, "get_params"):
                i_params = set(i_nested_val.get_params().keys())
                s_params = set(s_nested_val.get_params().keys())
                if i_params != s_params:
                    status = False
                    missing = i_params - s_params
                    extra = s_params - i_params
                    if missing:
                        msg_list.append(f"Missing parameters in {k}.{nested_k}: {missing}")
                    if extra:
                        msg_list.append(f"Extra parameters in {k}.{nested_k}: {extra}")

            elif isinstance(i_nested_val, np.ndarray):
                if s_nested_val.shape != i_nested_val.shape:
                    status = False
                    msg_list.append(
                        f"Shape mismatch for array '{k}.{nested_k}': "
                        f"expected {i_nested_val.shape}, got {s_nested_val.shape}"
                    )
                if s_nested_val.dtype != i_nested_val.dtype:
                    status = False
                    msg_list.append(
                        f"Dtype mismatch for array '{k}.{nested_k}': "
                        f"expected {i_nested_val.dtype}, got {s_nested_val.dtype}"
                    )

    return status, "\n".join(
        msg_list
    ) if msg_list else "Dictionary structure matches expected types"


def check_answer_dict_int_dict_str_any(
    student_answer: dict[int, dict],
    instructor_answer: dict[int, dict],
    rel_tol: float = 1e-04,
    keys: list[str] | None = None,
    exclude_keys: list[str] | None = None,
    partial_score_frac_l: list[float] = [0.0],
) -> tuple[bool, str]:
    """Check if student's nested dictionary values match instructor's.

    Checks a dictionary where each value is expected to be another dictionary
    mapping strings to any type. Compares actual values, not just structure.

    Args:
        student_answer: Student's submitted dictionary of dictionaries
        instructor_answer: Instructor's reference dictionary of dictionaries
        rel_tol: Relative tolerance for float comparisons
        keys: Optional list of keys to check. If None, checks all instructor keys
        exclude_keys: Optional list of keys to ignore during comparison
        partial_score_frac_l: List to store partial credit score fraction

    Returns:
        tuple[bool, str]: Status indicating if answers match and message detailing
            any mismatches

    """
    print("===> *** check_answer_dict_int_dict_str_any")
    print("\nstudent_answer=")
    pprint(student_answer)
    print("\ninstructor+answer")
    pprint(instructor_answer)
    print(f"{exclude_keys=}")
    print()

    msg_list = []
    status = True
    keys_list = list(instructor_answer.keys()) if keys is None else keys
    exclude_keys: set[str] = set(exclude_keys if exclude_keys is not None else [])

    ps_dict = init_partial_score_dict()
    total_checks = 0
    mismatches = 0

    for k in keys_list:
        if k not in student_answer:
            status = False
            msg_list.append(f"Missing top-level key '{k}' in student answer")
            continue

        s_val = student_answer[k]
        i_val = instructor_answer[k]

        if not isinstance(s_val, dict):
            status = False
            msg_list.append(
                f"Type mismatch for key '{k}': expected dict, got {type(s_val).__name__}"
            )
            continue

        # Check nested dictionary values, excluding ignored keys
        for nested_k, i_nested_val in i_val.items():
            # Skip this inner key if it's in ignore_keys
            if nested_k in exclude_keys:
                continue

            total_checks += 1

            if nested_k not in s_val:
                status = False
                mismatches += 1
                msg_list.append(f"Missing nested key '{nested_k}' in '{k}'")
                continue

            s_nested_val = s_val[nested_k]

            # Handle different types of values
            if isinstance(i_nested_val, (int, str)):
                if s_nested_val != i_nested_val:
                    status = False
                    mismatches += 1
                    msg_list.append(
                        f"Value mismatch for '{k}.{nested_k}': "
                        f"expected {i_nested_val}, got {s_nested_val}"
                    )

            # ... rest of the type checking logic remains the same ...

    if total_checks > 0:
        partial_score_frac_l[0] = 1.0 - (mismatches / total_checks)

    if not msg_list:
        msg_list = ["Answre parameters match expected values. "]

    return return_value(status, msg_list, student_answer, instructor_answer)


# ======================================================================


def check_structure_set_tuple_int(
    student_answer: set[tuple[int, ...]],
) -> tuple[bool, str]:
    """Check if student's answer is a set of tuples containing integers.

    Args:
        student_answer: Student's submitted answer
        instructor_answer: Instructor's reference answer (used for type hint only)

    Returns:
        tuple[bool, str]: Status indicating if structure matches and message detailing
            any type mismatches
    """
    msg_list = []
    status = True

    # Check if it's a set
    if not isinstance(student_answer, set):
        print("1. false", flush=True)
        return False, f"Expected type set, got {type(student_answer).__name__}"

    # Check each element
    for i, item in enumerate(student_answer):
        print(f"for, {i=}", flush=True)
        # Check if element is a tuple
        if not isinstance(item, tuple):
            status = False
            msg_list.append(f"Element {i} is not a tuple: got {type(item).__name__}")
            print("if not")
            continue

        # Check if all elements in tuple are integers
        if not all(isinstance(x, int) for x in item):
            print("if not all")
            status = False
            msg_list.append(f"Tuple {i} {item} contains non-integer values")

    print("msg_list= ", msg_list, flush=True)
    return status, "\n".join(
        msg_list
    ) if msg_list else "Structure matches expected set[tuple[int, ...]]"


# ======================================================================


def check_answer_set_tuple_int(
    student_answer: set[tuple[int, ...]],
    instructor_answer: set[tuple[int, ...]],
    partial_score_frac_l: list[float] = [0.0],
) -> tuple[bool, str]:
    """Check if student's set of integer tuples matches instructor's.

    Args:
        student_answer: Student's submitted set of integer tuples
        instructor_answer: Instructor's reference set of integer tuples
        partial_score_frac_l: List to store partial credit score fraction.
            The list permist its return via the argument.

    Returns:
        tuple[bool, str]: Status indicating if answers match and message detailing
            any mismatches
    """
    msg_list = []
    status = True

    print("===> *** check_answer_set_tuple_int")
    print(f"{student_answer=}")
    print(f"{instructor_answer=}")

    # First check structure
    structure_status, structure_msg = check_structure_set_tuple_int(student_answer)

    if not structure_status:
        print("Return due to bad structure check")
        return False, structure_msg

    # Check for missing tuples
    missing = instructor_answer - student_answer
    if missing:
        status = False
        msg_list.append(f"Missing tuples: {missing}")

    # Check for extra tuples
    extra = student_answer - instructor_answer
    if extra:
        status = False
        msg_list.append(f"Extra tuples: {extra}")

    # Calculate partial score based on correct tuples
    total = len(instructor_answer)
    if total > 0:
        correct = len(instructor_answer & student_answer)  # intersection
        partial_score_frac_l[0] = correct / total

    return return_value(status, msg_list, student_answer, instructor_answer)


def check_answer_logisticregression(
    student_answer,
    instructor_answer,
) -> tuple[bool, str]:
    """Check if student's LogisticRegression matches instructor's.

    Args:
        student_answer: Student's LogisticRegression object
        instructor_answer: Instructor's LogisticRegression object

    Returns:
        tuple[bool, str]: Status indicating if answers match and message detailing
            any mismatches
    """
    status = True
    msg_list = []

    # Check core parameters
    params_to_check = ["penalty", "C", "random_state", "solver", "max_iter"]
    for param in params_to_check:
        student_val = getattr(student_answer, param)
        instructor_val = getattr(instructor_answer, param)
        if student_val != instructor_val:
            status = False
            msg_list.append(
                f"Parameter '{param}' mismatch: expected {instructor_val}, got {student_val}"
            )

    if not msg_list:
        msg_list = ["LogisticRegression parameters match expected values"]

    return return_value(status, msg_list, student_answer, instructor_answer)


# ----------------------------------------------------------------------


def check_structure_list_tuple_float(
    student_answer: list[tuple[float, ...]],
) -> tuple[bool, str]:
    """Check if student's answer is a list of tuples containing floats.

    Args:
        student_answer: Student's submitted answer

    Returns:
        tuple[bool, str]: A tuple containing:
            - bool: True if structure matches, False otherwise
            - str: Message explaining any type mismatches
    """
    msg_list = []
    status = True

    print("INSIDE structure")
    print(f"{student_answer=}")

    # Check if it's a list
    if not isinstance(student_answer, list):
        print("not a list")
        return False, f"Expected type list, got {type(student_answer).__name__}"

    # Check each element
    for i, item in enumerate(student_answer):
        print(f"{i=}, {item=}")
        # Check if element is a tuple
        if not isinstance(item, tuple):
            status = False
            msg_list.append(f"Element {i} is not a tuple: got {type(item).__name__}")
            continue

        # Check if all elements in tuple are floats or can be cast to float
        if not all(isinstance(x, (float, int, np.floating)) for x in item):
            status = False
            msg_list.append(f"Tuple {i} {item} contains non-float values")

    print("status= ", status)
    return status, "\n".join(
        msg_list
    ) if msg_list else "Structure matches expected list[tuple[float, ...]]"


def check_answer_list_tuple_float(
    student_answer: list[tuple[float, ...]],
    instructor_answer: list[tuple[float, ...]],
    rel_tol: float,
    partial_score_frac_l: list[float] = [0.0],
) -> tuple[bool, str]:
    """Check if student's list of float tuples matches instructor's.

    Args:
        student_answer: Student's submitted list of float tuples
        instructor_answer: Instructor's reference list of float tuples
        rel_tol: Relative tolerance for float comparisons
        partial_score_frac: List to store partial credit score fraction.
            The list permits its return via the argument.

    Returns:
        tuple[bool, str]: Status indicating if answers match and message detailing
            any mismatches

    """
    print(f"...{rel_tol=}")
    print(f"{partial_score_frac_l=}")
    print(f"==> {student_answer=}")
    print(f"==> {instructor_answer=}")
    msg_list = []
    status = True
    ps_dict = init_partial_score_dict()

    # First check structure
    structure_status, structure_msg = check_structure_list_tuple_float(student_answer)
    print("===> EXIT check_structure_list_tuple_float")
    if not structure_status:
        return False, structure_msg

    # Check if lists have same length
    if len(student_answer) != len(instructor_answer):
        status = False
        msg_list.append(
            f"Length mismatch: expected {len(instructor_answer)}, got {len(student_answer)}"
        )
        return return_value(status, msg_list, student_answer, instructor_answer)

    ps_dict["nb_total"] = len(instructor_answer) * len(instructor_answer[0])

    # Check each tuple
    for i, (s_tuple, i_tuple) in enumerate(zip(student_answer, instructor_answer, strict=True)):
        # Check if tuples have same length
        if len(s_tuple) != len(i_tuple):
            status = False
            msg_list.append(
                f"Tuple {i} length mismatch: expected {len(i_tuple)}, got {len(s_tuple)}"
            )
            continue

        # Compare each float in the tuples
        print(f"==> {s_tuple=}, {i_tuple=}")
        for j, (s_val, i_val) in enumerate(zip(s_tuple, i_tuple, strict=True)):
            print(f"==> {j=}, {s_val=}, {i_val=}")
            print(f"{type(s_val)=}, {type(i_val)=}")
            status_, msg = check_float(i_val, s_val, rel_tol=rel_tol, abs_tol=1.0e-6)
            print(f"==> {status_=}, {msg=}")
            if not status_:
                status = False
                ps_dict["nb_mismatches"] += 1
                msg_list.append(f"Tuple {i}, element {j}: {msg}")

    try:
        partial_score_frac_l[0] = 1.0 - ps_dict["nb_mismatches"] / ps_dict["nb_total"]
        print(f"==> **** inside check_answer_list_tuple_float, {partial_score_frac_l=}")
    except ZeroDivisionError:
        partial_score_frac_l[0] = 1.0

    if not msg_list:
        msg_list = ["Answer matches expected values."]

    return return_value(status, msg_list, student_answer, instructor_answer)


# ----------------------------------------------------------------------
def check_answer_scatterplot2d(student_answer, instructor_answer, options, validation_functions):
    status = True
    msg_list = []

    s_answ = student_answer
    i_answ = instructor_answer

    s_plt = s_answ
    i_plt = i_answ

    at_least_val = options.get('at_least_validation', None)

    # print("s_answ= ", s_answ)
    # print(f"{type(s_answ)=}")
    s_fig = s_plt.figure
    i_fig = i_plt.figure
    # Assume only a single axis

    def check_grid_status(ax):
        # Check visibility of grid lines
        # Get a list of booleans indicating the visibility status of each gridline
        xgrid_visible = any([line.get_visible() for line in ax.xaxis.get_gridlines()])
        ygrid_visible = any([line.get_visible() for line in ax.yaxis.get_gridlines()])
        
        # If any of the grid lines are visible, we consider the grid "on"
        return xgrid_visible and ygrid_visible


    def fig_dict(answ):
        fig = answ.figure
        # print("==> answ= ", answ)
        # print("==> fig= ", fig)
        # print("==> ", dir(fig))
        ax = fig.axes[0]
        coll = ax.collections[0]
        xy = ax.collections[0].get_offsets()
        path_collection = answ
        face_colors = path_collection.get_facecolor() # RGBA
        s_face_colors_readable = [mcolors.to_hex(c) for c in face_colors]
        s_dict = {
            'ax': ax,
            'title': ax.get_title(),
            'xlabel': ax.get_xlabel(),
            'ylabel': ax.get_ylabel(),
            'x': xy[:, 0],
            'y': xy[:, 1],
            'colors': np.unique(s_face_colors_readable)
        }
        return s_dict

    s_dict = fig_dict(s_answ)
    i_dict = fig_dict(i_answ)

    s_grid = check_grid_status(s_dict['ax'])
    i_grid = check_grid_status(s_dict['ax'])

    title = s_dict['title']
    x_label = s_dict['xlabel']
    y_label = s_dict['ylabel']

    if clean_str_answer(x_label) == "" or clean_str_answer(y_label) == "":
        status = False
        msg_list.append("The plot is missing either xlabel or ylabel")

    if clean_str_answer(title) == "":
        status = False
        msg_list.append("The plot is missing the title")

    if at_least_val:
        count = at_least_val.get('count', 0)
        nb_points = len(s_dict['x'])
        if (nb_points < count):
            status = False
            msg_list.append(f"The 2D scatterplot should have at least {count} points")

    # print(f"{s_grid=}, {i_grid=}")

    return return_value(status, msg_list, student_answer, instructor_answer)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_scatterplot2d(student_answer):
    from matplotlib.collections import PathCollection

    status = True
    msg_list = []

    s_answ = student_answer

    if not isinstance(student_answer, PathCollection):
        status = False
        msg_list.append(
            "The answer type should be 'PathCollectdion', the type of the output of 'plt.scatter'."
        )

    xy = s_answ.get_offsets()
    #x, y = s_answ.get_offsets()
    #sxsy = x.data.astype(float), y.data.astype(float)
    #s_x, s_y = x.data.astype(float), y.data.astype(float)

    # x, y, z = i_answ._offsets3d
    # i_x, i_y, i_z = x.data.astype(float), y.data.astype(float), z.astype(float)

    """
    if i_x.shape == s_x.shape and i_y.shape == s_y.shape and i_z.shape == s_z.shape:
        status = False
        msg_list.append(f"The number of points ({s_x.shape[0]}) is incorrect")
    """

    return status, "\n".join(msg_list)


# ======================================================================


def check_answer_scatterplot3d(student_answer, instructor_answer, options, validation_functions):
    status = True
    msg_list = []

    s_answ = student_answer
    i_answ = instructor_answer

    rel_tol = options.get("rel_tol", 1.e-2)

    # Check for equality
    x, y, z = s_answ._offsets3d
    s_x, s_y, s_z = x.data.astype(float), y.data.astype(float), z.astype(float)

    x, y, z = i_answ._offsets3d
    i_x, i_y, i_z = x.data.astype(float), y.data.astype(float), z.astype(float)

    # print(f"==> {i_x=}, {i_y=}, {i_z=}")
    # print(f"==> {s_x=}, {s_y=}, {s_z=}")

    sum_i = np.sum(i_x) + np.sum(i_y) + np.sum(i_z)
    sum_s = np.sum(s_x) + np.sum(s_y) + np.sum(s_z)

    status, msg = check_float(sum_s, sum_i, rel_tol=rel_tol, abs_tol=1.e-5)
    msg_list.append(msg)

    return return_value(status, msg_list, s_answ, i_answ)


# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . .


def check_structure_scatterplot3d(student_answer):
    from matplotlib.collections import PathCollection

    status = True
    msg_list = []

    s_answ = student_answer

    if not isinstance(student_answer, PathCollection):
        status = False
        msg_list.append(
            "The answer type should be 'PathCollectdion', the type of the output of 'plt.scatter'."
        )

    x, y, z = s_answ._offsets3d
    s_x, s_y, s_z = x.data.astype(float), y.data.astype(float), z.astype(float)

    #x, y, z = i_answ._offsets3d
    #i_x, i_y, i_z = x.data.astype(float), y.data.astype(float), z.astype(float)

    #if i_x.shape == s_x.shape and i_y.shape == s_y.shape and i_z.shape == s_z.shape:
        #status = False
        #msg_list.append(f"The number of points ({s_x.shape[0]}) is incorrect")

    return status, "\n".join(msg_list)


# ======================================================================
