import pytest
import pytest
import assert_utilities
import numpy
from part1 import *
import yaml
import test_utils as u
from my_fixtures import *
from pytest_utils.decorators import max_score, visibility, partial_score, hide_errors

with open('type_handlers.yaml', 'r') as f:
    type_handlers = yaml.safe_load(f)

@hide_errors('')
def test_structure_compute_1A_colon_datasets_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack_lparen_rparen(run_compute):
    correct_answer = run_compute('part1', 'i')
    if '1A: datasets' not in correct_answer:
        explanation = "Key: '1A: datasets' not found.\n"
        test_structure_compute_1A_colon_datasets_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['1A: datasets']
    student_answer = run_compute('part1', 's')
    if '1A: datasets' not in student_answer:
        explanation = "Key: '1A: datasets' not found.\n"
        test_structure_compute_1A_colon_datasets_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['1A: datasets']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.01
    keys = None
    msg = "assert_utilities.check_structure_dict_string_Tuple_NDArray(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_compute_1A_colon_datasets_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_compute_1B_colon_fit_kmeans_function_lparen_rparen(run_compute):
    correct_answer = run_compute('part1', 'i')
    if '1B: fit_kmeans' not in correct_answer:
        explanation = "Key: '1B: fit_kmeans' not found.\n"
        test_structure_compute_1B_colon_fit_kmeans_function_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['1B: fit_kmeans']
    student_answer = run_compute('part1', 's')
    if '1B: fit_kmeans' not in student_answer:
        explanation = "Key: '1B: fit_kmeans' not found.\n"
        test_structure_compute_1B_colon_fit_kmeans_function_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['1B: fit_kmeans']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_function(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_compute_1B_colon_fit_kmeans_function_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_compute_1C_colon_cluster_successes_dict_lbrack_string_comma_set_rbrack_lparen_rparen(run_compute):
    correct_answer = run_compute('part1', 'i')
    if '1C: cluster successes' not in correct_answer:
        explanation = "Key: '1C: cluster successes' not found.\n"
        test_structure_compute_1C_colon_cluster_successes_dict_lbrack_string_comma_set_rbrack_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['1C: cluster successes']
    student_answer = run_compute('part1', 's')
    if '1C: cluster successes' not in student_answer:
        explanation = "Key: '1C: cluster successes' not found.\n"
        test_structure_compute_1C_colon_cluster_successes_dict_lbrack_string_comma_set_rbrack_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['1C: cluster successes']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_dict_string_set(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_compute_1C_colon_cluster_successes_dict_lbrack_string_comma_set_rbrack_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_compute_1C_colon_cluster_failures_set_lbrack_string_rbrack_lparen_rparen(run_compute):
    correct_answer = run_compute('part1', 'i')
    if '1C: cluster failures' not in correct_answer:
        explanation = "Key: '1C: cluster failures' not found.\n"
        test_structure_compute_1C_colon_cluster_failures_set_lbrack_string_rbrack_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['1C: cluster failures']
    student_answer = run_compute('part1', 's')
    if '1C: cluster failures' not in student_answer:
        explanation = "Key: '1C: cluster failures' not found.\n"
        test_structure_compute_1C_colon_cluster_failures_set_lbrack_string_rbrack_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['1C: cluster failures']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_set_string(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_compute_1C_colon_cluster_failures_set_lbrack_string_rbrack_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_compute_1D_colon_datasets_sensitive_to_initialization_set_lbrack_string_rbrack_lparen_rparen(run_compute):
    correct_answer = run_compute('part1', 'i')
    if '1D: datasets sensitive to initialization' not in correct_answer:
        explanation = "Key: '1D: datasets sensitive to initialization' not found.\n"
        test_structure_compute_1D_colon_datasets_sensitive_to_initialization_set_lbrack_string_rbrack_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['1D: datasets sensitive to initialization']
    student_answer = run_compute('part1', 's')
    if '1D: datasets sensitive to initialization' not in student_answer:
        explanation = "Key: '1D: datasets sensitive to initialization' not found.\n"
        test_structure_compute_1D_colon_datasets_sensitive_to_initialization_set_lbrack_string_rbrack_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['1D: datasets sensitive to initialization']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_set_string(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_compute_1D_colon_datasets_sensitive_to_initialization_set_lbrack_string_rbrack_lparen_rparen.explanation = explanation
    assert is_success
