import pytest
import pytest
import assert_utilities
import numpy
from part4 import *
import yaml
import test_utils as u
from my_fixtures import *
from pytest_utils.decorators import max_score, visibility, partial_score, hide_errors

with open('type_handlers.yaml', 'r') as f:
    type_handlers = yaml.safe_load(f)
@hide_errors('')
def test_structure_compute_4A_colon_datasets_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack_lparen_rparen(run_compute):
    correct_answer = run_compute('part4', 'i')
    if '4A: datasets' not in correct_answer:
        explanation = "Key: '4A: datasets' not found.\n"
        test_structure_compute_4A_colon_datasets_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['4A: datasets']
    student_answer = run_compute('part4', 's')
    if '4A: datasets' not in student_answer:
        explanation = "Key: '4A: datasets' not found.\n"
        test_structure_compute_4A_colon_datasets_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['4A: datasets']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.01
    keys = None
    msg = "assert_utilities.check_structure_dict_string_Tuple_NDArray(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_compute_4A_colon_datasets_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_compute_4A_colon_fit_hierarchical_cluster_function_lparen_rparen(run_compute):
    correct_answer = run_compute('part4', 'i')
    if '4A: fit_hierarchical_cluster' not in correct_answer:
        explanation = "Key: '4A: fit_hierarchical_cluster' not found.\n"
        test_structure_compute_4A_colon_fit_hierarchical_cluster_function_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['4A: fit_hierarchical_cluster']
    student_answer = run_compute('part4', 's')
    if '4A: fit_hierarchical_cluster' not in student_answer:
        explanation = "Key: '4A: fit_hierarchical_cluster' not found.\n"
        test_structure_compute_4A_colon_fit_hierarchical_cluster_function_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['4A: fit_hierarchical_cluster']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_function(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_compute_4A_colon_fit_hierarchical_cluster_function_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_compute_4B_colon_cluster_successes_list_lbrack_string_rbrack_lparen_rparen(run_compute):
    correct_answer = run_compute('part4', 'i')
    if '4B: cluster successes' not in correct_answer:
        explanation = "Key: '4B: cluster successes' not found.\n"
        test_structure_compute_4B_colon_cluster_successes_list_lbrack_string_rbrack_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['4B: cluster successes']
    student_answer = run_compute('part4', 's')
    if '4B: cluster successes' not in student_answer:
        explanation = "Key: '4B: cluster successes' not found.\n"
        test_structure_compute_4B_colon_cluster_successes_list_lbrack_string_rbrack_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['4B: cluster successes']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_list_string(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_compute_4B_colon_cluster_successes_list_lbrack_string_rbrack_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_compute_4C_colon_modified_function_function_lparen_rparen(run_compute):
    correct_answer = run_compute('part4', 'i')
    if '4C: modified function' not in correct_answer:
        explanation = "Key: '4C: modified function' not found.\n"
        test_structure_compute_4C_colon_modified_function_function_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['4C: modified function']
    student_answer = run_compute('part4', 's')
    if '4C: modified function' not in student_answer:
        explanation = "Key: '4C: modified function' not found.\n"
        test_structure_compute_4C_colon_modified_function_function_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['4C: modified function']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_function(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_compute_4C_colon_modified_function_function_lparen_rparen.explanation = explanation
    assert is_success
