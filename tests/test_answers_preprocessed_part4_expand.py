
import pytest
from pytest_utils.decorators import max_score, visibility, hide_errors
import instructor_code_with_answers.part4 as ic
from testing_utilities import assert_almost_equal
import assert_utilities  # <<< SHOULD be specified in config
from student_code_with_answers import *
import instructor_code_with_answers as sc
from my_fixtures import *   
# Not clear why 'import conftest' does not work
import tests.conftest as c
import test_utils as u
import random
import numpy as np
import yaml
import re
import my_fixtures

with open('type_handlers.yaml', 'r') as f:
    type_handlers = yaml.safe_load(f)

@max_score(20)
@hide_errors('')
def test_answers_compute_4A_colon_datasets_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack(run_compute):
    correct_answer = run_compute('part4', 'i')
    if '4A: datasets' not in correct_answer:
        explanation = "Key: '4A: datasets' not found.\n"
        test_answers_compute_4A_colon_datasets_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['4A: datasets']
    student_answer = run_compute('part4', 's')
    if '4A: datasets' not in student_answer:
        explanation = "Key: '4A: datasets' not found.\n"
        test_answers_compute_4A_colon_datasets_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack.explanation = explanation
        assert False
    else:
        student_answer = student_answer['4A: datasets']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.01
    keys = None
    msg_structure = "assert_utilities.check_structure_dict_string_Tuple_NDArray(student_answer, instructor_answer)"
    msg_answer = "assert_utilities.check_answer_dict_string_Tuple_NDArray(student_answer, instructor_answer, rel_tol)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
    else: 
        explanation_answer = "" 
    explanation = '\n'.join(['Structure tests:', explanation_structure, 'Answer tests:', explanation_answer])
    test_answers_compute_4A_colon_datasets_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_compute_4A_colon_fit_hierarchical_cluster_function(run_compute):
    correct_answer = run_compute('part4', 'i')
    if '4A: fit_hierarchical_cluster' not in correct_answer:
        explanation = "Key: '4A: fit_hierarchical_cluster' not found.\n"
        test_answers_compute_4A_colon_fit_hierarchical_cluster_function.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['4A: fit_hierarchical_cluster']
    student_answer = run_compute('part4', 's')
    if '4A: fit_hierarchical_cluster' not in student_answer:
        explanation = "Key: '4A: fit_hierarchical_cluster' not found.\n"
        test_answers_compute_4A_colon_fit_hierarchical_cluster_function.explanation = explanation
        assert False
    else:
        student_answer = student_answer['4A: fit_hierarchical_cluster']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.001
    keys = None
    msg_structure = "assert_utilities.check_structure_function(student_answer, instructor_answer)"
    msg_answer = "assert_utilities.check_answer_function(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
    else: 
        explanation_answer = "" 
    explanation = '\n'.join(['Structure tests:', explanation_structure, 'Answer tests:', explanation_answer])
    test_answers_compute_4A_colon_fit_hierarchical_cluster_function.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_compute_4B_colon_cluster_successes_list_lbrack_string_rbrack(run_compute):
    correct_answer = run_compute('part4', 'i')
    if '4B: cluster successes' not in correct_answer:
        explanation = "Key: '4B: cluster successes' not found.\n"
        test_answers_compute_4B_colon_cluster_successes_list_lbrack_string_rbrack.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['4B: cluster successes']
    student_answer = run_compute('part4', 's')
    if '4B: cluster successes' not in student_answer:
        explanation = "Key: '4B: cluster successes' not found.\n"
        test_answers_compute_4B_colon_cluster_successes_list_lbrack_string_rbrack.explanation = explanation
        assert False
    else:
        student_answer = student_answer['4B: cluster successes']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.001
    keys = None
    msg_structure = "assert_utilities.check_structure_list_string(student_answer, instructor_answer)"
    msg_answer = "assert_utilities.check_answer_list_string(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
    else: 
        explanation_answer = "" 
    explanation = '\n'.join(['Structure tests:', explanation_structure, 'Answer tests:', explanation_answer])
    test_answers_compute_4B_colon_cluster_successes_list_lbrack_string_rbrack.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_compute_4C_colon_modified_function_function(run_compute):
    correct_answer = run_compute('part4', 'i')
    if '4C: modified function' not in correct_answer:
        explanation = "Key: '4C: modified function' not found.\n"
        test_answers_compute_4C_colon_modified_function_function.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['4C: modified function']
    student_answer = run_compute('part4', 's')
    if '4C: modified function' not in student_answer:
        explanation = "Key: '4C: modified function' not found.\n"
        test_answers_compute_4C_colon_modified_function_function.explanation = explanation
        assert False
    else:
        student_answer = student_answer['4C: modified function']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.001
    keys = None
    msg_structure = "assert_utilities.check_structure_function(student_answer, instructor_answer)"
    msg_answer = "assert_utilities.check_answer_function(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
    else: 
        explanation_answer = "" 
    explanation = '\n'.join(['Structure tests:', explanation_structure, 'Answer tests:', explanation_answer])
    test_answers_compute_4C_colon_modified_function_function.explanation = explanation
    assert is_success


