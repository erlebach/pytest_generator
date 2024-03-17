
import pytest
from pytest_utils.decorators import max_score, visibility, hide_errors
import instructor_code_with_answers.part1 as ic
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
def test_answers_compute_1A_colon_datasets_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack(run_compute):
    correct_answer = run_compute('part1', 'i')
    if '1A: datasets' not in correct_answer:
        explanation = "Key: '1A: datasets' not found.\n"
        test_answers_compute_1A_colon_datasets_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['1A: datasets']
    student_answer = run_compute('part1', 's')
    if '1A: datasets' not in student_answer:
        explanation = "Key: '1A: datasets' not found.\n"
        test_answers_compute_1A_colon_datasets_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack.explanation = explanation
        assert False
    else:
        student_answer = student_answer['1A: datasets']
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
    test_answers_compute_1A_colon_datasets_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_compute_1B_colon_fit_kmeans_function(run_compute):
    correct_answer = run_compute('part1', 'i')
    if '1B: fit_kmeans' not in correct_answer:
        explanation = "Key: '1B: fit_kmeans' not found.\n"
        test_answers_compute_1B_colon_fit_kmeans_function.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['1B: fit_kmeans']
    student_answer = run_compute('part1', 's')
    if '1B: fit_kmeans' not in student_answer:
        explanation = "Key: '1B: fit_kmeans' not found.\n"
        test_answers_compute_1B_colon_fit_kmeans_function.explanation = explanation
        assert False
    else:
        student_answer = student_answer['1B: fit_kmeans']
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
    test_answers_compute_1B_colon_fit_kmeans_function.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_compute_1C_colon_cluster_successes_dict_lbrack_string_comma_set_rbrack(run_compute):
    correct_answer = run_compute('part1', 'i')
    if '1C: cluster successes' not in correct_answer:
        explanation = "Key: '1C: cluster successes' not found.\n"
        test_answers_compute_1C_colon_cluster_successes_dict_lbrack_string_comma_set_rbrack.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['1C: cluster successes']
    student_answer = run_compute('part1', 's')
    if '1C: cluster successes' not in student_answer:
        explanation = "Key: '1C: cluster successes' not found.\n"
        test_answers_compute_1C_colon_cluster_successes_dict_lbrack_string_comma_set_rbrack.explanation = explanation
        assert False
    else:
        student_answer = student_answer['1C: cluster successes']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.001
    keys = None
    msg_structure = "assert_utilities.check_structure_dict_string_set(student_answer, instructor_answer)"
    msg_answer = "assert_utilities.check_answer_dict_string_set(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
    else: 
        explanation_answer = "" 
    explanation = '\n'.join(['Structure tests:', explanation_structure, 'Answer tests:', explanation_answer])
    test_answers_compute_1C_colon_cluster_successes_dict_lbrack_string_comma_set_rbrack.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_compute_1C_colon_cluster_failures_set_lbrack_string_rbrack(run_compute):
    correct_answer = run_compute('part1', 'i')
    if '1C: cluster failures' not in correct_answer:
        explanation = "Key: '1C: cluster failures' not found.\n"
        test_answers_compute_1C_colon_cluster_failures_set_lbrack_string_rbrack.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['1C: cluster failures']
    student_answer = run_compute('part1', 's')
    if '1C: cluster failures' not in student_answer:
        explanation = "Key: '1C: cluster failures' not found.\n"
        test_answers_compute_1C_colon_cluster_failures_set_lbrack_string_rbrack.explanation = explanation
        assert False
    else:
        student_answer = student_answer['1C: cluster failures']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.001
    keys = None
    msg_structure = "assert_utilities.check_structure_set_string(student_answer, instructor_answer)"
    msg_answer = "assert_utilities.check_answer_set_string(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
    else: 
        explanation_answer = "" 
    explanation = '\n'.join(['Structure tests:', explanation_structure, 'Answer tests:', explanation_answer])
    test_answers_compute_1C_colon_cluster_failures_set_lbrack_string_rbrack.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_compute_1D_colon_datasets_sensitive_to_initialization_set_lbrack_string_rbrack(run_compute):
    correct_answer = run_compute('part1', 'i')
    if '1D: datasets sensitive to initialization' not in correct_answer:
        explanation = "Key: '1D: datasets sensitive to initialization' not found.\n"
        test_answers_compute_1D_colon_datasets_sensitive_to_initialization_set_lbrack_string_rbrack.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['1D: datasets sensitive to initialization']
    student_answer = run_compute('part1', 's')
    if '1D: datasets sensitive to initialization' not in student_answer:
        explanation = "Key: '1D: datasets sensitive to initialization' not found.\n"
        test_answers_compute_1D_colon_datasets_sensitive_to_initialization_set_lbrack_string_rbrack.explanation = explanation
        assert False
    else:
        student_answer = student_answer['1D: datasets sensitive to initialization']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.001
    keys = None
    msg_structure = "assert_utilities.check_structure_set_string(student_answer, instructor_answer)"
    msg_answer = "assert_utilities.check_answer_set_string(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
    else: 
        explanation_answer = "" 
    explanation = '\n'.join(['Structure tests:', explanation_structure, 'Answer tests:', explanation_answer])
    test_answers_compute_1D_colon_datasets_sensitive_to_initialization_set_lbrack_string_rbrack.explanation = explanation
    assert is_success


