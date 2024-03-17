
import pytest
from pytest_utils.decorators import max_score, visibility, hide_errors
import instructor_code_with_answers.part2 as ic
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
def test_answers_compute_2A_colon_blob_list_lbrack_NDArray_rbrack(run_compute):
    correct_answer = run_compute('part2', 'i')
    if '2A: blob' not in correct_answer:
        explanation = "Key: '2A: blob' not found.\n"
        test_answers_compute_2A_colon_blob_list_lbrack_NDArray_rbrack.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['2A: blob']
    student_answer = run_compute('part2', 's')
    if '2A: blob' not in student_answer:
        explanation = "Key: '2A: blob' not found.\n"
        test_answers_compute_2A_colon_blob_list_lbrack_NDArray_rbrack.explanation = explanation
        assert False
    else:
        student_answer = student_answer['2A: blob']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.01
    keys = None
    msg_structure = "assert_utilities.check_structure_list_NDArray(student_answer, instructor_answer)"
    msg_answer = "assert_utilities.check_answer_list_NDArray(student_answer, instructor_answer, rel_tol)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
    else: 
        explanation_answer = "" 
    explanation = '\n'.join(['Structure tests:', explanation_structure, 'Answer tests:', explanation_answer])
    test_answers_compute_2A_colon_blob_list_lbrack_NDArray_rbrack.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_compute_2B_colon_fit_kmeans_function(run_compute):
    correct_answer = run_compute('part2', 'i')
    if '2B: fit_kmeans' not in correct_answer:
        explanation = "Key: '2B: fit_kmeans' not found.\n"
        test_answers_compute_2B_colon_fit_kmeans_function.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['2B: fit_kmeans']
    student_answer = run_compute('part2', 's')
    if '2B: fit_kmeans' not in student_answer:
        explanation = "Key: '2B: fit_kmeans' not found.\n"
        test_answers_compute_2B_colon_fit_kmeans_function.explanation = explanation
        assert False
    else:
        student_answer = student_answer['2B: fit_kmeans']
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
    test_answers_compute_2B_colon_fit_kmeans_function.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_compute_2C_colon_SSE_plot_list_lbrack_list_lbrack_float_rbrack_rbrack(run_compute):
    correct_answer = run_compute('part2', 'i')
    if '2C: SSE plot' not in correct_answer:
        explanation = "Key: '2C: SSE plot' not found.\n"
        test_answers_compute_2C_colon_SSE_plot_list_lbrack_list_lbrack_float_rbrack_rbrack.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['2C: SSE plot']
    student_answer = run_compute('part2', 's')
    if '2C: SSE plot' not in student_answer:
        explanation = "Key: '2C: SSE plot' not found.\n"
        test_answers_compute_2C_colon_SSE_plot_list_lbrack_list_lbrack_float_rbrack_rbrack.explanation = explanation
        assert False
    else:
        student_answer = student_answer['2C: SSE plot']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.01
    keys = None
    msg_structure = "assert_utilities.check_structure_list_list_float(student_answer, instructor_answer)"
    msg_answer = "assert_utilities.check_answer_list_list_float(student_answer, instructor_answer, rel_tol)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
    else: 
        explanation_answer = "" 
    explanation = '\n'.join(['Structure tests:', explanation_structure, 'Answer tests:', explanation_answer])
    test_answers_compute_2C_colon_SSE_plot_list_lbrack_list_lbrack_float_rbrack_rbrack.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_compute_2D_colon_inertia_plot_list_lbrack_list_lbrack_float_rbrack_rbrack(run_compute):
    correct_answer = run_compute('part2', 'i')
    if '2D: inertia plot' not in correct_answer:
        explanation = "Key: '2D: inertia plot' not found.\n"
        test_answers_compute_2D_colon_inertia_plot_list_lbrack_list_lbrack_float_rbrack_rbrack.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['2D: inertia plot']
    student_answer = run_compute('part2', 's')
    if '2D: inertia plot' not in student_answer:
        explanation = "Key: '2D: inertia plot' not found.\n"
        test_answers_compute_2D_colon_inertia_plot_list_lbrack_list_lbrack_float_rbrack_rbrack.explanation = explanation
        assert False
    else:
        student_answer = student_answer['2D: inertia plot']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.01
    keys = None
    msg_structure = "assert_utilities.check_structure_list_list_float(student_answer, instructor_answer)"
    msg_answer = "assert_utilities.check_answer_list_list_float(student_answer, instructor_answer, rel_tol)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
    else: 
        explanation_answer = "" 
    explanation = '\n'.join(['Structure tests:', explanation_structure, 'Answer tests:', explanation_answer])
    test_answers_compute_2D_colon_inertia_plot_list_lbrack_list_lbrack_float_rbrack_rbrack.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_compute_2D_colon_do_ks_agree_ques_string(run_compute):
    correct_answer = run_compute('part2', 'i')
    if '2D: do ks agree?' not in correct_answer:
        explanation = "Key: '2D: do ks agree?' not found.\n"
        test_answers_compute_2D_colon_do_ks_agree_ques_string.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['2D: do ks agree?']
    student_answer = run_compute('part2', 's')
    if '2D: do ks agree?' not in student_answer:
        explanation = "Key: '2D: do ks agree?' not found.\n"
        test_answers_compute_2D_colon_do_ks_agree_ques_string.explanation = explanation
        assert False
    else:
        student_answer = student_answer['2D: do ks agree?']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.001
    keys = None
    msg_structure = "assert_utilities.check_structure_string(student_answer, instructor_answer)"
    msg_answer = "assert_utilities.check_answer_string(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
    else: 
        explanation_answer = "" 
    explanation = '\n'.join(['Structure tests:', explanation_structure, 'Answer tests:', explanation_answer])
    test_answers_compute_2D_colon_do_ks_agree_ques_string.explanation = explanation
    assert is_success


