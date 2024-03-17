import pytest
import pytest
import assert_utilities
import numpy
from part2 import *
import yaml
import test_utils as u
from my_fixtures import *
from pytest_utils.decorators import max_score, visibility, partial_score, hide_errors

with open('type_handlers.yaml', 'r') as f:
    type_handlers = yaml.safe_load(f)

@hide_errors('')
def test_structure_compute_2A_colon_blob_list_lbrack_NDArray_rbrack_lparen_rparen(run_compute):
    correct_answer = run_compute('part2', 'i')
    if '2A: blob' not in correct_answer:
        explanation = "Key: '2A: blob' not found.\n"
        test_structure_compute_2A_colon_blob_list_lbrack_NDArray_rbrack_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['2A: blob']
    student_answer = run_compute('part2', 's')
    if '2A: blob' not in student_answer:
        explanation = "Key: '2A: blob' not found.\n"
        test_structure_compute_2A_colon_blob_list_lbrack_NDArray_rbrack_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['2A: blob']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.01
    keys = None
    msg = "assert_utilities.check_structure_list_NDArray(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_compute_2A_colon_blob_list_lbrack_NDArray_rbrack_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_compute_2B_colon_fit_kmeans_function_lparen_rparen(run_compute):
    correct_answer = run_compute('part2', 'i')
    if '2B: fit_kmeans' not in correct_answer:
        explanation = "Key: '2B: fit_kmeans' not found.\n"
        test_structure_compute_2B_colon_fit_kmeans_function_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['2B: fit_kmeans']
    student_answer = run_compute('part2', 's')
    if '2B: fit_kmeans' not in student_answer:
        explanation = "Key: '2B: fit_kmeans' not found.\n"
        test_structure_compute_2B_colon_fit_kmeans_function_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['2B: fit_kmeans']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_function(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_compute_2B_colon_fit_kmeans_function_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_compute_2C_colon_SSE_plot_list_lbrack_list_lbrack_float_rbrack_rbrack_lparen_rparen(run_compute):
    correct_answer = run_compute('part2', 'i')
    if '2C: SSE plot' not in correct_answer:
        explanation = "Key: '2C: SSE plot' not found.\n"
        test_structure_compute_2C_colon_SSE_plot_list_lbrack_list_lbrack_float_rbrack_rbrack_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['2C: SSE plot']
    student_answer = run_compute('part2', 's')
    if '2C: SSE plot' not in student_answer:
        explanation = "Key: '2C: SSE plot' not found.\n"
        test_structure_compute_2C_colon_SSE_plot_list_lbrack_list_lbrack_float_rbrack_rbrack_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['2C: SSE plot']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.01
    keys = None
    msg = "assert_utilities.check_structure_list_list_float(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_compute_2C_colon_SSE_plot_list_lbrack_list_lbrack_float_rbrack_rbrack_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_compute_2D_colon_inertia_plot_list_lbrack_list_lbrack_float_rbrack_rbrack_lparen_rparen(run_compute):
    correct_answer = run_compute('part2', 'i')
    if '2D: inertia plot' not in correct_answer:
        explanation = "Key: '2D: inertia plot' not found.\n"
        test_structure_compute_2D_colon_inertia_plot_list_lbrack_list_lbrack_float_rbrack_rbrack_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['2D: inertia plot']
    student_answer = run_compute('part2', 's')
    if '2D: inertia plot' not in student_answer:
        explanation = "Key: '2D: inertia plot' not found.\n"
        test_structure_compute_2D_colon_inertia_plot_list_lbrack_list_lbrack_float_rbrack_rbrack_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['2D: inertia plot']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.01
    keys = None
    msg = "assert_utilities.check_structure_list_list_float(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_compute_2D_colon_inertia_plot_list_lbrack_list_lbrack_float_rbrack_rbrack_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_compute_2D_colon_do_ks_agree_ques_string_lparen_rparen(run_compute):
    correct_answer = run_compute('part2', 'i')
    if '2D: do ks agree?' not in correct_answer:
        explanation = "Key: '2D: do ks agree?' not found.\n"
        test_structure_compute_2D_colon_do_ks_agree_ques_string_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['2D: do ks agree?']
    student_answer = run_compute('part2', 's')
    if '2D: do ks agree?' not in student_answer:
        explanation = "Key: '2D: do ks agree?' not found.\n"
        test_structure_compute_2D_colon_do_ks_agree_ques_string_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['2D: do ks agree?']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_string(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_compute_2D_colon_do_ks_agree_ques_string_lparen_rparen.explanation = explanation
    assert is_success
