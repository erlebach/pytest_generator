
import pytest
from pytest_utils.decorators import max_score, visibility, hide_errors
import instructor_code_with_answers.part3 as ic
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
def test_answers_compute_3A_colon_toy_data_dict_lbrack_string_comma_NDArray_rbrack(run_compute):
    correct_answer = run_compute('part3', 'i')
    if '3A: toy data' not in correct_answer:
        explanation = "Key: '3A: toy data' not found.\n"
        test_answers_compute_3A_colon_toy_data_dict_lbrack_string_comma_NDArray_rbrack.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['3A: toy data']
    student_answer = run_compute('part3', 's')
    if '3A: toy data' not in student_answer:
        explanation = "Key: '3A: toy data' not found.\n"
        test_answers_compute_3A_colon_toy_data_dict_lbrack_string_comma_NDArray_rbrack.explanation = explanation
        assert False
    else:
        student_answer = student_answer['3A: toy data']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.001
    keys = ['X', 'y']
    msg_structure = "assert_utilities.check_structure_dict_string_NDArray(student_answer, instructor_answer, rel_tol, keys)"
    msg_answer = "assert_utilities.check_answer_dict_string_NDArray(student_answer, instructor_answer, rel_tol, keys)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
    else: 
        explanation_answer = "" 
    explanation = '\n'.join(['Structure tests:', explanation_structure, 'Answer tests:', explanation_answer])
    test_answers_compute_3A_colon_toy_data_dict_lbrack_string_comma_NDArray_rbrack.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_compute_3B_colon_linkage_NDArray(run_compute):
    correct_answer = run_compute('part3', 'i')
    if '3B: linkage' not in correct_answer:
        explanation = "Key: '3B: linkage' not found.\n"
        test_answers_compute_3B_colon_linkage_NDArray.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['3B: linkage']
    student_answer = run_compute('part3', 's')
    if '3B: linkage' not in student_answer:
        explanation = "Key: '3B: linkage' not found.\n"
        test_answers_compute_3B_colon_linkage_NDArray.explanation = explanation
        assert False
    else:
        student_answer = student_answer['3B: linkage']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.001
    keys = None
    msg_structure = "assert_utilities.check_structure_NDArray(student_answer, instructor_answer)"
    msg_answer = "assert_utilities.check_answer_NDArray(student_answer, instructor_answer, rel_tol)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
    else: 
        explanation_answer = "" 
    explanation = '\n'.join(['Structure tests:', explanation_structure, 'Answer tests:', explanation_answer])
    test_answers_compute_3B_colon_linkage_NDArray.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_compute_3B_colon_dendogram_dendrogram(run_compute):
    correct_answer = run_compute('part3', 'i')
    if '3B: dendogram' not in correct_answer:
        explanation = "Key: '3B: dendogram' not found.\n"
        test_answers_compute_3B_colon_dendogram_dendrogram.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['3B: dendogram']
    student_answer = run_compute('part3', 's')
    if '3B: dendogram' not in student_answer:
        explanation = "Key: '3B: dendogram' not found.\n"
        test_answers_compute_3B_colon_dendogram_dendrogram.explanation = explanation
        assert False
    else:
        student_answer = student_answer['3B: dendogram']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.001
    keys = None
    msg_structure = "assert_utilities.check_structure_dendrogram(student_answer, instructor_answer)"
    msg_answer = "assert_utilities.check_answer_dendrogram(student_answer, instructor_answer, rel_tol)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
    else: 
        explanation_answer = "" 
    explanation = '\n'.join(['Structure tests:', explanation_structure, 'Answer tests:', explanation_answer])
    test_answers_compute_3B_colon_dendogram_dendrogram.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_compute_3C_colon_iteration_int(run_compute):
    correct_answer = run_compute('part3', 'i')
    if '3C: iteration' not in correct_answer:
        explanation = "Key: '3C: iteration' not found.\n"
        test_answers_compute_3C_colon_iteration_int.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['3C: iteration']
    student_answer = run_compute('part3', 's')
    if '3C: iteration' not in student_answer:
        explanation = "Key: '3C: iteration' not found.\n"
        test_answers_compute_3C_colon_iteration_int.explanation = explanation
        assert False
    else:
        student_answer = student_answer['3C: iteration']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.001
    keys = None
    msg_structure = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    msg_answer = "assert_utilities.check_answer_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
    else: 
        explanation_answer = "" 
    explanation = '\n'.join(['Structure tests:', explanation_structure, 'Answer tests:', explanation_answer])
    test_answers_compute_3C_colon_iteration_int.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_compute_3D_colon_function_function(run_compute):
    correct_answer = run_compute('part3', 'i')
    if '3D: function' not in correct_answer:
        explanation = "Key: '3D: function' not found.\n"
        test_answers_compute_3D_colon_function_function.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['3D: function']
    student_answer = run_compute('part3', 's')
    if '3D: function' not in student_answer:
        explanation = "Key: '3D: function' not found.\n"
        test_answers_compute_3D_colon_function_function.explanation = explanation
        assert False
    else:
        student_answer = student_answer['3D: function']
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
    test_answers_compute_3D_colon_function_function.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_compute_3E_colon_clusters_set_lbrack_set_rbrack(run_compute):
    correct_answer = run_compute('part3', 'i')
    if '3E: clusters' not in correct_answer:
        explanation = "Key: '3E: clusters' not found.\n"
        test_answers_compute_3E_colon_clusters_set_lbrack_set_rbrack.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['3E: clusters']
    student_answer = run_compute('part3', 's')
    if '3E: clusters' not in student_answer:
        explanation = "Key: '3E: clusters' not found.\n"
        test_answers_compute_3E_colon_clusters_set_lbrack_set_rbrack.explanation = explanation
        assert False
    else:
        student_answer = student_answer['3E: clusters']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.001
    keys = None
    msg_structure = "assert_utilities.check_structure_set_set(student_answer, instructor_answer)"
    msg_answer = "assert_utilities.check_answer_set_set(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)
    if is_success:
        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)
    else: 
        explanation_answer = "" 
    explanation = '\n'.join(['Structure tests:', explanation_structure, 'Answer tests:', explanation_answer])
    test_answers_compute_3E_colon_clusters_set_lbrack_set_rbrack.explanation = explanation
    assert is_success



@max_score(20)
@hide_errors('')
def test_answers_compute_3F_colon_rich_get_richer_string(run_compute):
    correct_answer = run_compute('part3', 'i')
    if '3F: rich get richer' not in correct_answer:
        explanation = "Key: '3F: rich get richer' not found.\n"
        test_answers_compute_3F_colon_rich_get_richer_string.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['3F: rich get richer']
    student_answer = run_compute('part3', 's')
    if '3F: rich get richer' not in student_answer:
        explanation = "Key: '3F: rich get richer' not found.\n"
        test_answers_compute_3F_colon_rich_get_richer_string.explanation = explanation
        assert False
    else:
        student_answer = student_answer['3F: rich get richer']
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
    test_answers_compute_3F_colon_rich_get_richer_string.explanation = explanation
    assert is_success


