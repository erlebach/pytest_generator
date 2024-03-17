import pytest
import pytest
import assert_utilities
import numpy
from part3 import *
import yaml
import test_utils as u
from my_fixtures import *
from pytest_utils.decorators import max_score, visibility, partial_score, hide_errors

with open('type_handlers.yaml', 'r') as f:
    type_handlers = yaml.safe_load(f)

@hide_errors('')
def test_structure_compute_3A_colon_toy_data_dict_lbrack_string_comma_NDArray_rbrack_lparen_rparen(run_compute):
    correct_answer = run_compute('part3', 'i')
    if '3A: toy data' not in correct_answer:
        explanation = "Key: '3A: toy data' not found.\n"
        test_structure_compute_3A_colon_toy_data_dict_lbrack_string_comma_NDArray_rbrack_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['3A: toy data']
    student_answer = run_compute('part3', 's')
    if '3A: toy data' not in student_answer:
        explanation = "Key: '3A: toy data' not found.\n"
        test_structure_compute_3A_colon_toy_data_dict_lbrack_string_comma_NDArray_rbrack_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['3A: toy data']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = ['X', 'y']
    msg = "assert_utilities.check_structure_dict_string_NDArray(student_answer, instructor_answer, rel_tol, keys)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_compute_3A_colon_toy_data_dict_lbrack_string_comma_NDArray_rbrack_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_compute_3B_colon_linkage_NDArray_lparen_rparen(run_compute):
    correct_answer = run_compute('part3', 'i')
    if '3B: linkage' not in correct_answer:
        explanation = "Key: '3B: linkage' not found.\n"
        test_structure_compute_3B_colon_linkage_NDArray_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['3B: linkage']
    student_answer = run_compute('part3', 's')
    if '3B: linkage' not in student_answer:
        explanation = "Key: '3B: linkage' not found.\n"
        test_structure_compute_3B_colon_linkage_NDArray_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['3B: linkage']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_NDArray(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_compute_3B_colon_linkage_NDArray_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_compute_3B_colon_dendogram_dendrogram_lparen_rparen(run_compute):
    correct_answer = run_compute('part3', 'i')
    if '3B: dendogram' not in correct_answer:
        explanation = "Key: '3B: dendogram' not found.\n"
        test_structure_compute_3B_colon_dendogram_dendrogram_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['3B: dendogram']
    student_answer = run_compute('part3', 's')
    if '3B: dendogram' not in student_answer:
        explanation = "Key: '3B: dendogram' not found.\n"
        test_structure_compute_3B_colon_dendogram_dendrogram_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['3B: dendogram']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_dendrogram(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_compute_3B_colon_dendogram_dendrogram_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_compute_3C_colon_iteration_int_lparen_rparen(run_compute):
    correct_answer = run_compute('part3', 'i')
    if '3C: iteration' not in correct_answer:
        explanation = "Key: '3C: iteration' not found.\n"
        test_structure_compute_3C_colon_iteration_int_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['3C: iteration']
    student_answer = run_compute('part3', 's')
    if '3C: iteration' not in student_answer:
        explanation = "Key: '3C: iteration' not found.\n"
        test_structure_compute_3C_colon_iteration_int_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['3C: iteration']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_compute_3C_colon_iteration_int_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_compute_3D_colon_function_function_lparen_rparen(run_compute):
    correct_answer = run_compute('part3', 'i')
    if '3D: function' not in correct_answer:
        explanation = "Key: '3D: function' not found.\n"
        test_structure_compute_3D_colon_function_function_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['3D: function']
    student_answer = run_compute('part3', 's')
    if '3D: function' not in student_answer:
        explanation = "Key: '3D: function' not found.\n"
        test_structure_compute_3D_colon_function_function_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['3D: function']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_function(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_compute_3D_colon_function_function_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_compute_3E_colon_clusters_set_lbrack_set_rbrack_lparen_rparen(run_compute):
    correct_answer = run_compute('part3', 'i')
    if '3E: clusters' not in correct_answer:
        explanation = "Key: '3E: clusters' not found.\n"
        test_structure_compute_3E_colon_clusters_set_lbrack_set_rbrack_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['3E: clusters']
    student_answer = run_compute('part3', 's')
    if '3E: clusters' not in student_answer:
        explanation = "Key: '3E: clusters' not found.\n"
        test_structure_compute_3E_colon_clusters_set_lbrack_set_rbrack_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['3E: clusters']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_set_set(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_compute_3E_colon_clusters_set_lbrack_set_rbrack_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_compute_3F_colon_rich_get_richer_string_lparen_rparen(run_compute):
    correct_answer = run_compute('part3', 'i')
    if '3F: rich get richer' not in correct_answer:
        explanation = "Key: '3F: rich get richer' not found.\n"
        test_structure_compute_3F_colon_rich_get_richer_string_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['3F: rich get richer']
    student_answer = run_compute('part3', 's')
    if '3F: rich get richer' not in student_answer:
        explanation = "Key: '3F: rich get richer' not found.\n"
        test_structure_compute_3F_colon_rich_get_richer_string_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['3F: rich get richer']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_string(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_compute_3F_colon_rich_get_richer_string_lparen_rparen.explanation = explanation
    assert is_success
