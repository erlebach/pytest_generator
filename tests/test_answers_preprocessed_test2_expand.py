
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

@max_score(10)
@hide_errors('')
def test_answers_basic_int_test0_correct_int_i_answer_s_answer_0_0_int(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_i_answer_s_answer_0_0' not in correct_answer:
        explanation = "Key: 'correct_int_i_answer_s_answer_0_0' not found.\n"
        test_answers_basic_int_test0_correct_int_i_answer_s_answer_0_0_int.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_i_answer_s_answer_0_0']
    student_answer = run_compute('part1', 's')
    if 'correct_int_i_answer_s_answer_0_0' not in student_answer:
        explanation = "Key: 'correct_int_i_answer_s_answer_0_0' not found.\n"
        test_answers_basic_int_test0_correct_int_i_answer_s_answer_0_0_int.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_i_answer_s_answer_0_0']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_answer_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_answers_basic_int_test0_correct_int_i_answer_s_answer_0_0_int.explanation = explanation
    assert is_success



@max_score(10)
@hide_errors('')
def test_answers_basic_int_test0_correct_int_i_answer_s_answers_0_0_int(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_i_answer_s_answers_0_0' not in correct_answer:
        explanation = "Key: 'correct_int_i_answer_s_answers_0_0' not found.\n"
        test_answers_basic_int_test0_correct_int_i_answer_s_answers_0_0_int.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_i_answer_s_answers_0_0']
    student_answer = run_compute('part1', 's')
    if 'correct_int_i_answer_s_answers_0_0' not in student_answer:
        explanation = "Key: 'correct_int_i_answer_s_answers_0_0' not found.\n"
        test_answers_basic_int_test0_correct_int_i_answer_s_answers_0_0_int.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_i_answer_s_answers_0_0']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_answer_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_answers_basic_int_test0_correct_int_i_answer_s_answers_0_0_int.explanation = explanation
    assert is_success



@max_score(10)
@hide_errors('')
def test_answers_basic_int_test0_correct_int_i_answer_s_answers_0_1_int(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_i_answer_s_answers_0_1' not in correct_answer:
        explanation = "Key: 'correct_int_i_answer_s_answers_0_1' not found.\n"
        test_answers_basic_int_test0_correct_int_i_answer_s_answers_0_1_int.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_i_answer_s_answers_0_1']
    student_answer = run_compute('part1', 's')
    if 'correct_int_i_answer_s_answers_0_1' not in student_answer:
        explanation = "Key: 'correct_int_i_answer_s_answers_0_1' not found.\n"
        test_answers_basic_int_test0_correct_int_i_answer_s_answers_0_1_int.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_i_answer_s_answers_0_1']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_answer_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_answers_basic_int_test0_correct_int_i_answer_s_answers_0_1_int.explanation = explanation
    assert is_success



@max_score(10)
@hide_errors('')
def test_answers_basic_int_test1_correct_int_i_answers_s_answer_0_0_int(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_i_answers_s_answer_0_0' not in correct_answer:
        explanation = "Key: 'correct_int_i_answers_s_answer_0_0' not found.\n"
        test_answers_basic_int_test1_correct_int_i_answers_s_answer_0_0_int.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_i_answers_s_answer_0_0']
    student_answer = run_compute('part1', 's')
    if 'correct_int_i_answers_s_answer_0_0' not in student_answer:
        explanation = "Key: 'correct_int_i_answers_s_answer_0_0' not found.\n"
        test_answers_basic_int_test1_correct_int_i_answers_s_answer_0_0_int.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_i_answers_s_answer_0_0']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_answer_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_answers_basic_int_test1_correct_int_i_answers_s_answer_0_0_int.explanation = explanation
    assert is_success



@max_score(10)
@hide_errors('')
def test_answers_basic_int_test1_correct_int_i_answers_s_answer_1_0_int(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_i_answers_s_answer_1_0' not in correct_answer:
        explanation = "Key: 'correct_int_i_answers_s_answer_1_0' not found.\n"
        test_answers_basic_int_test1_correct_int_i_answers_s_answer_1_0_int.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_i_answers_s_answer_1_0']
    student_answer = run_compute('part1', 's')
    if 'correct_int_i_answers_s_answer_1_0' not in student_answer:
        explanation = "Key: 'correct_int_i_answers_s_answer_1_0' not found.\n"
        test_answers_basic_int_test1_correct_int_i_answers_s_answer_1_0_int.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_i_answers_s_answer_1_0']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_answer_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_answers_basic_int_test1_correct_int_i_answers_s_answer_1_0_int.explanation = explanation
    assert is_success



@max_score(10)
@hide_errors('')
def test_answers_basic_int_test2_correct_int_s_answers_i_answers_0_0_int(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_s_answers_i_answers_0_0' not in correct_answer:
        explanation = "Key: 'correct_int_s_answers_i_answers_0_0' not found.\n"
        test_answers_basic_int_test2_correct_int_s_answers_i_answers_0_0_int.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_s_answers_i_answers_0_0']
    student_answer = run_compute('part1', 's')
    if 'correct_int_s_answers_i_answers_0_0' not in student_answer:
        explanation = "Key: 'correct_int_s_answers_i_answers_0_0' not found.\n"
        test_answers_basic_int_test2_correct_int_s_answers_i_answers_0_0_int.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_s_answers_i_answers_0_0']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_answer_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_answers_basic_int_test2_correct_int_s_answers_i_answers_0_0_int.explanation = explanation
    assert is_success



@max_score(10)
@hide_errors('')
def test_answers_basic_int_test2_correct_int_s_answers_i_answers_1_0_int(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_s_answers_i_answers_1_0' not in correct_answer:
        explanation = "Key: 'correct_int_s_answers_i_answers_1_0' not found.\n"
        test_answers_basic_int_test2_correct_int_s_answers_i_answers_1_0_int.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_s_answers_i_answers_1_0']
    student_answer = run_compute('part1', 's')
    if 'correct_int_s_answers_i_answers_1_0' not in student_answer:
        explanation = "Key: 'correct_int_s_answers_i_answers_1_0' not found.\n"
        test_answers_basic_int_test2_correct_int_s_answers_i_answers_1_0_int.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_s_answers_i_answers_1_0']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_answer_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_answers_basic_int_test2_correct_int_s_answers_i_answers_1_0_int.explanation = explanation
    assert is_success



@max_score(10)
@hide_errors('')
def test_answers_basic_int_test2_correct_int_s_answers_i_answers_0_1_int(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_s_answers_i_answers_0_1' not in correct_answer:
        explanation = "Key: 'correct_int_s_answers_i_answers_0_1' not found.\n"
        test_answers_basic_int_test2_correct_int_s_answers_i_answers_0_1_int.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_s_answers_i_answers_0_1']
    student_answer = run_compute('part1', 's')
    if 'correct_int_s_answers_i_answers_0_1' not in student_answer:
        explanation = "Key: 'correct_int_s_answers_i_answers_0_1' not found.\n"
        test_answers_basic_int_test2_correct_int_s_answers_i_answers_0_1_int.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_s_answers_i_answers_0_1']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_answer_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_answers_basic_int_test2_correct_int_s_answers_i_answers_0_1_int.explanation = explanation
    assert is_success



@max_score(10)
@hide_errors('')
def test_answers_basic_int_test2_correct_int_s_answers_i_answers_1_1_int(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_s_answers_i_answers_1_1' not in correct_answer:
        explanation = "Key: 'correct_int_s_answers_i_answers_1_1' not found.\n"
        test_answers_basic_int_test2_correct_int_s_answers_i_answers_1_1_int.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_s_answers_i_answers_1_1']
    student_answer = run_compute('part1', 's')
    if 'correct_int_s_answers_i_answers_1_1' not in student_answer:
        explanation = "Key: 'correct_int_s_answers_i_answers_1_1' not found.\n"
        test_answers_basic_int_test2_correct_int_s_answers_i_answers_1_1_int.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_s_answers_i_answers_1_1']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_answer_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_answers_basic_int_test2_correct_int_s_answers_i_answers_1_1_int.explanation = explanation
    assert is_success



@max_score(10)
@hide_errors('')
def test_answers_basic_int_test3_correct_int_only_i_answers_0_0_int(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_only_i_answers_0_0' not in correct_answer:
        explanation = "Key: 'correct_int_only_i_answers_0_0' not found.\n"
        test_answers_basic_int_test3_correct_int_only_i_answers_0_0_int.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_only_i_answers_0_0']
    student_answer = run_compute('part1', 's')
    if 'correct_int_only_i_answers_0_0' not in student_answer:
        explanation = "Key: 'correct_int_only_i_answers_0_0' not found.\n"
        test_answers_basic_int_test3_correct_int_only_i_answers_0_0_int.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_only_i_answers_0_0']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_answer_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_answers_basic_int_test3_correct_int_only_i_answers_0_0_int.explanation = explanation
    assert is_success



@max_score(10)
@hide_errors('')
def test_answers_basic_int_test3_correct_int_only_i_answers_1_0_int(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_only_i_answers_1_0' not in correct_answer:
        explanation = "Key: 'correct_int_only_i_answers_1_0' not found.\n"
        test_answers_basic_int_test3_correct_int_only_i_answers_1_0_int.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_only_i_answers_1_0']
    student_answer = run_compute('part1', 's')
    if 'correct_int_only_i_answers_1_0' not in student_answer:
        explanation = "Key: 'correct_int_only_i_answers_1_0' not found.\n"
        test_answers_basic_int_test3_correct_int_only_i_answers_1_0_int.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_only_i_answers_1_0']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_answer_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_answers_basic_int_test3_correct_int_only_i_answers_1_0_int.explanation = explanation
    assert is_success



@max_score(10)
@hide_errors('')
def test_answers_basic_int_test3_correct_int_only_s_answers_0_0_int(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_only_s_answers_0_0' not in correct_answer:
        explanation = "Key: 'correct_int_only_s_answers_0_0' not found.\n"
        test_answers_basic_int_test3_correct_int_only_s_answers_0_0_int.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_only_s_answers_0_0']
    student_answer = run_compute('part1', 's')
    if 'correct_int_only_s_answers_0_0' not in student_answer:
        explanation = "Key: 'correct_int_only_s_answers_0_0' not found.\n"
        test_answers_basic_int_test3_correct_int_only_s_answers_0_0_int.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_only_s_answers_0_0']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_answer_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_answers_basic_int_test3_correct_int_only_s_answers_0_0_int.explanation = explanation
    assert is_success



@max_score(10)
@hide_errors('')
def test_answers_basic_int_test3_correct_int_only_s_answers_0_1_int(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_only_s_answers_0_1' not in correct_answer:
        explanation = "Key: 'correct_int_only_s_answers_0_1' not found.\n"
        test_answers_basic_int_test3_correct_int_only_s_answers_0_1_int.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_only_s_answers_0_1']
    student_answer = run_compute('part1', 's')
    if 'correct_int_only_s_answers_0_1' not in student_answer:
        explanation = "Key: 'correct_int_only_s_answers_0_1' not found.\n"
        test_answers_basic_int_test3_correct_int_only_s_answers_0_1_int.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_only_s_answers_0_1']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_answer_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_answers_basic_int_test3_correct_int_only_s_answers_0_1_int.explanation = explanation
    assert is_success



@max_score(10)
@hide_errors('')
def test_answers_basic_int_test4_correct_int_only_i_answer_0_0_int(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_only_i_answer_0_0' not in correct_answer:
        explanation = "Key: 'correct_int_only_i_answer_0_0' not found.\n"
        test_answers_basic_int_test4_correct_int_only_i_answer_0_0_int.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_only_i_answer_0_0']
    student_answer = run_compute('part1', 's')
    if 'correct_int_only_i_answer_0_0' not in student_answer:
        explanation = "Key: 'correct_int_only_i_answer_0_0' not found.\n"
        test_answers_basic_int_test4_correct_int_only_i_answer_0_0_int.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_only_i_answer_0_0']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_answer_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_answers_basic_int_test4_correct_int_only_i_answer_0_0_int.explanation = explanation
    assert is_success



@max_score(10)
@hide_errors('')
def test_answers_basic_int_test4_correct_int_only_s_answer_0_0_int(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_only_s_answer_0_0' not in correct_answer:
        explanation = "Key: 'correct_int_only_s_answer_0_0' not found.\n"
        test_answers_basic_int_test4_correct_int_only_s_answer_0_0_int.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_only_s_answer_0_0']
    student_answer = run_compute('part1', 's')
    if 'correct_int_only_s_answer_0_0' not in student_answer:
        explanation = "Key: 'correct_int_only_s_answer_0_0' not found.\n"
        test_answers_basic_int_test4_correct_int_only_s_answer_0_0_int.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_only_s_answer_0_0']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_answer_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_answers_basic_int_test4_correct_int_only_s_answer_0_0_int.explanation = explanation
    assert is_success


