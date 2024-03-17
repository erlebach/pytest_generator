import pytest
import pytest
import assert_utilities
from part1 import *
import yaml
import test_utils as u
from my_fixtures import *
from pytest_utils.decorators import max_score, visibility, partial_score, hide_errors

with open('type_handlers.yaml', 'r') as f:
    type_handlers = yaml.safe_load(f)

@hide_errors('')
def test_structure_basic_int_test0_correct_int_i_answer_s_answer_0_0_int_lparen_rparen(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_i_answer_s_answer_0_0' not in correct_answer:
        explanation = "Key: 'correct_int_i_answer_s_answer_0_0' not found.\n"
        test_structure_basic_int_test0_correct_int_i_answer_s_answer_0_0_int_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_i_answer_s_answer_0_0']
    student_answer = run_compute('part1', 's')
    if 'correct_int_i_answer_s_answer_0_0' not in student_answer:
        explanation = "Key: 'correct_int_i_answer_s_answer_0_0' not found.\n"
        test_structure_basic_int_test0_correct_int_i_answer_s_answer_0_0_int_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_i_answer_s_answer_0_0']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test0_correct_int_i_answer_s_answer_0_0_int_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_basic_int_test0_correct_int_i_answer_s_answers_0_0_int_lparen_rparen(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_i_answer_s_answers_0_0' not in correct_answer:
        explanation = "Key: 'correct_int_i_answer_s_answers_0_0' not found.\n"
        test_structure_basic_int_test0_correct_int_i_answer_s_answers_0_0_int_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_i_answer_s_answers_0_0']
    student_answer = run_compute('part1', 's')
    if 'correct_int_i_answer_s_answers_0_0' not in student_answer:
        explanation = "Key: 'correct_int_i_answer_s_answers_0_0' not found.\n"
        test_structure_basic_int_test0_correct_int_i_answer_s_answers_0_0_int_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_i_answer_s_answers_0_0']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test0_correct_int_i_answer_s_answers_0_0_int_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_basic_int_test0_correct_int_i_answer_s_answers_0_1_int_lparen_rparen(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_i_answer_s_answers_0_1' not in correct_answer:
        explanation = "Key: 'correct_int_i_answer_s_answers_0_1' not found.\n"
        test_structure_basic_int_test0_correct_int_i_answer_s_answers_0_1_int_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_i_answer_s_answers_0_1']
    student_answer = run_compute('part1', 's')
    if 'correct_int_i_answer_s_answers_0_1' not in student_answer:
        explanation = "Key: 'correct_int_i_answer_s_answers_0_1' not found.\n"
        test_structure_basic_int_test0_correct_int_i_answer_s_answers_0_1_int_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_i_answer_s_answers_0_1']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test0_correct_int_i_answer_s_answers_0_1_int_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_basic_int_test1_correct_int_i_answers_s_answer_0_0_int_lparen_rparen(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_i_answers_s_answer_0_0' not in correct_answer:
        explanation = "Key: 'correct_int_i_answers_s_answer_0_0' not found.\n"
        test_structure_basic_int_test1_correct_int_i_answers_s_answer_0_0_int_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_i_answers_s_answer_0_0']
    student_answer = run_compute('part1', 's')
    if 'correct_int_i_answers_s_answer_0_0' not in student_answer:
        explanation = "Key: 'correct_int_i_answers_s_answer_0_0' not found.\n"
        test_structure_basic_int_test1_correct_int_i_answers_s_answer_0_0_int_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_i_answers_s_answer_0_0']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test1_correct_int_i_answers_s_answer_0_0_int_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_basic_int_test1_correct_int_i_answers_s_answer_1_0_int_lparen_rparen(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_i_answers_s_answer_1_0' not in correct_answer:
        explanation = "Key: 'correct_int_i_answers_s_answer_1_0' not found.\n"
        test_structure_basic_int_test1_correct_int_i_answers_s_answer_1_0_int_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_i_answers_s_answer_1_0']
    student_answer = run_compute('part1', 's')
    if 'correct_int_i_answers_s_answer_1_0' not in student_answer:
        explanation = "Key: 'correct_int_i_answers_s_answer_1_0' not found.\n"
        test_structure_basic_int_test1_correct_int_i_answers_s_answer_1_0_int_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_i_answers_s_answer_1_0']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test1_correct_int_i_answers_s_answer_1_0_int_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_basic_int_test2_correct_int_s_answers_i_answers_0_0_int_lparen_rparen(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_s_answers_i_answers_0_0' not in correct_answer:
        explanation = "Key: 'correct_int_s_answers_i_answers_0_0' not found.\n"
        test_structure_basic_int_test2_correct_int_s_answers_i_answers_0_0_int_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_s_answers_i_answers_0_0']
    student_answer = run_compute('part1', 's')
    if 'correct_int_s_answers_i_answers_0_0' not in student_answer:
        explanation = "Key: 'correct_int_s_answers_i_answers_0_0' not found.\n"
        test_structure_basic_int_test2_correct_int_s_answers_i_answers_0_0_int_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_s_answers_i_answers_0_0']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test2_correct_int_s_answers_i_answers_0_0_int_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_basic_int_test2_correct_int_s_answers_i_answers_1_0_int_lparen_rparen(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_s_answers_i_answers_1_0' not in correct_answer:
        explanation = "Key: 'correct_int_s_answers_i_answers_1_0' not found.\n"
        test_structure_basic_int_test2_correct_int_s_answers_i_answers_1_0_int_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_s_answers_i_answers_1_0']
    student_answer = run_compute('part1', 's')
    if 'correct_int_s_answers_i_answers_1_0' not in student_answer:
        explanation = "Key: 'correct_int_s_answers_i_answers_1_0' not found.\n"
        test_structure_basic_int_test2_correct_int_s_answers_i_answers_1_0_int_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_s_answers_i_answers_1_0']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test2_correct_int_s_answers_i_answers_1_0_int_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_basic_int_test2_correct_int_s_answers_i_answers_0_1_int_lparen_rparen(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_s_answers_i_answers_0_1' not in correct_answer:
        explanation = "Key: 'correct_int_s_answers_i_answers_0_1' not found.\n"
        test_structure_basic_int_test2_correct_int_s_answers_i_answers_0_1_int_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_s_answers_i_answers_0_1']
    student_answer = run_compute('part1', 's')
    if 'correct_int_s_answers_i_answers_0_1' not in student_answer:
        explanation = "Key: 'correct_int_s_answers_i_answers_0_1' not found.\n"
        test_structure_basic_int_test2_correct_int_s_answers_i_answers_0_1_int_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_s_answers_i_answers_0_1']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test2_correct_int_s_answers_i_answers_0_1_int_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_basic_int_test2_correct_int_s_answers_i_answers_1_1_int_lparen_rparen(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_s_answers_i_answers_1_1' not in correct_answer:
        explanation = "Key: 'correct_int_s_answers_i_answers_1_1' not found.\n"
        test_structure_basic_int_test2_correct_int_s_answers_i_answers_1_1_int_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_s_answers_i_answers_1_1']
    student_answer = run_compute('part1', 's')
    if 'correct_int_s_answers_i_answers_1_1' not in student_answer:
        explanation = "Key: 'correct_int_s_answers_i_answers_1_1' not found.\n"
        test_structure_basic_int_test2_correct_int_s_answers_i_answers_1_1_int_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_s_answers_i_answers_1_1']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test2_correct_int_s_answers_i_answers_1_1_int_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_basic_int_test3_correct_int_only_i_answers_0_0_int_lparen_rparen(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_only_i_answers_0_0' not in correct_answer:
        explanation = "Key: 'correct_int_only_i_answers_0_0' not found.\n"
        test_structure_basic_int_test3_correct_int_only_i_answers_0_0_int_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_only_i_answers_0_0']
    student_answer = run_compute('part1', 's')
    if 'correct_int_only_i_answers_0_0' not in student_answer:
        explanation = "Key: 'correct_int_only_i_answers_0_0' not found.\n"
        test_structure_basic_int_test3_correct_int_only_i_answers_0_0_int_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_only_i_answers_0_0']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test3_correct_int_only_i_answers_0_0_int_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_basic_int_test3_correct_int_only_i_answers_1_0_int_lparen_rparen(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_only_i_answers_1_0' not in correct_answer:
        explanation = "Key: 'correct_int_only_i_answers_1_0' not found.\n"
        test_structure_basic_int_test3_correct_int_only_i_answers_1_0_int_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_only_i_answers_1_0']
    student_answer = run_compute('part1', 's')
    if 'correct_int_only_i_answers_1_0' not in student_answer:
        explanation = "Key: 'correct_int_only_i_answers_1_0' not found.\n"
        test_structure_basic_int_test3_correct_int_only_i_answers_1_0_int_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_only_i_answers_1_0']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test3_correct_int_only_i_answers_1_0_int_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_basic_int_test3_correct_int_only_s_answers_0_0_int_lparen_rparen(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_only_s_answers_0_0' not in correct_answer:
        explanation = "Key: 'correct_int_only_s_answers_0_0' not found.\n"
        test_structure_basic_int_test3_correct_int_only_s_answers_0_0_int_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_only_s_answers_0_0']
    student_answer = run_compute('part1', 's')
    if 'correct_int_only_s_answers_0_0' not in student_answer:
        explanation = "Key: 'correct_int_only_s_answers_0_0' not found.\n"
        test_structure_basic_int_test3_correct_int_only_s_answers_0_0_int_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_only_s_answers_0_0']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test3_correct_int_only_s_answers_0_0_int_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_basic_int_test3_correct_int_only_s_answers_0_1_int_lparen_rparen(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_only_s_answers_0_1' not in correct_answer:
        explanation = "Key: 'correct_int_only_s_answers_0_1' not found.\n"
        test_structure_basic_int_test3_correct_int_only_s_answers_0_1_int_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_only_s_answers_0_1']
    student_answer = run_compute('part1', 's')
    if 'correct_int_only_s_answers_0_1' not in student_answer:
        explanation = "Key: 'correct_int_only_s_answers_0_1' not found.\n"
        test_structure_basic_int_test3_correct_int_only_s_answers_0_1_int_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_only_s_answers_0_1']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test3_correct_int_only_s_answers_0_1_int_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_basic_int_test4_correct_int_only_i_answer_0_0_int_lparen_rparen(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_only_i_answer_0_0' not in correct_answer:
        explanation = "Key: 'correct_int_only_i_answer_0_0' not found.\n"
        test_structure_basic_int_test4_correct_int_only_i_answer_0_0_int_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_only_i_answer_0_0']
    student_answer = run_compute('part1', 's')
    if 'correct_int_only_i_answer_0_0' not in student_answer:
        explanation = "Key: 'correct_int_only_i_answer_0_0' not found.\n"
        test_structure_basic_int_test4_correct_int_only_i_answer_0_0_int_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_only_i_answer_0_0']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test4_correct_int_only_i_answer_0_0_int_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_basic_int_test4_correct_int_only_s_answer_0_0_int_lparen_rparen(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_int_only_s_answer_0_0' not in correct_answer:
        explanation = "Key: 'correct_int_only_s_answer_0_0' not found.\n"
        test_structure_basic_int_test4_correct_int_only_s_answer_0_0_int_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_int_only_s_answer_0_0']
    student_answer = run_compute('part1', 's')
    if 'correct_int_only_s_answer_0_0' not in student_answer:
        explanation = "Key: 'correct_int_only_s_answer_0_0' not found.\n"
        test_structure_basic_int_test4_correct_int_only_s_answer_0_0_int_lparen_rparen.explanation = explanation
        assert False
    else:
        student_answer = student_answer['correct_int_only_s_answer_0_0']
    print(f'is_fixture=True, is_student_file=True, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test4_correct_int_only_s_answer_0_0_int_lparen_rparen.explanation = explanation
    assert is_success
