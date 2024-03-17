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
def test_structure_basic_int_test0_correct_int_0_0_int_lparen_rparen():
    correct_answer = eval('u.decode_data("Mw==")')
    student_answer = eval('u.decode_data("Mw==")')
    print(f'is_fixture=False, is_student_file=False, is_instructor_file=False')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test0_correct_int_0_0_int_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_basic_int_test0_correct_int_0_0_int_lparen_rparen():
    correct_answer = eval('u.decode_data("Mw==")')
    student_answer = eval('u.decode_data("Mw==")')
    print(f'is_fixture=False, is_student_file=False, is_instructor_file=False')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test0_correct_int_0_0_int_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_basic_int_test0_correct_int_0_1_int_lparen_rparen():
    correct_answer = eval('u.decode_data("Mw==")')
    student_answer = eval('u.decode_data("IjMi")')
    print(f'is_fixture=False, is_student_file=False, is_instructor_file=False')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test0_correct_int_0_1_int_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_basic_int_test1_correct_int_0_0_int_lparen_rparen():
    correct_answer = eval('u.decode_data("Mw==")')
    student_answer = eval('u.decode_data("Mw==")')
    print(f'is_fixture=False, is_student_file=False, is_instructor_file=False')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test1_correct_int_0_0_int_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_basic_int_test1_correct_int_1_0_int_lparen_rparen():
    correct_answer = eval('u.decode_data("IjMi")')
    student_answer = eval('u.decode_data("Mw==")')
    print(f'is_fixture=False, is_student_file=False, is_instructor_file=False')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test1_correct_int_1_0_int_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_basic_int_test2_correct_int_0_0_int_lparen_rparen():
    correct_answer = eval('u.decode_data("Mw==")')
    student_answer = eval('u.decode_data("Mw==")')
    print(f'is_fixture=False, is_student_file=False, is_instructor_file=False')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test2_correct_int_0_0_int_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_basic_int_test2_correct_int_1_0_int_lparen_rparen():
    correct_answer = eval('u.decode_data("IjMi")')
    student_answer = eval('u.decode_data("Mw==")')
    print(f'is_fixture=False, is_student_file=False, is_instructor_file=False')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test2_correct_int_1_0_int_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_basic_int_test2_correct_int_0_1_int_lparen_rparen():
    correct_answer = eval('u.decode_data("Mw==")')
    student_answer = eval('u.decode_data("IjMi")')
    print(f'is_fixture=False, is_student_file=False, is_instructor_file=False')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test2_correct_int_0_1_int_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_basic_int_test2_correct_int_1_1_int_lparen_rparen():
    correct_answer = eval('u.decode_data("IjMi")')
    student_answer = eval('u.decode_data("IjMi")')
    print(f'is_fixture=False, is_student_file=False, is_instructor_file=False')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test2_correct_int_1_1_int_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_basic_int_test3_correct_int_0_0_int_lparen_rparen():
    correct_answer = eval('u.decode_data("Mw==")')
    print('not is_fixture and is_student_file: not implemented')
    print(f'is_fixture=False, is_student_file=True, is_instructor_file=False')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test3_correct_int_0_0_int_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_basic_int_test3_correct_int_1_0_int_lparen_rparen():
    correct_answer = eval('u.decode_data("IjMi")')
    print('not is_fixture and is_student_file: not implemented')
    print(f'is_fixture=False, is_student_file=True, is_instructor_file=False')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test3_correct_int_1_0_int_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_basic_int_test4_correct_int_0_0_int_lparen_rparen():
    correct_answer = eval('u.decode_data("Mw==")')
    print('not is_fixture and is_student_file: not implemented')
    print(f'is_fixture=False, is_student_file=True, is_instructor_file=False')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test4_correct_int_0_0_int_lparen_rparen.explanation = explanation
    assert is_success
