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
def test_structure_basic_int_test_correct_int_0_0_int_lparen_rparen(run_compute):
    correct_answer = eval('u.decode_data("Mw==")')
    student_answer = eval('u.decode_data("Mw==")')
    print(f'is_fixture=True, is_student_file=False, is_instructor_file=False')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test_correct_int_0_0_int_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_basic_int_test_correct_int_0_1_int_lparen_rparen(run_compute):
    correct_answer = eval('u.decode_data("Mw==")')
    student_answer = eval('u.decode_data("IjMi")')
    print(f'is_fixture=True, is_student_file=False, is_instructor_file=False')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test_correct_int_0_1_int_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_complex_dict_string_Tuple_NDArray_test_correct_complex_type_0_0_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack_lparen_rparen(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_complex_type_0_0' not in correct_answer:
        explanation = "Key: 'correct_complex_type_0_0' not found.\n"
        test_structure_complex_dict_string_Tuple_NDArray_test_correct_complex_type_0_0_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_complex_type_0_0']
    student_answer = eval('u.decode_data("eyJzdHJnMSI6ICIobnAuemVyb3MoMykiLCAibnAub25lcygyKSkiOiBudWxsLCAic3RyZzIiOiAiKG5wLnplcm9zKDIpKSJ9")')
    print(f'is_fixture=True, is_student_file=False, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    import numpy as np
    msg = "assert_utilities.check_structure_dict_string_Tuple_NDArray(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_complex_dict_string_Tuple_NDArray_test_correct_complex_type_0_0_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_complex_dict_string_Tuple_NDArray_test_correct_complex_type_0_1_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack_lparen_rparen(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_complex_type_0_1' not in correct_answer:
        explanation = "Key: 'correct_complex_type_0_1' not found.\n"
        test_structure_complex_dict_string_Tuple_NDArray_test_correct_complex_type_0_1_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_complex_type_0_1']
    student_answer = eval('u.decode_data("eyJzdHJnMSI6IFsibnAuemVyb3MoMykiLCAibnAub25lcygyKSJdfQ==")')
    print(f'is_fixture=True, is_student_file=False, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    import numpy as np
    msg = "assert_utilities.check_structure_dict_string_Tuple_NDArray(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_complex_dict_string_Tuple_NDArray_test_correct_complex_type_0_1_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_list_NDArray_test_correct_list_NDArray_0_0_list_lbrack_NDArray_rbrack_lparen_rparen(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_list_NDArray_0_0' not in correct_answer:
        explanation = "Key: 'correct_list_NDArray_0_0' not found.\n"
        test_structure_list_NDArray_test_correct_list_NDArray_0_0_list_lbrack_NDArray_rbrack_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_list_NDArray_0_0']
    student_answer = eval('u.decode_data("WyJucC56ZXJvcygzKSIsICJucC5vbmVzKDMpIl0=")')
    print(f'is_fixture=True, is_student_file=False, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    import numpy as np
    msg = "assert_utilities.check_structure_list_NDArray(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_list_NDArray_test_correct_list_NDArray_0_0_list_lbrack_NDArray_rbrack_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_list_NDArray_test_correct_list_NDArray_0_1_list_lbrack_NDArray_rbrack_lparen_rparen(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_list_NDArray_0_1' not in correct_answer:
        explanation = "Key: 'correct_list_NDArray_0_1' not found.\n"
        test_structure_list_NDArray_test_correct_list_NDArray_0_1_list_lbrack_NDArray_rbrack_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_list_NDArray_0_1']
    student_answer = eval('u.decode_data("WyJteSBzdHJpbmciLCAibnAub25lcygzKSJd")')
    print(f'is_fixture=True, is_student_file=False, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    import numpy as np
    msg = "assert_utilities.check_structure_list_NDArray(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_list_NDArray_test_correct_list_NDArray_0_1_list_lbrack_NDArray_rbrack_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_string_test_correct_string_0_0_string_lparen_rparen(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_string_0_0' not in correct_answer:
        explanation = "Key: 'correct_string_0_0' not found.\n"
        test_structure_string_test_correct_string_0_0_string_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_string_0_0']
    student_answer = eval('u.decode_data("Im15X3N0cmluZyI=")')
    print(f'is_fixture=True, is_student_file=False, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_string(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_string_test_correct_string_0_0_string_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_string_test_correct_string_0_1_string_lparen_rparen(run_compute):
    correct_answer = run_compute('part1', 'i')
    if 'correct_string_0_1' not in correct_answer:
        explanation = "Key: 'correct_string_0_1' not found.\n"
        test_structure_string_test_correct_string_0_1_string_lparen_rparen.explanation = explanation
        assert False
    else:
        correct_answer = correct_answer['correct_string_0_1']
    student_answer = eval('u.decode_data("NDU=")')
    print(f'is_fixture=True, is_student_file=False, is_instructor_file=True')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_string(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_string_test_correct_string_0_1_string_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_string_test_incorrect_string_0_0_string_lparen_rparen(run_compute):
    correct_answer = eval('u.decode_data("ImdlX3N0cmluZyI=")')
    student_answer = eval('u.decode_data("ImdmX3N0cmluZyI=")')
    print(f'is_fixture=True, is_student_file=False, is_instructor_file=False')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_string(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_string_test_incorrect_string_0_0_string_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_string_test_multiple_answers_0_0_string_lparen_rparen(run_compute):
    correct_answer = eval('u.decode_data("ImdlX3N0cmluZyI=")')
    student_answer = eval('u.decode_data("ImdmX3N0cmluZyI=")')
    print(f'is_fixture=True, is_student_file=False, is_instructor_file=False')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_string(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_string_test_multiple_answers_0_0_string_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_string_test_multiple_answers_1_0_string_lparen_rparen(run_compute):
    correct_answer = eval('u.decode_data("Mw==")')
    student_answer = eval('u.decode_data("ImdmX3N0cmluZyI=")')
    print(f'is_fixture=True, is_student_file=False, is_instructor_file=False')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_string(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_string_test_multiple_answers_1_0_string_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_string_test_multiple_answers_0_1_string_lparen_rparen(run_compute):
    correct_answer = eval('u.decode_data("ImdlX3N0cmluZyI=")')
    student_answer = eval('u.decode_data("Mw==")')
    print(f'is_fixture=True, is_student_file=False, is_instructor_file=False')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_string(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_string_test_multiple_answers_0_1_string_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_string_test_multiple_answers_1_1_string_lparen_rparen(run_compute):
    correct_answer = eval('u.decode_data("Mw==")')
    student_answer = eval('u.decode_data("Mw==")')
    print(f'is_fixture=True, is_student_file=False, is_instructor_file=False')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_string(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_string_test_multiple_answers_1_1_string_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_string_test_multiple_answers_0_2_string_lparen_rparen(run_compute):
    correct_answer = eval('u.decode_data("ImdlX3N0cmluZyI=")')
    student_answer = eval('u.decode_data("NDUuNw==")')
    print(f'is_fixture=True, is_student_file=False, is_instructor_file=False')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_string(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_string_test_multiple_answers_0_2_string_lparen_rparen.explanation = explanation
    assert is_success

@hide_errors('')
def test_structure_string_test_multiple_answers_1_2_string_lparen_rparen(run_compute):
    correct_answer = eval('u.decode_data("Mw==")')
    student_answer = eval('u.decode_data("NDUuNw==")')
    print(f'is_fixture=True, is_student_file=False, is_instructor_file=False')
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_string(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_string_test_multiple_answers_1_2_string_lparen_rparen.explanation = explanation
    assert is_success
