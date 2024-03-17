import pytest
import pytest
import assert_utilities
from part1 import *
import yaml
from my_fixtures import *
from pytest_utils.decorators import max_score, visibility, partial_score, hide_errors

with open('type_handlers.yaml', 'r') as f:
    type_handlers = yaml.safe_load(f)

@hide_errors('')
def test_structure_basic_int_test_correct_int_0_int_lparen_rparen(run_compute):
    student_answer = run_compute('part1', 's')
    student_answer = student_answer['correct_int_0']
    correct_answer = run_compute('part1', 'i')
    correct_answer = correct_answer['correct_int_0']
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test_correct_int_0_int_lparen_rparen.explanation = explanation

@hide_errors('')
def test_structure_basic_int_test_correct_int_1_int_lparen_rparen(run_compute):
    student_answer = run_compute('part1', 's')
    student_answer = student_answer['correct_int_1']
    correct_answer = run_compute('part1', 'i')
    correct_answer = correct_answer['correct_int_1']
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_basic_int_test_correct_int_1_int_lparen_rparen.explanation = explanation

@hide_errors('')
def test_structure_complex_dict_string_Tuple_NDArray_test_correct_complex_type_0_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack_lparen_rparen(run_compute):
    student_answer = run_compute('part1', 's')
    student_answer = student_answer['correct_complex_type_0']
    correct_answer = run_compute('part1', 'i')
    correct_answer = correct_answer['correct_complex_type_0']
    answer = student_answer
    tol = 0.001
    keys = None
    import numpy as np
    msg = "assert_utilities.check_structure_dict_string_Tuple_NDArray(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_complex_dict_string_Tuple_NDArray_test_correct_complex_type_0_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack_lparen_rparen.explanation = explanation

@hide_errors('')
def test_structure_complex_dict_string_Tuple_NDArray_test_correct_complex_type_1_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack_lparen_rparen(run_compute):
    student_answer = run_compute('part1', 's')
    student_answer = student_answer['correct_complex_type_1']
    correct_answer = run_compute('part1', 'i')
    correct_answer = correct_answer['correct_complex_type_1']
    answer = student_answer
    tol = 0.001
    keys = None
    import numpy as np
    msg = "assert_utilities.check_structure_dict_string_Tuple_NDArray(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_complex_dict_string_Tuple_NDArray_test_correct_complex_type_1_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack_lparen_rparen.explanation = explanation

@hide_errors('')
def test_structure_list_NDArray_test_correct_list_NDArray_0_list_lbrack_NDArray_rbrack_lparen_rparen(run_compute):
    student_answer = run_compute('part1', 's')
    student_answer = student_answer['correct_list_NDArray_0']
    correct_answer = run_compute('part1', 'i')
    correct_answer = correct_answer['correct_list_NDArray_0']
    answer = student_answer
    tol = 0.001
    keys = None
    import numpy as np
    msg = "assert_utilities.check_structure_list_NDArray(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_list_NDArray_test_correct_list_NDArray_0_list_lbrack_NDArray_rbrack_lparen_rparen.explanation = explanation

@hide_errors('')
def test_structure_list_NDArray_test_correct_list_NDArray_1_list_lbrack_NDArray_rbrack_lparen_rparen(run_compute):
    student_answer = run_compute('part1', 's')
    student_answer = student_answer['correct_list_NDArray_1']
    correct_answer = run_compute('part1', 'i')
    correct_answer = correct_answer['correct_list_NDArray_1']
    answer = student_answer
    tol = 0.001
    keys = None
    import numpy as np
    msg = "assert_utilities.check_structure_list_NDArray(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_list_NDArray_test_correct_list_NDArray_1_list_lbrack_NDArray_rbrack_lparen_rparen.explanation = explanation

@hide_errors('')
def test_structure_string_test_correct_string_0_string_lparen_rparen(run_compute):
    student_answer = run_compute('part1', 's')
    student_answer = student_answer['correct_string_0']
    correct_answer = run_compute('part1', 'i')
    correct_answer = correct_answer['correct_string_0']
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_string(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_string_test_correct_string_0_string_lparen_rparen.explanation = explanation

@hide_errors('')
def test_structure_string_test_correct_string_1_string_lparen_rparen(run_compute):
    student_answer = run_compute('part1', 's')
    student_answer = student_answer['correct_string_1']
    correct_answer = run_compute('part1', 'i')
    correct_answer = correct_answer['correct_string_1']
    answer = student_answer
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_structure_string(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_structure_string_test_correct_string_1_string_lparen_rparen.explanation = explanation
