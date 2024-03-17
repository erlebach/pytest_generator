
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
def test_answers_basic_int_test_correct_int_0_int(run_compute):
    student_answer, correct_answer = run_compute('part1')
    student_answer = student_answer['correct_int_0']
    correct_answer = correct_answer['correct_int_0']
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_answer_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_answers_basic_int_test_correct_int_0_int.explanation = explanation



@max_score(10)
@hide_errors('')
def test_answers_basic_int_test_correct_int_1_int(run_compute):
    student_answer, correct_answer = run_compute('part1')
    student_answer = student_answer['correct_int_1']
    correct_answer = correct_answer['correct_int_1']
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_answer_int(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_answers_basic_int_test_correct_int_1_int.explanation = explanation



@max_score(10)
@hide_errors('')
def test_answers_complex_dict_string_Tuple_NDArray_test_correct_complex_type_0_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack(run_compute):
    student_answer, correct_answer = run_compute('part1')
    student_answer = student_answer['correct_complex_type_0']
    correct_answer = correct_answer['correct_complex_type_0']
    tol = 0.001
    keys = None
    import numpy as np
    msg = "assert_utilities.check_answer_dict_string_Tuple_NDArray(student_answer, instructor_answer, rel_tol)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_answers_complex_dict_string_Tuple_NDArray_test_correct_complex_type_0_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack.explanation = explanation



@max_score(10)
@hide_errors('')
def test_answers_complex_dict_string_Tuple_NDArray_test_correct_complex_type_1_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack(run_compute):
    student_answer, correct_answer = run_compute('part1')
    student_answer = student_answer['correct_complex_type_1']
    correct_answer = correct_answer['correct_complex_type_1']
    tol = 0.001
    keys = None
    import numpy as np
    msg = "assert_utilities.check_answer_dict_string_Tuple_NDArray(student_answer, instructor_answer, rel_tol)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_answers_complex_dict_string_Tuple_NDArray_test_correct_complex_type_1_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack.explanation = explanation



@max_score(10)
@hide_errors('')
def test_answers_list_NDArray_test_correct_list_NDArray_0_list_lbrack_NDArray_rbrack(run_compute):
    student_answer, correct_answer = run_compute('part1')
    student_answer = student_answer['correct_list_NDArray_0']
    correct_answer = correct_answer['correct_list_NDArray_0']
    tol = 0.001
    keys = None
    import numpy as np
    msg = "assert_utilities.check_answer_list_NDArray(student_answer, instructor_answer, rel_tol)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_answers_list_NDArray_test_correct_list_NDArray_0_list_lbrack_NDArray_rbrack.explanation = explanation



@max_score(10)
@hide_errors('')
def test_answers_list_NDArray_test_correct_list_NDArray_1_list_lbrack_NDArray_rbrack(run_compute):
    student_answer, correct_answer = run_compute('part1')
    student_answer = student_answer['correct_list_NDArray_1']
    correct_answer = correct_answer['correct_list_NDArray_1']
    tol = 0.001
    keys = None
    import numpy as np
    msg = "assert_utilities.check_answer_list_NDArray(student_answer, instructor_answer, rel_tol)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_answers_list_NDArray_test_correct_list_NDArray_1_list_lbrack_NDArray_rbrack.explanation = explanation



@max_score(10)
@hide_errors('')
def test_answers_string_test_correct_string_0_string(run_compute):
    student_answer, correct_answer = run_compute('part1')
    student_answer = student_answer['correct_string_0']
    correct_answer = correct_answer['correct_string_0']
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_answer_string(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_answers_string_test_correct_string_0_string.explanation = explanation



@max_score(10)
@hide_errors('')
def test_answers_string_test_correct_string_1_string(run_compute):
    student_answer, correct_answer = run_compute('part1')
    student_answer = student_answer['correct_string_1']
    correct_answer = correct_answer['correct_string_1']
    tol = 0.001
    keys = None
    msg = "assert_utilities.check_answer_string(student_answer, instructor_answer)"
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}
    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)
    test_answers_string_test_correct_string_1_string.explanation = explanation


