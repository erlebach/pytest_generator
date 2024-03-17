# Answer 1A
def test_answers_compute_1A_colon_datasets_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack(run_compute):
    student_answer, correct_answer = run_compute('part1')
    student_answer = student_answer['1A: datasets']
    correct_answer = correct_answer['1A: datasets']
    tol = 0.01
    import numpy as np
    dct = {'instructor_answer': correct_answer, 'student_answer': student_answer, 'rel_tol': tol}
    error_msg = 'Correct answer: {instructor_answer}'.format(**dct)
    explanation = '{error_msg}\n'.format(error_msg=error_msg)
    test_answers_compute_1A_colon_datasets_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack.explanation = explanation
    msg = "assert_utilities.check_answer_dict_string_Tuple_NDArray(student_answer, instructor_answer, rel_tol)" ## .format(student_answer=student_answer, instructor_answer=correct_answer, rel_tol=tol)
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol}
    assert eval(msg, {'__builtins__':{}}, local_namespace)
----------------------------------------------------------------------
======================================================================
# Desired structure 1A
@hide_errors('')
def test_structure_compute_1A_colon_datasets_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack_lparen_rparen(run_compute):
    student_answer, correct_answer = run_compute('part1')
    student_answer = student_answer['1A: datasets']
    correct_answer = correct_answer['1A: datasets']
    tol = 0.01
    import numpy as np
    dct = {'instructor_answer': correct_answer, 'student_answer': student_answer, 'rel_tol': tol}
    error_msg = 'Correct answer: {instructor_answer}'.format(**dct)
    explanation = '{error_msg}\n'.format(error_msg=error_msg)
    test_answers_compute_1A_colon_datasets_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack.explanation = explanation
    #msg = "assert_utilities.check_answer_dict_string_Tuple_NDArray(student_answer, instructor_answer, rel_tol)" ## .format(student_answer=student_answer, instructor_answer=correct_answer, rel_tol=tol)
    # ONLY CHANGE
    msg = "assert_utilities.check_structure_dict_string_Tuple_NDArray(student_answer, instructor_answer, rel_tol)" ## .format(student_answer=student_answer, instructor_answer=correct_answer, rel_tol=tol)
    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol}
    assert eval(msg, {'__builtins__':{}}, local_namespace)

======================================================================
----------------------------------------------------------------------
# Current structure 1A
@hide_errors('')
def test_structure_compute_1A_colon_datasets_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack_lparen_rparen(run_compute):
    student_answer, correct_answer = run_compute('part1')
    student_answer = student_answer['1A: datasets']
    correct_answer = correct_answer['1A: datasets']
    answer = student_answer
    error_msg = type_handlers['types']['dict[string,Tuple[NDArray]]']['struct_msg'].format(answer_var='answer')
    explanation = '{error_msg}\n'.format(error_msg=error_msg)
    test_structure_compute_1A_colon_datasets_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack_lparen_rparen.explanation = explanation
    assert eval(type_handlers['types']['dict[string,Tuple[NDArray]]']['assert'].format(answer_var='answer')), type_handlers['types']['dict[string,Tuple[NDArray]]']['struct_msg'].format(answer_var='answer')
    dct = {'instructor_answer': correct_answer, 'student_answer': student_answer}
    error_msg = 'Correct answer: {instructor_answer}'.format(**dct)
    explanation = '{error_msg}\n'.format(error_msg=error_msg)
    test_structure_compute_1A_colon_datasets_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack_lparen_rparen.explanation = explanation
    msg = "assert_utilities.check_answer_dict_string_Tuple_NDArray(student_answer, instructor_answer, rel_tol)"    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer}
    assert eval(msg, {'__builtins__':{}}, local_namespace)
