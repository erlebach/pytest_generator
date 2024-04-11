#!/bin/bash

# either put full paths or path relative to current folder
export PYTHONPATH=/autograder/MAKE-STUDENT-OUTPUT/student_code:./pytest_utils:./student_code_with_answers:./instructor_code_with_answers:.:./tests

pytest -s \
tests/test_answers_preprocessed_part1_expand.py \
tests/test_answers_preprocessed_part2_expand.py \
tests/test_answers_preprocessed_part3_expand.py \
tests/test_answers_preprocessed_part4_expand.py

#pytest -s     tests/test_answers_preprocessed_part1_expand.py::test_answers_compute_1A_colon_datasets_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack
