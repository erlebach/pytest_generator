#!/bin/bash

# Runs on Gradescope server
# CODE: student code

# student_code_with_answers is on my local system
export PYTHONPATH=student_code_with_answers:/autograder/MAKE-STUDENT-OUTPUT/CODE:.:pytest_utils:instructor_code_with_answers
#export PYTHONPATH=.:pytest_utils

pytest --import-mode='prepend' \
    tests/test_answers_preprocessed_part1_expand.py \
    tests/test_answers_preprocessed_part2_expand.py \
    tests/test_answers_preprocessed_part3_expand.py \
    tests/test_answers_preprocessed_part4_expand.py

# pytest -s tests/test_answers_preprocessed_test1_expanded.py
#pytest -s tests/test_structure_preprocessed_test2_expand.py::test_structure_basic_int_test0_correct_int0_0_0_int_lparen_rparen

# All tests fail!
#pytest -s tests/test_structure_preprocessed_test2_expand.py
#pytest -s tests/test_structure_preprocessed_test3_expand.py
#pytest -s tests/test_structure_preprocessed_test4_expand.py

# read s_ and i_ from yaml files
#pytest -s tests/test_structure_preprocessed_test5_expand.py
pytest -s tests/test_answers_preprocessed_test5_expand.py
#pytest -s tests/test_answers_preprocessed_test4_expand.py::test_answers_basic_int_test0_correct_int_0_1_int

#pytest -s tests/test_structure_preprocessed_test5_expand.py::test_structure_basic_int_test3_correct_int_0_0_int_lparen_rparen
