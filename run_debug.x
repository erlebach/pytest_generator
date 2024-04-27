#!/bin/bash

# Runs on Gradescope server
# CODE: student code

# student_code_with_answers is on my local system
export PYTHONPATH=/autograder/MAKE-STUDENT-OUTPUT/student_code:./pytest_utils:./student_code_with_answers:./instructor_code_with_answers:.:./tests

#pytest -s tests/test_structure_preprocessed_hw4_expand.py
#pytest -s tests/test_answers_preprocessed_hw4_expand.py::test_answers_question1_a_string

#pytest -s tests/test_answers_preprocessed_hw4_expand.py::test_answers_question7_a_bool
#pytest -s tests/test_answers_preprocessed_hw4_expand.py

# --import-mode='append' is necessary. Othewise wrong file is included
pytest -s --import-mode='append'  tests/test_structure_preprocessed_hw4_expand.py::test_structure_question8_c_PX_1_1_plus_float

##pytest -s tests/test_structure_preprocessed_hw4_expand.py::test_structure_question8_a_PX_1_1_plus_float
