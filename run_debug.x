#!/bin/bash

# either put full paths or path relative to current folder
export PYTHONPATH=/autograder/MAKE-STUDENT-OUTPUT/student_code:./pytest_utils:./student_code_with_answers:./instructor_code_with_answers:.:./tests

# --import-mode='append' or 'prepend' has to do with the root folder
# 'append': Adds the root folder of the project to the end of the pythonpath
# 'prepend' and default adds the root folder of the project to the beginning and end of the pythonpath

#pytest -s --import-mode='append'  tests/test_structure_preprocessed_hw2_expand.py
pytest -s --import-mode='append'  tests/test_answers_preprocessed_hw2_expand.py::test_answers_question3_f_attr_for_splitting_str
