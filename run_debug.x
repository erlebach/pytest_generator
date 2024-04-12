#!/bin/bash

# either put full paths or path relative to current folder
export PYTHONPATH=/autograder/MAKE-STUDENT-OUTPUT/student_code:./pytest_utils:./student_code_with_answers:./instructor_code_with_answers:.:./tests

# --import-mode='append' or 'prepend' has to do with the root folder
# 'append': Adds the root folder of the project to the end of the pythonpath
# 'prepend' and default adds the root folder of the project to the beginning and end of the pythonpath

#pytest -s --import-mode='append' \
tests/test_answers_preprocessed_part1_expand.py \
tests/test_answers_preprocessed_part2_expand.py \
tests/test_answers_preprocessed_part3_expand.py \
tests/test_answers_preprocessed_part4_expand.py

#pytest --import-mode='append' -s tests/test_answers_preprocessed_part1_expand.py
#pytest --import-mode='append' -s tests/test_answers_preprocessed_part2_expand.py
#pytest --import-mode='append' -s tests/test_answers_preprocessed_part3_expand.py
pytest --import-mode='append' -s tests/test_answers_preprocessed_part4_expand.py

