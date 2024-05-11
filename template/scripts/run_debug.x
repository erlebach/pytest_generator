#!/bin/bash

# either put full paths or path releative to current folder
export PYTHONPATH=./pytest_utils:./student_code_with_answers:./instructor_code_with_answers:.:./tests


pytest -s --import-mode='prepend' tests/test_answers_preprocessed_example_expand.py


