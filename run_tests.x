#!/bin/bash

# Runs on Gradescope server
# CODE: student code

# student_code_with_answers is on my local system
export PYTHONPATH=/autograder/MAKE-STUDENT-OUTPUT/student_code:./pytest_utils:./student_code_with_answers:./instructor_code_with_answers:.:./tests

due_date="2025-03-04"

# Current date in YYYY-MM-DD format
current_date=$(date '+%Y-%m-%d')
echo "current_date" : $current_date
echo "due_date" : $due_date

if [[ "$current_date" < "$due_date" ]] || [[ "$current_date" == "$due_date" ]]; then
    echo "Current date is earlier than the due date."
    pytest -s --import-mode='append' tests/test_structure_preprocessed_hw4_expand.py
    # pytest -s --import-mode='append' tests/test_structure_preprocessed_hw4_expand.py::test_structure_question8_d_A_and_B_conditionally_independent_float
    # pytest -s --import-mode='append' tests/test_answers_preprocessed_hw4_expand.py

else 
    echo "Current date is later than the due date."
    pytest -s --import-mode='append' tests/test_answers_preprocessed_hw4_expand.py
fi
