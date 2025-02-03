#!/bin/bash

# Disable answer checking. 
# Change manually after due date

# Runs on Gradescope server
# CODE: student code

# student_code_with_answers is on my local system
export PYTHONPATH=student_code_with_answers:/autograder/MAKE-STUDENT-OUTPUT/CODE:.:pytest_utils:instructor_code_with_answers:tests

due_date="2025-01-24"

# Current date in YYYY-MM-DD format
current_date=$(date '+%Y-%m-%d')
echo "current_date" : $current_date
echo "due_date" : $due_date

if [[ "$current_date" < "$due_date" ]] || [[ "$current_date" == "$due_date" ]]; then
    echo "Current date is earlier than the due date."

    pytest -s --import-mode='append' tests/test_structure_preprocessed_hw1_expand.py


else 
    echo "Current date is later than the due date."

    # pytest -s --import-mode='append' tests/test_answers_preprocessed_hw1_expand.py
    # pytest -s --import-mode='append' tests/test_answers_preprocessed_hw1_expand.py::test_answers_question5_q5_1_dict_lbrack_str_comma_list_lbrack_int_rbrack_rbrack
    pytest -s --import-mode='append' tests/test_structure_preprocessed_hw1_expand.py::test_structure_question6_q6_1_dict_lbrack_str_comma_list_lbrack_str_rbrack_rbrack
    # pytest -s --import-mode='append' tests/test_answers_preprocessed_hw1_expand.py::test_answers_question6_q6_1_dict_lbrack_str_comma_list_lbrack_str_rbrack_rbrack
fi
