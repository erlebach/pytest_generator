#!/bin/bash

# Disable answer checking. 
# Change manually after due date

# Runs on Gradescope server
# CODE: student code

# student_code_with_answers is on my local system
export PYTHONPATH=student_code_with_answers:/autograder/MAKE-STUDENT-OUTPUT/CODE:.:pytest_utils:instructor_code_with_answers:tests

due_date="2025-03-04"

# Current date in YYYY-MM-DD format
current_date=$(date '+%Y-%m-%d')
echo "current_date" : $current_date
echo "due_date" : $due_date

if [[ "$current_date" < "$due_date" ]] || [[ "$current_date" == "$due_date" ]]; then
    echo "Current date is earlier than the due date."

    pytest -s --import-mode='append' tests/test_structure_preprocessed_hw3_expand.py
    # pytest -s --import-mode='append' tests/test_structure_preprocessed_hw3_expand.py::test_structure_question5_a_set_lbrack_str_rbrack
    # pytest -s --import-mode='append' tests/test_answers_preprocessed_hw3_expand.py
    #pytest -s --import-mode='append' tests/test_structure_preprocessed_all_questions_hw3_expand.py::test_structure_question5_a_set_lbrack_string_rbrack

else 
    echo "Current date is later than the due date."

    # # pytest -s --import-mode='append' tests/test_answers_preprocessed_hw1_expand.py
    #pytest -s --import-mode='append' tests/test_structure_preprocessed_hw1_expand.py::test_structure_question5_q5_4_dict_lbrack_str_comma_list_lbrack_int_rbrack_rbrack
    #pytest -s --import-mode='append' tests/test_answers_preprocessed_hw1_expand.py::test_answers_question5_q5_4_dict_lbrack_str_comma_list_lbrack_int_rbrack_rbrack

    #pytest -s --import-mode='append' tests/test_structure_preprocessed_hw1_expand.py
    pytest -s --import-mode='append' tests/test_answers_preprocessed_hw1_expand.py

fi

