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

    # # pytest -s --import-mode='append' tests/test_answers_preprocessed_hw1_expand.py
    #pytest -s --import-mode='append' tests/test_structure_preprocessed_hw1_expand.py::test_structure_question5_q5_4_dict_lbrack_str_comma_list_lbrack_int_rbrack_rbrack
    #pytest -s --import-mode='append' tests/test_answers_preprocessed_hw1_expand.py::test_answers_question5_q5_4_dict_lbrack_str_comma_list_lbrack_int_rbrack_rbrack

    #pytest -s --import-mode='append' tests/test_structure_preprocessed_hw1_expand.py
    #pytest -s --import-mode='append' tests/test_answers_preprocessed_hw1_expand.py
     pytest -s --import-mode='append' tests/test_answers_preprocessed_hw1_expand.py::test_answers_question4_q4_1_list_lbrack_str_rbrack

fi


# Fix all partial scoring
#OK FAILED tests/test_answers_preprocessed_hw1_expand.py::test_answers_question2_q2_4_set_lbrack_str_rbrack - assert False
#OK FAILED tests/test_answers_preprocessed_hw1_expand.py::test_answers_question3_q3_1_set_lbrack_str_rbrack - assert False
#OKFAILED tests/test_answers_preprocessed_hw1_expand.py::test_answers_question3_q3_2_set_lbrack_str_rbrack - assert False
#OK FAILED tests/test_answers_preprocessed_hw1_expand.py::test_answers_question3_q3_3_set_lbrack_str_rbrack - assert False
# OK FAILED tests/test_answers_preprocessed_hw1_expand.py::test_answers_question3_q3_6_set_lbrack_str_rbrack - assert False
# OK FAILED tests/test_answers_preprocessed_hw1_expand.py::test_answers_question3_q3_7_set_lbrack_str_rbrack - assert False
#FAILED tests/test_answers_preprocessed_hw1_expand.py::test_answers_question4_q4_1_list_lbrack_str_rbrack - assert False
#FAILED tests/test_answers_preprocessed_hw1_expand.py::test_answers_question5_q5_1_dict_lbrack_str_comma_list_lbrack_int_rbrack_rbrack - ValueError: zip() argument 2 is shorter than argument 1
#FAILED tests/test_answers_preprocessed_hw1_expand.py::test_answers_question5_q5_3_dict_lbrack_str_comma_list_lbrack_int_rbrack_rbrack - ValueError: zip() argument 2 is longer than argument 1
#FAILED tests/test_answers_preprocessed_hw1_expand.py::test_answers_question5_q5_4_dict_lbrack_str_comma_list_lbrack_int_rbrack_rbrack - assert False
#FAILED tests/test_answers_preprocessed_hw1_expand.py::test_answers_question6_q6_1_dict_lbrack_str_comma_list_lbrack_str_rbrack_rbrack - assert False
#FAILED tests/test_answers_preprocessed_hw1_expand.py::test_answers_question6_q6_2_dict_lbrack_str_comma_list_lbrack_str_rbrack_rbrack - assert False
#FAILED tests/test_answers_preprocessed_hw1_expand.py::test_answers_question6_q6_3_dict_lbrack_str_comma_list_lbrack_str_rbrack_rbrack - assert False
#FAILED tests/test_answers_preprocessed_hw1_expand.py::test_answers_question7_q7_2_str - assert False
#FAILED tests/test_answers_preprocessed_hw1_expand.py::test_answers_question7_q7_3_list_lbrack_tuple_lbrack_float_rbrack_rbrack - assert False
#FAILED tests/test_answers_preprocessed_hw1_expand.py::test_answers_question9_q9_1_str - assert False
#FAILED tests/test_answers_preprocessed_hw1_expand.py::test_answers_question9_q9_3_str - assert False
#FAILED tests/test_answers_preprocessed_hw1_expand.py::test_answers_question10_q10_2_bool - assert False
#FAILED tests/test_answers_preprocessed_hw1_expand.py::test_answers_question10_q10_7_bool - assert False]
