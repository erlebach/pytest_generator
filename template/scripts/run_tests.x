#!/bin/bash

# either put full paths or path releative to current folder
export PYTHONPATH=/autograder/MAKE-STUDENT-OUTPUT/student_code:./pytest_utils:./student_code_with_answers:.
#export PYTHONPATH=./CODE:./pytest_utils:.

due_date="2024-03-27"

# Current date in YYYY-MM-DD format
current_date=$(date '+%Y-%m-%d')
echo "current_date" : $current_date
echo "due_date" : $due_date

#----------------------------------------------------------------------
if [[ "$current_date" < "$due_date" ]] || [[ "$current_date" == "$due_date" ]]; then
    echo "Current date is earlier than the due date."
    # Only test structure if data < due_date

    # #### Structure tests ####
    # call prepare_dataset, which calls mnist

#pytest --import-mode='prepend' \
#tests/test_structure_preprocessed_part1_expand.py 
    
# Add -s to see my print statements
#tests/test_structure_preprocessed_part3_expand.py::test_structure_compute_3A_colon_toy_data_dict_lbrack_string_comma_NDArray_rbrack_lparen_rparen

pytest --import-mode='prepend' \
tests/test_structure_preprocessed_part3_expand.py # \
tests/test_structure_preprocessed_part1_expand.py \
tests/test_structure_preprocessed_part2_expand.py \
tests/test_structure_preprocessed_part4_expand.py

#pytest --import-mode='prepend' \
#tests/test_answers_preprocessed_part1_expand.py \
#tests/test_answers_preprocessed_part2_expand.py \
#tests/test_answers_preprocessed_part3_expand.py \
#tests/test_answers_preprocessed_part4_expand.py
    
#----------------------------------------------------------------------
else 
    echo "Current date is later than the due date."
    # Only test results if data < due_date
    pytest -s \
             tests/test_answers_preprocessed_part1_expand.py \
             tests/test_answers_preprocessed_part2_expand.py \
             tests/test_answers_preprocessed_part3_expand.py \
             tests/test_answers_preprocessed_part4_expand.py
fi

