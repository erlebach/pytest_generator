#!/bin/bash

# Runs on Gradescope server
export PYTHONPATH=/autograder/MAKE-STUDENT-OUTPUT/student_code:./pytest_utils:./student_code_with_answers:./instructor_code_with_answers:.:./tests

<<<<<<< HEAD
due_date="2025-05-01"
=======
>>>>>>> b6afed15545ba57dce1ba59049f0abd8666b1266
due_date="2025-03-20"

# Current date in YYYY-MM-DD format
current_date=$(date '+%Y-%m-%d')
echo "current_date" : $current_date
echo "due_date" : $due_date



if [[ "$current_date" < "$due_date" ]] || [[ "$current_date" == "$due_date" ]]; then
    echo "Current date is earlier than the due date."

# 8 tests
# pytest -s --import-mode='append' tests/test_structure_preprocessed_jarvis_patrick_expand.py 

# pytest -s --import-mode='append' tests/test_structure_preprocessed_em_expand.py 

pytest -s --import-mode='append' tests/test_structure_preprocessed_spectral_expand.py

else
    echo "Current date is later than the due date."

pytest -s --import-mode='append' \
    tests/test_answers_preprocessed_jarvis_patrick_expand.py \
    tests/test_answers_preprocessed_em_expand.py \
    tests/test_answers_preprocessed_spectral_expand.py

fi
