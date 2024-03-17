#!/bin/bash

# either put full paths or path releative to current folder
export PYTHONPATH=/autograder/MAKE-STUDENT-OUTPUT/student_code:./pytest_utils:./student_code_with_answers:instructor_code_with_answers:.


# All pass
#pytest -s --import-mode='prepend' tests/test_answers_part1.py  # all pass
#pytest -s --import-mode='prepend' tests/test_answers_part2.py
#pytest -s --import-mode='prepend' tests/test_answers_part3.py  # 7 pass
#pytest -s --import-mode='prepend' tests/test_answers_part4.py


#pytest -s --import-mode='prepend' tests/test_structure_part1.py   # all pass
#pytest -s --import-mode='prepend' tests/test_structure_part2.py   # all pass
#pytest -s --import-mode='prepend' tests/test_structure_part3.py   # 7 passed,(need hierarchical_toy.mat dataset)
#pytest -s --import-mode='prepend' tests/test_structure_part4.py   # 4 pass

#pytest -s --import-mode='prepend' \
#        tests/test_structure_part1.py \
#        tests/test_structure_part2.py \
#        tests/test_structure_part3.py \
#        tests/test_structure_part4.py 

pytest -s --import-mode='prepend' \
    tests/test_answers_part1.py \
    tests/test_answers_part2.py \
    tests/test_answers_part3.py \
    tests/test_answers_part4.py 

#pytest -s --import-mode='prepend' tests/test_answers_part1.py
#pytest -s --import-mode='prepend' tests/test_answers_part2.py
#pytest -s --import-mode='prepend' tests/test_answers_part3.py
#pytest -s --import-mode='prepend' tests/test_answers_part4.py

#pytest -s --import-mode='prepend' \
#tests/test_answers_part1.py::test_answers_compute_1C_colon_cluster_successes_dict_lbrack_string_comma_set_rbrack


exit 1

