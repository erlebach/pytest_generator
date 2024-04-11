#!/bin/bash

# either put full paths or path relative to current folder
export PYTHONPATH=/autograder/MAKE-STUDENT-OUTPUT/student_code:./pytest_utils:./student_code_with_answers:./instructor_code_with_answers:.:./tests

#pytest -s \
#         tests/test_answers_preprocessed_part1_expand.py \
#         tests/test_answers_preprocessed_part2_expand.py \
#         tests/test_answers_preprocessed_part3_expand.py \
#         tests/test_answers_preprocessed_part4_expand.py

#pytest -s tests/test_answers_preprocessed_part1_expand.py 
#pytest -s tests/test_answers_preprocessed_part2_expand.py   
#pytest -s tests/test_answers_preprocessed_part2_expand.py
#pytest -s tests/test_answers_preprocessed_part3_expand.py 
pytest -s tests/test_answers_preprocessed_part4_expand.py   
#pytest -s tests/test_answers_preprocessed_part4_expand.py::test_answers_compute_4B_colon_cluster_successes_list_lbrack_string_rbrack

#pytest -s tests/test_answers_preprocessed_part1_expand.py::test_answers_compute_1A_colon_datasets_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack

#pytest -s tests/test_answers_preprocessed_part4_expand.py::test_answers_compute_4A_colon_datasets_dict_lbrack_string_comma_Tuple_lbrack_NDArray_rbrack_rbrack 
#pytest -s tests/test_answers_preprocessed_part4_expand.py::test_answers_compute_4A_colon_fit_hierarchical_cluster_function 
#pytest -s tests/test_answers_preprocessed_part4_expand.py::test_answers_compute_4B_colon_cluster_successes_list_lbrack_string_rbrac
#pytest -s tests/test_answers_preprocessed_part4_expand.py::test_answers_compute_4C_colon_modified_function_function 

