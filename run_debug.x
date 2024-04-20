#!/bin/bash

# either put full paths or path relative to current folder
export PYTHONPATH=/autograder/MAKE-STUDENT-OUTPUT/student_code:./pytest_utils:./student_code_with_answers:./instructor_code_with_answers:.:./tests

# --import-mode='append' or 'prepend' has to do with the root folder
# 'append': Adds the root folder of the project to the end of the pythonpath
# 'prepend' and default adds the root folder of the project to the beginning and end of the pythonpath

#pytest -s --import-mode='append' tests/test_structure_preprocessed_spectral_expand.py
#pytest -s --import-mode='append' tests/test_structure_preprocessed_denclue_expand.py
#pytest -s --import-mode='append' tests/test_structure_preprocessed_jarvis_patrick_expand.py
#pytest -s --import-mode='append' tests/test_structure_preprocessed_em_expand.py

pytest -s --import-mode='append' tests/test_answers_preprocessed_em_expand.py::test_answers_gaussian_mixture_probability_2_amplitude_list_lbrack_float_rbrack

### FAILED tests/test_answers_preprocessed_em_expand.py:: test_answers_gaussian_mixture_probability_1_covariance_list_lbrack_ndarray_rbrack - TypeError: sequence item 0: expected str instance, list found
#### FAILED tests/test_answers_preprocessed_em_expand.py:: test_answers_gaussian_mixture_probability_2_covariance_list_lbrack_ndarray_rbrack - TypeError: sequence item 0: expected str instance, list found
#### FAILED tests/test_answers_preprocessed_em_expand.py:: test_answers_gaussian_mixture_probability_2_amplitude_list_lbrack_float_rbrack - assert False
# FAILED tests/test_answers_preprocessed_em_expand.py::test_answers_gaussian_mixture_SSE_list_lbrack_float_rbrack - assert False
