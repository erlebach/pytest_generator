#!/bin/bash

# Runs on Gradescope server
# CODE: student code

# student_code_with_answers is on my local system
export PYTHONPATH=/autograder/MAKE-STUDENT-OUTPUT/student_code:./pytest_utils:./student_code_with_answers:./instructor_code_with_answers:.:./tests

#pytest -s tests/test_structure_preprocessed_hw4_expand.py
pytest -s tests/test_answers_preprocessed_hw4_expand.py::test_answers_question1_a_str

# pytest -s --import-mode='append' tests/test_structure_preprocessed_spectral_expand.py
#pytest -s --import-mode='append' tests/test_structure_preprocessed_denclue_expand.py

# pytest -s --import-mode='append' tests/test_structure_preprocessed_jarvis_patrick_expand.py::test_structure_jarvis_patrick_clustering_jarvis_patrick_function_function

#pytest -s --import-mode='append' tests/test_structure_preprocessed_em_expand.py

# pytest -s --import-mode='append' tests/test_answers_preprocessed_em_expand.py::test_answers_gaussian_mixture_plot_original_cluster_scatterplot2d

##pytest -s tests/test_structure_preprocessed_hw4_expand.py::test_structure_question8_a_PX_1_1_plus_float
