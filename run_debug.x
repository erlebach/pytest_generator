#!/bin/bash

# either put full paths or path relative to current folder
export PYTHONPATH=/autograder/MAKE-STUDENT-OUTPUT/student_code:./pytest_utils:./student_code_with_answers:./instructor_code_with_answers:.:./tests

# --import-mode='append' or 'prepend' has to do with the root folder
# 'append': Adds the root folder of the project to the end of the pythonpath
# 'prepend' and default adds the root folder of the project to the beginning and end of the pythonpath

# pytest -s --import-mode='append' tests/test_structure_preprocessed_spectral_expand.py
#pytest -s --import-mode='append' tests/test_structure_preprocessed_denclue_expand.py

pytest -s --import-mode='append' tests/test_structure_preprocessed_jarvis_patrick_expand.py::test_structure_jarvis_patrick_clustering_jarvis_patrick_function_function

# OK
#pytest -s --import-mode='append' tests/test_structure_preprocessed_em_expand.py

# pytest -s --import-mode='append' tests/test_answers_preprocessed_em_expand.py::test_answers_gaussian_mixture_plot_original_cluster_scatterplot2d

