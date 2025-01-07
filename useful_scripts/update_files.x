#!/bin/bash

# Update files from pytest_generator

\cp pytest_generator/generator_config.yaml .
\cp pytest_generator/type_handlers.yaml .
\cp pytest_generator/*.yaml .
\cp pytest_generator/assert_utilities.py .
\cp pytest_generator/function_dictionaries.py .
\cp pytest_generator/testing_utilities.py .
\cp pytest_generator/test_utils.py .
\cp pytest_generator/yaml_expand.py .
\cp pytest_generator/tests/test_*.py tests/
\cp pytest_generator/tests/my_fixtures.py tests/
\cp -rf pytest_generator/pytest_utils .
\cp pytest_generator/run_tests.x .
\cp pytest_generator/student_code_with_answers/*.py student_code_with_answers/
\cp pytest_generator/instructor_code_with_answers/*.py instructor_code_with_answers/
\cp pytest_generator/student_github_template/*.py student_github_template/
