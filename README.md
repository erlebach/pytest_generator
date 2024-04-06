# pytest_generator
Automatic generation of pytest unit test for student grading
----------------------------------------------------------------------
2024-04-01: new in branch simplify_generator/
2024-04-02: branch simplify_generator no longer active. DELETE IT. 
2024-04-02: Deal with local_namespace for eval in eval_float. 
In yaml file, add required namespaces, first at the top of the file, or perhaps 
in the generator (keeps the yaml file cleaner). 
----------------------------------------------------------------------
2024-04-02_16:10
## TODO
### Add libraries
I should not download files that change from homework to homework. 
Instead, create a default setup in template_folder/ with instructor folder, 
    student folder, etc., for the user to copy to the current folder via
    cp -Rf template_folder/* .
files+folders inside template_folder/: 
    instructor_code_with_answers/
       part1.py
       all_questions.py
       i_utils.py
    student_github_template/
    useful_scripts/
       update.x
    grade.sh
    # cp pyproject.toml to parent folder to template_folder/
    pyproject.toml  # template for most homeworks
    yaml_expand.py

Assignment3SolPDF.docx
answer_part1.py
generator_utils.py
results.json
test_utils.py
DEBUG.md
assert_utilities.py
homework_template.yaml
run_debug.x
testing_utilities.py
HOW_TO_DEBUG.md
bak_assignment3
run_tests.x
tests
INSTALL.md
example_yaml_files
load_yaml.py
scripts
type_handlers.yaml
LICENSE
generate_all_tests.x
part1.yaml
student_code_with_answers
types_list.py
NOTES.md
generate_answers.x
part2.yaml
Pipfile
generate_homework_tests.x
preprocessed_part1.yaml
student_github_template
yaml_expand.py
README.md
generate_structure.x
preprocessed_part1_expand.yaml
template_code_generator.py
__pycache__
generator_config.yaml
pytest_utils
test_generator.py
