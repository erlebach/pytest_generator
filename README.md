# pytest_generator
Automatic generation of pytest unit test for student grading

----------------------------------------------------------------------
2024-04-01: new in branch simplify_generator/
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

----------------------------------------------------------------------
2024-04-07_16:24
- TODO: 
    - answer_type is null should never happen
    - in yaml_expand.py, remove spaces from types (fixes instructor errors)

----------------------------------------------------------------------
2025-01-08
- The `pytest_generator` repository should be copied to a working folder without .git/ to make sure that I can't inadvertendly write to the remote repository. But what if I make corrections to the pytest_generator folder? It is not yet clear how to organize my files. Therefore, I am not yet changing anything. 
I close pytest_generator repository when located in the grading repository. I make changes to the testing repo, and then I execute `update_files.x` to copy files from `pytest_generator/` to `pytest_generator/..`

----------------------------------------------------------------------
