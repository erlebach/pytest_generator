import functools
import pytest
import importlib
import sys
import os

# ----------------------------------------------------------------------
"""
# Cache the results. Necessary since otherwise the import within the function
# forces the call to the student and instructor code to repeat.
@functools.cache
def get_module_results(module_name):
    student_module_name = f"student_code_with_answers.{module_name}"
    sc = importlib.import_module(student_module_name)
    instructor_module_name = f"instructor_code_with_answers.{module_name}"
    ic = importlib.import_module(instructor_module_name)
    student_answer = sc.compute()
    correct_answer = ic.compute()
    return student_answer, correct_answer

# Usage: use `run_compute` as a parameter to the test function
#        sa, ca = run_compute(part1)
#    to return the student and correct answers
@pytest.fixture(scope='module')
def run_compute():
    def _module(module_name):
        return get_module_results(module_name)
    return _module
"""
# ----------------------------------------------------------------------
"""
To use @functools.cache safely, the function it is applied to must be a pure function, 
whose results only depend on the input arguments. This is likely correct in a testing 
environment when using serial calculations. Each test is run independently, and
all I am doing is loading modules and running student and instructor code, which are
self-contained. 
Nonetheless, use is risky. If I have errors in the testing framework, I should disable
the cache and check whether the errors persist. . 
"""
# ----------------------------------------------------------------------
@functools.cache
def with_custom_sys_path(path, func, *args, **kwargs):
    """
    Temporarily prepend a directory to sys.path, execute a function,
    and then restore sys.path to its original state.

    :param path: The directory to temporarily add to sys.path
    :param func: The function to execute while the directory is added
    :param args: Arguments to pass to the function
    :param kwargs: Keyword arguments to pass to the function
    """
    original_sys_path = list(sys.path)
    try:
        sys.path.insert(0, path)
        return func(*args, **kwargs)
    finally:
        sys.path = original_sys_path

# ----------------------------------------------------------------------
@functools.cache
def load_and_run_module(module_name, directory):
    """
    Loads a module from a specific directory and executes its `compute` function.

    :param module_name: Name of the module to load
    :param directory: Directory from which to load the module
    :return: The result of the module's `compute` function
    """
    original_cwd = os.getcwd()
    os.chdir(directory)
    try:
        module = importlib.import_module(module_name)
        result = module.compute()
    finally:
        os.chdir(original_cwd)
    return result
    #return module.compute()

# ----------------------------------------------------------------------
# Usage within your existing setup
@functools.cache
def get_module_results(module_name, ret='both'):
    student_directory = "./student_code_with_answers"
    instructor_directory = "./instructor_code_with_answers"

    if ret == 'both':
        student_result = with_custom_sys_path(student_directory, load_and_run_module, module_name, student_directory)
        instructor_result = with_custom_sys_path(instructor_directory, load_and_run_module, module_name, instructor_directory)
        return student_result, instructor_result
    elif ret == 's':
        return with_custom_sys_path(student_directory, load_and_run_module, module_name, student_directory)
    else:  # ret == 'i'
        return with_custom_sys_path(instructor_directory, load_and_run_module, module_name, instructor_directory)

# ----------------------------------------------------------------------
"""
# Cache the results. Necessary since otherwise the import within the function
# forces the call to the student and instructor code to repeat.
@functools.cache
def get_module_results(module_name, ret='both'):
    '''
    ret: 'both', return both student and instructor answers
    ret: 's', return student answer
    ret: 'i', return instructor
    '''
    if ret == 'both':
        student_module_name = f"student_code_with_answers.{module_name}"
        sc = importlib.import_module(student_module_name)
        instructor_module_name = f"instructor_code_with_answers.{module_name}"
        ic = importlib.import_module(instructor_module_name)
        student_answer = sc.compute()
        correct_answer = ic.compute()
        return student_answer, correct_answer
    elif ret == 's': 
        student_module_name = f"student_code_with_answers.{module_name}"
        sc = importlib.import_module(student_module_name)
        student_answer = sc.compute()
        return student_answer
    else:   # ret == 'i'
        instructor_module_name = f"instructor_code_with_answers.{module_name}"
        ic = importlib.import_module(instructor_module_name)
        correct_answer = ic.compute()
        return correct_answer
"""

@pytest.fixture(scope='module')
def run_compute():
    def _module(module_name, ret):
        return get_module_results(module_name, ret)
    return _module
