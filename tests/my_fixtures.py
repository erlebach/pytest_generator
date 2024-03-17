import functools
import pytest
import importlib
import sys
import os

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
def load_and_run_module(module_name, directory, function_name, *args, **kwargs):
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
        """
        To execute, `result = module.question1()`
        invoke  `load_and_run_module(module, directory, 'question1')
        """
        func_to_run = getattr(module, function_name)
        result = func_to_run(*args, **kwargs)
    finally:
        os.chdir(original_cwd)
    return result

# ----------------------------------------------------------------------
# Usage within your existing setup
@functools.cache
def get_module_results(module_name, function_name, ret='both', *args, **kwargs):
    # Hardcoded folder names. These could be included in the generator_config.yaml file. NOT DONE.
    student_directory = "./student_code_with_answers"
    instructor_directory = "./instructor_code_with_answers"

    if ret == 'both':
        student_result = with_custom_sys_path(student_directory, load_and_run_module, module_name, student_directory, function_name, *args, **kwargs)
        instructor_result = with_custom_sys_path(instructor_directory, load_and_run_module, module_name, instructor_directory, function_name, *args, **kwargs)
        return student_result, instructor_result
    elif ret == 's':
        return with_custom_sys_path(student_directory, load_and_run_module, module_name, student_directory, function_name, *args, **kwargs)
    else:  # ret == 'i'
        return with_custom_sys_path(instructor_directory, load_and_run_module, module_name, instructor_directory, function_name, *args, **kwargs)

# ----------------------------------------------------------------------

@pytest.fixture(scope='module')
def run_compute():
    def _module(module_name, function_name, ret, *args, **kwargs):
        return get_module_results(module_name, function_name, ret, *args, **kwargs)
    return _module

#----------------------------------------------------------------------
