import inspect
import functools
import matplotlib.pyplot as plt
import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import importlib
import sys
import os
### WHAT IS THIS?
from contextlib import contextmanager
# Don't I have to import this in the patching software? 
# import spectral_clustering

# ----------------------------------------------------------------------

@pytest.fixture(scope='function', autouse=True)
def disable_plot_show(mocker):
    mock_show = mocker.patch('matplotlib.pyplot.show')
    return mock_show

# ----------------------------------------------------------------------
"""
To use @functools.cache safely, the function it is applied to must be a pure function, 
whose results only depend on the input arguments. This is likely correct in a testing 
environment when using serial calculations. Each test is run independently, and
all I am doing is loading modules and running student and instructor code, which are
self-contained. 
Nonetheless, use is risky. If I have errors in the testing framework, I should disable
the cache and check whether the errors persist.
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
        print(f"==> with_custom_sys_path args, {path=}")
        print(f"==> with_custom_sys_path args , {func=}")
        print(f"==> with_custom_sys_path args , {args=}")
        print(f"==> with_custom_sys_path args , {kwargs=}")
        print('==> args= ', args)
        return func(*args, **kwargs)
    finally:
        sys.path = original_sys_path

# ----------------------------------------------------------------------
@contextmanager
def temporary_directory_change(new_directory):
    """
    Context manager to temporarily change the working directory.
    """
    current_directory = os.getcwd()
    try:
        os.chdir(new_directory)
        yield
    finally:
        os.chdir(current_directory)

# ----------------------------------------------------------------------

@functools.cache
def load_and_run_module(module_name, directory, function_name, *args, **kwargs):
    """
    Loads a module from a specific directory and executes its `compute` function.

    :param module_name: Name of the module to load
    :param directory: Directory from which to load the module
    :return: The result of the module's `compute` function
    """

    with temporary_directory_change(directory):
        # print(f"==> new CWD: {os.getcwd()=}")
        module = importlib.import_module(module_name)
        if hasattr(module, function_name):
            func_to_run = getattr(module, function_name)
            print(f"==> {func_to_run=}") # modify_args_decorator.<locals>.decorator.<locals>.wrapper
            # print(f"==> {args=}")  # No args.
            return func_to_run(*args)  # Removed **kwargs for simplification in this example
        else:
            raise AttributeError(f"{function_name} not found in {module_name}")
    # print(f"==> after with temporary, CWD: {os.getcwd()=}")
    # return result

# ----------------------------------------------------------------------

def modify_args_decorator(slice_lg=300):
    def decorator(func):
        print(f"==> enter decorator: {func=}, {func.__module__=}")
        def wrapper(*args, **kwargs):

            # get the current call stack
            current_stack = inspect.stack()
            # The first element of the stack is this wrapper function,
            # so the second element is where this function was called from.
            caller = current_stack[1]  # Get caller information from the stack
            #print(f"****** Function {func.__name__} called from {caller.filename} at line {caller.lineno}")
            #print(f"****** Function args: {args}, kwargs: {kwargs}")

            # Get the current module where the function is being called
            current_module = inspect.getmodule(inspect.currentframe())
            caller_module = inspect.getmodule(inspect.currentframe().f_back)
            #print(f"Function {func.__name__} called from module: {current_module.__name__}")
            #print(f"Called by module: {caller_module.__name__}")
            # END DEBUGING

            #print(f"decorator::wrapper: {func=}, {args=}, {kwargs=}")
            if len(args) >= 2 and len(args[0]) > slice_lg  and len(args[1]) > slice_lg:
                # 1st array is sometimes 1D, sometimes 2D
                modified_args = (args[0][:slice_lg], args[1][:slice_lg]) + args[2:]

                # Check if 'c' is in keyword arguments and modify it if it's a list or a numpy array
                if 'c' in kwargs:
                    c_arg = kwargs['c']
                    if isinstance(c_arg, (list, np.ndarray)) and len(c_arg) > slice_lg:  # Ensure 'c' is list or numpy array
                        kwargs['c'] = c_arg[:slice_lg]
                return func(*modified_args, **kwargs)

            print(f"modify_args_decorator: {func=}, {args=}, {kwargs=}")
            return func(*args, **kwargs)
        return wrapper
    return decorator

# ----------------------------------------------------------------------

# Patching spectral and scatter using unittest.mock directly
# Context manager to manage multiple patches
@contextmanager
def apply_patches(*patches):
    patched_objects = []
    try:
        for patch_obj in patches:
            print(f"... starting patch: {patch_obj}")
            patched_objects.append(patch_obj.start())
        yield patched_objects
    finally:
        for patch_obj in patches:
            print(f"... stopping patch: {patch_obj}")
            patch_obj.stop()

def patch_functions(module_name, function_dict, slice_lg=200):
    print("\n\n: ENTER patch_functions")
    print(f"===> {module_name=}")
    module = importlib.import_module(module_name)
    patches = []
    for func_name, func in function_dict.items():
        print(f"===> patch_functions: {func_name=}, {func=}")
        if func is None:
            # If no function object is supplied, get the function from the module
            print("==> func is None, module: ", module)
            func = getattr(module, func_name)
            print(f"==> {func=}, {func.__module__=}")
        else:
            print("==> func is not None")
            # Use the provided function directly
            func = func

        # print(f"==> modify_decorator: {func=}")
        # print(f"+++==> modify_decorator: {func.__dict__=}")
        patched_func = modify_args_decorator(slice_lg=slice_lg)(func)

        # Check where the function comes from to apply the correct patch
        if func in [plt.scatter, plt.plot]:
            # Since these are matplotlib functions, patch them directly in 'matplotlib.pyplot'
            # This sill handle ax.scatter and plt.scatter?
            patcher = patch('matplotlib.pyplot.' + func_name, new=patched_func)
        else:
            # Patch the function assumed to be from the provided module
            print(f"===> patcher: {func_name=}, {patched_func=}")
            patcher = patch.object(module, func_name, new=patched_func)

        patches.append(patcher)
    return patches

@pytest.fixture(scope='module')
def run_compute():
    # def _module(module_name, function_dict, ret, *args, **kwargs):
    def _module(patch_dict, ret, *args, **kwargs):
        print("\n==> patch_dict: ", patch_dict)
        module_name = patch_dict['module']
        function_name = patch_dict['function_name']
        function_dict = patch_dict['patched_functions']

        print(f"Executing in module: {repr(module_name)} function: {repr(function_name)}")

        patches = patch_functions(module_name, function_dict)
        with apply_patches(*patches):
            # results = get_module_results(module_name, function_dict.keys()[0], ret, *args, **kwargs)
            #print(f"{list(function_dict.keys())=}")
            #print(f"{list(function_dict.keys())[0]=}")
            print(f"Executing get_module_results with patches active for function: {function_name}")
            results = get_module_results(module_name, function_name, ret, *args, **kwargs)
        return results
    return _module
# ----------------------------------------------------------------------

# Using custom sys path to get module results
def get_module_results(module_name, function_name, ret='both', *args, **kwargs):
    directories = kwargs.get('student_directory'), kwargs.get('instructor_directory')
    results = []
    for directory in directories:
        print(f"==> get_module_results: {directory=}, {module_name=}, {function_name=}")
        result = with_custom_sys_path(directory, load_and_run_module, module_name, directory, function_name, *args, **kwargs)
        results.append(result)
    return results if ret == 'both' else results[0] if ret == 's' else results[1]

# ----------------------------------------------------------------------
