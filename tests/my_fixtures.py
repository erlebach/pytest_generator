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
    # print("INSIDE load_and_run_module")
    # print(f"{directory=}")
    # print(f"{module_name=}")
    # print(f"{function_name=}")
    # original_cwd = os.getcwd()
    # print(f"==> original CWD: {original_cwd=}")

    with temporary_directory_change(directory):
        print(f"==> new CWD: {os.getcwd()=}")
        module = importlib.import_module(module_name)
        if hasattr(module, function_name):
            func_to_run = getattr(module, function_name)
            return func_to_run(*args)  # Removed **kwargs for simplification in this example
        else:
            raise AttributeError(f"{function_name} not found in {module_name}")
    # print(f"==> after with temporary, CWD: {os.getcwd()=}")
    # return result

# ----------------------------------------------------------------------

def modify_args_decorator(slice_lg=300):
    def decorator(func):
        def wrapper(*args, **kwargs):
            if len(args) >= 2:
                # 1st array is sometimes 1D, sometimes 2D
                modified_args = (args[0][:slice_lg], args[1][:slice_lg]) + args[2:]

                # Check if 'c' is in keyword arguments and modify it if it's a list or a numpy array
                if 'c' in kwargs:
                    c_arg = kwargs['c']
                    if isinstance(c_arg, (list, np.ndarray)):  # Ensure 'c' is list or numpy array
                        kwargs['c'] = c_arg[:slice_lg]
                return func(*modified_args, **kwargs)
            return func(*args, **kwargs)
        return wrapper
    return decorator

# ----------------------------------------------------------------------

# Patching spectral and scatter using unittest.mock directly
def apply_patches(module_name):
    spectral_module = importlib.import_module(module_name)
    original_spectral = getattr(spectral_module, 'spectral')
    original_scatter = plt.scatter

    patched_spectral = modify_args_decorator(slice_lg=200)(original_spectral)
    patched_scatter = modify_args_decorator(slice_lg=200)(original_scatter)

    spectral_patch = patch.object(spectral_module, 'spectral', new=patched_spectral)
    scatter_patch = patch.object(plt, 'scatter', new=patched_scatter)

    spectral_patch.start()
    scatter_patch.start()
    return spectral_patch, scatter_patch

# ----------------------------------------------------------------------

# Fixture to run computations
@pytest.fixture(scope='module')
def run_compute():
    patches = []
    def _module(module_name, function_name, ret, *args, **kwargs):
        if function_name == 'spectral_clustering':
            spectral_patch, scatter_patch = apply_patches(module_name)
            patches.extend([spectral_patch, scatter_patch])
        results = get_module_results(module_name, function_name, ret, *args, **kwargs)
        for patch in patches:
            patch.stop()
        patches.clear()
        return results
    return _module

# ----------------------------------------------------------------------

# Using custom sys path to get module results
def get_module_results(module_name, function_name, ret='both', *args, **kwargs):
    directories = kwargs.get('student_directory'), kwargs.get('instructor_directory')
    results = []
    for directory in directories:
        result = with_custom_sys_path(directory, load_and_run_module, module_name, directory, function_name, *args, **kwargs)
        results.append(result)
    return results if ret == 'both' else results[0] if ret == 's' else results[1]

# ----------------------------------------------------------------------
