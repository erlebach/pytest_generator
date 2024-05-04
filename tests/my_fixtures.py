## My fixtures

import sys
import os
import inspect
import pytest
import functools
from functools import wraps
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from unittest.mock import patch, MagicMock
import importlib

### WHAT IS THIS?
from contextlib import contextmanager
import os
import sys

# Don't I have to import this in the patching software?
# import spectral_clustering


# @pytest.fixture
def load_data_labels(nb_slices: int):
    # Load your data here, using numpy or any suitable library
    base_path = Path(__file__).parent

    # Define the paths to your data and labels files
    data_file_path = (
        base_path / "question1_cluster_data.npy"
    )  # Adjust the file name as needed
    labels_file_path = (
        base_path / "question1_cluster_labels.npy"
    )  # Adjust the file name as needed

    data = np.load(data_file_path)[:nb_slices]
    labels = np.load(labels_file_path)[:nb_slices]
    return data, labels


# ----------------------------------------------------------------------


def data_decorator(f):
    # @wraps(f)
    def wrapper(*args, **kwargs):
        print("==> data_decorator:  is happening before the function is called.")
        return f(*args, **kwargs)

    return wrapper


# ----------------------------------------------------------------------


@pytest.fixture(scope="function", autouse=True)
def disable_plot_show(mocker):
    mock_show = mocker.patch("matplotlib.pyplot.show")
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
        # print(f"==> with_custom_sys_path args , {args=}")
        # print(f"==> with_custom_sys_path args , {kwargs=}")
        # print("==> args= ", args)
        # print(f"==> {sys.path=}")

        # Folder information
        # module_path = inspect.getfile(func.__module__)
        module_path = inspect.getfile(sys.modules[func.__module__])
        folder = os.path.dirname(module_path)
        print(f"==> {folder=}, {module_path=}")
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
        module_path = inspect.getfile(sys.modules[module_name])
        folder = os.path.dirname(module_path)

        # folder_path = ".." + directory + "." + module_name
        ## Make sure that directory does not have any "/" or "./".
        ## Make sure that both tests/ and tests/.. have an __init__.py
        ## Any folder with a __init__.py is a package
        folder_path = directory + "." + module_name
        module = importlib.import_module(folder_path)
        if hasattr(module, function_name):
            func_to_run = getattr(module, function_name)
            return func_to_run(
                *args
            )  # Removed **kwargs for simplification in this example
        else:
            raise AttributeError(f"{function_name} not found in {module_name}")


# ----------------------------------------------------------------------


def substitute_args_decorator(arg1, arg2):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Replace the first two arguments
            print(f"==> substitute_args_decorator: {arg1.shape=}, {arg2.shape=}")
            args = (arg1, arg2) + args[2:]
            return func(*args, **kwargs)

        return wrapper

    return decorator


def modify_args_decorator(slice_lg=300):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            current_stack = inspect.stack()
            caller = current_stack[1]  # Get caller information from the stack

            if True:
                args = [arg[:slice_lg] if len(arg) > slice_lg else arg for arg in args]
                kwargs = {
                    k: (
                        v[:slice_lg]
                        if isinstance(v, (list, np.ndarray)) and len(v) > slice_lg
                        else v
                    )
                    for k, v in kwargs.items()
                }
                modified_args = args
                return func(*modified_args, **kwargs)
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


def patch_functions(module_name, function_dict, arg1=None, arg2=None, slice_lg=None):
    if slice_lg is None:
        slice_lg = 200

    # print(f"===> patch_functions, module_name= ", module_name)
    module = importlib.import_module(module_name)
    patches = []
    for func_name, func in function_dict.items():
        if func is None:
            # If no function object is supplied, get the function from the module
            func = getattr(module, func_name)
        else:
            # Use the provided function directly
            func = func

        # Check if we need to replace the first two arguments
        if func_name == "spectral" and (arg1 is not None and arg2 is not None):
            print(f"===> {arg1.shape=}, {arg2.shape=}")
            patched_func = substitute_args_decorator(arg1, arg2)(func)
            print("data is substituted")  # why isn't patch applied?
        else:
            patched_func = modify_args_decorator(slice_lg=slice_lg)(func)

        # Check where the function comes from to apply the correct patch
        if func in [plt.scatter, plt.plot]:
            # Since these are matplotlib functions, patch them directly in 'matplotlib.pyplot'
            # This sill handle ax.scatter and plt.scatter?
            patcher = patch("matplotlib.pyplot." + func_name, new=patched_func)
        else:
            # Patch the function assumed to be from the provided module
            patcher = patch.object(module, func_name, new=patched_func)

        patches.append(patcher)
    return patches


@pytest.fixture(scope="module")
def run_compute():
    def _module(patch_dict, ret, *args, **kwargs):
        module_name = patch_dict["module"]
        function_name = patch_dict["function_name"]
        function_dict = patch_dict["patched_functions"]

        # Replace first two arguments of spectral by my own data
        nb_samples = 200

        ### My patches are not applied. WHY?

        data, labels = load_data_labels(nb_slices=nb_samples)
        # Replace first two args of spectral by my own data. This avoids some randomness.

        patches = patch_functions(
            module_name, function_dict, arg1=data, arg2=labels, slice_lg=nb_samples
        )
        # patches = patch_functions(module_name, function_dict, arg1=None, arg2=None, slice_lg=nb_samples.)

        with apply_patches(*patches):
            results = get_module_results(
                module_name, function_name, ret, *args, **kwargs
            )
        return results

    return _module


# ----------------------------------------------------------------------


# Using custom sys path to get module results
def get_module_results(module_name, function_name, ret="both", *args, **kwargs):
    directories = kwargs.get("student_directory"), kwargs.get("instructor_directory")
    results = []
    for directory in directories:
        result = with_custom_sys_path(
            directory,
            load_and_run_module,
            module_name,
            directory,
            function_name,
            *args,
            **kwargs,
        )
        results.append(result)
    return results if ret == "both" else results[0] if ret == "s" else results[1]


# ----------------------------------------------------------------------
