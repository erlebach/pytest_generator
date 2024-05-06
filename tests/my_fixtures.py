## My fixtures

import sys
import os
import inspect
import functools
from pathlib import Path
from contextlib import contextmanager, ExitStack


import pytest
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
from unittest.mock import patch
import importlib

# Don't I have to import this in the patching software?
# import spectral_clustering


### NEW
def load_data_labels(data_filename, labels_filename, nb_slices):
    base_path = Path(__file__).parent
    data_file_path = base_path / data_filename
    labels_file_path = base_path / labels_filename

    data = np.load(data_file_path)[:nb_slices]
    labels = np.load(labels_file_path)[:nb_slices]
    return data, labels



# @pytest.fixture
def load_data_labels(nb_slices: int):
    # I should be able to control the dataset to load
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


@pytest.fixture(scope="function", autouse=True)
def disable_plot_show(mocker):
    # mock_show = mocker.patch("matplotlib.pyplot.show")

    with ExitStack() as stack:
        # Mock plt.show() with no specific side effect
        mock_show = stack.enter_context(patch('matplotlib.pyplot.show'))

        # Replace plt.clf() with plt.close()
        stack.enter_context(patch('matplotlib.pyplot.clf', new=lambda: plt.close()))

        yield mock_show


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

        # Folder information
        # module_path = inspect.getfile(func.__module__)
        module_path = inspect.getfile(sys.modules[func.__module__])
        folder = os.path.dirname(module_path)
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
        module = importlib.import_module(directory + "." + module_name)
        module_path = inspect.getfile(module)
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
            args = (arg1, arg2) + args[2:]
            return func(*args, **kwargs)

        return wrapper

    return decorator

def modify_args_using_slice(args, kwargs, slice_lg=300):
    modified_args = []
    for arg in args:
        if isinstance(arg, np.ndarray) and arg.shape[0] > slice_lg:
            modified_args.append(arg[0:slice_lg])
        else:
            modified_args.append(arg)

    modified_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray) and v.shape[0] > slice_lg:
            modified_kwargs[k] = v[0:slice_lg]
        else:
            modified_kwargs[k] = v

    return modified_args, modified_kwargs


def modify_args_decorator(slice_lg=300):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            """
            Once the decorated function is called, the arguments revert to their original size
            """
            modified_args, modified_kwargs = modify_args_using_slice(args, kwargs, slice_lg=slice_lg)

            return func(*modified_args, **modified_kwargs)

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
            patched_objects.append(patch_obj.start())
        yield patched_objects
    finally:
        for patch_obj in patches:
            patch_obj.stop()


def patch_functions(directory, module_name, function_dict, arg1=None, arg2=None, slice_lg=None):
    if slice_lg is None:
        slice_lg = 200

    # module = importlib.import_module("student_code_with_answers." + module_name)  # NEW
    module = importlib.import_module(directory + "." + module_name)  
    patches = []

    for func_name, provided_func in function_dict.items():
        # Use the provided function directly or fetch from module if None
        func = (
            provided_func if provided_func is not None else getattr(module, func_name)
        )

        # Decide which decorator to apply based on whether specific arguments are provided
        # if func_name == 'spectral':
        if True:
            if arg1 is not None and arg2 is not None:
                # Apply substitute_args_decorator for 'spectral' function if arg1 and arg2 are provided
                patched_func = substitute_args_decorator(arg1, arg2)(func)
            else:
                # Apply modify_args_decorator for other cases or when slice_lg is specified
                patched_func = modify_args_decorator(slice_lg=slice_lg)(func)

        # Determine the patching strategy based on function origin (matplotlib or custom module)
        if func in [plt.scatter, plt.plot]:
            # Patch matplotlib plotting functions directly
            patcher = patch("matplotlib.pyplot." + func_name, new=patched_func)
        else:
            # Patch other functions from the provided module
            patcher = patch.object(module, func_name, new=patched_func)

        patches.append(patcher)
    return patches


@pytest.fixture(scope="module")
def run_compute():
    def _module(patch_dict, ret, *args, **kwargs):
        module_name = patch_dict["module"]
        function_name = patch_dict["function_name"]
        function_dict = patch_dict["patched_functions"]

        directories = kwargs.get("student_directory"), kwargs.get("instructor_directory")

        if ret == 's':
            directory = directories[0]
        elif ret == 'i':
            directory = directories[1]
        else:
            print("Only ret='i' or 's' implemented")
            raise NotImplementedError("Only 'i' or 's' implemented")


        # Replace first two arguments of spectral by my own data
        nb_samples = 200

        ### My patches are not applied. WHY?

        data, labels = load_data_labels(nb_slices=nb_samples)
        # Replace the first two args of spectral by my own data. This avoids some randomness.

        # Only work with modification module
        patches = patch_functions(
            directory, module_name, function_dict, arg1=None, arg2=None, slice_lg=nb_samples
        )
        # patches = patch_functions(module_name, function_dict, arg1=data, arg2=labels, slice_lg=nb_samples)

        with apply_patches(*patches):
            results = get_module_results(
                # High level function (typically the question in a student assignment)
                module_name,
                function_name,
                ret,
                *args,
                **kwargs,
            )
        return results

    return _module


# ----------------------------------------------------------------------


# Using custom sys path to get module results
def get_module_results(module_name, function_name, ret="both", *args, **kwargs):
    directories = kwargs.get("student_directory"), kwargs.get("instructor_directory")
    results = []
    if ret == 's':
        results = with_custom_sys_path(
            directories[0],
            load_and_run_module,
            module_name,
            directories[0],
            function_name,
            *args,
            **kwargs,
    )
    elif ret == 'i':
        results = with_custom_sys_path(
            directories[1],  # arguments passed to with_custom_sys_path
            load_and_run_module,
            module_name,
            directories[1],  # arguments passed to load_and_run_module
            function_name,
            *args,
            **kwargs,
        )

    else:
        print("return not handled: 'both'")
        raise NotImplementedError("Only 'i' or 's' implemented")
    
    return results


# ----------------------------------------------------------------------
