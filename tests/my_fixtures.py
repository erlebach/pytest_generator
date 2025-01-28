## My fixtures

import functools
import importlib
import inspect
import os
import sys
from contextlib import ExitStack, contextmanager
from functools import wraps
from pathlib import Path
from typing import Any
from unittest.mock import patch

import matplotlib.pyplot as plt
import numpy as np
import pytest

# Don't I have to import this in the patching software?
# import spectral_clustering


### NEW
"""
def load_data_labels(data_filename, labels_filename, nb_slices):
    print("==> load_data_labels")
    base_path = Path(__file__).parent
    data_file_path = base_path / data_filename
    labels_file_path = base_path / labels_filename

    data = np.load(data_file_path)[:nb_slices]
    labels = np.load(labels_file_path)[:nb_slices]
    return data, labels
"""


'''
# @pytest.fixture
def load_data_labels(nb_slices: int) -> tuple[Any, Any] | None:
    # I should be able to control the dataset to load
    # Load your data here, using numpy or any suitable library
    base_path = Path(__file__).parent

    # Define the paths to your data and labels files
    data_file_path = base_path / "question1_cluster_data.npy"  # Adjust the file name as needed
    labels_file_path = base_path / "question1_cluster_labels.npy"  # Adjust the file name as needed

    try:
        data = np.load(data_file_path)[:nb_slices]
        labels = np.load(labels_file_path)[:nb_slices]
    except (FileNotFoundError, OSError, ValueError) as e:
        print(f"Error loading data_labels: {e}")
        return None, None
    else:
        return data, labels
'''


# ----------------------------------------------------------------------


@pytest.fixture(scope="function", autouse=True)
def disable_plot_show(mocker):
    # mock_show = mocker.patch("matplotlib.pyplot.show")

    with ExitStack() as stack:
        # Mock plt.show() with no specific side effect
        mock_show = stack.enter_context(patch("matplotlib.pyplot.show"))

        # Replace plt.clf() with plt.close()
        stack.enter_context(patch("matplotlib.pyplot.clf", new=lambda: plt.close()))

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
            return func_to_run(*args)  # Removed **kwargs for simplification in this example
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
            modified_args, modified_kwargs = modify_args_using_slice(
                args, kwargs, slice_lg=slice_lg
            )

            return func(*modified_args, **modified_kwargs)

        return wrapper

    return decorator


# ----------------------------------------------------------------------


# Patching spectral and scatter using unittest.mock directly
# Context manager to manage multiple patches
@contextmanager
def apply_patches(*patches):
    # print("==> patches: ", patches)
    patched_objects = []
    try:
        for patch_obj in patches:
            try:
                # print(f"Attempting to patch: {patch_obj}")  # Just print the patch object
                patched_obj = patch_obj.start()
                patched_objects.append(patched_obj)
            except AttributeError as e:
                print(f"Failed to patch {patch_obj}: {e}")
                continue  # Skip failed patches
        yield patched_objects
    finally:
        for patch_obj in patches:
            try:
                patch_obj.stop()
            except Exception as e:
                print(f"Error stopping patch {patch_obj}: {e}")


""" ORIG
def patch_functions(
    directory, module_name: str, function_dict: dict, arg1=None, arg2=None, slice_lg=None, global_patches: dict=None
):
    if slice_lg is None:
        slice_lg = 200

    module = importlib.import_module(directory + "." + module_name)
    patches = []

    for func_name, provided_func in function_dict.items():
        # Use the provided function directly or fetch from module if None
        func = provided_func if provided_func is not None else getattr(module, func_name)

        # Always use modify_args_decorator if no specific args provided
        patched_func = modify_args_decorator(slice_lg=slice_lg)(func)

        # Create patch based on function type
        if func in [plt.scatter, plt.plot]:
            patcher = patch("matplotlib.pyplot." + func_name, new=patched_func)
        else:
            patcher = patch.object(module, func_name, new=patched_func)
        patches.append(patcher)

    # Add global variable patches
    if global_patches:
        for var_name, value in global_patches.items():
            patcher = patch.object(module, var_name, value)
            patches.append(patcher)

    return patches
"""


def patch_functions(
    directory,
    module_name: str,
    function_dict: dict,
    arg1=None,
    arg2=None,
    slice_lg=None,
    global_patches: dict = None,
):
    if slice_lg is None:
        slice_lg = 200

    module = importlib.import_module(directory + "." + module_name)
    patches = []

    # Only try to patch if there are functions to patch
    if function_dict and "patched_functions" in function_dict:
        for func_name, provided_func in function_dict["patched_functions"].items():
            # Skip if no function provided
            if provided_func is None:
                continue

            # Create patch based on function type
            if provided_func in [plt.scatter, plt.plot]:
                patched_func = modify_args_decorator(slice_lg=slice_lg)(provided_func)
                patcher = patch("matplotlib.pyplot." + func_name, new=patched_func)
            else:
                patcher = patch.object(module, func_name, new=provided_func)
            patches.append(patcher)

    # Add global variable patches
    if global_patches:
        for var_name, value in global_patches.items():
            patcher = patch.object(module, var_name, value)
            patches.append(patcher)

    return patches


@pytest.fixture(scope="module")
def run_compute():
    # def _module(patch_dict, ret, *args, **kwargs): # orig: 2025-01-05
    def _module(patch_dict, function_name, ret, *args, **kwargs):  # 2025-01-06
        # module_name = patch_dict["module"] # old
        module_dict = patch_dict  # new
        # function_name = patch_dict["function_name"]
        # print("++==> run_compute")
        # print(f"  {module_dict=}")
        # print(f"  {function_name=}")
        # print(f"  {module_dict[function_name]=}")
        function_dict = patch_dict[function_name]
        module_name = patch_dict["module_name"]
        global_patches = patch_dict.get("global_patches", {})  # Get global patches if they exist
        # print(f"{module_name=}")
        # print("after patch_dict")

        directories = kwargs.get("student_directory"), kwargs.get("instructor_directory")

        if ret == "s":
            directory = directories[0]
        elif ret == "i":
            directory = directories[1]
        else:
            print("Only ret='i' or 's' implemented")
            raise NotImplementedError("Only 'i' or 's' implemented")

        # Replace first two arguments of spectral by my own data
        # CHECK THAT SPECTRAL EXISTS
        nb_samples = 200

        ### My patches are not applied. WHY?

        '''
        ## load_data_labels is not normally required.
        try:
            data, labels = load_data_labels(nb_slices=nb_samples)
        except (FileNotFoundError, OSError, ValueError) as e:
            print(f"Error loading data_labels: {e}")
        '''
        # Replace the first two args of spectral by my own data. This avoids some randomness.

        # Only work with modification module
        patches = patch_functions(
            directory,
            module_name,
            function_dict,
            arg1=None,
            arg2=None,
            slice_lg=nb_samples,
            global_patches=global_patches,
        )
        '''
        patches = patch_functions(module_name, function_dict, arg1=data, arg2=labels, slice_lg=nb_samples)
        '''

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
    if ret == "s":
        results = with_custom_sys_path(
            directories[0],
            load_and_run_module,
            module_name,
            directories[0],
            function_name,
            *args,
            **kwargs,
        )
    elif ret == "i":
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
