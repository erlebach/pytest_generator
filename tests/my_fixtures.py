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
from importlib import reload

### WHAT IS THIS?
from contextlib import contextmanager
import os
import sys

# Don't I have to import this in the patching software?
# import spectral_clustering


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


def data_decorator(f):
    # @wraps(f)
    def wrapper(*args, **kwargs):
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
    print(f"==> enter with_custom_sys_path, {len(args)=}")  # 3
    try:
        sys.path.insert(0, path)

        # Folder information
        # module_path = inspect.getfile(func.__module__)
        module_path = inspect.getfile(sys.modules[func.__module__])
        folder = os.path.dirname(module_path)
        print("=== func= ", func)
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
    print("\n\n==> load_and_run_module, directory: ", directory)

    print("==> enter load_and_run_module, len(args): ", len(args))

    with temporary_directory_change(directory):
        # module = sys.modules["student_MWE." + module_name]
        #module = importlib.import_module("student_MWE." + module_name)
        print(f"===> load_and_module, import: {directory + '.' + module_name=}")
        print(f"===> load_and_module, {function_name=}")
        module = importlib.import_module(directory + "." + module_name)
        print(f"===> load_and_module after import, {module=}")
        module_path = inspect.getfile(module)
        # module_path = inspect.getfile(sys.modules["student_code_with_answers." + module_name])
        folder = os.path.dirname(module_path)

        # folder_path = ".." + directory + "." + module_name
        ## Make sure that directory does not have any "/" or "./".
        ## Make sure that both tests/ and tests/.. have an __init__.py
        ## Any folder with a __init__.py is a package

        folder_path = directory + "." + module_name
        module = importlib.import_module(folder_path)
        if hasattr(module, function_name):
            func_to_run = getattr(module, function_name)
            print("-==> load_and_run_module, nb_args: ", len(args))
            return func_to_run(
                *args
            )  # Removed **kwargs for simplification in this example
        else:
            raise AttributeError(f"{function_name} not found in {module_name}")


# ----------------------------------------------------------------------


def substitute_args_decorator(arg1, arg2):
    def decorator(func):
        # @wraps(func)
        def wrapper(*args, **kwargs):
            # Replace the first two arguments
            args = (arg1, arg2) + args[2:]
            return func(*args, **kwargs)

        return wrapper

    return decorator

def modify_args_using_shape(args, kwargs, slice_lg=300):
    modified_args = []
    for arg in args:
        if isinstance(arg, np.ndarray) and arg.shape[0] > slice_lg:
            #new_shape = (slice_lg,) + arg.shape[1:]  # Create a new shape that keeps other dimensions intact
            #arg.resize(new_shape, refcheck=False)  # Resize in place, modifying the original array
            modified_args.append(arg[0:slice_lg])
        else:
            modified_args.append(arg)

    modified_kwargs = {}
    for k, v in kwargs.items():
        if isinstance(v, np.ndarray) and v.shape[0] > slice_lg:
            #new_shape = (slice_lg,) + v.shape[1:]  # Same as above for kwargs
            #v.resize(new_shape, refcheck=False)
            modified_kwargs[k] = v[0:slice_lg]
        else:
            modified_kwargs[k] = v

    return modified_args, modified_kwargs


def modify_args_decorator(slice_lg=300):
    def decorator(func):
        # @wraps(func)
        def wrapper(*args, **kwargs):
            """
            Once the decorated function is called, the arguments revert to their original size
            """
            # current_stack = inspect.stack()
            # caller = current_stack[1]  # Get caller information from the stack

            modified_args, modified_kwargs = modify_args_using_shape(args, kwargs, slice_lg=slice_lg)

            return func(*modified_args, **modified_kwargs)

            """
            if False:
                args = [
                    (
                        arg[:slice_lg]
                        if isinstance(arg, np.ndarray) and len(arg) > slice_lg
                        else arg
                    )
                    for arg in args
                ]
                kwargs = {
                    k: (
                        v[:slice_lg]
                        if isinstance(v, np.ndarray) and len(v) > slice_lg
                        else v
                    )
                    for k, v in kwargs.items()
                }
                return func(*args, **kwargs)

            return func(*args, **kwargs)
            """

        print("before return wrapper")
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
            print("... start")
        yield patched_objects
    finally:
        for patch_obj in patches:
            patch_obj.stop()
            print("... stop")


def patch_functions(directory, module_name, function_dict, arg1=None, arg2=None, slice_lg=None):
    if slice_lg is None:
        slice_lg = 200

    # module = importlib.import_module("student_code_with_answers." + module_name)  # NEW
    print(f"\n===> patch_functions: {module_name=}")
    print(f"\n===> patch_functions: {function_dict=}")
    print(f"\n===> patch_functions: {directory=}")
    #module = sys.modules[module_name]
    #print("==> patch_functions: module= ", module)
    # module = importlib.import_module("student_MWE." + module_name)  # NEED THE FOLDER
    module = importlib.import_module(directory + "." + module_name)  # NEED THE FOLDER
    print(f"\n===> patch_function: {module=}")
    patches = []

    for func_name, provided_func in function_dict.items():
        # Use the provided function directly or fetch from module if None
        print("patch_functions: func_name= ", func_name)
        print("patch_functions: provided_func= ", provided_func)
        func = (
            provided_func if provided_func is not None else getattr(module, func_name)
        )
        print("patch_functions: func= ", func)

        # Decide which decorator to apply based on whether specific arguments are provided
        # if func_name == 'spectral':
        if True:
            if arg1 is not None and arg2 is not None:
                # Apply substitute_args_decorator for 'spectral' function if arg1 and arg2 are provided
                patched_func = substitute_args_decorator(arg1, arg2)(func)
            else:
                # Apply modify_args_decorator for other cases or when slice_lg is specified
                print(f"modify_args_decorator: {func=}")
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
        print(f"run_compute: {directories=}")

        if ret == 's':
            directory = directories[0]
        elif ret == 'i':
            directory = directories[1]
        else:
            print("Only ret='i' or 's' implemented")
            raise "Not implemented error"

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
            # reload("student_code_with_answers.spectral_clustering") # <<< NEW, 2024-05-04
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
    print(f"get_module_results: {directories=}")
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
        raise "FAIL"
    
    return results

    """
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
    """


# ----------------------------------------------------------------------
