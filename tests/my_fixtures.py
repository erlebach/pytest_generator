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
    #print("==> with_custom_sys_path, args: ", args)
    #print("==> func: ", func)
    #print("==> args: ", args)
    #print("==> **kwargs: ", **kwargs)
    try:
        sys.path.insert(0, path)
        #print("\n===> sys, func= ", func)
        #print("\n===> sys, func.__name__= ", func.__name__)
        #print("\n===> args= ", args)
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
    print("INSIDE load_and_run_module")
    print(f"{directory=}")
    print(f"{module_name=}")
    print(f"{function_name=}")
    original_cwd = os.getcwd()
    """
    try:
        os.chdir(directory)
        try:
            module = importlib.import_module(directory + "." + module_name)
        except:
            print("Import error in fixture")

        # To execute, `result = module.question1()`
        # invoke  `load_and_run_module(module, directory, 'question1')
        try:
            if hasattr(module, function_name):
                func_to_run = getattr(module, function_name)
            else:
                raise AttributeError(f"{function_name} not found in {module_name}")
        except Exception as e: 
            print(f"Error encountered: {e}")
            result = None


        result = func_to_run(*args)
        os.chdir(original_cwd)
    """
    try:
        os.chdir(directory)
        module = importlib.import_module(module_name)
        if hasattr(module, function_name):
            print("==> module, function name: ", module, function_name)
            func_to_run = getattr(module, function_name)
            print(f"==> {func_to_run=}")
        else:
            raise AttributeError(f"{function_name} not found in {module_name}")
        print("kwargs= ", kwargs)
        print("args= ", args)
        result = func_to_run(*args) #, **kwargs)
    except Exception as e:
        print(f"Error encountered: {e}")
        # Instead of quitting, handle the error to allow cleanup
        result = None
    finally:
        os.chdir(original_cwd)  # Ensure directory is always reset
    return result

# ----------------------------------------------------------------------
# Usage within your existing setup
@functools.cache
def get_module_results(module_name, function_name, ret='both', *args, **kwargs):
    # Hardcoded folder names. These could be included in the generator_config.yaml file. NOT DONE.

    # REMOVE HARDCODING. Add to configuration file
    #student_directory = "student_code_with_answers"
    ##student_directory = "student_github_template"   # for solution without correct answers
    #instructor_directory = "instructor_code_with_answers"

    #print("===> 0 get_module_results")
    if 'student_directory' in kwargs:
        student_directory = kwargs['student_directory']
    if 'instructor_directory' in kwargs:
        instructor_directory = kwargs['instructor_directory']

    if ret == 'both':
        student_result = with_custom_sys_path(student_directory, load_and_run_module, module_name, student_directory, function_name, *args, **kwargs)
        instructor_result = with_custom_sys_path(instructor_directory, load_and_run_module, module_name, instructor_directory, function_name, *args, **kwargs)
        return student_result, instructor_result

    elif ret == 's':
        return with_custom_sys_path(student_directory, load_and_run_module, module_name, student_directory, function_name, *args, **kwargs)

    else:  # ret == 'i'
        #print("===> get_module_results")
        return with_custom_sys_path(instructor_directory, load_and_run_module, module_name, instructor_directory, function_name, *args, **kwargs)

# ----------------------------------------------------------------------

@pytest.fixture(scope='module')
def run_compute():
    # Include key args: 'student_directory'= and 'instructor_directory'=
    def _module(module_name, function_name, ret, *args, **kwargs):
        return get_module_results(module_name, function_name, ret, *args, **kwargs)
    return _module

#----------------------------------------------------------------------

@pytest.fixture(scope='function', autouse=True)
def disable_plot_show(mocker):
    mock_show = mocker.patch('matplotlib.pyplot.show')
    return mock_show
#----------------------------------------------------------------------

#### LOCAL FIXTURES FOR THE SPECIFIC ASSIGNMENT

# Assuming 'spectral' is imported from the module where it's defined

"""
def mock_spectral(original_func, *args, **kwargs):
    # Modify only the first two arguments: data and labels
    if len(args) >= 2:

        data, labels = args[0], args[1]
        sliced_data = data[:300, :]  # Slice data to the first 300 samples
        sliced_labels = labels[:300]  # Slice labels to the first 300 samples
        args = (sliced_data, sliced_labels) + args[2:]  # Replace and preserve remaining args

    # Call the original function with modified args and all kwargs
    return original_func(*args, **kwargs)

@pytest.fixture
def mock_spectral_fixture(mocker, request):
    # Dynamically determine the module to mock based on test parameter
    module_name = request.param
    module = importlib.import_module(module_name)
    original_func = getattr(module, 'spectral')
    # Patch the 'spectral' function in the specified module
    mocker.patch(f"{module_name}.spectral", side_effect=lambda *args, **kwargs: mock_spectral(original_func, *args, **kwargs))
"""
