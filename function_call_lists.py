import assert_utilities as au
import yaml

def load_yaml_file(file_path):
    with open(file_path, "r") as file:
        questions_data = yaml.safe_load(file)
    return questions_data

# I should allow chaining of multiple validation functions. 

# in spectral.yaml
#       options:
#           float_error: {'rel_tol': 3.e-2, 'abs_tol': 1.e-5}
#           float_range: {'min': -3., 'max': 4.}

# assert_utilities prepends student_answer and instructor_answer
validation_function_templates = {
    'float_error': {
        'function': 'check_float',
        'parameters': ['rel_tol', 'abs_tol']
        #'parameters': [('rel_tol', 'abs_tol')]
    },
    'float_range': {
        'function': 'check_float_range',
        'parameters': ['min', 'max']
    },
    # my_array: 1D list or 1D array
    'monotone_increasing': {
        'function': 'check_list_float_monotone_increasing',
        'parameters': [] # could validate it is an array
    },
    # my_array: 1D list or 1D array
    'monotone_decreasing': {
        'function': 'check_list_float_monotone_decreasing',
        'parameters': []
    },
    'is_probability': {
        'function': 'check_list_float_is_probability',
        'parameters': ['rel_tol', 'abs_tol']
    },
    'at_least_count': {
        'function': 'check_list_at_least',
        'parameters': ['count']
    },
    # Apply validations on a particular key
    'range_validation': {
        'function': 'check_inner_key',
        'parameters': ['key_pos', 'key', 'min', 'max', 'validations']
    },
    'apply_validations_to_key': {
        'function': 'apply_validations_to_key',
        # key1, key2, list of functions, list of dicts
        'parameters': ['key', 'key_pos', 'validation_list', 'options_list']
    },
    # Other predefined functions...
}

def parse_yaml(data):
    questions = data["questions"]
    for question in questions:
        parts = question["parts"]
        for part in parts:
            print(f"\n{part=}")

def validate(data, validation_type):
    validation_info = validation_functions[validation_type]
    func = getattr(au, validation_info['function'])
    params = (data[param] for param in validation_info['parameters'])
    return func(*params)

def construct_function_dict(options):
    results = []
    for validation_type, options in options.items():
        if validation_type in validation_functions:
            # Construct actual parameters based on the part data
            constructed_params = {
                'my_float': options.get('my_float', None),  # Example placeholder
                'frange': (options.get('min'), options.get('max')),
                'ferror': (options.get('rel_tol', 1e-2), options.get('abs_tol', 1e-5)),
                'my_array': options.get('my_array', [])  # Example placeholder
            }
            result = validate(constructed_params, validation_type, options)
            results.append((validation_type, result))
            print(f"Validation result for {validation_type} on part {part['id']}: {result}")

def parse_yaml(data):
    questions = data["questions"]
    for question in questions:
        parts = question["parts"]
        for part in parts:
            print(f"\nHandling part: {part['id']}")
            options = part.get(options, {})
            result = construct_function_dict(options)


# ----------------------------------------------------------------------
if __name__ == '__main__':
    data = load_yaml_file("validation.yaml")
    parse_yaml(data)


"""
def check_msg_status(status, msg_list, status_, msg_):
def init_partial_score_dict() -> dict[str, float | int]:
def check_missing_keys(missing_keys, msg_list):
def check_float_range(s_el, frange):
def check_float(i_el, s_el, rel_tol=1.0e-2, abs_tol=1.0e-5):
def check_int(i_el, s_el):
def check_list_float_monotone_increasing(s_arr):
def check_list_float_monotone_decreasing(s_arr):
def check_list_float_is_probability(s_arr, rel_tol, abs_tol):
def check_list_float(i_arr, s_arr, rel_tol, abs_tol, ps_dict: dict[str, float | int], exclude_indices: list[int]=[]):
def check_list_int(i_arr, s_arr, ps_dict: dict[str, float | int]):
def check_set_int(i_set: set[int], s_set: set[int], ps_dict: dict[str, float | int]):
def check_str(i_str, s_str, str_choices: list[str] | None =None, remove_spaces: bool=None):
def check_list_str(i_list, s_list, ps_dict: dict[str, float | int]):
def check_dict_str_str(
def update_score(ps_dict: dict[str, float | int]) -> None:
def check_dict_str_float_range(keys, s_dict, range_val, ps_dict):
def check_dict_str_float(
def check_dict_int(keys, i_dict, s_dict, ps_dict: dict[str, float | int]):
def is_explain(answer):
def clean_str_answer(answer):
def load_yaml_file(file_path):
def extract_config_dict():
def fmt_ifstr(x):
def return_value(status, msg_list, s_answ, i_answ):
def are_sets_equal(set1, set2, rtol=1e-5, atol=1e-6):
def check_key_structure(s_dict, i_dict):
"""

