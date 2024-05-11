import assert_utilities as au
import yaml

def load_yaml_file(file_path):
    with open(file_path, "r") as file:
        questions_data = yaml.safe_load(file)
    return questions_data


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

### CODE BELOW NOT USED EXCEPT when executing this file.
"""
def parse_yaml(data):
    questions = data["questions"]
    for question in questions:
        parts = question["parts"]
        for part in parts:
            print(f"\n{part=}")
"""

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


