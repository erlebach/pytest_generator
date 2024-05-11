from pprint import pprint
import re
import yaml
import pytest
import argparse
from types_list import types_list
from generator_utils import sanitize_function_name, get_decoded_str, evaluate_answers
from function_call_lists import validation_function_templates as validation_functions

# In spectral.yaml
#     options:
#         float_error: {'rel_tol': 3.e-2, 'abs_tol': 1.e-5}
#         float_range: {'min': -3., 'max': 4.}
#
# in function_call_lists
#    'float_error': {
#        'function': 'check_float',
#        'parameters': [('rel_tol', 'abs_tol')]
#    },

# GPT-4 version
def validate_complex_function(key, value):
    """
    (Every 'function' key is a key in `validation_functions`)
    Parameters: 
    ----------
    value: {
        'key': 'ARI', 'key_pos': 'inner', 
        'options_list': [{'max': 1, 'min': -1}, {'abs_tol': '1e-5', 'rel_tol': 0.03}], 
        'validation_list': ['float_range', 'float_error']
           }
    Return: dict
    ------
       {
          'outer_function': 'apply_validations_to_key', 'args': ['inner', 'ARI'], 
          'inner_functions': [
              {'function': 'fct1', 'args': [.01, .3]},  
              {'function': 'fct2', 'args': [-1., 3.]}
          ]
       }
    """
    loc_key = value.get('key', "")
    key_pos = value.get('key_pos', "inner")
    validation_list = value['validation_list']
    options_list = value['options_list']

    if loc_key == "":
        print("validate_complex_function: missing key 'key' in function definition")
        validation_details = {}

    inner_functions = []

    # Generate a list of functions to apply
    for function_key, params in zip(validation_list, options_list):
        function_template = validation_functions[function_key]
        function_name = function_template['function']
        parameters = function_template['parameters']
        args = [params[param] for param in parameters]

        # Append the details of each validation function
        inner_functions.append({
            'function': function_name,
            'args': args
        })

    # In the future, there could be other complex functions beyond 'apply_validations_to_key'
    # Assemble the complex function details including both the outer and inner function calls
    validation_details = {
        'outer_function': 'apply_validations_to_key',
        'args': [key_pos, loc_key],
        'inner_functions': inner_functions
    }

    return validation_details



def validate_simple_function(key, value):
    """
    Return: dict
    -------
        {'function': 'check_float', 'args': [0.01, 1e-05]}
    """
    args = []
    for k,v in value.items():
        args.append(k)

    # Handle simple validation configurations
    function_template = validation_functions[key]
    function_name = function_template['function']
    function_args = [value[param] for param in function_template['parameters']]

    validation_details = {
        'function': function_name,
        'args': function_args
    }
    return validation_details


def generate_validations(part):
    """
    Return: 
    ------
    validation_functions=[{'function': 'check_float', 'args': [0.01, 1e-05]}, {'function': 'check_float_range', 'args': [-1.0, 2]}]
    """
    options = part.get('options', {})
    validations = []

    # Iterate over all entries in the options dictionary
    for key, value in options.items():
        if isinstance(value, list):
            for value_el in value:
                if 'validation_list' in value_el and 'options_list' in value_el:
                    print(".... if")
                    details = validate_complex_function(key, value_el)
                    validations.append(details)
        else:
            print(".... else")
            details = validate_simple_function(key, value)
            validations.append(details)

    return validations


# ----------------------------------------------------------------------
def add_attribute(name, attr):
    if attr is not None:
        test_code = f"    {name} = {attr}\n"
        test_code += f"    local_namespace['{name}'] = {name}\n"
    else:
        test_code = ""
    return test_code

def apply_options(defaults: dict, overrides: dict) -> dict:
    result = defaults.copy()  # Start with the defaults
    result.update(overrides)  # Apply overrides
    return result

with open("type_handlers.yaml") as f:
    type_handlers = yaml.safe_load(f)

def load_yaml_file(file_path):
    with open(file_path, "r") as file:
        questions_data = yaml.safe_load(file)
    return questions_data

def load_validations_options_file(file_path):
    with open(file_path, "r") as file:
        validations_options = yaml.safe_load(file)
    return validations_options

def load_configuration_file(file_path):
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def create_config_dict():
    with open("generator_config.yaml", "r") as f:
        config = yaml.safe_load(f)

    option_defaults = config.get("option_defaults", {})
    types = config.get("types", {})
    config_dict = {}
    config_dict['answer_type'] = config.get("all_tests").get("type", "float")
    config_dict["partial_score_frac"] = config.get("all_tests", {}).get(
        "partial_score_frac", {}
    )

    config_dict["student_folder_name"] = config.get("all_tests").get(
        "student_folder_name", "student_code_with_answers"
    )
    config_dict["instructor_folder_name"] = config.get("all_tests").get(
        "instructor_folder_name", "instructor_code_with_answers"
    )
    config_dict["i_answer_source"] = config.get("all_tests").get(
        "instructor_answer", "instructor_file"
    )
    config_dict["s_answer_source"] = config.get("all_tests").get(
        "student_answer", "student_file"
    )
    return config, config_dict

config, config_dict = create_config_dict()

# student_folder_name: student_code_with_answers
# instructor_folder_name: instructor_code_with_answers

# i_answer_source: instructor_file
# s_answer_source: student_file

# How to access an element of config dict and set to default value for non-existent key?
gen_config = config["test_answers"]
assert_false = gen_config.get("assert_false", False)
fixture_import_file = gen_config.get("fixture_import_file", None)

function_header_str = f"""
from pytest_utils.decorators import max_score, visibility, hide_errors
import assert_utilities  # <<< SHOULD be specified in config
from {fixture_import_file} import *   
# meant to handle fixtures (ideally auto-generated, but not yet)
from function_dictionaries import *
#import {fixture_import_file}
import numpy as np
from tests.pytest_utilities import get_current_function
import yaml
# pytest might change the python path. Make sure to import it last. 
# import pytest

with open('type_handlers.yaml', 'r') as f:
    type_handlers = yaml.safe_load(f)
"""

# ----------------------------------------------------------------------

def generate_test_answers_code(questions_data, sim_type, output_file="test_answers.py"):
    """
    """
    # Update the questions_data with the keys of config_dict that are not present in questions_data
    for k, v in config['all_tests'].items():
        if k not in questions_data:
            questions_data[k] = v

    if sim_type == "answers":
        # Fill in data from configuration file from the 'test_answers' section of the config dict
        for k, v in config["test_answers"].items():
            if k not in questions_data:
                questions_data[k] = v
    elif sim_type == "structure":
        # Fill in data from configuration file from the 'test_structure' section of the config dict
        for k, v in config["test_structure"].items():
            if k not in questions_data:
                questions_data[k] = v

    module_ = questions_data["module"]
    test_code = function_header_str
    max_score = questions_data.get("max_score", 0.0)

    fixture = questions_data.get("fixtures", {})
    fixture_name = fixture.get("name", "")
    fixture_args = fixture.get("args", [])

    if "fixtures" in questions_data:
        fixture = questions_data["fixtures"]["fixture"]
        fixture_name = fixture["name"]
        fixture_args = fixture["args"]  # list of strings
    else:
        fixture = None

    for question in questions_data["questions"]:
        max_score_q = question.get("max_score", max_score)
        part_question_id = question.get("id", None)
        if part_question_id == None:
            print("Question does not have an id")
            quit()

        # print("==> question_id: ", part_question_id)
        if "fixture" in question:
            fixture = question["fixture"]
            fixture_name = fixture["name"]
            fixture_args = fixture["args"]  # list of strings

        for part in question["parts"]:
            options = part.get('options', {})
            print("=========================================")
            print(f"\n===> {part=}")

            answer_type = config_dict['answer_type']
            part_type = part.get("type", answer_type)
            # I will need all fields in lower-level function
            part["type"] = part_type

            if "fixture" in part:
                fixture = part["fixture"]
                fixture_name = fixture["name"]
                fixture_args = fixture["args"]  # list of strings

            part_id = part["id"]
            part_id_sanitized = (
                part_id.replace(" ", "_")
                .replace("(", "")
                .replace(")", "")
                .replace("|", "_")
                .replace("=", "_")
            )
            max_score_part = part.get("max_score", max_score_q)

            if sim_type == "answers":
                function_name = (
                    f"test_answers_{question['id']}_{part_id_sanitized}_{part_type}"
                )
            else:
                function_name = (
                    f"test_structure_{question['id']}_{part_id_sanitized}_{part_type}"
                )

            function_name = sanitize_function_name(function_name)

            # print("==> part: ", part)
            decode_i_call_str = get_decoded_str(
                questions_data, part, "i_answer", "instructor_file"
            )
            decode_s_call_str = get_decoded_str(
                questions_data, part, "s_answer", "student_file"
            )

            if sim_type == "answers":
                test_code += f"\n@max_score({max_score_part})\n"
            test_code += "@hide_errors('')\n"

            if fixture:
                test_code += f"def {function_name}({fixture_name}):\n"
            else:
                test_code += f"def {function_name}():\n"

            test_code += f"    function_name = get_current_function()\n"

            is_fixture = (
                fixture is not None
                and isinstance(fixture_args, list)
                and len(fixture_args) > 0
            )
            is_instructor_file = (
                questions_data.get("i_answer_source", "yaml_file") == "instructor_file"
            )
            is_student_file = (
                questions_data.get("s_answer_source", "yaml_file") == "student_file"
            )

            if decode_i_call_str is None:
                # get instructor answer from instructor file
                is_instructor_file = True

            if decode_s_call_str is None:
                # get student answer from student file
                is_student_file = True

            question_id = f"{repr(part['id'])}"

            fixture_name = fixture["name"] if is_fixture else None
            if is_fixture and fixture_name is None:
                raise "Fixture name is not defined"

            test_code = evaluate_answers(
                questions_data,
                question,
                test_code,
                is_fixture,
                is_instructor_file,
                is_student_file,
                decode_i_call_str,
                decode_s_call_str,
                fixture,
                part,
                function_name,
            )

            if part_type in types_list:
                test_code += "    local_namespace = {}\n"

                import_file = f"type_handlers['types']['{part_type}']['import']"

                validation_functions = generate_validations(part)
                print(f"\nRETURN, {validation_functions=}")

                test_code += f"    validation_functions = {validation_functions}\n"
                test_code += f"    options = {options}\n"

                assertion_answer = eval(
                    f"type_handlers['types']['{part_type}']['assert_answer']"
                )  # Only difference
                assertion_structure = eval(
                    f"type_handlers['types']['{part_type}']['assert_structure']"
                )  # Only difference

                if eval(import_file):
                    test_code += f"    import {eval(import_file)}\n"

                # Check structures
                test_code += f'    msg_structure = "{assertion_structure}"\n'
                # Check answers
                test_code += f'    msg_answer = "{assertion_answer}"\n'

                note = part.get("note", None)
                if sim_type == "answers" and note is not None:
                    test_code += f"    note = {repr(note)}\n"
                    test_code += "    function_name.note = note\n"

                explanation = part.get("explanation", None)
                if sim_type == "answers" and explanation is not None:
                    test_code += f"    answer_note = {repr(explanation)}\n"
                    test_code += "    function_name.answer_note = answer_note\n"

                test_code += f"    answer_type = {repr(part_type)}\n"
                test_code += f"    question_id = {repr(part_question_id)}\n"
                test_code += f"    subquestion_id = {repr(part_id)}\n"

                test_code += f"    partial_score_frac_l = [0.]\n"

                test_code += "    local_namespace.update({'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'options': options, 'validation_functions': validation_functions, 'partial_score_frac_l': partial_score_frac_l})\n"

                test_code += "    function_name.answer_type = answer_type\n"
                test_code += "    function_name.question_id = question_id\n"
                test_code += "    function_name.subquestion_id = subquestion_id\n"
                test_code += (
                    "    function_name.partial_score_frac = partial_score_frac_l[0]\n"
                )

                test_code += "    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)\n"

                if sim_type == "answers":
                    test_code += "    if is_success:\n"
                    test_code += "        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)\n"
                    test_code += "        if is_success is True:\n"
                    test_code += "            function_name.partial_score_frac = 1.0\n"
                    test_code += "        else:\n"
                    test_code += "            function_name.partial_score_frac = partial_score_frac_l[0]\n"
                    test_code += "    else: \n"
                    test_code += "        explanation_answer = 'Failed structural tests, No grade for answer component\\n.' \n"
                    test_code += "        explanation_answer += f'Instructor answer: {repr(correct_answer)}\\n'\n"
                    test_code += "        explanation_answer += f'Student answer: {repr(student_answer)}'\n"
                    test_code += "        function_name.partial_score_frac = partial_score_frac_l[0]\n"

                if sim_type == "answers":
                    test_code += "    explanation = '\\n'.join(['==Structure tests==:', explanation_structure, '==Answer tests==:', explanation_answer])\n"
                else:
                    test_code += "    explanation = '\\n'.join(['==Structure tests==:', explanation_structure])\n"

                test_code += f"    function_name.explanation = explanation\n"
                test_code += f"    assert is_success\n"

            else:
                test_code += f"    print('type {part['type']} NOT HANDLED!')\n"

            if assert_false:
                test_code += f"    assert False\n"

            test_code += f"\n\n"

    with open(f"tests/{output_file}", "w") as file:
        file.write(test_code)


# ----------------------------------------------------------------------
def main(yaml_name, sim_type):
    """
    sim_type = ['answers', 'structure']
    """

    print("\n******* main ******")
    questions_data = load_yaml_file(yaml_name)
    generate_test_answers_code(
        questions_data, sim_type, f"test_{sim_type}_{yaml_name[:-5]}.py"
    )


# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pass in the name of the input yaml file."
    )
    parser.add_argument("-y", "--yaml", help="Name of the yaml file", required=True)
    parser.add_argument(
        "-t", "--simtype", help="'answers' or 'structure'", required=True
    )
    args = parser.parse_args()

    # print("++++++++++++++++++++++++++++++++++++++++++++++")
    # print("++++++++++++++++++++++++++++++++++++++++++++++")
    # print("+++++ START MAIN ++++")
    main(args.yaml, args.simtype)
