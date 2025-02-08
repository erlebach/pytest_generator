from pprint import pprint
import re
import yaml
import pytest
import argparse
from types_list import types_list
from generator_utils import sanitize_function_name, get_decoded_str, evaluate_answers

def add_attribute(name, attr):
    if attr is not None:
        test_code = f"    {name} = {attr}\n"
        test_code += f"    local_namespace['{name}'] = {name}\n"
    else:
        test_code = ""
    return test_code

def apply_options(defaults: dict, overrides: dict) -> dict:
    result = defaults.copy()  # Start with the defaults
    print(f"{overrides=}")
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
    config_dict["rel_tol"] = (
        config.get("types", {}).get("float", {}).get("rel_tol", 0.01)
    )
    config_dict["abs_tol"] = (
        config.get("types", {}).get("float", {}).get("abs_tol", 0.01)
    )
    str_choices = config.get("test_answers", {}).get("str_choices", [])
    config_dict["str_choices"] = config.get("types", {}).get("str_choices", [])
    config_dict["dict_float_choices"] = config.get("types", {}).get(
        "dict_float_choices", {}
    )
    config_dict["dict_int_choices"] = config.get("types", {}).get(
        "dict_int_choices", {}
    )
    config_dict["partial_score_frac"] = config.get("all_tests", {}).get(
        "partial_score_frac", {}
    )
    print(f"{config_dict['partial_score_frac']=}")
    print(f"{config_dict['dict_int_choices']=}")

    config_dict["remove_spaces"] = config_dict.get("option_defaults", {}).get(
        "remove_spaces", False
    )
    config_dict["monotone_increasing"] = config_dict.get("monotone_increasing", {}).get(
        "monotone_increasing", False
    )

    config_dict["exclude_indices"] = (
        config.get("types", {}).get("list[string]", {}).get("exclude_indices", [])
    )
    config_dict["exclude_keys"] = (
        config.get("types", {}).get("list[string]", {}).get("exclude_keys", [])
    )
    config_dict["include_indices"] = (
        config.get("types", {}).get("list[string]", {}).get("include_indices", [])
    )
    config_dict["outer_key_choices"] = []  # default
    config_dict["inner_key_choices"] = []  # default
    config_dict["key_choices"] = {}  # default

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
import yaml
import matplotlib.pyplot as plt
import test_utils as u  # for decode_data()
# pytest might change the python path. Make sure to import it last. 
# import pytest

with open('type_handlers.yaml', 'r') as f:
    type_handlers = yaml.safe_load(f)
"""

# ----------------------------------------------------------------------

def generate_test_answers_code(questions_data, sim_type, output_file="test_answers.py"):
    global rel_tol, abs_tol, exclude_indices, include_indices
    global str_choices, dict_float_choices, dict_int_choices

    # Fill in data from configuration file from the 'all_tests' section
    """
    questions_data["student_folder_name"] = config_dict.get(
        "student_folder_name", "student_code_with_answers"
    )
    questions_data["instructor_folder_name"] = config_dict.get(
        "instructor_folder_name", "instructor_code_with_answers"
    )
    questions_data["i_answer_source"] = config_dict.get(
        "i_answer_source", "instructor_file"
    )
    questions_data["s_answer_source"] = config_dict.get(
        "s_answer_source", "student_file"
    )
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

    """
    config_dict["student_folder_name"] = config.get(
        "student_folder_name", "student_code_with_answers"
    )
    config_dict["instructor_folder_name"] = config.get(
        "instructor_folder_name", "instructor_code_with_answers"
    )
    config_dict["i_answer_source"] = config.get(
        "instructor_answer", "instructor_code_with_answers"
    )
    config_dict["s_answer_source"] = config.get(
        "student_answer", "student_code_with_answers"
    )
    """
    print(f"==> {questions_data=}")

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
            answer_type = config_dict['answer_type']
            part_type = part.get("type", answer_type)
            # I will need all fields in lower-level function
            part["type"] = part_type

            if "fixture" in part:
                fixture = part["fixture"]
                fixture_name = fixture["name"]
                fixture_args = fixture["args"]  # list of strings

            part_id = part["id"]
            print("==> part_id= ", part_id)
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

            test_code += f"    function_name = {function_name}\n"

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

            #print(f"{part_type=}")
            #print("types_list= ", types_list)
            if part_type in types_list:
                print("part_type is in types_list")
                test_code += "    local_namespace = {}\n"

                import_file = f"type_handlers['types']['{part_type}']['import']"
                rel_tol = part.get("rel_tol", config_dict["rel_tol"])
                abs_tol = part.get("abs_tol", config_dict["abs_tol"])
                str_choices = part.get("str_choices", config_dict["str_choices"])
                dict_key_choices = part.get("key_choices", config_dict["key_choices"])
                dict_int_choices = part.get(
                    "dict_int_choices", config_dict["dict_int_choices"]
                )
                dict_float_choices = part.get(
                    "dict_float_choices", config_dict["dict_float_choices"]
                )
                monotone_increasing = part.get(
                    "monotone_increasing", config_dict["monotone_increasing"]
                )
                remove_spaces = part.get(
                    "remove_spaces",
                    question.get("remove_spaces", config_dict["remove_spaces"]),
                )

                # indices to exclude from grading for list[float]
                # Ignore index if in exclude list
                exclude_indices = part.get(
                    "exclude_indices", config_dict["exclude_indices"]
                )
                test_code += f"    exclude_indices = {exclude_indices}\n"
                test_code += (
                    "    local_namespace['exclude_indices'] = exclude_indices\n"
                )

                exclude_keys = part.get(
                    "exclude_keys", config_dict["exclude_keys"]
                )
                test_code += f"    exclude_keys = {exclude_keys}\n"
                test_code += (
                    "    local_namespace['exclude_keys'] = exclude_keys\n"
                )

                # indices to include from grading for list[float]
                # Ignore index if not in include list
                include_indices = part.get(
                    "include_indices", config_dict["include_indices"]
                )
                test_code += f"    include_indices = {include_indices}\n"
                test_code += (
                    f"    local_namespace['include_indices'] = include_indices\n"
                )

                test_code += f"    rel_tol = {rel_tol}\n"
                test_code += f"    abs_tol = {abs_tol}\n"

                test_code += add_attribute("options", options)
                test_code += add_attribute("str_choices", str_choices)
                test_code += add_attribute("monotone_increasing", monotone_increasing)

                if part_type in ["dict[str,int]"]:
                    test_code += f"    dict_int_choices = {dict_int_choices}\n"
                    test_code +=  "    local_namespace['dict_int_choices'] = dict_int_choices\n"

                if part_type in ["dict[str,float]", "dict[str,dict[str,float]]"]:
                    test_code += f"    dict_float_choices = {dict_float_choices}\n"
                    test_code +=  "    local_namespace['dict_float_choices'] = dict_float_choices\n"

                if part_type in ["str", "dict[str,float]", "list[str]"]:
                    test_code += f"    remove_spaces = {remove_spaces}\n"
                    test_code += (
                         "    local_namespace['remove_spaces'] = remove_spaces\n"
                    )

                if part_type in ["dict[str,list[str]]"]:
                    test_code += f"    dict_key_choices = {dict_key_choices}\n"
                    test_code += (
                         "    local_namespace['key_choices'] = dict_key_choices\n"
                    )


                test_code +=  "    local_namespace['rel_tol'] = rel_tol\n"
                test_code +=  "    local_namespace['abs_tol'] = abs_tol\n"

                strg = f"type_handlers['types']['{part_type}']['assert_answer']"

                assertion_answer = eval(
                    f"type_handlers['types']['{part_type}']['assert_answer']"
                )  # Only difference
                assertion_structure = eval(
                    f"type_handlers['types']['{part_type}']['assert_structure']"
                )  # Only difference
                keys = part.get(
                    "keys", None
                )  ### <<<< different: optional keys to consider (temporary)
                test_code += f"    keys = {keys}\n"

                if eval(import_file):
                    test_code += f"    {eval(import_file)}\n" # new: 2025-01-27

                # Check structures
                test_code += f'    msg_structure = "{assertion_structure}"\n'
                # Check answers
                test_code += f'    msg_answer = "{assertion_answer}"\n'

                test_code += "    local_namespace.update({'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'keys':keys})\n"

                print("********")
                local_vars_dict = part.get("locals", None)
                test_code += add_attribute("local_vars_dict", local_vars_dict)

                # One of a finite number of choices for string type
                choices = part.get("choices", [])
                if choices == [] and (type == "string" or type == "str"):
                    choices = []
                test_code += add_attribute("choices", choices)

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

                test_code += "    partial_score_frac_l = [0.]\n"
                test_code += "    local_namespace['partial_score_frac_l'] = partial_score_frac_l\n"

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
                    # test_code += "        print(f'FAILURE, partial score: {function_name.partial_score_frac}')\n"

                if sim_type == "answers":
                    test_code += "    explanation = '\\n'.join(['==Structure tests==:', explanation_structure, '==Answer tests==:', explanation_answer])\n"
                else:
                    test_code += "    explanation = '\\n'.join(['==Structure tests==:', explanation_structure])\n"

                test_code += f"    {function_name}.explanation = explanation\n"
                test_code += f"    assert is_success\n"

            else:
                test_code += f"    print('type {part['type']} NOT HANDLED!')\n"
                test_code +=  "    assert False\n"

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
    main(args.yaml, args.simtype)
