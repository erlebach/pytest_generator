
import re
import yaml
import pytest
import argparse
from types_list import types_list
from generator_utils import sanitize_function_name
from generator_utils import get_decoded_str
from generator_utils import evaluate_answers

with open('type_handlers.yaml') as f:
    type_handlers = yaml.safe_load(f)

def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        questions_data = yaml.safe_load(file)
    return questions_data

with open("generator_config.yaml", "r") as f:
    config = yaml.safe_load(f)
# How to access an element of config dict and set to default value for non-existent key?
gen_config = config['test_answers']
assert_false = gen_config.get("assert_false", False)
fixture_import_file = gen_config.get("fixture_import_file", None)

function_header_str = f"""
import pytest
from pytest_utils.decorators import max_score, visibility, hide_errors
import assert_utilities  # <<< SHOULD be specified in config
from {fixture_import_file} import *   
import {fixture_import_file}
import numpy as np
import yaml

with open('type_handlers.yaml', 'r') as f:
    type_handlers = yaml.safe_load(f)
"""


def generate_test_answers_code(questions_data, sim_type, output_file='test_answers.py'):
    module_ = questions_data["module"]
    test_code = function_header_str
    max_score = questions_data.get('max_score', 0.)

    fixture = questions_data.get("fixtures", {})
    fixture_name = fixture.get('name', "")
    fixture_args = fixture.get('args', [])

    if 'fixtures' in questions_data: 
        fixture = questions_data['fixtures']['fixture']
        fixture_name = fixture['name']
        fixture_args = fixture['args']  # list of strings
    else:
        fixture = None

    for question in questions_data['questions']:
        max_score_q = question.get('max_score', max_score)

        if 'fixture' in question: 
            fixture = question['fixture']
            fixture_name = fixture['name']
            fixture_args = fixture['args']  # list of strings

        for part in question['parts']:
            if 'fixture' in part: 
                fixture = part['fixture']
                fixture_name = fixture['name']
                fixture_args = fixture['args']  # list of strings

            part_id_sanitized = part['id'].replace(' ', '_').replace('(', '').replace(')', '').replace('|', '_').replace('=', '_')
            max_score_part = part.get('max_score', max_score_q)
            function_name = f"test_answers_{question['id']}_{part_id_sanitized}_{part['type']}"
            function_name = sanitize_function_name(function_name)

            decode_i_call_str = get_decoded_str(questions_data, part, 'i_answer', 'instructor_file')
            decode_s_call_str = get_decoded_str(questions_data, part, 's_answer', 'student_file')

            if sim_type == 'answers':
                test_code += f"\n@max_score({max_score_part})\n"
            test_code +=  "@hide_errors('')\n"

            if fixture:
                test_code +=  f"def {function_name}({fixture_name}):\n"
            else:
                test_code += f"def {function_name}():\n"

            is_fixture = fixture is not None and isinstance(fixture_args, list) and len(fixture_args) > 0
            is_instructor_file = questions_data.get('i_answer_source', 'yaml_file') == "instructor_file"
            is_student_file = questions_data.get('s_answer_source', 'yaml_file') == "student_file"

            if decode_i_call_str is None:
                # get instructor answer from instructor file
                is_instructor_file = True

            if decode_s_call_str is None:
                # get student answer from student file
                is_student_file = True

            question_id = f"{repr(part['id'])}"

            fixture_name = fixture['name'] if is_fixture else None
            if is_fixture and fixture_name is None:
                raise "Fixture name is not defined"

            test_code = evaluate_answers(questions_data, question['id'], test_code, is_fixture, is_instructor_file, is_student_file, 
                                         decode_i_call_str, decode_s_call_str, fixture, part, function_name)

            if part['type'] in types_list: 
                import_file = f"type_handlers['types']['{part['type']}']['import']"
                part_type = repr(f"{part['type']}")
                tol = part.get('rel_tol', 0.001)
                test_code += f"    tol = {tol}\n"
                assertion_answer = eval(f"type_handlers['types']['{part['type']}']['assert_answer']")  # Only difference
                assertion_structure = eval(f"type_handlers['types']['{part['type']}']['assert_structure']")  # Only difference
                keys = part.get('keys', None) ### <<<< different: optional keys to consider (temporary)
                test_code += f"    keys = {keys}\n"

                if eval(import_file):
                    test_code += f"    import {eval(import_file)}\n"

                # Check structures
                test_code += f"    msg_structure = \"{assertion_structure}\"\n"
                # Check answers
                test_code += f"    msg_answer = \"{assertion_answer}\"\n"

                test_code +=  "    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}\n"

                local_vars_dict = part.get('locals', None)
                if local_vars_dict:
                    #if 'locals' in part:
                    test_code += f"    local_vars_dict = {local_vars_dict}\n"
                    test_code +=  "    local_namespace['local_vars_dict'] = local_vars_dict\n"

                # One of a finite number of choices for string type
                choices = part.get('choices', [])
                #if choices == [] and (part_type == 'string' or part_type == 'str'):
                    #choices = []

                if choices is not None:
                    test_code += f"    choices = {choices}\n"
                    test_code +=  "    local_namespace['choices'] = choices\n"

                #keys = part.get('keys', [])
                #print("keys: ", keys)
                #if keys == [] and (part_type == 'string' or part_ == 'str'):
                    #keys = []

                #if keys is not None:
                    #test_code += f"    keys = {keys}\n"
                    #test_code +=  "    local_namespace['keys'] = keys\n"

                test_code +=  "    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)\n"

                if sim_type == 'answers':
                    test_code +=  "    if is_success:\n"
                    test_code +=  "        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)\n"
                    test_code +=  "    else: \n"
                    test_code +=  "        explanation_answer += 'Failed structural tests, No grade for answer component\\n.' \n"
                    test_code +=  "        explanation_answer += f'Instructor answer: {repr(correct_answer)}\\n'\n"
                    test_code +=  "        explanation_answer += f'Student answer: {repr(student_answer)}'\n"

                test_code +=  "    explanation = '\\n'.join(['Structure tests:', explanation_structure])\n"
                test_code += f"    {function_name}.explanation = explanation\n"
                test_code += f"    assert is_success\n"

            else:
                test_code += f"    print('type {part['type']} NOT HANDLED!')\n"

            if assert_false:
                test_code += f"    assert False\n"

            test_code += f"\n\n"
            
    with open(f"tests/{output_file}", 'w') as file:
        file.write(test_code)


# Usage example:
# questions_data = load_yaml_file('path_to_questions_answers.yaml')
# generate_test_answers_code(questions_data)

# NEW
def main(yaml_name, sim_type):
    """
    sim_type = ['answers', 'structure']
    """
    questions_data = load_yaml_file(yaml_name) 
    generate_test_answers_code(questions_data, sim_type, f"test_{sim_type}_{yaml_name[:-5]}.py")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass in the name of the input yaml file.")
    parser.add_argument("-y", "--yaml", help="Name of the yaml file", required=True)
    parser.add_argument("-t", "--simtype", help="'answers' or 'structure'", required=True)
    args = parser.parse_args()
    main(args.yaml, args.simtype)

