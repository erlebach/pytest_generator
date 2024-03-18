import yaml
import pytest


# def create_test_answers_generator_script(yaml_file, output_path="test_answers_generator.py"):
def create_test_answers_generator_script(output_path="test_answers_generator.py"):
    generator_code = """
import yaml
import pytest
import re
import test_utils as u
import argparse
from collections import defaultdict
from generator_utils import sanitize_function_name

with open('type_handlers.yaml') as f:
    type_handlers = yaml.safe_load(f)

def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        questions_data = yaml.safe_load(file)
    return questions_data

def get_decoded_str(questions_data, part, answer_key, source_file):
    # Ensure the encoded answer is properly escaped for inclusion in a double-quoted string
    # keys 'i_answer_source' and 's_answer_source' should be in yaml file
    if questions_data.get(answer_key+'_source', 'yaml_file') == source_file:
        encoded_answer_str = None
    else:
        # test that ['answer'] is in part
        #print("part= ", part)
        if answer_key in part and isinstance(part[answer_key], str):
            encoded_answer_str = part[answer_key].replace('"', '\"')
            #print(f"{encoded_answer_str=}")
        else:
            print(f"'part' should contain the key {repr(answer_key)} (str)")
            print(f"'part' read answer from the instructor/student file")
            encoded_answer_str = None   # <<< WRONG

        # Construct the call to decode_data as a string
        # encoded_answer_str not yet defined. 
        # So I need to handle multiple answers. 
        decode_call_str = f'''u.decode_data("{encoded_answer_str}")''' if encoded_answer_str else None
        #print(f"{decode_call_str=}")
        return decode_call_str

def evaluate_answers(question_id, test_code, is_fixture, is_instructor_file, is_student_file, 
                    decode_i_call_str, decode_s_call_str, fixture, part, function_name):
    if not is_fixture and not is_instructor_file:  # yaml file
        test_code += f"    correct_answer = eval('{decode_i_call_str}')\\n"
    elif not is_fixture and is_instructor_file:
        test_code +=  "    print('not is_fixture and is_instructor_file: not implemented')\\n"
        #test_code += f"    correct_answer = eval('{decode_call_str}')\\n"
    elif is_fixture and is_instructor_file:
        fixture_args = fixture['args']
        fixture_name = fixture['name']
        module_function_name = question_id   # name of function in student/instructor module
        part_id = f"{repr(part['id'])}"
        test_code += f"    correct_answer = {fixture_name}({repr(fixture_args[0])}, {repr(module_function_name)}, 'i')\\n"
        test_code += f"    if {part_id} not in correct_answer:\\n"
        explanation = repr(f"Key: {part_id} not found.\\n")  # Change in accordance to structure check
        test_code += f"        explanation = {explanation}\\n"
        test_code += f"        {function_name}.explanation = explanation\\n"
        test_code += f"        assert False\\n"
        test_code += f"    else:\\n"
        test_code += f"        correct_answer = correct_answer[{part_id}]\\n"
    else:  # fixture, yaml file
        test_code += f"    correct_answer = eval('{decode_i_call_str}')\\n"

    if not is_fixture and not is_student_file:  # yaml file
        test_code += f"    student_answer = eval('{decode_s_call_str}')\\n"
    elif not is_fixture and is_student_file:
        test_code +=  "    print('not is_fixture and is_student_file: not implemented')\\n"
        #test_code += f"    student_answer = eval('{decode_call_str}')\\n"
    elif is_fixture and is_student_file:
        fixture_args = fixture['args']
        #fixture_name = fixture['name']
        fixture_name = fixture['name']
        module_function_name = question_id   # name of function in student/instructor module
        part_id = f"{repr(part['id'])}"
        test_code += f"    student_answer = {fixture_name}({repr(fixture_args[0])}, {repr(module_function_name)}, 's')\\n"
        test_code += f"    if {part_id} not in student_answer:\\n"
        explanation = repr(f"Key: {part_id} not found.\\n")  # Change in accordance to structure check
        test_code += f"        explanation = {explanation}\\n"
        test_code += f"        {function_name}.explanation = explanation\\n"
        test_code += f"        assert False\\n"
        test_code += f"    else:\\n"
        test_code += f"        student_answer = student_answer[{part_id}]\\n"
    else:  # fixture, yaml file
        test_code += f"    student_answer = eval('{decode_s_call_str}')\\n"
    return test_code

with open("generator_config.yaml", "r") as f:
    config = yaml.safe_load(f)
# How to access an element of config dict and set to default value for non-existent key?
gen_config = config['test_answers']
assert_false = gen_config.get("assert_false", False)
fixture_import_file = gen_config.get("fixture_import_file", None)

types_list = [
    "dict[string,dict[str,list]]",
    "dict[string,Tuple[NDArray]]", 
    "dict[string,NDArray]", 
    "list[list[float]]", 
    "dict[string,set]",
    "explain_string", 
    "list[NDArray]", 
    "list[string]",
    "set[NDArray]", 
    "set[string]",
    "dendrogram", 
    "function",
    "set[set]",
    "NDArray", 
    "string",
    "float", 
    "bool",
    "dict",
    "int",
    #"float_range",
    #"eval_float",
    #"float",
    #"choice",
    #"set",
    #"list"
]

def generate_test_answers_code(questions_data, output_file='test_answers.py'):
    module_ = questions_data["module"]
    test_code = f'''
import pytest
from pytest_utils.decorators import max_score, visibility, hide_errors
import instructor_code_with_answers.{module_} as ic
from testing_utilities import assert_almost_equal
import assert_utilities  # <<< SHOULD be specified in config
from student_code_with_answers import *
import instructor_code_with_answers as sc
from {fixture_import_file} import *   
# Not clear why 'import conftest' does not work
import tests.conftest as c
import test_utils as u
import random
import numpy as np
import yaml
import re
import {fixture_import_file}

with open('type_handlers.yaml', 'r') as f:
    type_handlers = yaml.safe_load(f)
'''
    if 'max_score' in questions_data: 
        max_score = questions_data['max_score']
    else:
        max_score = 0.

    if 'fixtures' in questions_data: 
        fixture = questions_data['fixtures']['fixture']
        fixture_name = fixture['name']
        fixture_args = fixture['args']  # list of strings
    else:
        fixture = None

    for question in questions_data['questions']:
        if 'max_score' in question: 
            max_score_q = question['max_score']
        else:
            max_score_q = max_score

        if 'fixture' in question: 
            fixture = question['fixture']
            fixture_name = fixture['name']
            fixture_args = fixture['args']  # list of strings

        for part in question['parts']:
            part_id_sanitized = part['id'].replace(' ', '_').replace('(', '').replace(')', '').replace('|', '_').replace('=', '_')
            function_name = f"test_answers_{question['id']}_{part_id_sanitized}_{part['type']}"
            function_name = sanitize_function_name(function_name)

            decode_i_call_str = get_decoded_str(questions_data, part, 'i_answer', 'instructor_file')
            decode_s_call_str = get_decoded_str(questions_data, part, 's_answer', 'student_file')

            '''  # I am not sure this is required. Not in structure genreation
            # Ensure the encoded answer is properly escaped for inclusion in a double-quoted string
            if questions_data.get('answer_source', 'yaml_file') == "instructor_file":
                encoded_answer_str = None
            else:
                # test that ['answer'] is in part
                try:
                    encoded_answer_str = part['answer'].replace('"', '\\"')
                except KeyError as e:
                    print(f"Error for question {repr(part['id'])}:")   # <<<< why this happening. 
                    print("Question: Missing required key: 'answer' since answer_source is not 'instructor_file'")
                    encoded_answer_str = None
            '''

            ## Construct the call to decode_data as a string
            #decode_call_str = f'''u.decode_data("{encoded_answer_str}")'''

            test_code += f"\\n@max_score({max_score_q})\\n"
            test_code +=  "@hide_errors('')\\n"
            test_code +=  f"def {function_name}({fixture_name}):\\n"

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
            test_code = evaluate_answers(question['id'], test_code, is_fixture, is_instructor_file, is_student_file, 
                                         decode_i_call_str, decode_s_call_str, fixture, part, function_name)

            test_code += f"    print(f'{is_fixture=}, {is_student_file=}, {is_instructor_file=}')\\n"
            #test_code += f"    answer = student_answer\\n"

            if part['type'] in types_list: 
                import_file = f"type_handlers['types']['{part['type']}']['import']"
                part_type = repr(f"{part['type']}")
                tol = part.get('rel_tol', 0.001)
                test_code += f"    tol = {tol}\\n"
                ## Answer is the student answer
                # QUESTION: perhaps I should first check for structure, and if structure passes, then check answer. 
                # If structure does not pass, give a zero for the answer. 
                assertion_answer = eval(f"type_handlers['types']['{part['type']}']['assert_answer']")  # Only difference
                assertion_structure = eval(f"type_handlers['types']['{part['type']}']['assert_structure']")  # Only difference
                keys = part.get('keys', None) ### <<<< different: optional keys to consider (temporary)
                test_code += f"    keys = {keys}\\n"
                if eval(import_file):
                    test_code += f"    import {eval(import_file)}\\n"


                # Check structures
                test_code += f"    msg_structure = \\"{assertion_structure}\\"\\n"
                # Check answers
                test_code += f"    msg_answer = \\"{assertion_answer}\\"\\n"

                test_code +=  "    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}\\n"
                test_code +=  "    is_success, explanation_structure = eval(msg_structure, {'__builtins__':{}}, local_namespace)\\n"
                test_code +=  "    if is_success:\\n"
                test_code +=  "        is_success, explanation_answer    = eval(msg_answer,    {'__builtins__':{}}, local_namespace)\\n"
                test_code +=  "    else: \\n"
                test_code +=  "        explanation_answer = \\"\\" \\n"
                test_code +=  "    explanation = '\\\\n'.join(['Structure tests:', explanation_structure, 'Answer tests:', explanation_answer])\\n"
                test_code += f"    {function_name}.explanation = explanation\\n"
                test_code += f"    assert is_success\\n"


            else:
                test_code += f"    print('type {part['type']} NOT HANDLED!')\\n"

            if assert_false:
                test_code += f"    assert False\\n"

            test_code += f"\\n\\n"
            
    with open(f"tests/{output_file}", 'w') as file:
        file.write(test_code)


# Usage example:
# questions_data = load_yaml_file('path_to_questions_answers.yaml')
# generate_test_answers_code(questions_data)

# NEW
def main(yaml_name):
    questions_data = load_yaml_file(yaml_name) 
    generate_test_answers_code(questions_data, f"test_answers_{yaml_name[:-5]}.py")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Pass in the name of the input yaml file.")
    parser.add_argument("-y", "--yaml", help="Name of the yaml file", required=True)
    args = parser.parse_args()
    main(args.yaml)

"""

# ----------------------------------------------------------------------
    with open(output_path, "w") as file:
        file.write(generator_code)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    create_test_answers_generator_script()

