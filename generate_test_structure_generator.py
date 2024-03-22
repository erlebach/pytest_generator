import yaml
import pytest


def create_test_structure_generator_script(output_path="test_structure_generator.py"):
    generator_code = """
import yaml
import pytest
from generator_utils import sanitize_function_name
import argparse
import test_utils as u

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

def evaluate_answers(questions_data, question_id, test_code, is_fixture, is_instructor_file, is_student_file, 
                    decode_i_call_str, decode_s_call_str, fixture, part, function_name):

    student_directory = questions_data['student_folder_name']
    instructor_directory = questions_data['instructor_folder_name']

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
        test_code += f"    kwargs = {{'student_directory': {repr(student_directory)} , 'instructor_directory': {repr(instructor_directory)}}}\\n"
        test_code += f"    correct_answer = {fixture_name}({repr(fixture_args[0])}, {repr(module_function_name)}, 'i', **kwargs)\\n"
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
        fixture_name = fixture['name']
        module_function_name = question_id   # name of function in student/instructor module
        part_id = f"{repr(part['id'])}"
        test_code += f"    student_answer = {fixture_name}({repr(fixture_args[0])}, {repr(module_function_name)}, 's', **kwargs)\\n"
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
gen_config = config['test_structure']
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
    "eval_float",
    "set[string]",
    "dendrogram", 
    "function",
    "set[set]",
    "integer",
    "NDArray", 
    "string",
    "float", 
    "dict",
    "bool",
    "int",
    #"float_range",
    #"choice"
    #"eval_float",  # no
    #"list", # no
]

def generate_test_structure_code(questions_data, output_file='test_structure.py'):
    module_ = questions_data["module"]
    test_code = "import pytest\\n"
    test_code += "import pytest\\n"
    test_code += "import assert_utilities\\n"
    test_code += "import numpy as np\\n"
    test_code += f"from {module_} import *\\n"
    test_code += "import yaml\\n"
    test_code += "import test_utils as u\\n"
    test_code += f"from {fixture_import_file} import *\\n"
    test_code += "from pytest_utils.decorators import max_score, visibility, partial_score, hide_errors\\n"
    test_code += "\\nwith open('type_handlers.yaml', 'r') as f:\\n"
    test_code += "    type_handlers = yaml.safe_load(f)\\n"

    if 'fixtures' in questions_data: 
        fixture = questions_data['fixtures']['fixture']
        fixture_name = fixture['name']
        fixture_args = fixture['args']  # list of strings
    else:
        fixture = None


    for question in questions_data['questions']:

        if 'fixture' in question: 
            fixture = question['fixture']
            fixture_name = fixture['name']
            fixture_args = fixture['args']  # list of strings

        for part in question['parts']:
            #print("===> part['type'] = ", part['type'])
            if 'fixture' in part: 
                fixture = part['fixture']
                fixture_name = fixture['name']
                fixture_args = fixture['args']  # list of strings

            part_id_sanitized = part['id'].replace(' ', '_').replace('(', '').replace(')', '').replace('|', '_').replace('=', '_')
            function_name = f"test_structure_{question['id']}_{part_id_sanitized}_{part['type']}()"
            function_name = sanitize_function_name(function_name)

            decode_i_call_str = get_decoded_str(questions_data, part, 'i_answer', 'instructor_file')
            decode_s_call_str = get_decoded_str(questions_data, part, 's_answer', 'student_file')

            # Hide assert errors
            test_code += f"\\n@hide_errors('')\\n"
            if fixture:
                test_code += f"def {function_name}({fixture_name}):\\n"
            else:
                test_code += f"def {function_name}():\\n"

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

            #test_code += f"    print(f'{is_fixture=}, {is_student_file=}, {is_instructor_file=}')\\n"
            test_code += f"    answer = student_answer\\n"

            #assertion = f"type_handlers['types']['{part['type']}']['assert'].format(answer_var='answer')"
            #struct_msg = f"type_handlers['types']['{part['type']}']['struct_msg'].format(answer_var='answer')"

            if part['type'] == 'float_range':
                min_value, max_value = part['range']
                # orig
                #error_msg = f"'{part['id']} not in range [{min_value}, {max_value}]'"
                error_msg = repr(f"{part['id']} not in range [{min_value}, {max_value}]")
                test_code += f"    error_msg = {error_msg}\\n"
                explanation = repr(f"{{error_msg}}\\n")  # Change in accordance to structure check
                # Using format helps avoid embedded f-strings
                test_code += f"    explanation = {explanation}.format(error_msg=error_msg)\\n"
                test_code += f"    {function_name}.explanation = explanation\\n"
                test_code += f"    assert isinstance(answer, float) and {min_value} <= answer <= {max_value}, {error_msg}\\n"

            elif part['type'] == 'choice':
                choices_repr = repr(part['options'])  # Safely represent the choices list as a string
                error_msg = f"'answer not in {choices_repr}'"
                test_code += f"    {function_name}.explanation = {error_msg!r}\\n"  # !r use repr() to escape special characters
                test_code += f"    assert answer in {choices_repr}, {error_msg}\\n"
                test_code += f"    print('type {choice} NOT YET HANDLED')\\n"

            elif part['type'] in types_list: 
                import_file = f"type_handlers['types']['{part['type']}']['import']"
                part_type = repr(f"{part['type']}")
                tol = part.get('rel_tol', 0.001)
                test_code += f"    tol = {tol}\\n"
                ## Answer is the student answer
                assertion = eval(f"type_handlers['types']['{part['type']}']['assert_structure']")  # Only difference
                keys = part.get('keys', None) ### <<<< different: optional keys to consider (temporary)
                test_code += f"    keys = {keys}\\n"

                if eval(import_file):
                    test_code += f"    import {eval(import_file)}\\n"

                test_code += f"    msg = \\"{assertion}\\"\\n"
                test_code +=  "    local_namespace={'array': np.array, 'assert_utilities': assert_utilities, 'student_answer': student_answer, 'instructor_answer': correct_answer, 'rel_tol':tol, 'keys':keys}\\n"

                local_vars_dict = part.get('local_vars_dict', None)
                if local_vars_dict:
                    test_code += f"    local_vars_dict = {local_vars_dict}\\n"
                    test_code +=  "    local_namespace['local_vars_dict'] = local_vars_dict\\n"

                # One of a finite number of choices for string type
                choices = part.get('choices', None)
                if not choices and (part['type'] == 'string' or part['type'] == 'str'):
                    choices = []

                if choices is not None:
                    test_code += f"    choices = {choices}\\n"
                    test_code +=  "    local_namespace['choices'] = choices\\n"

                test_code +=  "    is_success, explanation = eval(msg, {'__builtins__':{}}, local_namespace)\\n"
                test_code += f"    {function_name}.explanation = explanation\\n"
                test_code += f"    assert is_success\\n"

            else:
                test_code += f"    print('ELSE: NOT HANDLED, type: {part['type']}')\\n"
                assertion = f"type_handlers['types']['{part['type']}']['assert'].format(answer_var='answer')"
                #error_msg = repr(f"{part['id']} not in valid")
                # Do not add additional quotes if they are already present
                print("ELSE, part= ", part)
                error_msg = str(struct_msg)
                test_code += f"    error_msg = {error_msg}\\n"
                explanation = repr(f"{{error_msg}}\\n")  # Change in accordance to structure check
                test_code += f"    explanation = {explanation}.format(error_msg=error_msg)\\n"
                test_code += f"    {function_name}.explanation = explanation\\n"
                test_code += f"    assert eval({assertion}), {error_msg}\\n"

            ### WHAT IS THIS? 
#            if 'choices' in part:
#                part_choices = repr(part['choices'])
#                test_code += "    for el in answer:\\n"
#                # Join the choices into a single string with each choice quoted, separated by commas
#                choices_str = ', '.join([f"'{choice}'" for choice in part['choices']])
#                # Use the constructed choices_str in the assertion message
#                # TODO: return to an earlier solution using the f\\" techqnique. 
#                test_code += f"        assert el in {repr(part['choices'])}, f'Element {{repr(el)}} not in [{choices_str}]'\\n"
#                test_code +=  "    print('choices NOT HANDLED')\\n"

            if assert_false:
                test_code += f"    assert False\\n"

    with open(f"tests/{output_file}", 'w') as file:
        file.write(test_code)


# Usage example:
# questions_data = load_yaml_file('path_to_questions_answers.yaml')
# generate_test_structure_code(questions_data)

def main(yaml_name):
    questions_data = load_yaml_file(yaml_name) 
    generate_test_structure_code(questions_data, f"test_structure_{yaml_name[:-5]}.py")

# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Usage example:
    # >>> all_questions.yaml might change <<<
    #questions_data = load_yaml_file('all_questions.yaml')
    #questions_data = load_yaml_file('part4.yaml')  # <<<<  SHOULD be an argument
    #generate_test_structure_code(questions_data)

    parser = argparse.ArgumentParser(description="Pass in the name of the input yaml file.")
    parser.add_argument("-y", "--yaml", help="Name of the yaml file", required=True)
    args = parser.parse_args()
    main(args.yaml)

"""

    with open(output_path, "w") as file:
        file.write(generator_code)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    create_test_structure_generator_script()
