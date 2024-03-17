import yaml
import pytest

def create_test_structure_generator_script(output_path="test_structure_generator.py"):
    generator_code = """
import yaml
import pytest
from generator_utils import sanitize_function_name

with open('type_handlers.yaml') as f:
    type_handlers = yaml.safe_load(f)

def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        questions_data = yaml.safe_load(file)
    return questions_data

def generate_test_structure_code(questions_data, output_file='test_structure.py'):
    test_code = '''\
import pytest
import yaml
with open('type_handlers.yaml', 'r') as f:
    type_handlers = yaml.safe_load(f)
from all_questions import *     ### Module with questions. Read in. 
'''

    for question in questions_data['questions']:
        for part in question['parts']:
            part_id_sanitized = part['id'].replace(' ', '_').replace('(', '').replace(')', '').replace('|', '_').replace('=', '_')
            function_name = f"test_structure_{question['id']}_{part_id_sanitized}_{part['type']}()"
            function_name = sanitize_function_name(function_name)
            test_code += f'''

def {function_name}():
    #answer = getattr({question['id']}(), '{part['id']}')
    answer = {question['id']}()['{part['id']}']
'''
            if part['type'] == 'float_range':
                min_value, max_value = part['range']
                test_code += f'''    assert isinstance(answer, float) and {min_value} <= answer <= {max_value}, '{part['id']} not in range [{min_value}, {max_value}]'\\n'''
            elif part['type'] == 'choice':
                choices = part['options']
                test_code += f'''    assert answer in {choices}, answer not in {choices} \\n'''
            else:
                assertion = f"type_handlers['types']['{part['type']}']['assert'].format(answer_var='answer')"
                test_code += f'''    assert eval({assertion}), 'Assertion for {part['id']} failed'\n'''

    with open("test_structure.py", 'w') as file:
        file.write(test_code)


# Usage example:
# questions_data = load_yaml_file('path_to_questions_answers.yaml')
# generate_test_structure_code(questions_data)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    # Usage example:
    # >>> all_questions.yaml might change <<<
    questions_data = load_yaml_file('all_questions.yaml')
    #questions_data = load_yaml_file('all_questions_hw3.yaml')
    generate_test_structure_code(questions_data)
"""

    with open(output_path, "w") as file:
        file.write(generator_code)


# ----------------------------------------------------------------------
if __name__ == "__main__":
    create_test_structure_generator_script()
