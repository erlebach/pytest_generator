
import yaml
import pytest
from sanitize import sanitize_function_name

with open('type_handlers.yaml') as f:
    type_handlers = yaml.safe_load(f)

def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        questions_data = yaml.safe_load(file)
    return questions_data

def handle_type(the_type):
    if the_type == 'float_range':
        part_type = 'float'
        value = 0.
    elif the_type == 'choice':
        part_type = 'string'
        value = ""
    elif the_type == 'list':
        part_type = 'list'
        value = []
    elif the_type == 'set':
        part_type = 'set'
        value = set()
    else:
        part_type = f"{the_type}"
        value = 0

    return part_type, value

def generate_test_structure_code(questions_data, output_file='test_structure.py'):
    test_code = '''import pytest
from all_questions import *
import yaml
with open('type_handlers.yaml', 'r') as f:
    type_handlers = yaml.safe_load(f)
'''

    for question in questions_data['questions']:
        test_code += "\n\n\n"
        test_code += "-----------------------------------------------------------\n"
        test_code += f"def {question['id']}():\n"
        test_code += "    all_answers = {}\n"

        # content of question
        for part in question['parts']:
            test_code += f"\n    # type: {part['type']}\n"
            part_type, value = handle_type(part['type'])
            test_code += f"    answer[{part['id']}] = {value}\n"

    with open("template_code.py", 'w') as file:
        file.write(test_code)


# Usage example:
# questions_data = load_yaml_file('path_to_questions_answers.yaml')
# generate_test_structure_code(questions_data)

def main():
    # Usage example:
    # >>> all_questions.yaml might change <<<
    questions_data = load_yaml_file('hw4.yaml')
    #questions_data = load_yaml_file('all_questions_hw3.yaml')
    generate_test_structure_code(questions_data)

# ----------------------------------------------------------------------
if __name__ == "__main__":
    main()
