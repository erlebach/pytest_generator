import yaml
import pprint


def generate_python_code_from_yaml(yaml_file_path):
    with open(yaml_file_path, "r") as file:
        questions_data = yaml.safe_load(file)
    print(questions_data)

    python_code = ""
    for question in questions_data["questions"]:
        print(f"{question=}")
        question_id = question["id"]
        python_code += f"def {question_id}():\n"
        python_code += "    answers = {}\n"
        for part in question["parts"]:
            part_id = part["id"]
            if part["type"] in ["choice", "string"]:
                default_value = '""'
            elif part["type"] in ["integer", "float", "float_range"]:
                default_value = str(part["answer"])
            python_code += f'    answers["{part_id}"] = {default_value}\n'
        python_code += "    return answers\n\n"

    return python_code


# Replace 'path_to_your_yaml_file.yaml' with the actual path to your YAML file
yaml_file_path = "hw4.yaml"
python_code = generate_python_code_from_yaml(yaml_file_path)

# Print or save the generated Python code
print(python_code)
# To save to a file, uncomment the following lines:
# with open('all_questions.py', 'w') as file:
#     file.write(python_code)
