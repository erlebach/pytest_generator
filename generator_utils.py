import re


def sanitize_function_name(name):
    # Mapping of special characters to unique strings
    special_char_mapping = {
        "+": "_plus_",
        "-": "_minus_",
        "=": "_equals_",
        ">": "_gt_",
        "<": "_lt_",
        "*": "_star_",
        "/": "_slash_",
        "\\": "_backslash_",
        ".": "_dot_",
        ",": "_comma_",
        ":": "_colon_",
        ";": "_semicolon_",
        "!": "_excl_",
        "?": "_ques_",
        "@": "_at_",
        "#": "_hash_",
        "$": "_dollar_",
        "%": "_percent_",
        "^": "_caret_",
        "&": "_amp_",
        "|": "_pipe_",
        "~": "_tilde_",
        "`": "_backtick_",
        "[": "_lbrack_",
        "]": "_rbrack_",
        "{": "_lbrace_",
        "}": "_rbrace_",
        "(": "_lparen_",
        ")": "_rparen_",
        "'": "_squote_",
        '"': "_dquote_",
    }

    # Replace each special character using the mapping
    for char, replacement in special_char_mapping.items():
        name = name.replace(char, replacement)

    # Replace sequences of underscores with a single underscore
    name = re.sub(r"_+", "_", name)

    # Remove leading or trailing underscores
    name = name.strip("_")

    # Ensure the name starts with a letter or underscore
    if not re.match(r"^[a-zA-Z_]", name):
        name = "_" + name

    return name


# ----------------------------------------------------------------------
def get_decoded_str(questions_data, part, answer_key, source_file):
    # Ensure the encoded answer is properly escaped for inclusion in a double-quoted string
    # keys 'i_answer_source' and 's_answer_source' should be in yaml file
    if questions_data.get(answer_key + "_source", "yaml_file") == source_file:
        encoded_answer_str = None
    else:
        # test that ['answer'] is in part
        # print("part= ", part)
        # print("get_decoded_str")
        # print("  answer_key: ", answer_key)
        # print("part: ", part)
        if answer_key in part and isinstance(part[answer_key], str):
            encoded_answer_str = part[answer_key].replace('"', '"')
            # print(f"{encoded_answer_str=}")
        else:
            print(f"'part' should contain the key {repr(answer_key)} (str)")
            print(f"'part' read answer from the instructor/student file")
            encoded_answer_str = None  # <<< WRONG

        # Construct the call to decode_data as a string
        # encoded_answer_str not yet defined.
        # So I need to handle multiple answers.
        decode_call_str = (
            f"""u.decode_data("{encoded_answer_str}")""" if encoded_answer_str else None
        )
        # print(f"{decode_call_str=}")
        return decode_call_str


# ----------------------------------------------------------------------
def evaluate_answers(
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
):
    student_directory = questions_data["student_folder_name"]
    instructor_directory = questions_data["instructor_folder_name"]
    part_id = part.get("id", None)
    if part_id is None:
        print("part_id is missing. Abort.")
    # If no type, use default type (often float)
    part_type = part.get("type")
    part_question_id = question.get("id", None)
    question_id = part_id = part["id"]

    if not is_fixture and not is_instructor_file:  # yaml file
        test_code += f"    correct_answer = eval('{decode_i_call_str}')\n"
    elif not is_fixture and is_instructor_file:
        test_code += "    print('not is_fixture and is_instructor_file: not implemented')\n"
        # test_code += f"    correct_answer = eval('{decode_call_str}')\n"
    elif is_fixture and is_instructor_file:
        fixture_args = fixture["args"]
        fixture_name = fixture["name"]
        module_function_name = part_question_id  # name of function in student/instructor module
        # print(f"{module_function_name=}")
        part_id = f"{repr(part['id'])}"
        test_code += f"    kwargs = {{'student_directory': {repr(student_directory)} , 'instructor_directory': {repr(instructor_directory)}}}\n"
        # 2024-03-19
        # I should be able to generalize this so that the first argument is *fixture_args, which would allow fixtures with
        # either no args or multiple args before module_function_name, 'i', and **kwargs
        # Orig
        # test_code += f"    correct_answer = {fixture_name}({repr(fixture_args[0])}, {repr(module_function_name)}, 'i', **kwargs)\n"
        # New patch mechanism
        # print(f"fixture_args= {fixture_args}")
        print(f"{fixture_name=}, {type(fixture_name)=}")
        print(f"{fixture_args[0]=}, {type(fixture_args[0])=}")
        print(f"==> correct_answer = {fixture_name}({fixture_args[0]}, {repr(module_function_name)}, 'i', **kwargs)\n")
        # 2025-01-06: added quotes around fixture_args[0]
        # 2025-01-06: the 2nd argument to {fixture_name} is the question name as a string
        # print(f"{module_function_name=}")
        # print(f"    correct_answer = {fixture_name}({fixture_args[0]}, {repr(module_function_name)}, 'i', **kwargs)\n")
        test_code += f"    correct_answer = {fixture_name}({fixture_args[0]}, {repr(module_function_name)}, 'i', **kwargs)\n"
        test_code += f"    if {part_id} not in correct_answer:\n"
        explanation = repr(
            f"Key: {part_id} not found in instructor answer!\n"
        )  # Change in accordance to structure check
        test_code += f"        explanation = {explanation}\n"
        test_code += f"        {function_name}.explanation = explanation\n"
        test_code += f"        answer_type = {repr(part_type)}\n"
        test_code += f"        question_id = {repr(part_question_id)}\n"
        test_code += f"        subquestion_id = {part_id}\n"
        test_code += f"        {function_name}.answer_type = answer_type\n"
        test_code += f"        {function_name}.question_id = question_id\n"
        test_code += f"        {function_name}.subquestion_id = subquestion_id\n"
        test_code += f"        assert False\n"
        test_code += f"    else:\n"
        test_code += f"        correct_answer = correct_answer[{part_id}]\n"
    else:  # fixture, yaml file
        test_code += f"    correct_answer = eval('{decode_i_call_str}')\n"

    if not is_fixture and not is_student_file:  # yaml file
        test_code += f"    student_answer = eval('{decode_s_call_str}')\n"
    elif not is_fixture and is_student_file:
        test_code += "    print('not is_fixture and is_student_file: not implemented')\n"
        # test_code += f"    student_answer = eval('{decode_call_str}')\n"
    elif is_fixture and is_student_file:
        fixture_args = fixture["args"]
        # fixture_name = fixture['name']
        fixture_name = fixture["name"]
        module_function_name = part_question_id  # name of function in student/instructor module
        part_id = f"{repr(part['id'])}"
        # Original code
        # test_code += f"    student_answer = {fixture_name}({repr(fixture_args[0])}, {repr(module_function_name)}, 's', **kwargs)\n"
        # New patch mechanism
        test_code += f"    student_answer = {fixture_name}({fixture_args[0]}, {repr(module_function_name)}, 's', **kwargs)\n"
        test_code += f"    if {part_id} not in student_answer:\n"
        explanation = repr(
            f"Key: {part_id} not found in student answer!\n"
        )  # Change in accordance to structure check
        test_code += f"        explanation = {explanation}\n"
        test_code += f"        {function_name}.explanation = explanation\n"
        test_code += f"        answer_type = {repr(part_type)}\n"
        test_code += f"        question_id = {repr(part_question_id)}\n"
        test_code += f"        subquestion_id = {part_id}\n"
        test_code += f"        {function_name}.answer_type = answer_type\n"
        test_code += f"        {function_name}.question_id = question_id\n"
        test_code += f"        {function_name}.subquestion_id = subquestion_id\n"
        test_code += f"        assert False\n"
        test_code += f"    else:\n"
        test_code += f"        student_answer = student_answer[{part_id}]\n"
    else:  # fixture, yaml file
        test_code += f"    student_answer = eval('{decode_s_call_str}')\n"
    return test_code


# ----------------------------------------------------------------------
