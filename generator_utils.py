import re

def sanitize_function_name(name):
    # Mapping of special characters to unique strings
    special_char_mapping = {
        '+': '_plus_',
        '-': '_minus_',
        '=': '_equals_',
        '>': '_gt_',
        '<': '_lt_',
        '*': '_star_',
        '/': '_slash_',
        '\\': '_backslash_',
        '.': '_dot_',
        ',': '_comma_',
        ':': '_colon_',
        ';': '_semicolon_',
        '!': '_excl_',
        '?': '_ques_',
        '@': '_at_',
        '#': '_hash_',
        '$': '_dollar_',
        '%': '_percent_',
        '^': '_caret_',
        '&': '_amp_',
        '|': '_pipe_',
        '~': '_tilde_',
        '`': '_backtick_',
        '[': '_lbrack_',
        ']': '_rbrack_',
        '{': '_lbrace_',
        '}': '_rbrace_',
        '(': '_lparen_',
        ')': '_rparen_',
        "'": '_squote_',
        '"': '_dquote_'
    }

    # Replace each special character using the mapping
    for char, replacement in special_char_mapping.items():
        name = name.replace(char, replacement)

    # Replace sequences of underscores with a single underscore
    name = re.sub(r'_+', '_', name)

    # Remove leading or trailing underscores
    name = name.strip('_')

    # Ensure the name starts with a letter or underscore
    if not re.match(r'^[a-zA-Z_]', name):
        name = '_' + name
    
    return name
#----------------------------------------------------------------------
def get_decoded_str(questions_data, part, answer_key, source_file):
    # Ensure the encoded answer is properly escaped for inclusion in a double-quoted string
    # keys 'i_answer_source' and 's_answer_source' should be in yaml file
    if questions_data.get(answer_key+'_source', 'yaml_file') == source_file:
        encoded_answer_str = None
    else:
        # test that ['answer'] is in part
        #print("part= ", part)
        if answer_key in part and isinstance(part[answer_key], str):
            encoded_answer_str = part[answer_key].replace('"', '"')
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
#----------------------------------------------------------------------
