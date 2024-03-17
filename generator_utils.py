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

