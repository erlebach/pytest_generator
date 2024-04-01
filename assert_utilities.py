#" assert_utilities.py

import ast
import yaml
import re
import numpy as np
import inspect  # <<<<
import random
import math

# ......................................................................

def is_explain(answer):
    """
    Is 'explain' in the cleaned answer string?
    Not currently used. 
    """
    return 'explain' in clean_str_answer(answer)

# ......................................................................

def clean_str_answer(answer):
    answer =  answer.lower().strip()
    # Transform double spaces to single space
    answer =  re.sub(r'\s+', ' ', answer)
    return answer

# ......................................................................

def load_yaml_file(file_path):
    with open(file_path, 'r') as file:
        questions_data = yaml.safe_load(file)
    return questions_data

config_dict = load_yaml_file("generator_config.yaml")
max_nb_words = config_dict["test_structure"]["max_nb_words"]


#----------------------------------------------------------------------
def fmt_ifstr(x):
    return repr(x) if isinstance(x, str) else str(x)
#----------------------------------------------------------------------
def return_value(status, msg_list, s_answ, i_answ):
    """
    Used when checking the answer for a question,
    as opposed to the structure.
    """
    if status:
        msg_list.append("Answer is correct")
    else:
        msg_list.append("Answer is incorrect.")
    msg_list.append(f"Instructor answer: {fmt_ifstr(i_answ)}")
    msg_list.append(f"Student answer: {fmt_ifstr(s_answ)}")

    return status, "\n".join(msg_list)

#----------------------------------------------------------------------
def are_sets_equal(set1, set2, rtol=1e-5, atol=1e-6):
  """
  Compares two sets of floats for equality within a relative and absolute tolerance.

  Args:
      set1: The first set of floats.
      set2: The second set of floats.
      rtol: The relative tolerance (default: 1e-5).
      atol: The absolute tolerance (default: 1e-8).

  Returns:
      True if the sets are equal within the specified tolerances, False otherwise.
  """
  if len(set1) != len(set2):
    return False  # Sets must have the same length
  for x, y in zip(set1, set2):
    if not np.isclose(x, y, rtol=rtol, atol=atol):
      return False
  return True

# ======================================================================
def check_answer_float(student_answer, instructor_answer, rel_tol):
    """
    Check answer correctness. Assume the structure is correct.
    """
    abs_err = math.fabs(student_answer - instructor_answer)
    is_abs = False  # irrelevant if rel error criteria is satisfied

    if math.fabs(instructor_answer) > 1.e-8:
        is_rel = (student_answer - instructor_answer) / instructor_answer < rel_tol
    else:
        is_abs = abs_err < 1.e-8
        is_rel = False

    if is_rel:
        status = True
        msg_list = [f"Answer is correct to within relative error: {100*rel_tol}%"]
    elif is_abs:
        status = True
        msg_list = [f"Answer is correct to within absolute error: {1.e-8}"]
    else:
        status = False
        msg_list = [f"Answer is incorrect"]
    return return_value(status, msg_list, student_answer, instructor_answer)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def check_structure_float(student_answer, instructor_answer):
    if isinstance(student_answer, float):
        status = True
        msg_list = ["Answer is of type float as expected."]
    else:
        status = False
        msg_list = [f"Answer should be of type float. It is of type {type(student_answer)}"]
    return status, "\n".join(msg_list)
# ======================================================================
def check_answer_eval_float(student_answer, instructor_answer, local_vars_dict, rel_tol):
    msg_list = []
    status = True
    s_answ = student_answer
    i_answ = instructor_answer
    #s_answ = s_answ.replace('^', '**')
    #s_answ = s_answ.replace('x', '*')
    #s_answ = s_answ.replace('X', '*')
    #i_answ = i_answ.replace('^', '**')
    random_values = {}
    local_dct = {}

    for i, _ in enumerate(range(3)):
        for var, (lower, upper) in local_vars_dict.items():
            random_values[var] = random.uniform(lower, upper)
            local_dct[var] = random_values[var]
        s_float = eval(s_answ, {}, local_dct)
        i_float = eval(i_answ, {}, local_dct)
        if math.fabs(i_float) < 1.e-5:
            abs_err = math.fabs(i_float - s_float)
            ae_status = abs_err < 1.e-5
            status *= ae_status
            if not ae_status:
                msg_list += ["Absolute error > 1.e-5"]
        else:
            rel_err = math.fabs((s_float - i_float) / i_float)
            re_status = rel_err < rel_tol
            status *= re_status
            if not re_status:
                msg_list += [f"Relative error > {rel_tol}"]
            else:
                msg_list += [f"Relative error < {rel_tol}"]
        
    return return_value(status, msg_list, s_answ, i_answ)

# ======================================================================
def check_structure_eval_float(student_answer, instructor_answer, local_vars_dict, rel_tol):
    print("\nENTER check_structure_eval_float")
    print("student_answer= ", student_answer)
    if not isinstance(student_answer, str):
        return False, "Student_answer is {student_answer}. Should be string defining a valid Python expression."

    try:
        ast.parse(student_answer, mode='eval')
        return True, "Valid python expression"
    except SyntaxError:
        print("===> expression is false")
        return False, "Your valid expression is not valid Python"

# ======================================================================

def check_answer_dict_string_dict_str_list(student_answer, instructor_answer):
    """
    The type is a dict[str, dict[str, list]]
    """
    if not isinstance(student_answer, dict):
        return False

    for k, v in student_answer.items():
        if not (isinstance(k, str) and isinstance(v, dict)):
            return False

        for inner_k, inner_v in v.items():
            if not (isinstance(inner_k, str) and isinstance(inner_v, list)):
                return False

    return True

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def check_structure_dict_string_dict_str_list(student_answer, instructor_answer):
    msg_list = []
    status = True
    i_ans = instructor_answer
    s_ans = student_answer

    missing_keys = set(i_ans.keys()) - set(s_ans.keys())
    if len(missing_keys) > 0: 
        return False, f"- Missing keys: {[repr(k) for k in missing_keys]}."
    
    for k, v in instructor_answer.items():
        if not isinstance(v, dict): 
            msg_list.append(f"- answer[{repr(k)}] must be of type 'dict'")
            status *= False
            continue
        # v is a dict
        for kk, vv in v.items():
            if not (isinstance(kk, str) and isinstance(vv, list)): 
                msg_list.append(f"- answer[{repr(k)}] must have keys of type 'str' and values of type 'list'")
                status *= False

    if status == True:
        msg_list.append("Type 'dict[str, dict[str, list]' is correct.")

    return status, "\n".join(msg_list)

# ======================================================================

def check_answer_dict(student_answer, instructor_answer):
    # This function must be thoroughly debugged. I don't trust it. 
    """
    Uses recursive calls. Can return status, but how to return msg_list?
    """
    msg_list = []
    s_dict = student_answer
    i_dict = instructor_answer
    """
    Compares two generic dictionaries for equality down to two levels.
    ### NOT TESTED. Written by GPT-4, modified by GE

    Args:
        i_dict = dict1: The first dictionary.
        s_dict = dict2: The second dictionary.

    Returns:
        True if the dictionaries are equal down to two levels, False otherwise.
    """

    # Check the top-level keys match
    if set(i_dict.keys()) != set(dicts.keys()):
        return False

    # Iterate through keys and compare values
    for key in i_dict:
        # If both values are dictionaries, compare their keys and values
        if isinstance(i_dict[key], dict) and isinstance(s_dict[key], dict):
            if not check_answer_dict(s_dict[key], i_dict[key]):
                return False
        else:
            # If the values are not both dictionaries, directly compare them
            if i_dict[key] != s_dict[key]:
                return False

    return return_value(status, msg_list, s_answ, i_answ)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def check_structure_dict(student_answer, instructor_answer):
    """
    Checks if the key structure of two dictionaries matches down to two levels,
    with the second dictionary (i_dict) considered as the gold standard.

    Args:
        s_dict: The first dictionary (student-generated).
        i_dict: The second dictionary (instructor-generated, gold standard).

    Returns:
        True if the key structures match down to two levels, False otherwise.
    """

    #print("==> enter dict struct check")
    msg_list = []
    s_answ = student_answer
    i_answ = instructor_answer

    if not isinstance(s_answ, dict): 
        return False, f"- Answer must be of type 'dict'"

    # Check the top-level keys match
    missing_keys = set(s_answ.keys()) - set(i_answ.keys())
    if set(s_answ.keys()) != set(i_answ.keys()):
        if len(missing_keys) > 0: 
            return False, f"- Missing keys: {[repr(k) for k in missing_keys]}."

    # Iterate through keys and check structures
    for key in i_answ:
        # If both values are dictionaries, compare their key sets
        if isinstance(s_answ.get(key), dict) and isinstance(i_answ[key], dict):
            ret = check_key_structure(s_dict[key], i_dict[key])
            if not ret[0]:
                return ret
        elif isinstance(s_answ.get(key), dict) != isinstance(i_answ[key], dict):
            # One is a dict and the other is not, key structure does not match
            return False, "- Mismatch! Student and instructor answers are of different types."

    return True, "The dictionary elements types matches that of the instructor"

# ======================================================================

def check_answer_string(student_answer, instructor_answer):
    msg_list = []
    s_answ = clean_str_answer(student_answer)
    i_answ = clean_str_answer(instructor_answer)

    status = True if s_answ == i_answ else False
    msg_list.append("Strings are lowered and stripped")
    return return_value(status, msg_list, s_answ, i_answ)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

# MUST FIX
def check_structure_string(student_answer, instructor_answer, choices):
    """
    choices: list of strings
    """
    status = True
    msg_list = []
    choices = [clean_str_answer(c) for c in choices]

    # Ideally, should be done when yaml file is preprocessed
    # All strings should be lowered at that time. 
    #choices = [clean_str_answer(s) for s in choices] 

    if not isinstance(student_answer, str):
        status = False
        msg_list +=["- Answer must be of type 'str'"]
    else:
        msg_list += ["- type 'str' is correct"]
        # clean choices (lower, strip, '  ' -> ' ')
        student_answer = clean_str_answer(student_answer)

    if status and choices != []:
        if not student_answer in choices: 
            status = False
            msg_list += [f"- Answer must be one of {choices}"]
        else:
            msg_list += [f"- Answer {repr(student_answer)} is among the valid choices"]

    print("\n".join(msg_list))
    return status, "\n".join(msg_list)

# ======================================================================

def check_answer_explain_string(student_answer, instructor_answer):   
    msg_list = []
    status = True
    return return_value(status, msg_list, student_answer, instructor_answer)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def check_structure_explain_string(student_answer, instructor_answer):   
    """
    The type is an explain_string
    The string should have a minimum number of words stored in "type_handlers.yaml"
    """
    msg_list = []
    status = True
    if isinstance(student_answer, str):
        status = True
        msg_list.append("- Type 'str' is correct")
    else: 
        status = False
        msg_list.append("- Type must be of type 'str'")

    # Check the number of words in the string
    max_nb_words = 4  # WHERE IS THIS SET. Should be in configuration file. 

    if status:
        is_nb_words = len(student_answer.split()) >= max_nb_words
        if is_nb_words:
            msg_list.append("- Length of answer is sufficient")
        else:
            status = False
            msg_list.append(f"- Length of answer must be > {max_nb_words}")

    return status, "\n".join(msg_list)

# ======================================================================

# SOMETHING WRONG. 
'''
def check_answer_set_string(student_answer, instructor_answer, choices=None):
    """
    s_answ: student answer: set of strings
    i_answ: instructor answer: set of strings
    """
    msg_list = []
    status = True
    s_answ = {i.lower().strip() for i in student_answer}
    i_answ = {i.lower().strip() for i in instructor_answer}
    status = True if s_answ == i_answ else False

    msg_list.append("Strings are lower-cased on stripped")
    return return_value(status, msg_list, s_answ, i_answ)
'''
def check_answer_set_string(student_answer, instructor_answer, choices=None):
    """
    s_answ: student answer: set of strings
    i_answ: instructor answer: set of strings
    choices: one of several choices
    """
    msg_list = []
    status = True

    #print("===> set_string, choices: ", choices)

    s_answ = {i.lower().strip() for i in student_answer}
    i_answ = {i.lower().strip() for i in instructor_answer}

    if choices and isinstance(choices[0], list):
        #print("===> choices: ", choices)
        choices = [set(c) for c in choices]

    #print("===> set_string, after set, choices: ", choices)

    if choices and isinstance(choices[0], set):
        #print("===> choices= ", choices)
        for i, a_set in enumerate(choices):
            choices[i] = {el.lower().strip() for el in a_set}

    if choices and isinstance(choices[0], set):
        status = True if s_answ in choices else False
    else:
        status = s_answ == i_answ

    msg_list.append("Strings are lower-cased and stripped")
    if choices and isinstance(choices[0], set):
        msg_list.append(f"Student answer is one of {choices}")
    return return_value(status, msg_list, s_answ, i_answ)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def check_structure_set_string(student_answer, instructor_answer):
    """
    """
    msg_list = []
    status = True

    if isinstance(student_answer, set) or isinstance(student_answer, list):
        status = True
        msg_list.append("- Type is either 'list' or 'set' (correct).")
    else:
        status = False
        msg_list.append("- Answer must be of type 'set' or 'list'.")

    if status:
        are_all_str = True
        for s in student_answer:
            if not isinstance(s, str):
                msg_list.append("- Set element {repr(s)} must be of type 'str'")
                status = False
                are_all_str = False

    if are_all_str:
        msg_list.append("- All elements are 'str', as required")
        status = True

    return status, "\n".join(msg_list)

# ======================================================================

def check_answer_dict_string_set(student_answer, instructor_answer):
    """
    student answer: dictionary with keys:str, values: a set of objects
    instructor answer: dictionary with keys:str, values: a set of objects

    Even if the student returns a dict[str] = list, the list is cast to a set
    """
    msg_list = []
    s_answ = student_answer
    i_answ = instructor_answer
    status = True
    for k in instructor_answer.keys():
        s_val = student_answer[k]
        i_val = instructor_answer[k]
        status *= set(s_val) == set(i_val)

    return return_value(status, msg_list, student_answer, instructor_answer)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def check_structure_dict_string_set(student_answer, instructor_answer):
    """
    TODO: provide a list of keys to check as an argument keys (default None)
    """
    status = True
    msg_list = []

    if not isinstance(student_answer, dict):
        msg_list.append("- Answer should be of type 'dict'.")
    else:
        msg_list.append("- Type 'dict' is correct")

    keys = set(instructor_answer.keys())
    student_keys = set(student_answer.keys())
    missing_keys = keys - student_keys

    if status:
        if len(missing_keys) > 0:
            msg_list.append(f"- Missing keys: {[repr(k) for k in missing_keys]}.")
            status = False
        else:
            msg_list.append(f"- No missing keys")

    if status:
        is_item_type_list = True
        for k, v in student_answer.items():
            if k in keys:
                if not isinstance(v, (set, list)): 
                    msg_list.append(f"- Answer[{repr(k)}] must be of type 'set' or 'list'.")
                    # The answer is cast to a set when checked for accuracy
                    status = False
                    is_item_type_list = False

        if is_item_type_list:
            msg_list.append("- All list elements are of type 'list' or 'set' as expected")
            status = True

    return status, "\n".join(msg_list)

# ======================================================================

def check_answer_dict_string_float(student_answer, instructor_answer, rel_tol, keys):
    """
    student answer: dictionary with keys:str, values: an NDArray
    instructor answer: dictionary with keys:str, values: a set of objects
    rel_tol: tolerance on the matrix norm
    keys: None if all keys should be considered
    """
    msg_list = []
    status = True
    keys = list(instructor_answer.keys()) if keys == None else keys

    # Need an exception in case the student key is not found
    for k in keys:
        s_float = student_answer[k]
        i_float = instructor_answer[k]
        if math.fabs(i_float) < 1.e-6:
            if math.fabs(s_flaot) > 1.e-6:
                status = False
            else:
                status = True
        else:
            rel_err = math.fabs(s_float - i_float) / math.fabs(i_float)
            status *= rel_err < rel_tol

    return return_value(status, msg_list, student_answer, instructor_answer)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def check_structure_dict_string_float(student_answer, instructor_answer, rel_tol, keys=None):
    """
    student answer: dictionary with keys:str, values: float
    instructor answer: dictionary with keys:str, values: a set of objects
    rel_tol: tolerance on the matrix norm
    keys: None if all keys should be considered
    """
    status = True
    msg_list = []

    if not isinstance(instructor_answer, dict):
        msg_list += ["Instructor answer should be a dict"]
        status = False

    if status and not isinstance(student_answer, dict):
        msg_list += ["Student answer should be a dict"]
        status = False

    if status:
        keys = keys if keys else list(instructor_answer.keys())
        instructor_keys = set(keys)
        instructor_answer = {k: v for k, v in instructor_answer.items() if k in keys}
        student_keys = set(student_answer.keys())
        missing_keys = list(instructor_keys - student_keys)

        if len(missing_keys) > 0:
            msg_list.append(f"- Missing keys: {[repr(k) for k in missing_keys]}.")
            status = False
        else:
            msg_list.append(f"- No missing keys.")

    if status:
        # some keys are filtered. Student is allowed to have 
        # keys not in the instructor set
        for k, v in instructor_answer.items():
            vs = student_answer[k]
            if not isinstance(vs, float):
                msg_list.append(f"- answer[{repr(k)}] should be a float.")
                status = False

        if status:
            msg_list.append(f"- All elements are of type float as expected.")

    if status:
        msg_list.append(f"Type 'dict[str, float]' is correct")

    return status, "\n".join(msg_list)


# ======================================================================

def check_answer_dict_string_NDArray(student_answer, instructor_answer, rel_tol, keys):
    """
    student answer: dictionary with keys:str, values: an NDArray
    instructor answer: dictionary with keys:str, values: a set of objects
    rel_tol: tolerance on the matrix norm
    keys: None if all keys should be considered
    """
    #print("check_answer_dict_string_NDArray")
    msg_list = []
    status = True
    i_dict_norm = {}
    s_dict_norm = {}
    keys = list(instructor_answer.keys()) if keys == None else keys

    # Need an exception in case the student key is not found
    for k in keys:
        s_arr = student_answer[k]
        i_arr = instructor_answer[k]
        s_norm = np.linalg.norm(s_arr)
        i_norm = np.linalg.norm(i_arr)
        i_dict_norm[k] = i_norm
        s_dict_norm[k] = s_norm
        rel_err = math.fabs(s_norm - i_norm) / math.fabs(i_norm)
        status *= rel_err < rel_tol

    return return_value(status, msg_list, student_answer, instructor_answer)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def check_structure_dict_string_NDArray(student_answer, instructor_answer, rel_tol, keys=None):
    """
    student answer: dictionary with keys:str, values: an NDArray
    instructor answer: dictionary with keys:str, values: a set of objects
    rel_tol: tolerance on the matrix norm
    keys: None if all keys should be considered
    """
    #print("check_structure_dict_string_NDArray")
    status = True
    msg_list = []

    #print(f"arg: {keys=}")
    print(f"===> {instructor_answer=}")
    print(f"===> {student_answer=}")

    if not isinstance(instructor_answer, dict):
        msg_list += ["Instructor answer should be a dict"]
        status = False

    if status and not isinstance(student_answer, dict):
        msg_list += ["Student answer should be a dict"]
        status = False

    if status:
        keys = keys if keys else list(instructor_answer.keys())
        instructor_keys = set(keys)
        #print(f"{instructor_keys=}")
        instructor_answer = {k: v for k, v in instructor_answer.items() if k in keys}
        student_keys = set(student_answer.keys())
        missing_keys = list(instructor_keys - student_keys)

        if len(missing_keys) > 0:
            msg_list.append(f"- Missing keys: {[repr(k) for k in missing_keys]}.")
            status = False
        else:
            msg_list.append(f"- No missing keys.")

    if status:
        # some keys are filtered. Student is allowed to have 
        # keys not in the instructor set
        for k, v in instructor_answer.items():
            #print("==> key: ", k)
            vs = student_answer[k]
            if not isinstance(vs, type(np.zeros(1))): 
                msg_list.append(f"- answer[{repr(k)}] should be a numpy array.")
                status = False

        if status:
            msg_list.append(f"- All elements are of type NDArray as expected.")

    if status:
        msg_list.append(f"Type 'dict[str, NDArray]' is correct")

    return status, "\n".join(msg_list)

# ======================================================================

def check_answer_list_NDArray(student_answer, instructor_answer, rel_tol):
    """
    tol: max relative error on the L2 norm
    Check that all elements in the list have matching norms
    """
    msg_list = []
    status = True
    answ_eq_len = len(student_answer) == len(instructor_answer)
    i_norm_list = []
    s_norm_list = []

    if answ_eq_len:
        for s_arr, i_arr in zip(student_answer, instructor_answer):
            s_norm = np.linalg.norm(s_arr)
            i_norm = np.linalg.norm(i_arr)
            i_norm_list.append(i_norm)
            s_norm_list.append(s_norm)
            rel_error = (s_norm - i_norm) / i_norm
            status *= rel_error < rel_tol

    if not status: 
        msg_list.append("Replace the arrays by their norms")

    return return_value(status, msg_list, s_norm_list, i_norm_list)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def check_structure_list_NDArray(student_answer, instructor_answer):
    """
    Check that elements in the list are NDArrays
    """
    #print("==>  structure check_structure_list_NDArray....")
    status = True
    msg_list = []

    if not isinstance(student_answer, list):
        status = False
        msg_list.append(f"- The answer should be of type 'list'; your type is {type(student_answer)}")
    else:
        msg_list.append("- The answer is type list. Correct.")

    # Check length of list
    if status:
        if len(student_answer) != len(instructor_answer):
            status = False
            msg_list.append(f"- The length of your list is incorrect. Your list length is {len(student_answer)}. The length should be {len(instructor_answer)}.")
        else: 
            msg_list.append("- The length of the list is correct.")

    if status: 
        for i, s_arr in enumerate(instructor_answer):
            if not isinstance(s_arr, type(np.zeros(1))): 
                status = False
                msg_list.append("- Element {i} of your list should be of type 'numpy.array'.")
        
    if status:
        msg_list.append("- All list elements are type NDArray.")

    return status, "\n".join(msg_list)

# ======================================================================

# <<<<<<< NOT IN type_handlers >>>>>>
def check_answer_set_NDArray(student_answer, instructor_answer, rel_tol):
    """
    tol: max relative error on the L2 norm
    Check that all elements in the list have matching norms. Order does not 
    matter. So create a set of norms and match them. 
    """
    msg_list = []
    s_answ = list(student_answer)
    i_answ = list(instructor_answer)

    for i, arr in enumerate(s_answ):
        s_answ[i] = np.linalg.norm(arr)
    for i, arr in enumerate(i_answ):
        i_answ[i] = np.linalg.norm(arr)

    s_answ = set(s_answ)
    i_answ = set(i_answ)

    status = are_sets_equal(s_answ, i_answ, rtol=tol)
    if not status:
        msg_list.append("The arrays can be in any order")

    return return_value(status, msg_list, s_answ, i_answ)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def check_structure_set_NDArray(student_answer, instructor_answer):
    return True

# ======================================================================

def check_answer_NDArray(student_answer, instructor_answer, tol):
    """
    tol: max relative error on the L2 norm
    Check that all elements in the list have matching norms
    """
    msg_list = []
    status = True
    s_norm = np.linalg.norm(student_answer)
    i_norm = np.linalg.norm(instructor_answer)

    # Can i_norm be zero?
    status *= (s_norm - i_norm) / i_norm <= tol
    if not status:
        print("Replace the array by its norm")

    return return_value(status, msg_list, s_norm, i_norm)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def check_structure_NDArray(student_answer, instructor_answer):
    """
    Check that all elements in the list have matching norms
    instructor_answer: not used
    """
    if not isinstance(student_answer, type(np.zeros([1]))):
        return False, "- Answer should be a numpy array." 
    return True, "Type 'NDArray' is correct."

# ======================================================================

def check_answer_function(student_answer, instructor_answer):
    """
    Student and instructor functions. I will print out the source. 
    Ideally, we'd check the arguments, and check the execution of the 
    function. 
    """
    s_source = inspect.getsource(student_answer)
    i_source = inspect.getsource(instructor_answer)

    msg_list = []
    msg_list.append("Functions are not graded, unless not present.")
    msg_list.append("Instructor function Source")
    msg_list.append(s_source)

    status = True
    return return_value(status, msg_list, s_source, i_source)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def check_structure_function(student_answer, instructor_answer):
    print(f"==> inside function: {type(student_answer)=}")
    if not isinstance(student_answer, type(lambda: None)):
        return False, "- Answer should be a Python function."
    return True, "Type 'function' is correct."

# ======================================================================

def check_answer_list_list_float(student_answer, instructor_answer, tol):
    """
    Check two lists of lists of floats with each other
    """
    msg_list = []

    lg_eq = True
    status = True

    for s_lst, i_lst in zip(student_answer, instructor_answer):
        lg_eq *= len(s_lst) == len(i_lst)
        for i, el in enumerate(s_lst):
            if (i_lst[i] != 0.):
                rel_err = math.fabs(s_lst[i] - i_lst[i]) / math.fabs(i_lst[i])
            else:
                rel_err = math.fabs(s_lst[i] - i_lst[i]) 

            status *= rel_err < tol

    msg_list.append(f"Answer correct if relative error < {tol*100} percent")
    return return_value(status, msg_list, student_answer, instructor_answer)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def check_structure_list_list_float(student_answer, instructor_answer):
    """
    Check structure of student_answer. 
    instructor_answer: not used
    """
    msg_list = []
    status = True

    if not isinstance(student_answer, list):
        msg_list.append("- answer should be a list.")
        status = False
        return status, "\n".join(msg_list)

    if not len(student_answer) == len(instructor_answer):
        msg_list.append("- Number of elements in the answer is incorrect.")
        status = False

    for i, s_list in enumerate(student_answer):
        if not isinstance(s_list, list):
            msg_list.append(f"- answer[{i}] is not a list. Recheck all elements.")
            status = False
            continue

        for j, el in enumerate(s_list):
            if not isinstance(float(el), float): 
                msg_list.append(f"- answer[{i}][{j}] cannot be cast to a float. All elements must be castable to float." )
                status = False

    if status:
        msg_list.append("Type 'list[list[float]]' is correct.")

    return status, "\n".join(msg_list)

# ======================================================================

def check_answer_list_set(student_answer, instructor_answer):
    """
    Check two lists of sets (although the set is encoded as a list) with each other

    Both answers are list of sets or lists of list
    """
    msg_list = []

    for s_lst, i_lst in zip(student_answer, instructor_answer):
        s_set = set(s_lst)
        i_set = set(i_lst)
        is_true *= i_set == s_set

    msg_list.append("Answer is a list of sets; sets can be replaced by lists or tuples.")
    return return_value(status, msg_list, student_answer, instructor_answer)

# ======================================================================

def convert_to_set_of_sets(input_sequence):
    """
    Convert each inner sequence to a set, then the outer sequence to a set of sets
    """
    return {frozenset(inner) for inner in input_sequence}

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def check_answer_set_set(student_answer, instructor_answer):
    """
    Both student answer and instructor answer should be a set of sets or a structure
    that can be converted to a set of setsr 
    """
    msg_list = []
    seq_s = student_answer
    seq_i = instructor_answer
    # Convert both sequences to sets of sets
    # One might start with a list of lists [of objects]
    set_of_sets_s = convert_to_set_of_sets(seq_s)
    set_of_sets_i = convert_to_set_of_sets(seq_i)

    # Compare the sets of sets
    # What is actually compared? 
    status = True if set_of_sets_s == set_of_sets_i else False
    return return_value(status, msg_list, set_of_sets_s, set_of_sets_i)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def check_structure_set_set(student_answer, instructor_answer):
    """
    Created by GPT-4, modified by GE, 2024-03-06
    Both student answer and instructor answer should be a set of sets or a structure
    that can be converted to a set of setsr 
    Not verified. 
    """
    msg_list = []
    status = True
    seq_s = student_answer
    seq_i = instructor_answer

    # Function to check if an object is a sequence but not a string
    def is_sequence_but_not_string(obj):
        return isinstance(obj, (list, tuple, set)) and not isinstance(obj, str)

    # Check if the outer structures are sequences
    if not is_sequence_but_not_string(seq_s):
        msg_list.append("- The outer structure is not a sequence (list or set or tuple).")
        status = False
    else:
        msg_list.append("- The outer structure is a sequence (list or set or tuple).")

    # If outer structures are sequences, check each inner structure
    if status:
        #print("====> seq_s: ", seq_s)
        for i, seq in enumerate(seq_s, start=1):
            if not is_sequence_but_not_string(seq):
                msg_list_append("Element {i} of the outer set is not compatible with a set and has type {seq}.")
                status = False
                continue
        if status:
            msg_list.append(f"- All elements of the outer set are compatible with a set (which means I can coerce it into a set")
            msg_list.append(f"- Answer has the correct structure")

    #print("======> student answer: ", student_answer)
    #print("======> instructor answer: ", instructor_answer)
    return status, "\n".join(msg_list)

# ======================================================================

def check_answer_dict_string_Tuple_NDArray(student_answer, instructor_answer, rel_tol):
    """
    GE original function restructed by GPT-4 (2024-03-06)
    student_answer: dictionary with keys:str, values: tuple of NDArrays
    instructor_answer: dictionary with keys:str, values: a set of objects
    rel_tol: tolerance on the matrix norm
    """
    msg_list = []
    status = True  # Assuming correct until proven otherwise

    # Dictionaries to hold norms for student and instructor answers
    student_norms = {}
    instructor_norms = {}

    for k in instructor_answer.keys():
        # Initialize norms list for current key in both dicts
        student_norms[k] = []
        instructor_norms[k] = []

        try:
            s_tuple = student_answer[k]
        except KeyError as e:
            msg_list.append(f"Error: key {repr(k)} is missing")
            continue

        i_tuple = instructor_answer[k]
        for s_arr, i_arr in zip(s_tuple, i_tuple):
            # Calculate norms
            s_norm = np.linalg.norm(s_arr)
            i_norm = np.linalg.norm(i_arr)

            # Store norms
            student_norms[k].append(s_norm)
            instructor_norms[k].append(i_norm)

            # Check relative error against tolerance
            rel_err = math.fabs(s_norm - i_norm) / math.fabs(i_norm) if i_norm != 0 else s_norm
            status *= rel_err <= rel_tol

    msg_list.append("Only print the norms of the arrays")
    return return_value(status, msg_list, student_norms, instructor_norms)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def check_structure_dict_string_Tuple_NDArray(student_answer, instructor_answer):
    status = True
    msg_list = []
    for k, v in instructor_answer.items():
        # repr adds additional quotes; str does not. 
        if k not in student_answer:
            msg_list.append(f"- Missing key {repr(k)}") 
            status = False
            continue
        if not isinstance(v, (tuple, list)):
            msg_list.append(f"- dict[{repr(k)}] is not a tuple") 
            status = False
            continue
        for i, v_el in enumerate(v):
            if not isinstance(v_el, type(np.zeros(1))):
                msg_list.append(f"- dict[{repr(k)}][{i}] is not an numpy array")
                status = False

    if status:
        msg_list.append("Type 'dict[str, tuple(NDArray)]' is correct.")

    return status, "\n".join(msg_list)

# ======================================================================

def check_answer_dendrogram(student_dendro, instructor_dendro, rel_tol):
    """
    With help from GPT-4
    Compares two dendrogram dictionaries.

    Args:
        dend1: The first dendrogram dictionary.
        dend2: The second dendrogram dictionary.
        rel_tol: The relative tolerance for coordinate comparison.

    Returns:
        True if the dendrograms are considered equal within the tolerance,
        False otherwise.
    """
    status = True
    msg_list = []

    # from scipy.cluster.hierarchy import dendrogram

    dend1 = student_dendro
    dend2 = instructor_dendro

    # Might crash if structures don't match. To fix later. 
    # Coordinate comparison
    if not np.allclose(dend1['icoord'], dend2['icoord'], rtol=rel_tol):
        status = False
    if not np.allclose(dend1['dcoord'], dend2['dcoord'], rtol=rel_tol):
        status = False

    # Leaf order comparison
    if not np.array_equal(dend1['leaves'], dend2['leaves']):
        status = False

    # Optional: Color comparison
    """
    if 'color_list' in dend1 and 'color_list' in dend2:
        if not all(np.array_equal(c1, c2) for c1, c2 in zip(dend1['color_list'], dend2['color_list'])):
            return False
    """
    return return_value(status, msg_list, student_dendro, instructor_dendro)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def check_structure_dendrogram(student_dendro, instructor_dendro):
    """
    Checks if the structure and types of the student's dendrogram dictionary match
    the expected structure from scipy's dendrogram function.

    Args:
        student_dendro: The dendrogram dictionary from a student.
        instructor_dendro: not used. 

    Returns:
        A tuple (bool, str): True and an empty string if the structure matches,
        otherwise False and a message indicating the mismatch.
    """

    # Expected keys in a scipy dendrogram
    expected_keys = {'icoord', 'dcoord', 'leaves', 'ivl', 'color_list'}
    student_keys = set(student_dendro.keys())

    status = True
    msg_list = []

    # Check for missing keys in student dendrogram
    missing_keys = expected_keys - student_keys
    if missing_keys:
        return False, f"Missing keys in student dendrogram: {missing_keys}."

    # For each key, check if the values are lists and have a consistent structure
    for key in expected_keys:
        if key in student_keys:  # Check only if key is present, though missing keys were already caught
            value = student_dendro[key]

            # Ensure value is a list
            if not isinstance(value, list):
                return False, f"Expected a list for key '{key}', found {type(value)}."

            # Specific checks for 'icoord' and 'dcoord' which should contain lists of lists
            if key in ['icoord', 'dcoord']:
                if not all(isinstance(item, list) for item in value):
                    return False, f"Expected a list of lists for key '{key}'."

            # 'leaves' and 'ivl' checks could be added here, such as ensuring 'leaves' contains integers,
            # and 'ivl' contains strings, if necessary for the scope of your validation.

    # If we reach here, the structure is as expected
    return True, "Type Dendrogram has correct structure."

# ======================================================================

def check_answer_int(student_answer, instructor_answer):
    msg_list = []
    #print("===> inside check_answer_int, integer")

    if student_answer != instructor_answer:
        status = False
        msg_list = ["Answer is incorrect"]
    else:
        status = True
        msg_list = ["Answer is correct"]

    return return_value(status, msg_list, student_answer, instructor_answer)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def check_structure_int(student_answer, instructor_answer):
    """
    """
    #print("===> inside check_structure_int, integer")
    if not isinstance(student_answer, int):
        status = False
        msg_list = [f"Answer must be of type 'int'. Your answer is of type {type(student_answer)}."]
    else:
        status = True
        msg_list = [f"Answer is of type 'int' as expected."]

    return status, "\n".join(msg_list)

# ======================================================================

def check_answer_bool(student_answer, instructor_answer):
    msg_list = []
    status = True

    if student_answer != instructor_answer:
        status = False
        msg_list = ["Answer is incorrect."]
    else:
        status = True
        msg_list = ["Answer is correct."]

    #print("==> bool msg_list= ", msg_list)

    return return_value(status, msg_list, student_answer, instructor_answer)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def check_structure_bool(student_answer, instructor_answer):
    if not isinstance(student_answer, bool):
        status = False
        msg_list = [f"Answer must be of type 'bool'. Your answer is of type {type(student_answer)}."]
    else:
        status = True
        msg_list = [f"Answer is of type 'bool' as expected."]

    return status, "\n".join(msg_list)

# ======================================================================

def check_answer_list_string(student_answer, instructor_answer):
    # Normalize and compare the two lists
    msg_list = []
    status = True
    normalized_s_answ = [s.lower().strip() for s in student_answer]
    normalized_i_answ = [s.lower().strip() for s in instructor_answer]

    for s_a, i_a in zip(normalized_s_answ, normalized_i_answ):
        if s_a != i_a:
            status = False
            msg_list += ["Mismatched strings"]

    return return_value(status, msg_list, normalized_s_answ, normalized_i_answ)

# . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . . 

def check_structure_list_string(student_answer, instructor_answer):
    msg_list = []

    # Function to check if an object is a list of strings
    def is_list_of_strings(obj):
        return isinstance(obj, list) and all(isinstance(element, str) for element in obj)

    # Check if both sequences are lists of strings
    if not is_list_of_strings(student_answer):
        msg_list.append("Answer must be a list of strings.")
        status = False
    else:
        msg_list.append("Type 'list[str]' is correct")
        status = True

    return status, "\n".join(msg_list)

# ======================================================================
