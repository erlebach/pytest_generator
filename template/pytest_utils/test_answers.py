import pytest
from testing_utilities import assert_almost_equal
from all_questions import *
import instructor_code_with_answers.all_questions as ic
from pytest_utils.decorators import max_score, visibility
import all_questions as sc
import conftest as c
import utils as u
import yaml
import re

with open('type_handlers.yaml', 'r') as f:
    type_handlers = yaml.safe_load(f)

def clean_str_answer(answer):
    answer =  answer.lower().strip()
    answer =  re.sub(r'\s+', ' ', answer)
    return answer

def is_explain(answer):
    return 'explain' in clean_str_answer(answer)

def test_answers_question1_a_bool():
    student_answer = question1()['(a)']
    correct_answer = eval('u.decode_data("dHJ1ZQ==")')
    # Only needed for strings
    if 'explain' in '(a)'.lower():
        return
    print('type bool NOT HANDLED!')
def test_answers_question1_a_explain_string():
    student_answer = question1()['(a) explain']
    correct_answer = eval('u.decode_data("IlRvIGJlIGRldGVybWluZWQuXG4i")')
    # Only needed for strings
    if 'explain' in '(a) explain'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (a) explain'

def test_answers_question1_b_bool():
    student_answer = question1()['(b)']
    correct_answer = eval('u.decode_data("dHJ1ZQ==")')
    # Only needed for strings
    if 'explain' in '(b)'.lower():
        return
    print('type bool NOT HANDLED!')
def test_answers_question1_b_explain_string():
    student_answer = question1()['(b) explain']
    correct_answer = eval('u.decode_data("IlRoZXJlIGlzIG5vIHJhbmRvbSBlbGVtZW50IGluIHRoZSBhbGdvcml0aG1zIGZvciBhZ2dsb21lcmF0aXZlIGhpZXJhcmNoaWNhbCB0ZWNobmlxdWVzIHVubGVzcyB0aGVyZSBhcmUgdGllcyBpbiB0aGUgcHJveGltaXR5IHZhbHVlc1xuIg==")')
    # Only needed for strings
    if 'explain' in '(b) explain'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (b) explain'

def test_answers_question1_c_bool():
    student_answer = question1()['(c)']
    correct_answer = eval('u.decode_data("ZmFsc2U=")')
    # Only needed for strings
    if 'explain' in '(c)'.lower():
        return
    print('type bool NOT HANDLED!')
def test_answers_question1_c_explain_string():
    student_answer = question1()['(c) explain']
    correct_answer = eval('u.decode_data("IkFsdGhvdWdoIGstbWVhbnMgaXMgbW9yZSBjb21wdXRhdGlvbmFsbHkgZWZmaWNpZW50IHRoYW4gYWdnbG9tZXJhdGl2ZSAgaGllcmFyY2hpY2FsIGNsdXN0ZXJpbmcsIHRoZXJlIGFyZSBtb3JlIGVmZmljaWVudCBhbGdvcml0aG1zIHBvc3NpYmxlLCAgZS5nLiwgdGhlIGxlYWRlciBhbGdvcml0aG0uIChTZWUgRXhlcmNpc2UgMTIsIENoYXB0ZXIgNy4pXG4i")')
    # Only needed for strings
    if 'explain' in '(c) explain'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (c) explain'

def test_answers_question1_d_bool():
    student_answer = question1()['(d)']
    correct_answer = eval('u.decode_data("ZmFsc2U=")')
    # Only needed for strings
    if 'explain' in '(d)'.lower():
        return
    print('type bool NOT HANDLED!')
def test_answers_question1_d_explain_string():
    student_answer = question1()['(d) explain']
    correct_answer = eval('u.decode_data("IlNwbGl0dGluZyBkZWNyZWFzZXMgU1NFIGJlY2F1c2Ugd2UgaGF2ZSB0d28gY2VudHJvaWRzICBmb3IgdGhlIHNhbWUgc2V0IG9mIHBvaW50cywgd2hpY2ggd2lsbCByZWR1Y2UgdGhlIGRpc3RhbmNlICBvZiBwb2ludHMgdG8gdGhlIG5lYXJlc3QgY2VudHJvaWQuXG4i")')
    # Only needed for strings
    if 'explain' in '(d) explain'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (d) explain'

def test_answers_question1_e_bool():
    student_answer = question1()['(e)']
    correct_answer = eval('u.decode_data("dHJ1ZQ==")')
    # Only needed for strings
    if 'explain' in '(e)'.lower():
        return
    print('type bool NOT HANDLED!')
def test_answers_question1_e_explain_string():
    student_answer = question1()['(e) explain']
    correct_answer = eval('u.decode_data("IkZvciBLLW1lYW5zLCBTU0UgaXMgYW4gaW52ZXJzZSBtZWFzdXJlIG9mIHRoZSBjb2hlc2lvbiBvZiBjbHVzdGVycywgYW5kIHRodXMsIGFzIFNTRSBkZWNyZWFzZXMsIGNvaGVzaW9uIGluY3JlYXNlcyBhbmQgdmljZS12ZXJzYS5cbiI=")')
    # Only needed for strings
    if 'explain' in '(e) explain'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (e) explain'

def test_answers_question1_f_bool():
    student_answer = question1()['(f)']
    correct_answer = eval('u.decode_data("dHJ1ZQ==")')
    # Only needed for strings
    if 'explain' in '(f)'.lower():
        return
    print('type bool NOT HANDLED!')
def test_answers_question1_f_explain_string():
    student_answer = question1()['(f) explain']
    correct_answer = eval('u.decode_data("IkZvciBLLW1lYW5zIFNTQiAodGhlIGJldHdlZW4gc3VtIG9mIHNxdWFyZXMpIGlzIGEgZGlyZWN0ICBtZWFzdXJlIG9mIHRoZSBzZXBhcmF0aW9uIG9mIGNsdXN0ZXJzLCBhbmQgdGh1cywgYXMgU1NCICBpbmNyZWFzZXMsIHNlcGFyYXRpb24gaW5jcmVhc2VzIGFuZCB2aWNlLXZlcnNhLlxuIg==")')
    # Only needed for strings
    if 'explain' in '(f) explain'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (f) explain'

def test_answers_question1_g_bool():
    student_answer = question1()['(g)']
    correct_answer = eval('u.decode_data("ZmFsc2U=")')
    # Only needed for strings
    if 'explain' in '(g)'.lower():
        return
    print('type bool NOT HANDLED!')
def test_answers_question1_g_explain_string():
    student_answer = question1()['(g) explain']
    correct_answer = eval('u.decode_data("IkNvaGVzaW9uIGFuZCBzZXBhcmF0aW9uIGFyZSBpbmRlcGVuZGVudCBmb3IgSy1NZWFucywgaS5lLiwgIGltcHJvdmluZyBjb2hlc2lvbiAoc21hbGxlciBTU0UpIGRvZXNuXHUyMDE5dCBuZWNlc3NhcmlseSAgaW1wcm92ZSBzZXBhcmF0aW9uIChsYXJnZXIgYmV0d2VlbiBzdW0gb2Ygc3F1YXJlcyAoU1NCKSlcbiI=")')
    # Only needed for strings
    if 'explain' in '(g) explain'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (g) explain'

def test_answers_question1_h_bool():
    student_answer = question1()['(h)']
    correct_answer = eval('u.decode_data("dHJ1ZQ==")')
    # Only needed for strings
    if 'explain' in '(h)'.lower():
        return
    print('type bool NOT HANDLED!')
def test_answers_question1_h_explain_string():
    student_answer = question1()['(h) explain']
    correct_answer = eval('u.decode_data("IkZvciBLLW1lYW5zLCB0aGUgdG90YWwgc3VtIG9mIHNxdWFyZXMgKFRTUykgaXMgdGhlIHN1bSBvZiBTU0UgIChvciB3aXRoaW4gY2x1c3RlciBzdW0gb2Ygc3F1YXJlcykgYW5kIHRoZSBiZXR3ZWVuIHN1bSBvZiAgc3F1YXJlcyAoU1NCKSwgVFNTIGlzIGNvbnN0YW50IGR1cmluZyB0aGUgSy1tZWFucyBjbHVzdGVyaW5nIHByb2Nlc3MuICBTZWUgdGhlIGJvb2ssIHBhZ2UgNTc3XG4i")')
    # Only needed for strings
    if 'explain' in '(h) explain'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (h) explain'

def test_answers_question1_i_bool():
    student_answer = question1()['(i)']
    correct_answer = eval('u.decode_data("dHJ1ZQ==")')
    # Only needed for strings
    if 'explain' in '(i)'.lower():
        return
    print('type bool NOT HANDLED!')
def test_answers_question1_i_explain_string():
    student_answer = question1()['(i) explain']
    correct_answer = eval('u.decode_data("IkZvciBLLW1lYW5zLCB0aGUgdG90YWwgc3VtIG9mIHNxdWFyZXMgKFRTUykgaXMgdGhlIHN1bSBvZiAgU1NFICh0aGUgd2l0aGluIGNsdXN0ZXIgc3VtIG9mIHNxdWFyZXMpIGFuZCB0aGUgYmV0d2VlbiBzdW0gIG9mIHNxdWFyZXMgKFNTQikuIE5vdGUgVFNTIGlzIGNvbnN0YW50IGF0IGV2ZXJ5IHN0ZXAgb2YgdGhlICBLLW1lYW5zIGNsdXN0ZXJpbmcgcHJvY2Vzcy4gU2VlIHRoZSBib29rLCBwYWdlIDU3Ny4gICBTU0UgaXMgYW4gaW52ZXJzZSBtZWFzdXJlIG9mIGNsdXN0ZXIgY29oZXNpb24sIHdoaWxlIFNTQiBpcyAgYSBkaXJlY3QgbWVhc3VyZSBvZiBjbHVzdGVyIHNlcGFyYXRpb24uIFRodXMsIGFzIGNvaGVzaW9uICBpbmNyZWFzZXMsIFNTRSBkZWNyZWFzZXMsIGFuZCBTU0IgKHNlcGFyYXRpb24pIGluY3JlYXNlcyBzaW5jZSAgVFNTID0gU1NFICsgU1NCIGlzIGEgY29uc3RhbnQuIFdoZW4gU1NFIGlzIGF0IGEgbG9jYWwgbWluaW1hLCAgQlNTIGlzIGF0IGEgbG9jYWwgbWF4aW1hLlxuIg==")')
    # Only needed for strings
    if 'explain' in '(i) explain'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (i) explain'

def test_answers_question2_a_bool():
    student_answer = question2()['(a)']
    correct_answer = eval('u.decode_data("dHJ1ZQ==")')
    # Only needed for strings
    if 'explain' in '(a)'.lower():
        return
    print('type bool NOT HANDLED!')
def test_answers_question2_a_explain_string():
    student_answer = question2()['(a) explain']
    correct_answer = eval('u.decode_data("IlRoZSBjbHVzdGVycyBhcmUgdG9vIGZhciBhd2F5IGZvciBvbmUgY2VudHJvaWQgdG8gIGF0dHJhY3QgcG9pbnRzIGZyb20gYW5vdGhlci5cbiI=")')
    # Only needed for strings
    if 'explain' in '(a) explain'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (a) explain'

def test_answers_question2_b_bool():
    student_answer = question2()['(b)']
    correct_answer = eval('u.decode_data("ZmFsc2U=")')
    # Only needed for strings
    if 'explain' in '(b)'.lower():
        return
    print('type bool NOT HANDLED!')
def test_answers_question2_b_explain_string():
    student_answer = question2()['(b) explain']
    correct_answer = eval('u.decode_data("IlRoZSBmaW5hbCBjbHVzdGVycyB3aWxsIGhhdmUgcG9pbnRzIGZyb20gYm90aCBvZiB0aGUgdHdvICBzaGFkZWQgcmVnaW9ucyBzaW5jZSB0aGV5IGFyZSBjbG9zZSB0byBlYWNoIG90aGVyIGFuZCBub3Qgb2YgY2lyY3VsYXIgc2hhcGUuXG4i")')
    # Only needed for strings
    if 'explain' in '(b) explain'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (b) explain'

def test_answers_question2_c_bool():
    student_answer = question2()['(c)']
    correct_answer = eval('u.decode_data("dHJ1ZQ==")')
    # Only needed for strings
    if 'explain' in '(c)'.lower():
        return
    print('type bool NOT HANDLED!')
def test_answers_question2_c_explain_string():
    student_answer = question2()['(c) explain']
    correct_answer = eval('u.decode_data("IlRoZSBjZW50cm9pZCBhdCAxMi41IGlzIGZhcnRoZXIgYXdheSBmcm9tIGFsbCBwb2ludHMgdGhhbiAgYW55IG90aGVyIGNsdXN0ZXJzIGFuZCB3aWxsIGJlY29tZSBlbXB0eS5cbiI=")')
    # Only needed for strings
    if 'explain' in '(c) explain'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (c) explain'

def test_answers_question3_a_SSE_eval_float():
    student_answer = question3()['(a) SSE']
    correct_answer = eval('u.decode_data("bnVsbA==")')
    # Only needed for strings
    if 'explain' in '(a) SSE'.lower():
        return
    print('type eval_float NOT HANDLED!')
def test_answers_question3_b_SSE_eval_float():
    student_answer = question3()['(b) SSE']
    correct_answer = eval('u.decode_data("IjQgKiAoYSoqMiArIGIqKjIgKyBjKioyKSI=")')
    # Only needed for strings
    if 'explain' in '(b) SSE'.lower():
        return
    print('type eval_float NOT HANDLED!')
def test_answers_question3_c_SSE_eval_float():
    student_answer = question3()['(c) SSE']
    correct_answer = eval('u.decode_data("IjQqKFJeMiArIDBeMiArIChSLzIpXjIpIg==")')
    # Only needed for strings
    if 'explain' in '(c) SSE'.lower():
        return
    print('type eval_float NOT HANDLED!')
def test_answers_question4_a_comma_Circle_a_integer():
    student_answer = question4()['(a), Circle (a)']
    correct_answer = eval('u.decode_data("MQ==")')
    # Only needed for strings
    if 'explain' in '(a), Circle (a)'.lower():
        return
    print('type integer NOT HANDLED!')
def test_answers_question4_a_comma_Circle_b_integer():
    student_answer = question4()['(a), Circle (b)']
    correct_answer = eval('u.decode_data("MQ==")')
    # Only needed for strings
    if 'explain' in '(a), Circle (b)'.lower():
        return
    print('type integer NOT HANDLED!')
def test_answers_question4_a_comma_Circle_c_integer():
    student_answer = question4()['(a), Circle (c)']
    correct_answer = eval('u.decode_data("MQ==")')
    # Only needed for strings
    if 'explain' in '(a), Circle (c)'.lower():
        return
    print('type integer NOT HANDLED!')
def test_answers_question4_a_explain_string():
    student_answer = question4()['(a) explain']
    correct_answer = eval('u.decode_data("IkFsbCBvZiBjaXJjbGUgQVx1MjAxOXMgcG9pbnRzIHdpbGwgYmUgYXNzaWduZWQgdG8gdGhlIGNlbnRyb2lkIGluIEEuICBBYm91dCAxLzMgb2YgY2lyY2xlIEJcdTIwMTlzIHBvaW50cyAodGhlIG9uZXMgaW4gdGhlIGxlZnQgdGhpcmQgb2YgY2lyY2xlICBCKSB3aWxsIGJlIGFzc2lnbmVkIHRvIHRoZSBjZW50cm9pZCBvbiB0aGUgbGVmdCBpbiBjaXJjbGUgQi4gIFRoZSByZW1haW5pbmcgMi8zIG9mIHRoZSBwb2ludHMgaW4gQiBhbmQgYWxsIHRoZSBwb2ludHMgaW4gQyB3aWxsICBiZSBhc3NpZ25lZCB0byB0aGUgY2VudHJvaWQgaW4gdGhlIGNlbnRlciBvZiBCLiBUaGlzIHdpbGwgY2F1c2UgdGhlICByaWdodCBjZW50cm9pZCBpbiBCIHRvIG1vdmUgdG8gY2lyY2xlIEMgc2luY2UgQyBoYXMgbWFueSBtb3JlIHBvaW50cyAgdGhhbiBCLiAgSW4gdGhlIG5leHQgaXRlcmF0aW9uLCBhbGwgcG9pbnRzIGluIEEsQiwgYW5kIEMgd2lsbCBiZSAgYXNzaWduZWQgdG8gdGhlIGNlbnRyb2lkcyBsb2NhdGVkIGluIHRoZWlyIG93biBjaXJjbGVzIGFuZCBLLW1lYW5zICB3aWxsIGNvbnZlcmdlLlxuIg==")')
    # Only needed for strings
    if 'explain' in '(a) explain'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (a) explain'

def test_answers_question4_b_comma_Circle_a_integer():
    student_answer = question4()['(b), Circle (a)']
    correct_answer = eval('u.decode_data("MQ==")')
    # Only needed for strings
    if 'explain' in '(b), Circle (a)'.lower():
        return
    print('type integer NOT HANDLED!')
def test_answers_question4_b_comma_Circle_b_integer():
    student_answer = question4()['(b), Circle (b)']
    correct_answer = eval('u.decode_data("MQ==")')
    # Only needed for strings
    if 'explain' in '(b), Circle (b)'.lower():
        return
    print('type integer NOT HANDLED!')
def test_answers_question4_b_comma_Circle_c_integer():
    student_answer = question4()['(b), Circle (c)']
    correct_answer = eval('u.decode_data("MQ==")')
    # Only needed for strings
    if 'explain' in '(b), Circle (c)'.lower():
        return
    print('type integer NOT HANDLED!')
def test_answers_question4_b_explain_string():
    student_answer = question4()['(b) explain']
    correct_answer = eval('u.decode_data("IlNpbmNlIGNpcmNsZXMgQSBhbmQgQiBhcmUgY2xvc2UgdG9nZXRoZXIgYW5kIHF1aXRlIGZhciBhd2F5IGZyb20gIGNpcmNsZSBDLCB0aGUgcG9pbnRzIGZyb20gYm90aCBBIGFuZCBCIHdpbGwgYmUgYXNzaWduZWQgdG8gdGhlICBjZW50cm9pZCB0aGF0IGlzIGluIEEuIFRoZSBwb2ludHMgaW4gQyB3aWxsIGJlIHNwbGl0IGJldHdlZW4gdGhlICB0d28gY2VudHJvaWRzIGluIEMsIHdpdGggZWFjaCBjZW50cm9pZCBoYXZpbmcgNTAsMDAwIHBvaW50cy4gIFNpbmNlICBBIGFuZCBCIGhhdmUgdGhlIHNhbWUgbnVtYmVyIG9mIHBvaW50cyBlYWNoLCB0aGUgY2VudHJvaWQgaW4gQSB3aWxsICBtb3ZlIGJldHdlZW4gQSBhbmQgQi4gVGhlIGNlbnRyb2lkcyBpbiBDIHdpbGwgbW92ZSBhcGFydCBzbGlnaHRseSAgYnV0IGJvdGggd2lsbCByZW1haW4gaW4gQywgZWFjaCBoYXZpbmcgaGFsZiBvZiBDXHUyMDE5cyBwb2ludHMuXG4i")')
    # Only needed for strings
    if 'explain' in '(b) explain'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (b) explain'

def test_answers_question4_c_comma_Circle_a_integer():
    student_answer = question4()['(c), Circle (a)']
    correct_answer = eval('u.decode_data("MA==")')
    # Only needed for strings
    if 'explain' in '(c), Circle (a)'.lower():
        return
    print('type integer NOT HANDLED!')
def test_answers_question4_c_comma_Circle_b_integer():
    student_answer = question4()['(c), Circle (b)']
    correct_answer = eval('u.decode_data("MA==")')
    # Only needed for strings
    if 'explain' in '(c), Circle (b)'.lower():
        return
    print('type integer NOT HANDLED!')
def test_answers_question4_c_comma_Circle_c_integer():
    student_answer = question4()['(c), Circle (c)']
    correct_answer = eval('u.decode_data("Ij0gMiI=")')
    # Only needed for strings
    if 'explain' in '(c), Circle (c)'.lower():
        return
    print('type integer NOT HANDLED!')
def test_answers_question4_c_explain_string():
    student_answer = question4()['(c) explain']
    correct_answer = eval('u.decode_data("IlNpbmNlIGNpcmNsZXMgQSBhbmQgQiBhcmUgY2xvc2UgdG9nZXRoZXIgYW5kIHF1aXRlIGZhciBhd2F5IGZyb20gIGNpcmNsZSBDLCB0aGUgcG9pbnRzIGZyb20gYm90aCBBIGFuZCBCIHdpbGwgYmUgYXNzaWduZWQgdG8gdGhlICBjZW50cm9pZCB0aGF0IGlzIGluIEEuIFRoZSBwb2ludHMgaW4gQyB3aWxsIGJlIHNwbGl0IGJldHdlZW4gdGhlICB0d28gY2VudHJvaWRzIGluIEMsIHdpdGggZWFjaCBjZW50cm9pZCBoYXZpbmcgNTAsMDAwIHBvaW50cy4gICBTaW5jZSBBIGFuZCBCIGhhdmUgdGhlIHNhbWUgbnVtYmVyIG9mIHBvaW50cyBlYWNoLCB0aGUgY2VudHJvaWQgIGluIEEgd2lsbCBtb3ZlIGJldHdlZW4gQSBhbmQgQi4gVGhlIGNlbnRyb2lkcyBpbiBDIHdpbGwgbW92ZSBhcGFydCAgc2xpZ2h0bHkgYnV0IGJvdGggd2lsbCByZW1haW4gaW4gQywgZWFjaCBoYXZpbmcgaGFsZiBvZiBDXHUyMDE5cyBwb2ludHMuXG4i")')
    # Only needed for strings
    if 'explain' in '(c) explain'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (c) explain'

def test_answers_question5_a_list_lbrack_string_rbrack():
    student_answer = question5()['(a)']
    correct_answer = eval('u.decode_data("WyJHcm91cCBBIiwgIkdyb3VwIEIiXQ==")')
    # Only needed for strings
    if 'explain' in '(a)'.lower():
        return
    print('type list[string] NOT HANDLED!')
def test_answers_question5_a_explain_string():
    student_answer = question5()['(a) explain']
    correct_answer = eval('u.decode_data("Ikdyb3VwcyBBIGFuZCBCIHdpbGwgYmUgbWVyZ2VkIHNpbmNlIHRoZXkgaGF2ZSB0aGUgc21hbGxlc3QgIHNpbmdsZSBsaW5rIGRpc3RhbmNlIChiZXR3ZWVuIHRoZSByaWdodC1tb3N0IHBvaW50IG9mIEEgIGFuZCBsZWZ0LW1vc3QgcG9pbnQgb2YgQiksIGFzIGNvbXBhcmVkIHRvIEdyb3VwcyBBIGFuZCBDLCAgYW5kIEdyb3VwcyBCIGFuZCBDXG4i")')
    # Only needed for strings
    if 'explain' in '(a) explain'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (a) explain'

def test_answers_question5_b_list_lbrack_string_rbrack():
    student_answer = question5()['(b)']
    correct_answer = eval('u.decode_data("WyJHcm91cCBBIiwgIkdyb3VwIEMiXQ==")')
    # Only needed for strings
    if 'explain' in '(b)'.lower():
        return
    print('type list[string] NOT HANDLED!')
def test_answers_question5_b_explain_string():
    student_answer = question5()['(b) explain']
    correct_answer = eval('u.decode_data("Ikdyb3VwcyBBIGFuZCBDIHdpbGwgYmUgbWVyZ2VkIHNpbmNlIHRoZXkgaGF2ZSB0aGUgc21hbGxlc3QgIGNvbXBsZXRlIGxpbmsgZGlzdGFuY2UgKGJldHdlZW4gdGhlIHJpZ2h0LW1vc3QgcG9pbnQgb2YgQSAgYW5kIHRoZSBmYXJ0aGVzdCBwb2ludCBpbiBDKSwgYXMgY29tcGFyZWQgdG8gdGhlIGNvbXBsZXRlICBsaW5rIGRpc3RhbmNlIG9mIEdyb3VwcyBBIGFuZCBCIChiZXR3ZWVuIHRoZSBsZWZ0LW1vc3QgcG9pbnQgIGluIEEgYW5kIHJpZ2h0LW1vc3QgcG9pbnQgaW4gQiksIGFuZCBHcm91cHMgQiBhbmQgQyAoYmV0d2VlbiAgcmlnaHRtb3N0LXBvaW50IGluIEIgYW5kIHRoZSBmYXJ0aGVzdCBwb2ludCBpbiBDKS5cIlxuIg==")')
    # Only needed for strings
    if 'explain' in '(b) explain'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (b) explain'

def test_answers_question6_a_core_set():
    student_answer = question6()['(a) core']
    correct_answer = eval('u.decode_data("WyJCIiwgIkMiLCAiRSIsICJGIiwgIkkiLCAiSiIsICJMIiwgIk0iXQ==")')
    # Only needed for strings
    if 'explain' in '(a) core'.lower():
        return
    assert eval('set(student_answer) == set(correct_answer)'), 'Incorrect answer for (a) core'
def test_answers_question6_a_boundary_set():
    student_answer = question6()['(a) boundary']
    correct_answer = eval('u.decode_data("WyJEIiwgIkciXQ==")')
    # Only needed for strings
    if 'explain' in '(a) boundary'.lower():
        return
    assert eval('set(student_answer) == set(correct_answer)'), 'Incorrect answer for (a) boundary'
def test_answers_question6_a_noise_set():
    student_answer = question6()['(a) noise']
    correct_answer = eval('u.decode_data("WyJBIiwgIkgiXQ==")')
    # Only needed for strings
    if 'explain' in '(a) noise'.lower():
        return
    assert eval('set(student_answer) == set(correct_answer)'), 'Incorrect answer for (a) noise'
def test_answers_question6_b_cluster_1_set():
    student_answer = question6()['(b) cluster 1']
    correct_answer = eval('u.decode_data("WyJCIiwgIkMiLCAiRCIsICJFIiwgIkYiLCAiRyJd")')
    # Only needed for strings
    if 'explain' in '(b) cluster 1'.lower():
        return
    assert eval('set(student_answer) == set(correct_answer)'), 'Incorrect answer for (b) cluster 1'
def test_answers_question6_b_cluster_2_set():
    student_answer = question6()['(b) cluster 2']
    correct_answer = eval('u.decode_data("WyJJIiwgIkoiLCAiTCIsICJNIl0=")')
    # Only needed for strings
    if 'explain' in '(b) cluster 2'.lower():
        return
    assert eval('set(student_answer) == set(correct_answer)'), 'Incorrect answer for (b) cluster 2'
def test_answers_question6_b_cluster_3_set():
    student_answer = question6()['(b) cluster 3']
    correct_answer = eval('u.decode_data("W10=")')
    # Only needed for strings
    if 'explain' in '(b) cluster 3'.lower():
        return
    assert eval('set(student_answer) == set(correct_answer)'), 'Incorrect answer for (b) cluster 3'
def test_answers_question6_b_cluster_4_set():
    student_answer = question6()['(b) cluster 4']
    correct_answer = eval('u.decode_data("W10=")')
    # Only needed for strings
    if 'explain' in '(b) cluster 4'.lower():
        return
    assert eval('set(student_answer) == set(correct_answer)'), 'Incorrect answer for (b) cluster 4'
def test_answers_question6_c_minus_a_core_set():
    student_answer = question6()['(c)-a core']
    correct_answer = eval('u.decode_data("WyJCIiwgIkMiLCAiRSIsICJGIiwgIkkiLCAiSiIsICJMIiwgIk0iLCAiRCIsICJHIl0=")')
    # Only needed for strings
    if 'explain' in '(c)-a core'.lower():
        return
    assert eval('set(student_answer) == set(correct_answer)'), 'Incorrect answer for (c)-a core'
def test_answers_question6_c_minus_a_boundary_set():
    student_answer = question6()['(c)-a boundary']
    correct_answer = eval('u.decode_data("WyJBIiwgIkgiXQ==")')
    # Only needed for strings
    if 'explain' in '(c)-a boundary'.lower():
        return
    assert eval('set(student_answer) == set(correct_answer)'), 'Incorrect answer for (c)-a boundary'
def test_answers_question6_c_minus_a_noise_set():
    student_answer = question6()['(c)-a noise']
    correct_answer = eval('u.decode_data("W10=")')
    # Only needed for strings
    if 'explain' in '(c)-a noise'.lower():
        return
    assert eval('set(student_answer) == set(correct_answer)'), 'Incorrect answer for (c)-a noise'
def test_answers_question6_c_minus_b_cluster_1_set():
    student_answer = question6()['(c)-b cluster 1']
    correct_answer = eval('u.decode_data("WyJBIiwgIkIiLCAiQyIsICJEIiwgIkUiLCAiRiIsICJHIiwgIkgiLCAiSSIsICJKIiwgIkwiLCAiTSJd")')
    # Only needed for strings
    if 'explain' in '(c)-b cluster 1'.lower():
        return
    assert eval('set(student_answer) == set(correct_answer)'), 'Incorrect answer for (c)-b cluster 1'
def test_answers_question6_c_minus_b_cluster_2_set():
    student_answer = question6()['(c)-b cluster 2']
    correct_answer = eval('u.decode_data("W10=")')
    # Only needed for strings
    if 'explain' in '(c)-b cluster 2'.lower():
        return
    assert eval('set(student_answer) == set(correct_answer)'), 'Incorrect answer for (c)-b cluster 2'
def test_answers_question6_c_minus_b_cluster_3_set():
    student_answer = question6()['(c)-b cluster 3']
    correct_answer = eval('u.decode_data("W10=")')
    # Only needed for strings
    if 'explain' in '(c)-b cluster 3'.lower():
        return
    assert eval('set(student_answer) == set(correct_answer)'), 'Incorrect answer for (c)-b cluster 3'
def test_answers_question6_c_minus_b_cluster_4_set():
    student_answer = question6()['(c)-b cluster 4']
    correct_answer = eval('u.decode_data("W10=")')
    # Only needed for strings
    if 'explain' in '(c)-b cluster 4'.lower():
        return
    assert eval('set(student_answer) == set(correct_answer)'), 'Incorrect answer for (c)-b cluster 4'
def test_answers_question7_a_string():
    student_answer = question7()['(a)']
    correct_answer = eval('u.decode_data("IkNsdXN0ZXIgNCI=")')
    # Only needed for strings
    if 'explain' in '(a)'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (a)'

def test_answers_question7_a_explain_string():
    student_answer = question7()['(a) explain']
    correct_answer = eval('u.decode_data("IkNsdXN0ZXIgNCBpcyB0aGUgbW9zdCByYW5kb20uIg==")')
    # Only needed for strings
    if 'explain' in '(a) explain'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (a) explain'

def test_answers_question7_b_string():
    student_answer = question7()['(b)']
    correct_answer = eval('u.decode_data("IkNsdXN0ZXIgMSI=")')
    # Only needed for strings
    if 'explain' in '(b)'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (b)'

def test_answers_question7_b_explain_string():
    student_answer = question7()['(b) explain']
    correct_answer = eval('u.decode_data("IkNsdXN0ZXIgNCBpcyB0aGUgbGVhc3QgcmFuZG9tLiI=")')
    # Only needed for strings
    if 'explain' in '(b) explain'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (b) explain'

def test_answers_question8_a_Matrix_1_string():
    student_answer = question8()['(a) Matrix 1']
    correct_answer = eval('u.decode_data("IkRhdGFzZXQgWiI=")')
    # Only needed for strings
    if 'explain' in '(a) Matrix 1'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (a) Matrix 1'

def test_answers_question8_a_explain_diag_entries_comma_Matrix_1_string():
    student_answer = question8()['(a) explain diag entries, Matrix 1']
    correct_answer = eval('u.decode_data("IjIgZGlhZ29uYWwgZW50cmllcyBhcmUgbW9yZSBibHVlIGFuZCBjcmlzcCBjb21wYXJlZCB0byB0aGUgIG90aGVyIDIsIGluZGljYXRpbmcgMiBjbHVzdGVycyBoYXZlIGJldHRlciBjb2hlc2lvbiAgKEIgYW5kIEMpIHRoYW4gdGhlIG90aGVyIDIgKEEgYW5kIEQpLlxuIg==")')
    # Only needed for strings
    if 'explain' in '(a) explain diag entries, Matrix 1'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (a) explain diag entries, Matrix 1'

def test_answers_question8_a_explain_non_minus_diag_entries_comma_Matrix_1_string():
    student_answer = question8()['(a) explain non-diag entries, Matrix 1']
    correct_answer = eval('u.decode_data("IjEuIFJvd3MgMSBhbmQgMyBjb3JyZXNwb25kIHRvIGNsdXN0ZXJzIEEgYW5kIEMuIFRoaXMgaXMgYmVjYXVzZSAgdGhlIGNvbG9ycyBvZiB0aGUgb2ZmLWRpYWdvbmFsIGVudHJpZXMgZm9yIHRoZXNlIHR3byByb3dzIGFyZSAgYWxsIGRpZmZlcmVudCwgaW5kaWNhdGluZyB0aGUgZGlmZmVyZW50IGRpc3RhbmNlcyBiZXR3ZWVuIGNsdXN0ZXIgIEEgKG9yIEMpXHUyMDE5IHMgZGlzdGFuY2VzIHRvIGFsbCBvdGhlciBjbHVzdGVycyAoaS5lOiBBIGlzIGNsb3Nlc3QgdG8gQyAgKGJsdWUgb2ZmLWRpYWdvbmFsKTsgZm9sbG93ZWQgYnkgQiAoZ3JlZW4gb2ZmLWRpYWdvbmFsKTsgYW5kIGlzIHRoZSAgZnVydGhlc3QgZnJvbSBEICh5ZWxsb3cgb2ZmLWRpYWdvbmFsKTsgc2ltaWxhciBleHBsYW5hdGlvbiBmb3IgQykuIDIuIFJvdyAyIGNvcnJlc3BvbmQgdG8gY2x1c3RlciBCLiBTYW1lIGRpc3RhbmNlcyB0byBBIGFuZCBDIChncmVlbiAgb2ZmLWRpYWdvbmFsKSwgZnVydGhlc3QgZGlzdGFuY2UgZnJvbSBBIChyZWQgb2ZmLWRpYWdvbmFsKS4gMy4gUm93IDQgY29ycmVzcG9uZCB0byBjbHVzdGVyIEQuIFNhbWUgZGlzdGFuY2VzIHRvIEEgYW5kIEMgKHllbGxvdyAgb2ZmLWRpYWdvbmFsKSwgZnVydGhlc3QgZGlzdGFuY2UgZnJvbSBCIChyZWQgb2ZmLWRpYWdvbmFsKS5cbiI=")')
    # Only needed for strings
    if 'explain' in '(a) explain non-diag entries, Matrix 1'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (a) explain non-diag entries, Matrix 1'

def test_answers_question8_a_Matrix_2_string():
    student_answer = question8()['(a) Matrix 2']
    correct_answer = eval('u.decode_data("IkRhdGFzZXQgWCI=")')
    # Only needed for strings
    if 'explain' in '(a) Matrix 2'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (a) Matrix 2'

def test_answers_question8_a_explain_diag_entries_comma_Matrix_2_string():
    student_answer = question8()['(a) explain diag entries, Matrix 2']
    correct_answer = eval('u.decode_data("IlR3byBkaWFnb25hbCBlbnRyaWVzIGFyZSBtb3JlIGJsdWUgYW5kIGNyaXNwIGNvbXBhcmVkIHRvIHRoZSBvdGhlciB0d28sICBpbmRpY2F0aW5nIHR3byBjbHVzdGVycyBoYXZlIGJldHRlciBjb2hlc2lvbiAoQiBhbmQgQykgdGhhbiB0aGUgb3RoZXIgIHR3byAoQSBhbmQgRClcbiI=")')
    # Only needed for strings
    if 'explain' in '(a) explain diag entries, Matrix 2'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (a) explain diag entries, Matrix 2'

def test_answers_question8_a_explain_non_minus_diag_entries_comma_Matrix_2_string():
    student_answer = question8()['(a) explain non-diag entries, Matrix 2']
    correct_answer = eval('u.decode_data("IjEuIFJvd3Mgd2l0aCBsZXNzIGNyaXNwIGRpYWdvbmFsIGVudHJpZXMgKHJvd3MgMSBhbmQgNCkgaGF2ZSBhbGwgIGRpZmZlcmVudCBjb2xvcnMsIGluZGljYXRpbmcgdGhhdCBhbGwgb3RoZXIgY2x1c3RlcnMgaGF2ZSBkaWZmZXJlbnQgIGRpc3RhbmNlcyBmcm9tIHRoZXNlIGNsdXN0ZXJzIChlLmc6IENsdXN0ZXIgQSBpcyB0aGUgbmVhcmVzdCB0byBCLCAgZm9sbG93ZWQgYnkgQyBhbmQgdGhlbiBELCBubyAyIGNsdXN0ZXJzIGhhdmUgc2FtZSBkaXN0YW5jZSB0byBjbHVzdGVyIEEpLiAgMi4gUm93cyB3aXRoIG1vcmUgY3Jpc3AgZGlhZ29uYWwgZW50cmllcyBoYXZlIDIgc2FtZSBjb2xvcnMgKG90aGVyIHRoYW4gIHRoZSBkaWFnb25hbCksIGluZGljYXRpbmcgdGhhdCBpdCBoYXMgc2FtZSBkaXN0YW5jZSB0byAyIGNsdXN0ZXJzLCAgYW5kIGlzIHRoZSBmdXJ0aGVzdCBmcm9tIDEgY2x1c3RlciAoZS5nOiBCXHUyMDE5cyBkaXN0YW5jZSB0byBBIGFuZCBDIGlzICBzaW1pbGFyLCBidXQgaXMgdGhlIGZ1cnRoZXJzdCBmcm9tIEQpLiBcbiI=")')
    # Only needed for strings
    if 'explain' in '(a) explain non-diag entries, Matrix 2'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (a) explain non-diag entries, Matrix 2'

def test_answers_question8_a_Matrix_3_string():
    student_answer = question8()['(a) Matrix 3']
    correct_answer = eval('u.decode_data("IkRhdGFzZXQgWSI=")')
    # Only needed for strings
    if 'explain' in '(a) Matrix 3'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (a) Matrix 3'

def test_answers_question8_a_explain_diag_entries_comma_Matrix_3_string():
    student_answer = question8()['(a) explain diag entries, Matrix 3']
    correct_answer = eval('u.decode_data("IlR3byBkaWFnb25hbCBlbnRyaWVzIGFyZSBtb3JlIGJsdWUgYW5kIGNyaXNwIGNvbXBhcmVkIHRvIHRoZSBvdGhlciB0d28sICBpbmRpY2F0aW5nIHR3byBjbHVzdGVycyBoYXZlIGJldHRlciBjb2hlc2lvbiAoQiBhbmQgQykgdGhhbiB0aGUgb3RoZXIgIHR3byAoQSBhbmQgRCkuIFxuIg==")')
    # Only needed for strings
    if 'explain' in '(a) explain diag entries, Matrix 3'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (a) explain diag entries, Matrix 3'

def test_answers_question8_a_explain_non_minus_diag_entries_comma_Matrix_3_string():
    student_answer = question8()['(a) explain non-diag entries, Matrix 3']
    correct_answer = eval('u.decode_data("IkFsbCByb3dzIGhhdmUgdHdvIHNpbWlsYXIgYW5kIDEgZGlmZmVyZW50IGNvbG9ycyBvZmYgZGlhZ29uYWxzIGVudHJpZXMuICBUaGlzIGluZGljYXRlcyBlYWNoIGNsdXN0ZXIgaGFzIHR3byBvdGhlciBjbHVzdGVycyByZWxhdGl2ZWx5IGNsb3NlciAgdG8gaXQgdGhhbiB0aGUgcmVtYWluaW5nIDEgY2x1c3RlciAoZS5nOiBCIGlzIHNpbWlsYXJseSBjbG9zZSB0byBBIGFuZCAgQyBjb21wYXJlZCB0byB3aXRoIEQuXG4i")')
    # Only needed for strings
    if 'explain' in '(a) explain non-diag entries, Matrix 3'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (a) explain non-diag entries, Matrix 3'

def test_answers_question8_b_Row_1_string():
    student_answer = question8()['(b) Row 1']
    correct_answer = eval('u.decode_data("IkNsdXN0ZXIgQSI=")')
    # Only needed for strings
    if 'explain' in '(b) Row 1'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (b) Row 1'

def test_answers_question8_b_Row_2_string():
    student_answer = question8()['(b) Row 2']
    correct_answer = eval('u.decode_data("IkNsdXN0ZXIgQiI=")')
    # Only needed for strings
    if 'explain' in '(b) Row 2'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (b) Row 2'

def test_answers_question8_b_Row_3_string():
    student_answer = question8()['(b) Row 3']
    correct_answer = eval('u.decode_data("IkNsdXN0ZXIgQyI=")')
    # Only needed for strings
    if 'explain' in '(b) Row 3'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (b) Row 3'

def test_answers_question8_b_Row_4_string():
    student_answer = question8()['(b) Row 4']
    correct_answer = eval('u.decode_data("IkNsdXN0ZXIgRCI=")')
    # Only needed for strings
    if 'explain' in '(b) Row 4'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (b) Row 4'

def test_answers_question8_b_Row_1_explain_string():
    student_answer = question8()['(b) Row 1 explain']
    correct_answer = eval('u.decode_data("IkRpYWdvbmFsIGVudHJ5IGlzIGxlc3MgY3Jpc3AsIG1lYW5pbmcgdGhlIGNsdXN0ZXIgaXMgbGVzcyBjb2hlc2l2ZS4gIEFsbCBvZmYtIGRpYWdvbmFsIGVudHJpZXMgaGF2ZSBkaWZmZXJlbnQgY29sb3JzLCBpbmRpY2F0aW5nIGFsbCAgb3RoZXIgY2x1c3RlcnMgaGF2ZSBkaWZmZXJlbnQgZGlzdGFuY2VzIGZyb20gaXMgKGNsb3Nlc3QgdG8gQiwgIGZvbGxvd2VkIGJ5IEMsIGFuZCBmdXJ0aGVzdCBmcm9tIEEpLiBcbiI=")')
    # Only needed for strings
    if 'explain' in '(b) Row 1 explain'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (b) Row 1 explain'

def test_answers_question8_b_Row_2_explain_string():
    student_answer = question8()['(b) Row 2 explain']
    correct_answer = eval('u.decode_data("IkRpYWdvbmFsIGVudHJ5IGlzIG1vcmUgY3Jpc3AsIGluZGljYXRpbmcgdGhlIGNsdXN0ZXIgaXMgY29oZXNpdmUuICAyLzMgb2ZmLWRpYWdvbmFsIGVudHJpZXMgaGF2ZSB0aGUgc2FtZSBjb2xvciwgaW5kaWNhdGluZyAyIG90aGVyICBjbHVzdGVycyBhcmUgY2xvc2VyIHRvIGl0IChBIGFuZCBDLCBldmVudGhvdWdoIHRoZSBvZmYtZGlhZ29uYWwgIGluZGljYXRpbmcgZGlzdGFuY2VzIHdpdGggQSBpcyBsZXNzIGNyaXNwKSwgYW5kIGlzIHRoZSBmdXJ0aGVzdCAgZnJvbSAxIG90aGVyIGNsdXN0ZXIgKEQpLiBcbiI=")')
    # Only needed for strings
    if 'explain' in '(b) Row 2 explain'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (b) Row 2 explain'

def test_answers_question8_b_Row_3_explain_string():
    student_answer = question8()['(b) Row 3 explain']
    correct_answer = eval('u.decode_data("IlRoZSBleHBsYW5hdGlvbiBpcyBzaW1pbWlsYXIgdG8gcm93IDIuIg==")')
    # Only needed for strings
    if 'explain' in '(b) Row 3 explain'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (b) Row 3 explain'

def test_answers_question8_b_Row_4_explain_string():
    student_answer = question8()['(b) Row 4 explain']
    correct_answer = eval('u.decode_data("IlRoZSBFeHBsYW5hdGlvbiBpcyBzaW1pbGFyIHRvIHJvdyAxIGluIGludmVydGVkIG9yZGVyLiI=")')
    # Only needed for strings
    if 'explain' in '(b) Row 4 explain'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (b) Row 4 explain'

def test_answers_question9_a_list():
    student_answer = question9()['(a)']
    correct_answer = eval('u.decode_data("WyJIaWVyYXJjaGljYWwiLCAib3ZlcmxhcHBpbmciLCAicGFydGlhbCJd")')
    # Only needed for strings
    if 'explain' in '(a)'.lower():
        return
    print('type list NOT HANDLED!')
def test_answers_question9_b_list():
    student_answer = question9()['(b)']
    correct_answer = eval('u.decode_data("WyJQYXJ0aXRpb25hbCIsICJleGNsdXNpdmUiLCAiY29tcGxldGUiXQ==")')
    # Only needed for strings
    if 'explain' in '(b)'.lower():
        return
    print('type list NOT HANDLED!')
def test_answers_question9_c_list():
    student_answer = question9()['(c)']
    correct_answer = eval('u.decode_data("WyJQYXJ0aXRpb25hbCIsICJmdXp6eSIsICJjb21wbGV0ZSJd")')
    # Only needed for strings
    if 'explain' in '(c)'.lower():
        return
    print('type list NOT HANDLED!')
def test_answers_question9_d_list():
    student_answer = question9()['(d)']
    correct_answer = eval('u.decode_data("WyJIaWVyYXJjaGljYWwiLCAib3ZlcmxhcHBpbmciLCAicGFydGlhbCJd")')
    # Only needed for strings
    if 'explain' in '(d)'.lower():
        return
    print('type list NOT HANDLED!')
def test_answers_question9_e_list():
    student_answer = question9()['(e)']
    correct_answer = eval('u.decode_data("WyJQYXJ0aXRpb25hbCIsICJFeGNsdXNpdmUiLCAicGFydGlhbCJd")')
    # Only needed for strings
    if 'explain' in '(e)'.lower():
        return
    print('type list NOT HANDLED!')
def test_answers_question9_e_explain_string():
    student_answer = question9()['(e) explain']
    correct_answer = eval('u.decode_data("IlNvbWUgc3R1ZGVudHMgaW4gdGhlIENTIGRlcHQgd291bGRuXHUyMDE5dCBoYXZlICB0YWtlbiB0aGUgRE0gY2xhc3MgYW5kIHRodXMgY2FuXHUyMDE5dCBiZSBncm91cGVkLlxuIg==")')
    # Only needed for strings
    if 'explain' in '(e) explain'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (e) explain'

def test_answers_question10_a_Figure_a_string():
    student_answer = question10()['(a) Figure (a)']
    correct_answer = eval('u.decode_data("Im5vIg==")')
    # Only needed for strings
    if 'explain' in '(a) Figure (a)'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (a) Figure (a)'

def test_answers_question10_a_Figure_b_string():
    student_answer = question10()['(a) Figure (b)']
    correct_answer = eval('u.decode_data("InllcyI=")')
    # Only needed for strings
    if 'explain' in '(a) Figure (b)'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (a) Figure (b)'

def test_answers_question10_a_explain_string():
    student_answer = question10()['(a) explain']
    correct_answer = eval('u.decode_data("IkRCU0NBTiBjYW4gd29yayBvbmx5IGZvciAoYikgYmVjYXVzZSBpbiAoYikgdGhlIHBvaW50cyBpbiB0aGUgbm9zZSwgIGV5ZXMsIGFuZCBtb3V0aCBhcmUgbXVjaCBjbG9zZXIgdG9nZXRoZXIgdGhhbiB0aGUgcG9pbnRzIGJldHdlZW4gIHRoZXNlIGFyZWFzLCBhbmQgREJTQ0FOIGNvdWxkIGRpc3Rpbmd1aXNoIHRoZXNlIGFyZWFzLiBGb3IgKGEpLCAgdGhlIG5vaXNlIGlzIG11Y2ggZGVuc2VyIHRoYW4gdGhlIGludGVyZXN0IHBhdHRlcm5zLCBzbyB0aGUgbm9zZSwgIGV5ZXMsIGFuZCBtb3V0aCB3aWxsIGJlIGVsaW1pbmF0ZWQgYnkgREJTQ0FOLlxuIg==")')
    # Only needed for strings
    if 'explain' in '(a) explain'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (a) explain'

def test_answers_question10_b_Figure_a_string():
    student_answer = question10()['(b) Figure (a)']
    correct_answer = eval('u.decode_data("Im5vIg==")')
    # Only needed for strings
    if 'explain' in '(b) Figure (a)'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (b) Figure (a)'

def test_answers_question10_b_Figure_b_string():
    student_answer = question10()['(b) Figure (b)']
    correct_answer = eval('u.decode_data("InllcyI=")')
    # Only needed for strings
    if 'explain' in '(b) Figure (b)'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (b) Figure (b)'

def test_answers_question10_b_explain_string():
    student_answer = question10()['(b) explain']
    correct_answer = eval('u.decode_data("IkstbWVhbnMgY2FuIHdvcmsgZm9yIChiKSBhcyBsb25nIGFzIHRoZSBudW1iZXIgb2YgY2x1c3RlcnMgIHdhcyBzZXQgdG8gNCwgYWx0aG91Z2ggdGhlIGxvd2VyIGRlbnNpdHkgcG9pbnRzIHdvdWxkIGFsc28gIGJlIGluY2x1ZGVkLiBLLW1lYW5zIGRvZXMgbm90IHdvcmsgZm9yIChhKS5cbiI=")')
    # Only needed for strings
    if 'explain' in '(b) explain'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (b) explain'

def test_answers_question10_c_string():
    student_answer = question10()['(c)']
    correct_answer = eval('u.decode_data("IlRha2UgdGhlIHJlY2lwcm9jYWwgb2YgdGhlIGRlbnNpdHkgYXMgdGhlIG5ldyBkZW5zaXR5IGFuZCB1c2UgREJTQ0FOLiI=")')
    # Only needed for strings
    if 'explain' in '(c)'.lower():
        return
    student_answer = clean_str_answer(student_answer)
    correct_answer = clean_str_answer(correct_answer)
    assert eval('student_answer == correct_answer'), 'Incorrect answer for (c)'
