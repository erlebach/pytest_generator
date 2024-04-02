import pytest
import json


@pytest.mark.hookwrapper
def pytest_runtest_makereport(item):
    x = yield
    x._result.max_score = getattr(item._obj, 'max_score', 0)
    x._result.visibility = getattr(item._obj, 'visibility', 'visible')
    x._result.partial_score = getattr(item._obj, 'partial_score', '0')
    x._result.explanation = getattr(item._obj, 'explanation', None)
    x._result.answer_note = getattr(item._obj, 'answer_note', None)
    x._result.note = getattr(item._obj, 'note', None)
    x._result.hide_errors = getattr(item._obj, 'hide_errors', None)

def pytest_terminal_summary(terminalreporter, exitstatus):
    json_results = {
        'output': 'Text relevant to the entire submission',
        'output_format': 'simple_format',
        'test_output_format': 'text',
        'test_name_format': 'text',
        'stdout_visibility': 'hidden',
        'visibility': 'visible',
        'explanation': '',
        'note': '',
        'answer_note': '',
        'hide_errors': '',
        'partial_score': 0,
        'extra_data': {},
        'tests': []
    }

    all_tests = []
    total_obtained_score = 0
    total_max_score = 0

    if ('failed' in terminalreporter.stats):
        all_tests = all_tests + terminalreporter.stats['failed']
    if ('passed' in terminalreporter.stats):
        all_tests = all_tests + terminalreporter.stats['passed']

    #### Where is s.outcome determined?

    # First, calculate total scores
    for s in all_tests:
        total_max_score += s.max_score
        if s.outcome == 'passed':
            total_obtained_score += s.max_score

    for s in all_tests:
        # Printed after each message
        output = "" #'Nothing to output (debugging)'
        rescaled_score = 0
        score = s.max_score

        """
        # Not a good idea to provide links to the test code, because I do not 
        # wish to students to see it. 
        # Construct a clickable link using the file path and line number
        file_path = s.location[0]  # The file path of the test
        line_number = s.location[1]  # The line number of the test
        clickable_link = f"{file_path}:{line_number}"
        # Include the clickable link in the output
        output += f"\nClick here to view the test: {clickable_link}"
        """
        #output += f"\nExplanation: {s.explanation!s}"  # str()

        if True:
            if s.explanation is not None:
                # print additional explanation
                output += f"\nExplanation: {s.explanation!s}"  

            if s.answer_note is not None and s.answer_note is not "":
                output += f"\nAdditional note: {s.answer_note!s}"  

            if hasattr(s, 'hide_errors') and s.hide_errors is not None:
                output += s.hide_errors
            else:
                # print assert error message
                error_message = str(s.longrepr.reprcrash.message)
                output += error_message

        if (s.outcome == 'failed'):
            score = 0
            status = 'failed'  # not sure if needed
        elif (s.outcome == 'passed'):
            rescaled_score = (s.max_score / (total_max_score+.01)) * 100
            status = 'passed'
        else:
            output = ''
            status = 'failed'

        json_results["tests"].append(
            {
                'score': rescaled_score,
                'partial_score': s.partial_score,
                'non-scaled_score': score,
                'max_score': s.max_score,
                'name': s.location[2],
                'output': output,
                'visibility': s.visibility,
                'explanation': s.explanation,
                'status': status
            }
        )

    with open('results.json', 'w') as results:
        #print(json_results)
        results.write(json.dumps(json_results, indent=4))
