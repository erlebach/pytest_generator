import pytest

# Monitor exceptions

#"""
def pytest_exception_interact(node, call, report):
    if call.excinfo is not None:
        # Formatting the traceback as a string
        tb_string = "".join(traceback.format_exception(*call.excinfo._excinfo))

        # Store the traceback string using the test name as the key
        captured_tracebacks[node.name] = tb_string

@pytest.fixture(autouse=True)
def check_traceback(request):
    yield  # Let the test run
    # Check if this test had an exception and print its traceback
    tb_string = captured_tracebacks.get(request.node.name)
    if tb_string:
        # Insert your condition here. For example, always print for demonstration
        print(f"Traceback for {request.node.name}:\n{tb_string}")
#"""
