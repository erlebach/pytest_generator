import sys

def get_current_function():
    """ Return the current function object. """
    frame = sys._getframe(1)  # Get the frame of the caller function
    return frame.f_globals[frame.f_code.co_name]

