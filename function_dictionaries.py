import matplotlib.pyplot as plt

# Address each question. Ideally, there should be a default question
#   if there are no patches and no attributes: 
module_name = {
    'default_question': {
        # changed attributes
        # 'nb_samples': 200,
        'patched_functions': {}
    }
}

# TODO: 
#  The dictionary name should be unique
#  dictionary name here: `testing_questions` and should not have
#  to be the same name as the module

testing_questions = {
    'module_name': 'testing_questions',
    'question1': { 'patched_functions': {} },
    'question2': { 'patched_functions': {} },
    'question3': { 'patched_functions': {} },
}

all_questions = {
    'module_name': 'all_questions',
    'question1': { 'nb_samples': 200, 'patched_functions': {} },
    'question2': { 'patched_functions': {} },
    'question3': { 'patched_functions': {} },
    'question4': { 'patched_functions': {} },
    'question5': { 'patched_functions': {} },
    'question6': { 'patched_functions': {} },
    'question7': { 'patched_functions': {} },
    'question8': { 'patched_functions': {} },
    'question9': { 'patched_functions': {} },
    'question10': { 'patched_functions': {} },
    'question11': { 'patched_functions': {} },
    'question12': { 'patched_functions': {} },
    'question13': { 'patched_functions': {} },
    'question14': { 'patched_functions': {} },
    'question15': { 'patched_functions': {} },
    'question16': { 'patched_functions': {} },
    'question17': { 'patched_functions': {} },
    'question18': { 'patched_functions': {} },
    'question19': { 'patched_functions': {} },
    'question20': { 'patched_functions': {} },
}

# Stress test the typing system
# Uniformize the feedback messages
testing_questions = {
    'module_name': 'testing_questions',
    'question1': {
        'patched_functions': {}
    },
    'question2': {
        'patched_functions': {}
    }
}

# More generally, I should provide the container module for each patched function
#all_questions_dict = [{
#    'module': 'all_questions',
#    'function_name': 'question1',
#    'patched_functions': {}
#},{
#    'module': 'all_questions',
#    'function_name': 'question2',
#    'patched_functions': {}
#}

spectral_patch_dict = {
    'module': 'all_questions',
    'function_name': 'question2',
    'patched_functions': {}
}

spectral_patch_dict = {
    'module': 'spectral_clustering',
    'function_name': 'spectral_clustering',
    'nb_samples': 200,
    'patched_functions': {
        'spectral': None,  # Patch the 'spectral' function found in the spectral_clustering module
        'scatter': plt.scatter,  # Patch plt.scatter from matplotlib.pyplot
        'plot': plt.plot  # Patch plt.plot from matplotlib.pyplot
    }
}

jarvis_patrick_patch_dict = {
    'module': 'jarvis_patrick_clustering',
    'function_name': 'jarvis_patrick_clustering',
    'patched_functions': {
        'jarvis_patrick': None,
        'scatter': plt.scatter,  
        'plot': plt.plot  # 
    }
}


em_patch_dict = {
    'module': 'expectation_maximization',
    'function_name': 'gaussian_mixture',
    # Error if key not present
    # I wnat to be to handle 1 or 2 arguments to patch up
    'nb_samples': 200,
    'patched_functions': {
        # 'em_algorithm': None,  
        #'scatter': plt.scatter,  
        #'plot': plt.plot  
    }
}
