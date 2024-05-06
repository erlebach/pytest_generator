import matplotlib.pyplot as plt

# More generally, I should provide the container module for each patched function
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
