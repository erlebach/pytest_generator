import numpy as np
import matplotlib.pyplot as plt

def question1():
    answers = {}
    an = answers

    # type: ndarray
    an['nda_1'] = np.zeros(3)
    an['nda_2'] = np.array([3, 4])
    an['nda_3'] = np.zeros([3,4])
    an['nda_3'] = [3,4]
    an['nda_4'] = 45.3

    # type: explain_str
    an['es_1'] = (
      "The requested explanation of the question is a very complex"
      "proposition in as much as it conflicts with my daily activities.")
    an['es_2'] = "request explanation"
    an['es_3'] = 35
    an['es_4'] = ["message"]

    # type: list[str]
    an['ls_1'] = ["this", "is", "a"]
    an['ls_2'] = ["this", "is", "a", "string"]
    an['ls_3'] = ["this", 4, "a"]
    an['ls_4'] = 45

    # type: list[list[float]]
    an['llf_1'] = [[3.5,3.4], [5.2], [3.2, -4.5]]
    an['llf_2'] = [[3.5,3.4], [5.4], [3.2, -4.5]]
    an['llf_3'] = [[3.5,3.4], [3.2, -4.5]]
    an['llf_4'] = 45

    # type: dict[int,list[float]]
    an['dilf_1'] = {3:[4.5, 3.2], 5:[7.2, 9.1, 3.2]}
    an['dilf_2'] = {7:[4.5, 3.2], 5:[7.2, 9.1, 3.2]}
    an['dilf_3'] = {3:[4.5, 3.2]}
    an['dilf_4'] = 45

   # type: eval_float
   # an['ev_1'] = 
   # an['ev_2'] = 
   # an['ev_3'] = 

    # type: set[str]
    an['ss_1'] = {'my', 'house', 'is'}
    an['ss_2'] = {'house', 'is', 'my', 'my'} # pass
    an['ss_3'] = {'my', 'house'} # fail
    an['ss_4'] = {'my', 'house', 45} # fail
    an['ss_5'] = 45 # fail

##    # type: set[set[int]]
##    # set of set is unhashable. Not allowed for now. 
##    an['ssi_1'] = {{3,4,5},{8,-7}}
##    an['ssi_2'] = {{4,3,5},{-7,8,8}}  # pass
##    an['ssi_3'] = {{3,4},{8,-7}} # fail
##    an['ssi_4'] = 45  # fail

    # type: dict[str,set]
    an['dss_1'] = {'a':{3,4,5}, 'b':{2,8.2,'gor'}}
    an['dss_2'] = {'a':{4,3,5,3}, 'b':{'gor',2,8.2}} # pass
    an['dss_3'] = {30:{4,3,5,3}, 'b':{'gor',2,8.2}} # fail
    an['dss_4'] = 45 # fail

    # type: set[ndarray]  (specify type of array, or error tolerance?)
    # sets of hashables will require frozen sets. 
    ## an['snda_1'] = {np.zeros(3), np.zeros(4)}
    ## an['snda_2'] = {np.zeros(3), np.zeros(4), np.zeros(4)} # pass
    ## an['snda_2'] = {np.zeros(5), np.zeros(4)} # fail
    ## an['snda_3'] = 45 # fail

    # type: dict[str,tuple[ndarray]]
    an['dtnda_1'] = {'a': (np.zeros(3), np.array([3,4])), 
                     'b': (np.array([4.5,5.5]), np.array([-2,3]))}
    an['dtnda_2'] = {'a': (np.zeros(3), np.array([3,4])), 
                     'b': (np.array([14.5,5.5,6.5]), np.array([-2,3]))} # fail
    an['dtnda_3'] = {'a': (np.zeros(3), np.array([3,4])),  
                     'b': (np.array([4.5,5.5]), np.array([-2,3])),
                     'c': (np.array([4,5]))} # fail
    an['dtnda_4'] = {'a': (np.zeros(3), np.array([3,4])),  
                     'b1': (np.array([4.5,5.5]), np.array([-2,3])),
                     'c': (np.array([4,5]))} # fail
    an['dtnda_5'] = 45 # fail

    # type: dict[int,ndarray]
    an['dinda_1'] = {3: np.zeros(3), 4: np.array([3,4])} # pass
    an['dinda_2'] = {'3': np.zeros(3), 5: np.array([3,4])}  # fail
    an['dinda_3'] = 45

    # type: dict[tuple[int],ndarray]
    an['dtinda_1'] = {(3,4): np.zeros(3), (5,): np.array([3,4.4])}
    an['dtinda_2'] = {3: np.zeros(3), (5): np.array([3,4.4])}
    an['dtinda_3'] = 45

    # type: list[ndarray]
    an['lnda_1'] = [np.zeros(3), np.array([4, 5])] # pass
    # results.json thinks that all elements are ndarray. WRONG.
    an['lnda_2'] = [{3,4},  np.array([4, 5])] # fail
    # results.json thinks that all elements are ndarray. WRONG.
    an['lnda_3'] = [45.3,  np.array([4, 5])] # fail
    an['lnda_4'] = 45 # fail
    an['lnda_5'] = [np.zeros(3), np.array([4, 5]), np.array([3,2])] # fail (wrong length)

    # type: dict[str,set[int]]
    an['dssi_1'] = {'a':{3,4}, 'b': {5,2,3}}
    an['dssi_2'] = {'a':{3,4,4}, 'b': {5,2,3}}
    # structural test does not check size of set
    an['dssi_3'] = {'a':{3,4,5}, 'b': {5,2,3}} # struct pass, answ fail
    # Only check required keys in dicts. Extra keys allowed.. 
    an['dssi_4'] = {'a':{3,4}, 'b': {5,2,3}, 'c': 42}
    an['dssi_5'] = {3:{3,4}, 'b':{5,2,3}}
    an['dssi_6'] = 45

    def test_sum(arg1: int, arg2: int) -> int:
        return arg1 + arg2

    # type: function
    an['fct_1'] = test_sum
    an['fct_2'] = 45

    x = list(range(10))
    y = list(range(10))
    plot = plt.plot(x, y)
    scat = plt.scatter(x, y)

    # type: lineplot
    an['lineplot_1'] = plot # pass
    an['lineplot_2'] = scat # fail
    an['lineplot_3'] = 45 # fail

    # type: scatterplot2d
    an['scatter2d_1'] = scat # pass
    an['scatter2d_2'] = plot # fail
    an['scatter2d_3'] = 45

    # type: scatterplot3d
    an['scatter3d_1'] = None
    an['scatter3d_2'] = None
    an['scatter3d_3'] = None

    # type: dendogram
    # an['dendo_1'] = None
    # an['dendo_2'] = None
    # an['dendo_3'] = None

    return answers
