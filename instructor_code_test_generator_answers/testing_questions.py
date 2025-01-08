import numpy as np
import matplotlib.pyplot as plt

def question1():
    answers = {}
    an = answers

    # type: ndarray
    an['nda_1'] = np.zeros(3)
    an['nda_2'] = np.zeros(3)
    an['nda_3'] = np.zeros(3)
    an['nda_4'] = np.zeros(3)

    # type: explain_str
    an['es_1'] = (
      "The requested explanation of the question is a very complex"
      "proposition in as much as it conflicts with my daily activities.")
    an['es_2'] = (
      "The requested explanation of the question is a very complex"
      "proposition in as much as it conflicts with my daily activities.")
    an['es_3'] = (
      "The requested explanation of the question is a very complex"
      "proposition in as much as it conflicts with my daily activities.")
    an['es_4'] = (
      "The requested explanation of the question is a very complex"
      "proposition in as much as it conflicts with my daily activities.")

    # type: list[str]
    an['ls_1'] = ["this", "is", "a"]
    an['ls_2'] = an['ls_1']
    an['ls_3'] = an['ls_1']
    an['ls_4'] = an['ls_1']

    # type: list[list[float]]
    an['llf_1'] = [[3.5,3.4], [5.2], [3.2, -4.5]]
    an['llf_2'] = an['llf_1']
    an['llf_3'] = an['llf_1']
    an['llf_4'] = an['llf_1']

    # type: dict[int,list[float]]
    an['dilf_1'] = {3:[4.5, 3.2], 5:[7.2, 9.1, 3.2]}
    an['dilf_2'] = an['dilf_1']
    an['dilf_3'] = an['dilf_1']
    an['dilf_4'] = an['dilf_1']

   # type: eval_float
   # an['ev_1'] = 
   # an['ev_2'] = an['ev_2']
   # an['ev_3'] = an['ev_2']

    # type: set[str]
    an['ss_1'] = {'my', 'house', 'is'}
    an['ss_2'] = an['ss_1']
    an['ss_3'] = an['ss_1']
    an['ss_4'] = an['ss_1']
    an['ss_5'] = an['ss_1']

    # type: set[set[int]]
    # set of set is unhashable, so cannot be handled
##    an['ssi_1'] = {{3,4,5},{8,-7}}
##    an['ssi_2'] = an['ssi_1']
##    an['ssi_3'] = an['ssi_1']
##    an['ssi_4'] = an['ssi_1']

    # type: dict[str,set]
    an['dss_1'] = {'a':{3,4,5}, 'b':{2,8.2,'gor'}}
    an['dss_2'] = an['dss_1']
    an['dss_3'] = an['dss_1']

    # type: set[ndarray]
    ## an['snda_1'] = {np.zeros(3), np.zeros(4)}
    ## an['snda_2'] = an['snda_1'] 
    ## an['snda_3'] = an['snda_1']

    # type: dict[str,tuple[ndarray]]
    an['dtnda_1'] = {'a': (np.zeros(3), np.array([3,4])), 
                     'b': (np.array([4.5,5.5]), np.array([-2,3]))}
    an['dtnda_2'] = an['dtnda_1']
    an['dtnda_3'] = an['dtnda_1']
    an['dtnda_4'] = an['dtnda_1']
    an['dtnda_5'] = an['dtnda_1']

    # type: dict[int,ndarray]
    an['dinda_1'] = {3: np.zeros(3), 4: np.array([3,4])}
    an['dinda_2'] = an['dinda_1']
    an['dinda_3'] = an['dinda_1']

    # type: dict[tuple[int],ndarray]
    an['dtinda_1'] = {(3,4): np.zeros(3), (5,): np.array([3,4.4])}
    an['dtinda_2'] = an['dtinda_1']
    an['dtinda_3'] = an['dtinda_1']

    # type: list[ndarray]
    an['lnda_1'] = [np.zeros(3), np.array([4, 5])]
    an['lnda_2'] = an['lnda_1']
    an['lnda_3'] = an['lnda_1']
    an['lnda_4'] = an['lnda_1']
    an['lnda_5'] = an['lnda_1']

    # type: dict[str,set[int]]
    an['dssi_1'] = {'a':{3,4}, 'b': {5,2,3}}
    an['dssi_2'] = an['dssi_1']
    an['dssi_3'] = an['dssi_1']
    an['dssi_4'] = an['dssi_1']
    an['dssi_5'] = an['dssi_1']
    an['dssi_6'] = an['dssi_1']

    def test_sum(arg1: int, arg2: int) -> int:
        return arg1 + arg2

    # type: function
    an['fct_1'] = test_sum
    an['fct_2'] = an['fct_1']

    x = list(range(10))
    y = list(range(10))
    plot = plt.plot(x, y)
    scat = plt.scatter(x, y)

    # type: lineplot
    an['lineplot_1'] = plot
    an['lineplot_2'] = an['lineplot_1']
    an['lineplot_3'] = an['lineplot_1']

    # type: scatterplot2d
    an['scatter2d_1'] = scat
    an['scatter2d_2'] = an['scatter2d_1']
    an['scatter2d_3'] = an['scatter2d_1']

    # type: scatterplot3d
    an['scatter3d_1'] = None
    an['scatter3d_2'] = an['scatter3d_1']
    an['scatter3d_3'] = an['scatter3d_1']

    # type: dendogram
    # an['dendo_1'] = None
    # an['dendo_2'] = an['dendo_1']
    # an['dendo_3'] = an['dendo_1']

    return answers

#    "dict[int,list[float]]",
#    "eval_float",
#    "explain_str",
#    "function",
#    "lineplot",
#    "list[float]",
#    "list[int]",
#    "list[list[float]]",
#    "list[str]",

#    "scatterplot3d",
#    "scatterplot2d",
#    "dendrogram",
#    "set[str]",
#    "dict[str,set]",
#    "set[set[int]]",
#    "dict[str,tuple[ndarray]]",
#    "dict[int,ndarray]",
#    "dict[tuple[int],ndarray]",
#    "list[ndarray]",
#    "set[ndarray]",
#    "dict[str,set[int]]"
