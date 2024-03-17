from pprint import pprint

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

import myplots as myplt
import time
import warnings
import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster, datasets, mixture
from sklearn.datasets import make_blobs
from sklearn.neighbors import kneighbors_graph
from sklearn.preprocessing import StandardScaler
from itertools import cycle, islice
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage  #

# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u

# ----------------------------------------------------------------------
"""
Part 2
Comparison of Clustering Evaluation Metrics: 
In this task you will explore different methods to find a good value for k
"""


def fit_kmeans_sse(dataset, *, k, seed=42):
    # Is there a random_tate argument?
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    X, y, centers = dataset
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    kmeans = cluster.KMeans(n_clusters=k, init="random", random_state=seed)
    kmeans.fit(X)
    # y_kmeans = kmeans.predict(X)
    # Compute SSE without inertia
    sse = 0
    for i in range(k):
        sse += np.sum((X[kmeans.labels_ == i] - kmeans.cluster_centers_[i]) ** 2)
    return sse


def fit_kmeans_inertia(dataset, *, k, seed=42):
    # Is there a random_tate argument?
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    X, y, centers = dataset
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    kmeans = cluster.KMeans(n_clusters=k, init="random", random_state=seed)
    kmeans.fit(X)
    # y_kmeans = kmeans.predict(X)
    return kmeans.inertia_


def compute():
    # ---------------------
    answers = {}

    """
    A.	Call the make_blobs function with following parameters :(center_box=(-20,20), n_samples=20, centers=5, random_state=12).
    """

    # X, y, centers
    # data is a list of tuple or list of list of lists
    data = datasets.make_blobs(
        n_samples=20,
        centers=5,
        center_box=(-20, 20),
        random_state=12,
        return_centers=True,
    )
    #print("make_blobs, data= ", data)
    # dct: return value from the make_blobs function in sklearn, expressed as a list of three numpy arrays
    # Comment to myself: the third value returned might not be used. The
    # students should choose arguments that allow three values to be returned.
    # Test whether list(data) has 3 terms, and whether the third term has
    # length centers (5)
    dct = answers["2A: blob"] = list(data)  # in case data is a tuple

    """
    B. Modify the fit_kmeans function to return the SSE (see Equations 8.1 and 8.2 in the book).
    """

    # dct value: the `fit_kmeans` function
    dct = answers["2B: fit_kmeans"] = fit_kmeans_sse

    """
    C.	Plot the SSE as a function of k for k=1,2,….,8, and choose the optimal k based on the elbow method.
    """

    SSE = []
    ks = list(range(1, 9))
    for k in ks:
        sse = fit_kmeans_sse(data, k=k)
        SSE.append([k, sse])

    # dct value: a list of tuples, e.g., [[0, 100.], [1, 200.]]
    # Each tuple is a (k, SSE) pair
    dct = answers["2C: SSE plot"] = SSE

    """
    D.	Repeat part 2.C for inertia (note this is an attribute in the kmeans estimator called _inertia). Do the optimal k’s agree?
    """

    Inertia = []
    ks = list(range(1, 9))
    for k in ks:
        inertia = fit_kmeans_inertia(data, k=k)
        Inertia.append([k, inertia])

    # print(f"{Inertia=}")
    # print(f"{SSE=}")

    # dct value has the same structure as in 2C
    dct = answers["2D: inertia plot"] = Inertia

    # dct value should be a string, e.g., "yes" or "no"
    dct = answers["2D: do ks agree?"] = "yes"

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part2.pkl", "wb") as f:
        pickle.dump(answers, f)
