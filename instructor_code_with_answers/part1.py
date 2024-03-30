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

from sklearn.datasets import make_moons, make_circles, make_blobs
from sklearn.cluster import KMeans


# import plotly.figure_factory as ff
import math
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u


# ----------------------------------------------------------------------
"""
Part 1: 
Evaluation of k-Means over Diverse Datasets: 
In the first task, you will explore how k-Means perform on datasets with diverse structure.
"""

# Fill this function with code at this location. Do NOT move it. 
# Change the arguments and return according to 
# the question asked. 

def fit_kmeans(dataset, n_clusters):

    # Unpack the dataset into its components
    data, _ = dataset  # We ignore the labels for K-means clustering
    
    # Standardize the data
    scaler = StandardScaler()
    data_standardized = scaler.fit_transform(data)
    
    # Initialize and fit the KMeans model
    kmeans = KMeans(n_clusters=n_clusters, init='random', random_state=42)
    kmeans.fit(data_standardized)

    return kmeans.labels_


def compute():
    answers = {}

    """
    A.	Load the following 5 datasets with 100 samples each: noisy_circles (nc), noisy_moons (nm), blobs with varied variances (bvv), Anisotropicly distributed data (add), blobs (b). Use the parameters from (https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html), with any random state. (with random_state = 42). Not setting the correct random_state will prevent me from checking your results.
    """

    random_state = 42

    n_samples = 100
    noisy_circles = make_circles(n_samples=n_samples, factor=.5, noise=.05, random_state=random_state)
    noisy_moons = make_moons(n_samples=n_samples, noise=.05, random_state=random_state)
    blobs = make_blobs(n_samples=n_samples, random_state=random_state)
    varied = make_blobs(n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state)

    # Anisotropicly distributed data
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(blobs[0], transformation)
    aniso = (X_aniso, blobs[1])

    datasets = {
        "nc": noisy_circles,
        "nm": noisy_moons,
        "bvv": varied,
        "add": aniso,
        "b": blobs
        }
    
    # Scaling all datasets for consistency
    for key in datasets.keys():
        datasets[key] = (StandardScaler().fit_transform(datasets[key][0]), datasets[key][1])

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # 'nc', 'nm', 'bvv', 'add', 'b'. keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    dct = answers["1A: datasets"] = datasets

    #answers["1A: datasets"]["nc"] = datasets["nc"]
    #answers["1A: datasets"]["nm"] = datasets["nm"]
    #answers["1A: datasets"]["bvv"] = datasets["bvv"]
    #answers["1A: datasets"]["add"] = datasets["add"]
    #answers["1A: datasets"]["b"] = datasets["b"]

    """
   B. Write a function called fit_kmeans that takes dataset (before any processing on it), i.e., pair of (data, label) Numpy arrays, and the number of clusters as arguments, and returns the predicted labels from k-means clustering. Use the init='random' argument and make sure to standardize the data (see StandardScaler transform), prior to fitting the KMeans estimator. This is the function you will use in the following questions. 
    """

    # dct value:  the `fit_kmeans` function
    dct = answers["1B: fit_kmeans"] = fit_kmeans


    """
    C.	Make a big figure (4 rows x 5 columns) of scatter plots (where points are colored by predicted label) with each column corresponding to the datasets generated in part 1.A, and each row being k=[2,3,5,10] different number of clusters. For which datasets does k-means seem to produce correct clusters for (assuming the right number of k is specified) and for which datasets does k-means fail for all values of k? 
    
    Create a pdf of the plots and return in your report. 
    """
    ks = [2, 3, 5, 10]

    plt.figure(figsize=(25, 20))

    # dataset titles
    dataset_names = {"nc": "Noisy Circles", "nm": "Noisy Moons", "bvv": "Varied Variances", "add": "Anisotropic", "b": "Blobs"}

    # Loop over each k value and dataset to plot
    for i, k in enumerate(ks):
        for j, (dataset_abbr, dataset) in enumerate(datasets.items()):
            predicted_labels = fit_kmeans(dataset, k)
            plt.subplot(len(ks), len(datasets), i*len(datasets) + j + 1)
            plt.scatter(dataset[0][:, 0], dataset[0][:, 1], c=predicted_labels, s=50, cmap='viridis')
            plt.title(f"{dataset_names[dataset_abbr]} (k={k})")

    plt.tight_layout()
    plt.show()

    #answers["1C: cluster successes"] = {
    #    "bvv": [3], 
    #    "add": [3],  
    #    "b": [3]     
    #    }
    
    answers["1C: cluster failures"] = ["nc", "nm"]

    # dct value: return a dictionary of one or more abbreviated dataset names (zero or more elements) 
    # and associated k-values with correct clusters.  key abbreviations: 'nc', 'nm', 'bvv', 'add', 'b'. 
    # The values are the list of k for which there is success. Only return datasets where the list of cluster size k is non-empty.
    dct = answers["1C: cluster successes"] = {
        "bvv": [3], 
        "add": [3],  
        "b": [3]     
        }

    # dct value: return a list of 0 or more dataset abbreviations (list has zero or more elements, 
    # which are abbreviated dataset names as strings)
    dct = answers["1C: cluster failures"] = ["nc", "nm"]

    """
    D. Repeat 1.C a few times and comment on which (if any) datasets seem to be sensitive to the choice of initialization for the k=2,3 cases. You do not need to add the additional plots to your report.

    Create a pdf of the plots and return in your report. 
    """

    # dct value: list of dataset abbreviations
    # Look at your plots, and return your answers.
    # The plot is part of your report, a pdf file name "report.pdf", in your repository.
    dct = answers["1D: datasets sensitive to initialization"] = ["The Noisy Circles and Noisy Moons datasets are sensitive to the choice of initialization because of their inherent shapes and overlaps. These datasets consist of points that form non-linear and intertwined patterns, which don't conform to the spherical cluster assumption that methods like k-means rely on. Because of this, the initial placement of centroids can heavily influence which points are captured by each cluster, especially since there are no clear boundaries."]

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)
