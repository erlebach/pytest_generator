from sklearn.preprocessing import StandardScaler
from sklearn import cluster
import matplotlib.pyplot as plt
import numpy as np
import utils as u
import pickle


# ----------------------------------------------------------------------
"""
Part 1: 
Evaluation of k-Means over Diverse Datasets: 
In the first task, you will explore how k-Means perform on datasets with diverse structure.
"""

# Fill this function with code at this location. Do NOT move it.
# Change the arguments and return according to
# the question asked.


def fit_kmeans(dataset, *, k, seed=42):
    # Is there a random_state argument?
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    X, y = dataset
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    kmeans = cluster.KMeans(n_clusters=k, init="random", random_state=seed)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    # returns an int [0, 1, ..., k-1]
    return y_kmeans


def plot_data(data, ks, file_nm, seed=42):
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    colors = np.array(u.colors)
    for data_id, (key, dataset) in enumerate(data.items()):
        for k_id, k in enumerate(ks):
            # labels
            y_kmeans = fit_kmeans(dataset, k=k, seed=seed)
            #print(f"{key=}, {k=}, {seed=}, {y_kmeans=}")
            # X: list of (x,y) coordinates
            X, y = dataset
            ax = axes[k_id, data_id]
            x, y = X[:, 0], X[:, 1]
            ax.scatter(x, y, c=colors[y_kmeans])
            ax.set_title(f"{key}, k= {k}")
    plt.tight_layout()
    plt.savefig(f"{file_nm}.pdf")


def compute():
    answers = {}

    """
    A.	Load the following 5 datasets with 100 samples each: noisy_circles (nc), noisy_moons (nm), blobs with varied variances (bvv), Anisotropicly distributed data (add), blobs (b). Use the parameters from (https://scikit-learn.org/stable/auto_examples/cluster/plot_cluster_comparison.html), with any random state. (with random_state = 42). Not setting the correct random_state will prevent me from checking your results.
    """

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    data = u.load_datasets(n_samples=100)
    dct = answers["1A: datasets"] = data

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
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    colors = np.array(u.colors)
    for data_id, (key, dataset) in enumerate(data.items()):
        for k_id, k in enumerate(ks):
            # labels
            y_kmeans = fit_kmeans(dataset, k=k)
            # X: list of (x,y) coordinates
            X, y = dataset
            ax = axes[k_id, data_id]
            x, y = X[:, 0], X[:, 1]
            ax.scatter(x, y, c=colors[y_kmeans])
            ax.set_title(f"{key}, k= {k}")
    plt.tight_layout()
    plt.savefig("Figure_1C.pdf")

    # dct value: return a dictionary of the abbreviated dataset names (zero or more elements) and associated k-values with correct clusters.
    # key abbreviations: 'nc', 'nm', 'bvv', 'add', 'b'. The values are the list of k for which there is success. Only return datasets where the list of cluster size k is non-empty.

    # 'add' almost works with k=3
    dct = answers["1C: cluster successes"] = {"b": [3]}  # correct
    # dct = answers["1C: cluster successes"] = {"b": [3, 5]}

    # dct value: return a list of dataset abbreviations (list has zero or more elements, which are abbreviated dataset names as strings)
    #dct = answers["1C: cluster failures"] = ["nc", "nm", "bvv", "add"] # orig
    # Test string (strip + lower)
    # Store as list, check as set
    dct = answers["1C: cluster failures"] = ["nM", "nc", "bvv", "add"] # for testing.  Works. But strings not yet lower-cased

    """
    D. Repeat 1.C a few times and comment on which (if any) datasets seem to be sensitive to the choice of initialization for the k=2,3 cases. You do not need to add the additional plots to your report.

    Create a pdf of the plots and return in your report. 
    """

    seeds = [343, 144, 245]
    ks = [2, 3]
    for repeat in range(3):
        plot_data(data, ks=ks, file_nm=f"Figure_1D_{repeat}", seed=seeds[repeat])

    # dct value: list of dataset abbreviations
    # Look at your plots, and return your answers.
    # The plot is part of your report, a pdf file name "report.pdf", in your repository.

    # All datasets appear insensitive to initialization when k=2, 3
    # When k=5, other datasets are sensitive to initialize.

    # No data 'significantly' sensitive to initialization
    dct = answers["1D: datasets sensitive to initialization"] = [""]

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part1.pkl", "wb") as f:
        pickle.dump(answers, f)
