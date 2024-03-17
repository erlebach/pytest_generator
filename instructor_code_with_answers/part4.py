import numpy as np
import matplotlib.pyplot as plt
from sklearn import cluster
from sklearn.preprocessing import StandardScaler
from scipy.cluster.hierarchy import linkage as linkage_fct

# to calculate the clusters given the threshold
from scipy.cluster.hierarchy import fcluster

# to smooth the data
from scipy.signal import savgol_filter

# import plotly.figure_factory as ff
from sklearn.cluster import AgglomerativeClustering
import pickle
import utils as u
from typing import Literal
from numpy.typing import NDArray

"""
Part 4.	
Evaluation of Hierarchical Clustering over Diverse Datasets:
In this task, you will explore hierarchical clustering over different datasets. You will also evaluate different ways to merge clusters and good ways to find the cut-off point for breaking the dendrogram.
"""

# Fill these two functions with code at this location. Do NOT move it.
# Change the arguments and return according to
# the question asked.


def fit_hierarchical_cluster(
    dataset: tuple[NDArray],
    linkage: Literal["ward", "average", "complete", "single"],
    k: int,
):
    X, y = dataset
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    agglom = AgglomerativeClustering(n_clusters=k, linkage=linkage, metric="euclidean")
    y_agglom = agglom.fit_predict(X)
    return y_agglom


def get_distance_threshold(Z):
    iterations = list(range(len(Z)))
    distance = Z[:, 2]

    # Look at plots to evaluate smoothness estimates
    smoothed_distance = savgol_filter(
        distance, window_length=5, polyorder=2, mode="nearest"
    )
    # smoothed_distance = distance
    #  Formulas assume that the iterations are equally spaced
    max_slope_index = np.argmax(np.gradient(smoothed_distance))
    max_curvature_index = np.argmax(np.gradient(np.gradient(smoothed_distance)))
    dct_slope = {}
    dct_curvature = {}
    dct_slope["max_index"] = max_slope_index
    dct_slope["thresh_dist"] = smoothed_distance[max_slope_index]
    dct_slope["smooth_dist"] = smoothed_distance
    dct_slope["dist"] = distance
    dct_curvature["smooth_dist"] = smoothed_distance
    dct_curvature["max_index"] = max_curvature_index
    dct_curvature["thresh_dist"] = smoothed_distance[max_curvature_index]
    dct_curvature["dist"] = distance
    return dct_slope, dct_curvature


def fit_modified(
    dataset: tuple[NDArray],
    linkage: Literal["ward", "average", "complete", "single"],
):
    X, y = dataset
    scaler = StandardScaler().fit(X)
    X = scaler.transform(X)
    Z = linkage_fct(X, linkage)
    slope_threshold, curvature_threshold = get_distance_threshold(Z)
    dct_slope, dct_curvature = get_distance_threshold(Z)
    slope_agglom = AgglomerativeClustering(
        n_clusters=None,
        linkage=linkage,
        metric="euclidean",
        distance_threshold=dct_slope["thresh_dist"],
    )
    curvature_agglom = AgglomerativeClustering(
        n_clusters=None,
        linkage=linkage,
        metric="euclidean",
        distance_threshold=dct_curvature["thresh_dist"],
    )
    y_slope_agglom = slope_agglom.fit_predict(X)
    y_curvature_agglom = curvature_agglom.fit_predict(X)
    return y_slope_agglom, y_curvature_agglom, dct_slope, dct_curvature


"""
def plot_data(data, ks, file_nm, seed=42):
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    colors = np.array(u.colors)
    for data_id, (key, dataset) in enumerate(data.items()):
        for k_id, k in enumerate(ks):
            # labels
            y_kmeans = fit_kmeans(dataset, k=k, seed=seed)
            print(f"{key=}, {k=}, {seed=}, {y_kmeans=}")
            # X: list of (x,y) coordinates
            X, y = dataset
            ax = axes[k_id, data_id]
            x, y = X[:, 0], X[:, 1]
            ax.scatter(x, y, c=colors[y_kmeans])
            ax.set_title(f"{key}, k= {k}")
    plt.tight_layout()
    plt.savefig(f"{file_nm}.pdf")
"""


def compute():
    answers = {}

    """
    A.	Repeat parts 1.A and 1.B with hierarchical clustering. That is, write a function called fit_hierarchical_cluster (or something similar) that takes the dataset, the linkage type and the number of clusters, that trains an AgglomerativeClustering sklearn estimator and returns the label predictions. Apply the same standardization as in part 1.B. Use the default distance metric (euclidean) and the default linkage (ward).
    """

    # Dictionary of 5 datasets. e.g., dct["nc"] = [data, labels]
    # keys: 'nc', 'nm', 'bvv', 'add', 'b' (abbreviated datasets)
    data = u.load_datasets(n_samples=100)
    dct = answers["4A: datasets"] = data

    # dct value:  the `fit_hierarchical_cluster` function
    dct = answers["4A: fit_hierarchical_cluster"] = fit_hierarchical_cluster

    """
    B.	Apply your function from 4.A and make a plot similar to 1.C with the four linkage types (single, complete, ward, centroid: rows in the figure), and use 2 clusters for all runs. Compare the results to problem 1, specifically, are there any datasets that are now correctly clustered that k-means could not handle?

    Create a pdf of the plots and return in your report. 
    """

    k = 2
    linkages = ["single", "complete", "ward", "average"]
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    colors = np.array(u.colors)
    for data_id, (key, dataset) in enumerate(data.items()):
        for link_id, linkage in enumerate(linkages):
            # labels
            #print("linkage: ", linkage, type(linkage))
            y_kmeans = fit_hierarchical_cluster(dataset, k=k, linkage=linkage)
            # X: list of (x,y) coordinates
            X, y = dataset
            ax = axes[link_id, data_id]
            x, y = X[:, 0], X[:, 1]
            ax.scatter(x, y, c=colors[y_kmeans])
            ax.set_title(f"{key}, linkage: {linkage}")
    plt.tight_layout()
    plt.savefig("Figure_4B.pdf")

    # dct value: list of dataset abbreviations (see 1.C)
    dct = answers["4B: cluster successes"] = ["nc", "nm"]

    """
    C.	There are essentially two main ways to find the cut-off point for breaking the diagram: specifying the number of clusters and specifying a maximum distance. The latter is challenging to optimize for without knowing and/or directly visualizing the dendrogram, however, sometimes simple heuristics can work well. The main idea is that since the merging of big clusters usually happens when distances increase, we can assume that a large distance change between clusters means that they should stay distinct. Modify the function from part 1.A to calculate a cut-off distance before classification. Specifically, estimate the cut-off distance as the maximum rate of change of the distance between successive cluster merges (you can use the scipy.hierarchy.linkage function to calculate the linkage matrix with distances). Apply this technique to all the datasets and make a plot similar to part 4.B.

    linkages = ["single", "complete", "ward", "average"]
    
    Create a pdf of the plots and return in your report. 
    """

    linkages = ["single", "complete", "ward", "average"]
    fig, axes = plt.subplots(4, 5, figsize=(20, 16))
    fig1, axes1 = plt.subplots(4, 5, figsize=(20, 16))
    # Elbow curve
    fig2, axes2 = plt.subplots(4, 5, figsize=(20, 16))
    colors = np.array(u.colors)
    for data_id, (key, dataset) in enumerate(data.items()):
        for link_id, linkage in enumerate(linkages):
            ax = axes[link_id, data_id]
            ax1 = axes1[link_id, data_id]
            ax2 = axes2[link_id, data_id]
            X, y = dataset
            y_labels_slope, y_labels_curvature, dct_slope, dct_curvature = fit_modified(
                dataset, linkage
            )

            X, y = dataset  # untransformed
            # Transform data in the same w as in fit_hiearchical_cluster
            scaler = StandardScaler().fit(X)

            X = scaler.transform(X)
            x, y = X[:, 0], X[:, 1]

            ax.scatter(x, y, c=colors[y_labels_slope])
            ax1.scatter(x, y, c=colors[y_labels_curvature])
            ax.set_title(f"{key}, linkage: {linkage}")
            ax1.set_title(f"{key}, linkage: {linkage}")
            ax2.set_title(f"Elbow: {key}, linkage: {linkage}")
            x_s, y_s = dct_slope["max_index"], dct_slope["thresh_dist"]
            x_c, y_c = dct_curvature["max_index"], dct_curvature["thresh_dist"]

            Z = linkage_fct(X, linkage)
            # ax2.plot(range(len(Z)), Z[:, 2])
            ax2.plot(range(len(Z)), dct_slope["smooth_dist"])
            ax2.plot(range(len(Z)), dct_slope["dist"])
            ax2.scatter(x_s, y_s, marker="o", color="red", label="max slope")
            ax2.scatter(x_c, y_c, marker="+", color="blue", label="max curvature")
            ax2.legend()
            ax.grid(True)
            ax1.grid(True)
            ax2.grid(True)
    fig.tight_layout()
    fig1.tight_layout()
    fig2.tight_layout()
    fig.savefig("Figure_slope_4C.pdf")
    fig1.savefig("Figure_curvature_4C.pdf")
    fig2.savefig("Elbow_curves_4C.pdf")

    # dct is the function described above in 4.C
    dct = answers["4C: modified function"] = fit_modified

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part4.pkl", "wb") as f:
        pickle.dump(answers, f)
