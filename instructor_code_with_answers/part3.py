import numpy as np
import matplotlib.pyplot as plt
import scipy.io as io
from scipy.cluster.hierarchy import dendrogram, linkage

# import plotly.figure_factory as ff
from sklearn.cluster import AgglomerativeClustering
import pickle

"""
Part 3.	
Hierarchical Clustering: 
Recall from lecture that agglomerative hierarchical clustering is a greedy iterative scheme that creates clusters, i.e., distinct sets of indices of points, by gradually merging the sets based on some cluster dissimilarity (distance) measure. Since each iteration merges a set of indices there are at most n-1 mergers until the all the data points are merged into a single cluster (assuming n is the total points). This merging process of the sets of indices can be illustrated by a tree diagram called a dendrogram. Hence, agglomerative hierarchal clustering can be simply defined as a function that takes in a set of points and outputs the dendrogram.
"""

# Fill this function with code at this location. Do NOT move it.
# Change the arguments and return according to
# the question asked.


def Euclidean_distance_matrix(points_I, points_J):
    dist_matrix = np.sqrt(
        ((points_I[:, np.newaxis, :] - points_J[np.newaxis, :, :]) ** 2).sum(axis=2)
    )
    return dist_matrix


def data_index_function(X, I: set, J: set) -> float:
    # Euclidean Distance matrix
    I = np.array(list(I))
    J = np.array(list(J))
    I = X[I]
    J = X[J]
    #print("I= ", I)
    dist_mat = Euclidean_distance_matrix(I, J)
    # dist_mat = np.sqrt(((I[:, np.newaxis, :] - J[np.newaxis, :, :]) ** 2).sum(axis=2))
    min_dist = np.min(dist_mat)
    return min_dist


"""
def plot_dendrogram(model, **kwargs):
    # Create linkage matrix and then plot the dendrogram

    # create the counts of samples under each node
    counts = np.zeros(model.children_.shape[0])
    n_samples = len(model.labels_)
    for i, merge in enumerate(model.children_):
        current_count = 0
        for child_idx in merge:
            if child_idx < n_samples:
                current_count += 1  # leaf node
            else:
                current_count += counts[child_idx - n_samples]
        counts[i] = current_count

    linkage_matrix = np.column_stack(
        [model.children_, model.distances_, counts]
    ).astype(float)

    # Plot the corresponding dendrogram
    dendrogram(linkage_matrix, **kwargs)
"""


def compute():
    answers = {}

    """
    A.	Load the provided dataset “hierarchical_toy_data.mat” using the scipy.io.loadmat function.
    """

    # return value of scipy.io.loadmat()
    data = io.loadmat("hierarchical_toy_data.mat")
    X, y = data["X"], data["y"]

    #print("data: ", data)
    #raise "error"

    answers["3A: toy data"] = data  # dict

    """
    B.	Create a linkage matrix Z, and plot a dendrogram using the scipy.hierarchy.linkage and scipy.hierachy.dendrogram functions, with “single” linkage.
    Include the dendrogram plot in your report (NOT IN ORIGINAL FILE)
    """
    linkage_matrix = linkage(X, "single")
    # Plot the corresponding dendrogram
    dct_dendo = dendrogram(Z=linkage_matrix)
    # Save the Dendrogram to a file
    plt.savefig("dendrogram_3B.pdf")
    print(f"{list(dct_dendo.keys())=}")

    # Answer: NDArray
    answers["3B: linkage"] = linkage_matrix
    #print(f"{linkage_matrix=}")

    # Answer: the return value of the dendogram function, dicitonary
    # Also accept 3B: dendrogram? We'll see
    answers["3B: dendogram"] = dct_dendo

    """
    C.	Consider the merger of the cluster corresponding to points with index sets {I={8,2,13}} J={1,9}}. At what iteration (starting from 0) were these clusters merged? That is, what row does the merger of A correspond to in the linkage matrix Z? The rows count from 0. 
    """

    # Plot the dendogram
    # setting distance_threshold=0 ensures we compute the full tree.
    model = AgglomerativeClustering(
        distance_threshold=0, n_clusters=None, linkage="single"
    )
    model = model.fit(X)
    plt.title("Hierarchical Clustering Dendrogram")
    print("after plot dendrogram")

    I = {8, 2, 13}
    J = {1, 9}

    # Answer type: integer
    # iteration 0: {2, 13} => 14
    # iteration 1: {8, {2, 13}} = {8, 14} => 15
    # iteration 2: {1, 9} => 16
    # iteration 3: {6, 14} => 17
    # iteration 3: {{8, {2, 13}}, {1, 9}} = {15, 16} => 18
    answers["3C: iteration"] = 4

    """
    D.	Write a function that takes the data and the two index sets {I,J} above, and returns the dissimilarity given by single link clustering using the Euclidian distance metric. The function should output the same value as the 3rd column of the row found in problem 2.C.
    """

    min_dist = data_index_function(X, I, J)
    # Answer type: a function defined above
    answers["3D: function"] = data_index_function
    answers["3D: min_dist"] = min_dist  # FORGOTTEN ANSWER in test

    """
    E.	In the actual algorithm, deciding which clusters to merge should consider all of the available clusters at each iteration. List all the clusters as index sets, using a list of lists, 
    e.g., [[0,1,2],[3,4],[5],[6],…],  that were available when the two clusters in part 2.D were merged.
    """

    # List the clusters. the [{0,1,2}, {3,4}, {5}, {6}, ...] represents a list of lists.
    answers["3E: clusters"] = [
        [12],
        [7],
        [3],
        [10],
        [0],
        [11],
        [5],
        [8, 2, 13],
        [1, 9],
        [4],
        [6, 14],
    ]

    """
    F.	Single linked clustering is often criticized as producing clusters where “the rich get richer”, that is, where one cluster is continuously merging with all available points. Does your dendrogram illustrate this phenomenon?
    """

    # Answer type: string. Insert your explanation as a string.
    answers["3F: rich get richer"] = (
        "Partially. The get richer concept is occuring with nodes 12,7,3,10,0, 11, and 5. Node 8,2,13,1,9,4,6,14 are not getting richer one at a time."
    )

    return answers


# ----------------------------------------------------------------------
if __name__ == "__main__":
    answers = compute()

    with open("part3.pkl", "wb") as f:
        pickle.dump(answers, f)
