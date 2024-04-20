import numpy as np

# ----------------------------------------------------------------------


def compute_SSE(data, labels):
    """
    Calculate the sum of squared errors (SSE) for a clustering.

    Parameters:
    - data: numpy array of shape (n, 2) containing the data points
    - labels: numpy array of shape (n,) containing the cluster assignments

    Returns:
    - sse: the sum of squared errors
    """
    # ADD STUDENT CODE
    sse = 0.0
    for i in np.unique(labels):
        cluster_points = data[labels == i]
        cluster_center = np.mean(cluster_points, axis=0)
        sse += np.sum((cluster_points - cluster_center) ** 2)
    return sse


# ----------------------------------------------------------------------

'''
def adjusted_rand_index(labels_true, labels_pred) -> float:
    """
    Compute the adjusted Rand index.

    Parameters:
    - labels_true: The true labels of the data points.
    - labels_pred: The predicted labels of the data points.

    Returns:
    - ari: The adjusted Rand index value.

    The adjusted Rand index is a measure of the similarity between two data clusterings.
    It takes into account both the similarity of the clusters themselves and the similarity
    of the data points within each cluster. The adjusted Rand index ranges from -1 to 1,
    where a value of 1 indicates perfect agreement between the two clusterings, 0 indicates
    random agreement, and -1 indicates complete disagreement.
    """
    # Create contingency table
    contingency_table = np.histogram2d(
        labels_true,
        labels_pred,
        bins=(np.unique(labels_true).size, np.unique(labels_pred).size),
    )[0]

    # Sum over rows and columns
    sum_combinations_rows = np.sum(
        [np.sum(nj) * (np.sum(nj) - 1) / 2 for nj in contingency_table]
    )
    sum_combinations_cols = np.sum(
        [np.sum(ni) * (np.sum(ni) - 1) / 2 for ni in contingency_table.T]
    )

    # Sum of combinations for all elements
    N = np.sum(contingency_table)
    sum_combinations_total = N * (N - 1) / 2

    # Calculate ARI
    ari = (
        np.sum([np.sum(n_ij) * (np.sum(n_ij) - 1) / 2 for n_ij in contingency_table])
        - (sum_combinations_rows * sum_combinations_cols) / sum_combinations_total
    ) / (
        (sum_combinations_rows + sum_combinations_cols) / 2
        - (sum_combinations_rows * sum_combinations_cols) / sum_combinations_total
    )

    return ari
'''

import numpy as np


def adjusted_rand_index(labels_true, labels_pred) -> float:
    """
    Compute the adjusted Rand index.

    Parameters:
    - labels_true: The true labels of the data points.
    - labels_pred: The predicted labels of the data points.

    Returns:
    - ari: The adjusted Rand index value.
    """
    # Create contingency table
    contingency_table = np.histogram2d(
        labels_true,
        labels_pred,
        bins=(np.unique(labels_true).size, np.unique(labels_pred).size),
    )[0]

    # Sum over rows and columns
    sum_combinations_rows = np.sum(
        (contingency_table.sum(axis=1) * (contingency_table.sum(axis=1) - 1)) / 2
    )
    sum_combinations_cols = np.sum(
        (contingency_table.sum(axis=0) * (contingency_table.sum(axis=0) - 1)) / 2
    )

    # Sum of combinations for all elements
    N = contingency_table.sum()
    sum_combinations_total = N * (N - 1) / 2

    # Sum of combinations for each cell in the contingency table
    sum_combinations_within = np.sum((contingency_table * (contingency_table - 1)) / 2)

    # Calculate ARI
    expected_index = (
        sum_combinations_rows * sum_combinations_cols / sum_combinations_total
    )
    max_index = (sum_combinations_rows + sum_combinations_cols) / 2
    ari = (sum_combinations_within - expected_index) / (max_index - expected_index)

    return ari


# ----------------------------------------------------------------------


def k_means(data, k, max_iter=100, tol=1e-4):
    """
    K-means clustering algorithm using only NumPy.

    Parameters:
    - data: numpy.ndarray, shape (n_samples, n_features)
    - k: int, number of clusters
    - max_iter: int, maximum number of iterations (default: 100)
    - tol: float, tolerance to declare convergence (default: 1e-4)

    Returns:
    - centroids: numpy.ndarray, shape (k, n_features)
    - labels: numpy.ndarray, shape (n_samples,)
    """
    # Randomly initialize centroids by selecting k data points
    indices = np.random.choice(data.shape[0], k, replace=False)
    centroids = data[indices]

    for iteration in range(max_iter):
        # Assignment step: Assign each point to the nearest centroid
        distances = np.sqrt(((data[:, np.newaxis] - centroids) ** 2).sum(axis=2))
        labels = np.argmin(distances, axis=1)

        # Update step: Compute new centroids from the means of the points
        new_centroids = np.array([data[labels == i].mean(axis=0) for i in range(k)])

        # Convergence check: If centroids do not change significantly, break
        if np.all(np.linalg.norm(new_centroids - centroids, axis=1) < tol):
            break

        centroids = new_centroids

    return centroids, labels


# ----------------------------------------------------------------------

import numpy as np
from sklearn.neighbors import NearestNeighbors

import numpy as np
from sklearn.neighbors import NearestNeighbors


def create_symmetric_knn_proximity_graph(data, k, sigma=None):
    """
    Create a symmetric k-nearest neighbor proximity graph using a Gaussian kernel from 2D data.

    Parameters:
    - data (numpy.ndarray): An array of shape (N, 2) containing N points in 2D.
    - k (int): The number of nearest neighbors to consider.
    - sigma (float, optional): The standard deviation used in the Gaussian kernel. If None,
                               it is estimated from the data.

    Returns:
    - A (numpy.ndarray): A symmetric weighted adjacency matrix using Gaussian kernel.
    """
    # Initialize NearestNeighbors with the specified number of neighbors
    nbrs = NearestNeighbors(n_neighbors=k + 1, algorithm="auto").fit(data)

    # Find the k nearest neighbors (including the point itself)
    distances, indices = nbrs.kneighbors(data)

    N = data.shape[0]
    A = np.zeros((N, N))

    if sigma is None:
        # Estimate sigma using the mean distance to the nearest neighbor
        sigma = np.mean(
            distances[:, 1]
        )  # using the second column because the first is zero (distance to itself)

    # Populate the adjacency matrix using Gaussian kernel weights
    for i in range(N):
        for j in range(1, k + 1):  # start from 1 to skip the point itself
            A[i, indices[i, j]] = np.exp(-(distances[i, j] ** 2) / (2 * sigma**2))

    # Symmetrize the matrix to make it undirected
    A = np.maximum(A, A.T)  # Take the maximum of A and its transpose

    return A


# Example usage
data = np.random.rand(100, 2)  # 100 points in 2D
k = 5  # Number of nearest neighbors
proximity_matrix = create_symmetric_knn_proximity_graph(data, k)

print("Symmetric Proximity Matrix Using Gaussian Kernel:\n", proximity_matrix)
# ----------------------------------------------------------------------

import numpy as np


def euclidean_distances(X, Y=None):
    """
    Compute the squared Euclidean distances between each point in X and Y.

    Parameters:
    - X (numpy.ndarray): An array of shape (n_samples_X, n_features) containing the dataset.
    - Y (numpy.ndarray): An optional array of shape (n_samples_Y, n_features) containing another dataset.
                         If None, Y is assumed to be the same as X.

    Returns:
    - distances (numpy.ndarray): An array of shape (n_samples_X, n_samples_Y) containing the squared distances.
    """
    if Y is None:
        Y = X
    # Calculate the squared differences expanded out in the Euclidean distance formula
    diff = X[:, np.newaxis, :] - Y[np.newaxis, :, :]
    distances = np.sum(diff**2, axis=2)
    return distances


# ----------------------------------------------------------------------


# NOT DEBUGGED
def nearest_neighbors(data, k=5):
    """
    Find k-nearest neighbors for each point in the dataset using only numpy.

    Parameters:
    - data (numpy.ndarray): The dataset of shape (N, features).
    - k (int): The number of nearest neighbors to find.

    Returns:
    - indices (numpy.ndarray): Indices of the k-nearest neighbors for each point.
    - distances (numpy.ndarray): Distances to the k-nearest neighbors for each point.
    """
    # Compute pairwise Euclidean distances
    dist_matrix = euclidean_distances(data)

    # We use np.argsort to get indices of sorted distances
    sorted_indices = np.argsort(dist_matrix, axis=1)

    # Select the first k indices for each point - note we use 1:k+1 to skip the point itself
    nearest_indices = sorted_indices[:, 1 : k + 1]
    nearest_distances = np.sort(dist_matrix, axis=1)[:, 1 : k + 1]

    return nearest_indices, nearest_distances


# Example usage:
data = np.random.rand(10, 2)  # 10 points in 2D
k = 3  # Number of nearest neighbors
indices, distances = nearest_neighbors(data, k)

print("Indices of Nearest Neighbors:\n", indices)
print("Distances to Nearest Neighbors:\n", distances)

# ----------------------------------------------------------------------
