from sklearn.preprocessing import StandardScaler
from sklearn import cluster
from sklearn import datasets
from typing import Any
import numpy as np
from numpy.typing import NDArray
from sklearn.cluster import AgglomerativeClustering

print("==============> instructor utils")


def load_datasets(n_samples: int) -> dict[str, Any]:
    data: dict[str, tuple[NDArray, NDArray]] = {}

    random_state = 42
    data["nc"] = datasets.make_circles(n_samples=n_samples, factor=0.5, noise=0.05, random_state=random_state)
    data["nm"] = datasets.make_moons(n_samples=n_samples, noise=0.05, random_state=random_state)

    # Anisotropically distributed data
    X, y = datasets.make_blobs(n_samples=n_samples, random_state=random_state)
    transformation = [[0.6, -0.6], [-0.4, 0.8]]
    X_aniso = np.dot(X, transformation)
    data["add"] = (X_aniso, y)

    # Blobs with varied variances random_state = 170 (bvv)
    random_state = 42
    data["bvv"] = datasets.make_blobs(
        n_samples=n_samples, cluster_std=[1.0, 2.5, 0.5], random_state=random_state
    )

    # blobs
    data["b"] = datasets.make_blobs(n_samples=n_samples, random_state=8)

    return data


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# NEW
def fit_transform_kmeans(data):
    # def fit_kmeans(data, k):
    """
    data: (Xtrain, labels)
    """
    xtrain, ytrain = data
    scaler = StandardScaler().fit(xtrain)
    Xtrain = scaler.transform(xtrain)
    # ks = [2, 3, 5, 10]
    y_kmeans = {}
    y_kmeans[k] = fit_kmeans(Xtrain, ytrain, k=k)
    return y_kmeans


# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
def fit_kmeans(X, y, *, k):
    # Is there a random_tate argument?
    # https://scikit-learn.org/stable/modules/generated/sklearn.cluster.KMeans.html
    kmeans = cluster.KMeans(n_clusters=k, init="random", random_state=42)
    kmeans.fit(X)
    y_kmeans = kmeans.predict(X)
    return y_kmeans


# the * means that all arguments following * are keyword arguments
def fit_hierarchical(X, y, *, linkage, k):
    # Is there a random_tate argument?
    cut_off_dist = 0  # ???
    aggclust = AgglomerativeClustering(
        n_clusters=k, linkage=linkage, distance_threshold=cut_off_dist
    )
    aggclust.fit(X)
    # kmeans = cluster.KMeans(n_clusters=k, init="random")
    # kmeans.fit(X)
    # y_kmeans = kmeans.predict(X)
    hier_means = aggclust.predict(X)  # Use training set?
    #print("hier_means= ", hier_means)
    return hier_means


# ----------------------------------------------------------------------
def fit_transform_kmeans(data):
    """
    data: (Xtrain, labels)
    """
    xtrain, ytrain = data
    scaler = StandardScaler().fit(xtrain)
    Xtrain = scaler.transform(xtrain)
    ks = [2, 3, 5, 10]
    y_kmeans = {}
    for k in ks:
        # calls cluster.KMeans with random_state=42
        y_kmeans[k] = fit_kmeans(Xtrain, ytrain, k=k)
    return y_kmeans


# ----------------------------------------------------------------------
def fit_transform_hierarchical(data, linkage=None):
    """
    data: (Xtrain, labels)
    """
    xtrain, ytrain = data
    scaler = StandardScaler().fit(xtrain)
    Xtrain = scaler.transform(xtrain)
    ks = [2]
    y_kmeans = {}
    for k in ks:
        y_kmeans[k] = fit_hierarchical(Xtrain, ytrain, linkage=linkage, k=k)
    return y_kmeans


# ----------------------------------------------------------------------
colors = [
    "red",
    "blue",
    "green",
    "yellow",
    "magenta",
    "orange",
    "brown",
    "plum",
    "gold",
    "lime",
    "slategray",
    "teal",
]

# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
# ----------------------------------------------------------------------
