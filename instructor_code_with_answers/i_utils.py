import math


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

    # blobs (I AM SUSPICIOUS
    data["b"] = datasets.make_blobs(n_samples=n_samples, random_state=random_state)

    return data

def log2(x):
    return math.log(x, 2)



class BinaryTree:
    def __init__(self, value):
        self.value = value
        self.left = None
        self.right = None

    def insert_left(self, value):
        if self.left is None:
            self.left = BinaryTree(value)
            return self.left
        else:
            new_node = BinaryTree(value)
            new_node.left = self.left
            self.left = new_node
            return new_node

    def insert_right(self, value):
        if self.right is None:
            self.right = BinaryTree(value)
            return self.right
        else:
            new_node = BinaryTree(value)
            new_node.right = self.right
            self.right = new_node
            return new_node

    def __repr__(self):
        return f"BinaryTree({self.value})"

    def __str__(self, level=0):
        ret = "\t" * level + repr(self) + "\n"
        if self.left is not None:
            ret += self.left.__str__(level + 1)
        if self.right is not None:
            ret += self.right.__str__(level + 1)
        return ret

    def print_tree(self):
        print(self.__str__())


# Example on how to create a binary tree
# A has two children: B and C
# B has two children: D and E
# C has two children: F and G
# Construct the binary tree:
def construct_binary_tree():
    root = BinaryTree("A")
    root.insert_left("B")
    root.insert_right("C")
    root.left.insert_left("D")
    root.left.insert_right("E")
    root.right.insert_left("F")
    root.right.insert_right("G")
    return root
