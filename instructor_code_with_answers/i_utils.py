import numpy as np
from tqdm import tqdm
import torch as torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt


def set_seed(seed):
    """
    Ensure consistent results
    Parameters:
    seed: choose an integer seed to ensure consistent results
        in torch and numpy
    """
    torch.manual_seed(seed)
    np.random.seed(seed)


def colors():
    return ["blue", "red", "green", "magenta", "cyan"] * 2


class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim=3, out_dim=1, activation=torch.sigmoid):
        super(Model, self).__init__()
        self.W1 = nn.Parameter(torch.rand(hidden_dim, input_dim))
        self.W2 = nn.Parameter(torch.rand(out_dim, hidden_dim))
        self.bias1 = nn.Parameter(torch.zeros(hidden_dim))
        self.bias2 = nn.Parameter(torch.zeros(out_dim))
        self.activation = activation

    def forward(self, inp):
        #print(f"forward, {type(inp)=}, {inp.shape=}") # 8 x 3
        #print(f"forward, {inp[0,0].dtype=}")  # int64
        #print(f"forward, {type(self.W1)=}, {self.W1.shape=}") # 3 x 3
        #print(f"forward, {self.W1[0,0]=}") 
        #print(f"forward, {self.W1[0,0].dtype=}") 
        #print(f"forward, {self.bias1[0].dtype=}")
        #print(f"forward, {type(self.bias1)=}, {self.bias1.shape=}") # 3
        y1 = batch_matvec(self.W1, inp)
        y1 = self.bias1 + y1
        # y2 = torch.sigmoid(batch_matvec(self.W2, torch.sigmoid(y1)))
        y2 = self.activation(batch_matvec(self.W2, torch.sigmoid(y1)) + self.bias2) # works better
        # print(y2.shape)
        return y1, y2.squeeze()

def run_model(model, B, target, nb_hid=4, d=3, n_iter=5000):
    # Run optimization of model and grab losses
    # losses = run_optimization(model, B, target, error_fn=lambda x: skewed_error(x,alpha=0), n_iter=1000)

    # I need 5 planes. That is surprising.
    # Plot the graph in log units along y

    # Target not used
    B = B.astype(np.float32)
    losses = run_optimization(model, B, target, n_iter=3000)
    plt.plot(losses)
    plt.yscale("log")
    plt.title("losses, d={d}, k={nb_hid}")
    plt.grid(True)
    file_nm = f"plot2_losses_d{d}_k{nb_hid}.pdf"
    plt.savefig(file_nm)
    return losses


# Implement training of network
def run_optimization(model, B, target, n_iter=1000, lr=1e-1):
    opt = Adam(model.parameters(), lr=lr)
    losses = []
    for k in tqdm(range(n_iter)):
        # print(type(B))
        #print(f"run_optimization, {B=}")
        #print(f"run_optimization, {B.shape=}")
        #print(f"run_optimization, {type(B)=}")  # numpy array
        #print(f"run_optimization, {B[0,0].dtype=}") # int64
        #print("B= ", B)
        BB = torch.tensor(B)
        #print(f"{type(BB)}")
        #print(f"{BB[0].dtype}")
        y1, out = model(BB)  ### ERROR
        opt.zero_grad()
        sqr_error = (target - out) ** 2
        loss = torch.sum(sqr_error)
        loss.backward()
        opt.step()
        losses.append(loss.item())
    return losses


# Create binary arrays
def make_binary_arrays(d=2):
    return np.array(
        list(
            map(lambda x: np.array(list("{0:b}".format(x).zfill(d))), range(int(2**d)))
        )
    ).astype(int)


def generalized_xor(values):
    """
    Perform a cascaded XOR operation in a vectorized manner using NumPy.

    :param values: A NumPy array of boolean values, of shape [nrows, dbits].
    :return: The result of the cascaded XOR operation.
    """
    # Convert boolean array to integers (True to 1, False to 0) and sum them up
    sum_of_values = np.sum(values.astype(int), axis=1)

    #print("sum_of_values= ", sum_of_values)

    # XOR is True (1) if the sum is odd, otherwise False (0)
    xor_result = sum_of_values % 2 == 1

    return xor_result.astype(int)


def vectorized_cascaded_xor_all_rows(values):
    """
    Perform a cascaded XOR operation in a vectorized manner using NumPy.

    :param values: A NumPy array of boolean values.
        - array of shape n_rows x nb_bits
    :return: The result of the cascaded XOR operation.
    """
    # Convert boolean array to integers (True to 1, False to 0) and sum them up
    sum_of_values = np.sum(values.astype(int), axis=1)

    # XOR is True (1) if the sum is odd, otherwise False (0)
    xor_result = (sum_of_values % 2 == 1).astype(int)

    return xor_result


# Example usage:
# values = np.array([True, False, True, True])
# print(vectorized_cascaded_xor(values))


# Example usage:
# values = [True, False, True, True]  # This example should return True
# print(cascaded_xor(values))


# Einstein summation for neural network function
def batch_matvec(A, B):
    """
    i: sample
    j: input_dim
    k: hidden_dim
    [hidden_dim,input_dim]
    [sample_size, input_dim]
    [sample_size, hidden_dim]
    """
    #print(f"{A.shape=}, {B.shape=}")  # 2,5  and 8,3
    return torch.einsum("kj,ij->ik", A, B)


def plot1a(iterations, losses):
    plt.plot(iterations, losses)
    plt.title("Plot of Losses vs. Iterations")
    plt.xlabel("(Iterations)")
    plt.ylabel("(losses)")
    # plt.show()
    plt.savefig("part1_losses_vs_iterations.pdf")


def plot1b(test, colors):
    # Plot outputs from model
    plt.scatter(test[:, 0], test[:, 1], color=colors[:])
    plt.scatter(0, 0, color="yellow")
    plt.scatter(0, 1, color="green")
    plt.scatter(1, 0, color="green")
    plt.scatter(1, 1, color="yellow")
    plt.title("Classification of Grid of Points")
    plt.xlabel("X")
    plt.ylabel("Y")
    # plt.show()
    plt.savefig("part1_classification_grid_points_a")


def plot1c(x, y_line, test, colors):
    plt.scatter(test[:, 0], test[:, 1], color=colors[:])
    plt.scatter(0, 0, color="yellow")
    plt.scatter(0, 1, color="green")
    plt.scatter(1, 0, color="green")
    plt.scatter(1, 1, color="green")
    plt.plot(x, y_line, color="black")
    # plt.plot(x,y_line2,color='black')
    plt.title("Classification of Grid of Points")
    plt.xlabel("X")
    plt.ylabel("Y")
    # plt.show()
    plt.savefig("part1_classification_grid_points_b")


def plot1d(y1_tt, y1_np, colors2, colors3):
    plt.scatter(y1_tt[:, 0], y1_tt[:, 1], color=colors2[:])
    plt.scatter(y1_np[:, 0], y1_np[:, 1], color=colors3[:])
    # plt.plot(x,y_line, color='black')
    plt.title("Classification of Grid of Points with Hidden Layer Before Sigmoid")
    plt.xlabel("y1_i")
    plt.ylabel("y1_j")
    # plt.show()
    plt.savefig("part1_classification_grid_points_hidden_before_sigmoid.pdf")
    # Plot after sigmoid


def plot1e(x, y_line, y_line2, y1_tts, y1_nps, colors2, colors3):
    # Plot outputs
    plt.scatter(y1_tts[:, 0], y1_tts[:, 1], color=colors2[:])
    plt.scatter(y1_nps[:, 0], y1_nps[:, 1], color=colors3[:])
    plt.plot(x, y_line, color="black")
    plt.plot(x, y_line2, color="black")
    plt.title("Classification of Grid of Points with Hidden Layer After Sigmoid")
    plt.xlabel("sigmoid(y1_i)")
    plt.ylabel("sigmoid(y1_j)")
    # plt.show()
    plt.savefig("part1_classification_grid_points_hidden_after_sigmoid.pdf")
    plt.close()


def plot2a(B, target):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    #print(f"plot2a, {target[:]=}")
    #print(f"plot2a, {target=}")
    target = np.asarray(target).astype(int)
    new_colors = [["red", "blue"][i] for i in target % 2]
    alpha = [1] * len(target)
    ax.scatter(B[:, 0], B[:, 1], B[:, 2], c=new_colors)
    plt.savefig("plot2a, titlexxx.pdf")
    plt.close()


def plot2b(kk, losses_2):
    plt.title("Minimum loss vs k-value(number of Hidden Dimensions)")
    plt.xlabel("Value of K")
    plt.ylabel("Minimum Loss")
    #print(f"{len(kk)=}, {len(losses_2)=}")
    plt.plot(kk[4:7], losses_2[4:7])
    plt.title("Minimum loss vs k-value(number of Hidden Dimensions)")
    plt.xlabel("Value of K")
    plt.ylabel("Minimum Loss")
    # plt.show()
    plt.savefig("plot2b, min loss vs k.pdf")
    plt.close()


def plot2c(x, y, z1, z2, z3, z4, z5, B2, colors):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")
    ax.scatter(B2[:, 0], B2[:, 1], B2[:, 2], color=colors[:])
    ax.plot_surface(x, y, z1)
    ax.plot_surface(x, y, z2)
    ax.plot_surface(x, y, z3)
    ax.plot_surface(x, y, z4)
    ax.plot_surface(x, y, z5)
    # plt.show()
    plt.savefig("plot2c.pdf")
    plt.close()
