import math
import numpy as np
from tqdm import tqdm
import torch as torch
import torch.nn as nn
from torch.optim import Adam
import matplotlib.pyplot as plt
from torch.nn import BCELoss

# Model:
#   Input: B x b_size  (B == number samples, single batch of size B))
#   out = W * input  (W: hid x b_size), (input: B x b_size  ==> out: B x hid

def savefig(filenm):
    plt.savefig(filenm)
    plt.close()



def setup_models(Bs, b_size, nb_hiddens, activation):
    # b_size: binary size. E.g. binary_size=4, each sample has 4 bits
    models_dict = {}
    # higher number of nodes in the hidden layer implies higher model
    # complexity
    for hid in nb_hiddens:
        models_dict[hid] = Model(
            input_dim=b_size, out_dim=1, hidden_dim=hid, activation=activation
        )
    return models_dict

def setup_models_3(Bs, b_size, nb_hiddens, activation):
    # b_size: binary size. E.g. binary_size=4, each sample has 4 bits
    models_dict = {}
    # higher number of nodes in the hidden layer implies higher model
    # complexity
    for hid in nb_hiddens:
        models_dict[hid] = Model_3layers(
            input_dim=b_size, out_dim=1, hidden_dim=hid, activation=activation
        )
    return models_dict

class Model(nn.Module):
    """
    A class representing a neural network model.

    Args:
        input_dim (int): The dimension of the input.
        hidden_dim (int, optional): The dimension of the hidden layer. Defaults to 3.
        out_dim (int, optional): The dimension of the output. Defaults to 1.
        activation (function, optional): The activation function to be used. Defaults to torch.sigmoid.

    Attributes:
        input_dim (int): The dimension of the input.
        hidden_dim (int): The dimension of the hidden layer.
        out_dim (int): The dimension of the output.
        W1 (nn.Parameter): The weight parameter for the first layer.
        W2 (nn.Parameter): The weight parameter for the second layer.
        bias1 (nn.Parameter): The bias parameter for the first layer.
        bias2 (nn.Parameter): The bias parameter for the second layer.
        activation (function): The activation function used in the model.
    """

    def __init__(self, input_dim, hidden_dim=3, out_dim=1, activation=torch.sigmoid):
        """
        Initializes the Model class.
        """
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.out_dim = out_dim

        # Xavier gain
        if activation.__name__ == 'sigmoid':
            self.gain = 1.
        elif activation.__name__ == 'relu':
            self.gain = math.sqrt(2.)

        # Parameters to train
        limit1 = np.sqrt(6. / (hidden_dim + input_dim))
        limit2 = np.sqrt(6. / (hidden_dim + out_dim))
        # Use Xavier initialization, programmed manually with a 
        # uniform distribution
        self.W1 = torch.empty(input_dim, hidden_dim)
        self.W2 = torch.empty(hidden_dim, out_dim)
        nn.init.xavier_uniform(self.W1, gain=self.gain)
        nn.init.xavier_uniform(self.W2, gain=self.gain)
        #self.W1 = nn.Parameter(torch.tensor(np.random.uniform(-limit1, limit1, size=(input_dim, hidden_dim)), dtype=torch.float32))
        #self.W2 = nn.Parameter(torch.tensor(np.random.uniform(-limit2, limit2, size=(hidden_dim, out_dim)), dtype=torch.float32))
        # Initialize bias to zero
        self.bias1 = nn.Parameter(torch.zeros(hidden_dim))
        self.bias2 = nn.Parameter(torch.zeros(out_dim))
        # Randomize the bias
        #self.bias1 = nn.Parameter(torch.rand(hidden_dim))
        #self.bias2 = nn.Parameter(torch.rand(out_dim))
        self.activation = activation

    def forward(self, inp):
        """
        Performs the forward pass of the neural network.

        Args:
            inp (torch.Tensor): Input tensor of shape (2**b_size, b_size).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
                - y1: Output tensor of shape (2**b_size, hid).
                - y2: Output tensor of shape (2**b_size).
        """
        y1 = torch.matmul(inp, self.W1) + self.bias1
        y2 = torch.sigmoid(torch.matmul(self.activation(y1), self.W2) + self.bias2)
        return y1, y2.squeeze()

    def my_train(self, inputs, target, n_iter=5000, lr=1.0e-2):
        """
        Trains the model using optimization and returns the list of losses.

        Parameters:
        - target: The target values for training.
        - n_iter: The number of iterations for optimization (default: 5000).
        - lr: The learning rate for optimization (default: 1.0e-2).

        Returns:
        - losses: A list of losses at each iteration.
        """
        print("my_train: self: ", self)
        # Run optimization of model and grab losses
        inp = torch.tensor(inputs.astype(np.float32))
        opt = Adam(self.parameters(), lr=lr)
        losses = []
        criterion = BCELoss()
        for _ in tqdm(range(n_iter)):
            results = self.forward(inp)
            #print("==> len(results): ", len(results))
            out = results[-1]
            #_, out = self.forward(inp)
            opt.zero_grad()
            bce_error = criterion(out, target).sum()  # scalar
            # Average over samples
            bce_error.backward()
            opt.step()
            losses.append(bce_error.item())

        return losses

    def accuracy(self, inputs, target):
        """
        What is the fraction of samples trained correctly?
        Evaluate the output of the model
        """
        _, out = self.forward(torch.tensor(inputs, dtype=torch.float32))
        
        # if out > 0.5, set to 1, out < 0.5, set to 0
        out = [0. if o < 0.5 else 1. for o in out]
        # out and target are 0 and 1
        my_error = np.sum([o - t for o,t in zip(out, target)])
        my_accuracy = 1. - np.abs(my_error) / len(target)
        return my_accuracy



class Model_3layers(Model):
    def __init__(self, input_dim, hidden_dim=3, out_dim=1, activation=torch.sigmoid):
        super(Model_3layers, self).__init__(input_dim, hidden_dim, out_dim, activation)

        # Parameters to train
        limit1 = np.sqrt(6. / (hidden_dim + input_dim))
        limit2 = np.sqrt(6. / (hidden_dim + hidden_dim))
        limit3 = np.sqrt(6. / (hidden_dim + out_dim))
        # Use Xavier initialization, programmed manually with a 
        # uniform distribution
        #self.W1 = nn.Parameter(torch.tensor(np.random.uniform(-limit1, limit1, size=(input_dim, hidden_dim)), dtype=torch.float32))
        #self.W2 = nn.Parameter(torch.tensor(np.random.uniform(-limit2, limit2, size=(hidden_dim, hidden_dim)), dtype=torch.float32))
        #self.W3 = nn.Parameter(torch.tensor(np.random.uniform(-limit2, limit2, size=(hidden_dim, out_dim)), dtype=torch.float32))
        self.W1 = torch.empty(input_dim, hidden_dim)
        self.W2 = torch.empty(hidden_dim, hidden_dim)
        self.W3 = torch.empty(hidden_dim, out_dim)
        nn.init.xavier_uniform(self.W1, gain=self.gain)
        nn.init.xavier_uniform(self.W2, gain=self.gain)
        nn.init.xavier_uniform(self.W3, gain=self.gain)
        # Initialize bias to zero
        self.bias1 = nn.Parameter(torch.zeros(hidden_dim))
        self.bias2 = nn.Parameter(torch.zeros(hidden_dim))
        self.bias3 = nn.Parameter(torch.zeros(out_dim))
        # Randomize the bias
        #self.bias1 = nn.Parameter(torch.rand(hidden_dim))
        #self.bias2 = nn.Parameter(torch.rand(hidden_dim))
        self.activation = activation

    def forward(self, inp):
        """
        Performs the forward pass of the neural network.

        Args:
            inp (torch.Tensor): Input tensor of shape (2**b_size, b_size).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]: A tuple containing two tensors:
                - y1: Output tensor of shape (2**b_size, hid).
                - y2: Output tensor of shape (2**b_size).
        """
        y1 = self.activation(torch.matmul(inp, self.W1) + self.bias1)
        y2 = self.activation(torch.matmul(y1,  self.W2) + self.bias2)
        y3 = torch.sigmoid(torch.matmul(y2,    self.W3))
        return y1, y2, y3.squeeze()

    def my_train(self, inputs, target, n_iter=5000, lr=1.0e-2):
        """
        Trains the model using optimization and returns the list of losses.

        Parameters:
        - target: The target values for training.
        - n_iter: The number of iterations for optimization (default: 5000).
        - lr: The learning rate for optimization (default: 1.0e-2).

        Returns:
        - losses: A list of losses at each iteration.
        """
        print("my_train: self: ", self)
        # Run optimization of model and grab losses
        inp = torch.tensor(inputs.astype(np.float32))
        opt = Adam(self.parameters(), lr=lr)
        losses = []
        criterion = BCELoss()
        for _ in tqdm(range(n_iter)):
            results = self.forward(inp)
            out = results[-1]
            opt.zero_grad()
            bce_error = criterion(out, target).sum()  # scalar
            # Average over samples
            bce_error.backward()
            opt.step()
            losses.append(bce_error.item())

        return losses

    def accuracy(self, inputs, target):
        """
        What is the fraction of samples trained correctly?
        Evaluate the output of the model
        """
        _, _, out = self.forward(torch.tensor(inputs, dtype=torch.float32))
        
        # if out > 0.5, set to 1, out < 0.5, set to 0
        out = [0. if o < 0.5 else 1. for o in out]
        # out and target are 0 and 1
        my_error = np.sum([o - t for o,t in zip(out, target)])
        my_accuracy = 1. - np.abs(my_error) / len(target)
        return my_accuracy

def plot_min_losses(hid_sizes, binary_size, min_losses, title):
    plt.close()
    plot = plt.plot(hid_sizes, min_losses)
    plt.grid(True)
    plt.title(f"{binary_size=}, {title}")
    savefig("part2_min_losses")

def plot2_accuracies(accur, accur3):
    # Plot both sets of accuracies for Models and Models_3
    accur_l = [accur[k] for k in accur.keys()]
    keys = [k for k in accur.keys()]
    accur3_l = [accur3[k] for k in accur3.keys()]
    keys3 = [k for k in accur3.keys()]
    plt.plot(keys, accur_l, color="blue", label="1 layer")
    plt.plot(keys3, accur3_l, color="red", label="2 layers")
    plt.legend()

def plot_model_losses(hidden_sizes, losses, binary_size, title):
    for k in hidden_sizes:
        plt.plot(losses[k], label=f"{k}")
        plt.yscale('log')
    plt.legend()
    plt.title(f"{binary_size=}, {title}")

def plot_losses(model):
    """
    Plots the losses of a given model.

    Parameters:
    model (object): The model object for which the losses will be plotted.

    Returns:
    None
    """
    plt.plot(model.losses)
    plt.yscale("log")
    plt.title(f"losses, d={model.input_dim}, k={model.hidden_dim}")
    plt.grid(True)
    file_nm = f"plot2_losses_d{model.input_dim}_k{model.hidden_dim}.pdf"
    plt.savefig(file_nm)
    plt.close()


# ----------------------------------------------------------------------


# Create binary arrays
def make_binary_arrays(d=2):
    return np.array(
        list(
            map(lambda x: np.array(list("{0:b}".format(x).zfill(d))), range(int(2**d)))
        )
    ).astype(int)


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


def generalized_xor(values):
    """
    Perform a cascaded XOR operation in a vectorized manner using NumPy.

    :param values: A NumPy array of boolean values, of shape [nrows, dbits].
    :return: The result of the cascaded XOR operation.
    """
    # Convert boolean array to integers (True to 1, False to 0) and sum them up
    sum_of_values = np.sum(values.astype(int), axis=1)

    # print("sum_of_values= ", sum_of_values)

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


def plot1a(iterations, losses):
    plt.plot(iterations, losses)
    plt.title("Plot of Losses vs. Iterations")
    plt.xlabel("(Iterations)")
    plt.ylabel("(losses)")
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
    plt.savefig("part1_classification_grid_points_a")


def plot1c(x, y_line, test, colors):
    plt.scatter(test[:, 0], test[:, 1], color=colors[:])
    plt.scatter(0, 0, color="yellow")
    plt.scatter(0, 1, color="green")
    plt.scatter(1, 0, color="green")
    plt.scatter(1, 1, color="green")
    plt.plot(x, y_line, color="black")
    plt.title("Classification of Grid of Points")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.savefig("part1_classification_grid_points_b")


def plot1d(y1_tt, y1_np, colors2, colors3):
    plt.scatter(y1_tt[:, 0], y1_tt[:, 1], color=colors2[:])
    plt.scatter(y1_np[:, 0], y1_np[:, 1], color=colors3[:])
    plt.title("Classification of Grid of Points with Hidden Layer Before Sigmoid")
    plt.xlabel("y1_i")
    plt.ylabel("y1_j")
    plt.savefig("part1_classification_grid_points_hidden_before_sigmoid.pdf")


def plot1e(x, y_line, y_line2, y1_tts, y1_nps, colors2, colors3):
    # Plot outputs
    plt.scatter(y1_tts[:, 0], y1_tts[:, 1], color=colors2[:])
    plt.scatter(y1_nps[:, 0], y1_nps[:, 1], color=colors3[:])
    plt.plot(x, y_line, color="black")
    plt.plot(x, y_line2, color="black")
    plt.title("Classification of Grid of Points with Hidden Layer After Sigmoid")
    plt.xlabel("sigmoid(y1_i)")
    plt.ylabel("sigmoid(y1_j)")
    plt.savefig("part1_classification_grid_points_hidden_after_sigmoid.pdf")
    plt.close()


def plot2a(B, target):
    fig = plt.figure()
    ax = fig.add_subplot(projection="3d")  # makes it 3D
    target = np.asarray(target).astype(int)
    new_colors = [["red", "blue"][i] for i in target % 2]
    plot = ax.scatter(B[:, 0], B[:, 1], B[:, 2], c=new_colors)
    plt.savefig("plot2a, titlexxx.pdf")
    plt.close()
    return plot


def plot2b(kk, losses_2):
    plt.title("Minimum loss vs k-value(number of Hidden Dimensions)")
    plt.xlabel("Value of K")
    plt.ylabel("Minimum Loss")
    plt.plot(kk[4:7], losses_2[4:7])
    plt.title("Minimum loss vs k-value(number of Hidden Dimensions)")
    plt.xlabel("Value of K")
    plt.ylabel("Minimum Loss")
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
    plt.savefig("plot2c.pdf")
    plt.close()
