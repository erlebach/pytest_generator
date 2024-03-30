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

class Model(nn.Module):
    def __init__(self, input_dim, hidden_dim=3, out_dim=1): 
        super(Model, self).__init__()
        self.W1 = nn.Parameter(torch.rand(hidden_dim, input_dim)) 
        self.W2 = nn.Parameter(torch.rand(out_dim, hidden_dim)) 
        self.bias1=nn.Parameter(torch.zeros(hidden_dim)) 
        self.bias2=nn.Parameter(torch.zeros(out_dim))

    def forward(self,inp):
        y1 = batch_matvec(self.W1, inp) 
        #print(y1.shape)
        y1= self.bias1+y1
        #y2 = torch.sigmoid(batch_matvec(self.W2, torch.sigmoid(y1)))
        y2 = torch.sigmoid(batch_matvec(self.W2, torch.sigmoid(y1))+self.bias2) 
        #print(y2.shape)
        return y1,y2.squeeze()


#Implement training of network
def run_optimization(model, B, target, n_iter=1000, lr=1e-1): 
    opt = Adam(model.parameters(), lr=lr)
    losses = []
    print(f"{B.shape=}")  # 8, 3
    for k in tqdm(range(n_iter)): 
        y1,out = model(torch.clone(B).detach())    ### ERROR
        opt.zero_grad()
        sqr_error = (target-out)**2 
        loss = torch.sum(sqr_error) 
        loss.backward()
        opt.step() 
        losses.append(loss.item())
    return losses

#Create binary arrays
def make_binary_arrays(d=2):
    return np.array(list(
        map(lambda x: np.array(list("{0:b}".format(x).zfill(d))),range(int(2**d)))
        )).astype(int)

#Einstein summation for neural network function 
def batch_matvec(A,B):
    """
    i: sample
    j: input_dim 
    k: hidden_dim
    [hidden_dim,input_dim] 
    [sample_size, input_dim] 
    [sample_size, hidden_dim]
    """
    return torch.einsum('kj,ij->ik', A, B)

def plot1a(iterations, losses):
    plt.plot(iterations,losses)
    plt.title("Plot of Losses vs. Iterations") 
    plt.xlabel("(Iterations)") 
    plt.ylabel("(losses)")
    #plt.show()
    plt.savefig("part1_losses_vs_iterations.pdf")

def plot1b(test, colors):
    #Plot outputs from model 
    plt.scatter(test[:,0],test[:,1],color=colors[:]) 
    plt.scatter(0,0, color='yellow') 
    plt.scatter(0,1, color='green') 
    plt.scatter(1,0, color='green') 
    plt.scatter(1,1, color='yellow') 
    plt.title("Classification of Grid of Points") 
    plt.xlabel("X")
    plt.ylabel("Y")
    #plt.show()
    plt.savefig("part1_classification_grid_points_a")

def plot1c(x, y_line, test, colors):
    plt.scatter(test[:,0],test[:,1],color=colors[:]) 
    plt.scatter(0,0, color='yellow') 
    plt.scatter(0,1, color='green') 
    plt.scatter(1,0, color='green') 
    plt.scatter(1,1, color='green') 
    plt.plot(x,y_line, color='black') 
    #plt.plot(x,y_line2,color='black') 
    plt.title("Classification of Grid of Points") 
    plt.xlabel("X")
    plt.ylabel("Y")
    #plt.show()
    plt.savefig("part1_classification_grid_points_b")

def plot1d(y1_tt, y1_np, colors2, colors3):
    plt.scatter(y1_tt[:,0],y1_tt[:,1],color=colors2[:])
    plt.scatter(y1_np[:,0],y1_np[:,1], color=colors3[:]) 
    #plt.plot(x,y_line, color='black')
    plt.title("Classification of Grid of Points with Hidden Layer Before Sigmoid") 
    plt.xlabel("y1_i")
    plt.ylabel("y1_j") 
    #plt.show()
    plt.savefig("part1_classification_grid_points_hidden_before_sigmoid.pdf")
    #Plot after sigmoid

def plot1e(x, y_line, y_line2, y1_tts, y1_nps, colors2, colors3):
    #Plot outputs 
    plt.scatter(y1_tts[:,0],y1_tts[:,1],color=colors2[:])
    plt.scatter(y1_nps[:,0],y1_nps[:,1], color=colors3[:]) 
    plt.plot(x,y_line, color='black')
    plt.plot(x,y_line2, color='black')
    plt.title("Classification of Grid of Points with Hidden Layer After Sigmoid") 
    plt.xlabel("sigmoid(y1_i)")
    plt.ylabel("sigmoid(y1_j)")
    #plt.show()
    plt.savefig("part1_classification_grid_points_hidden_after_sigmoid.pdf")

def plot2a(B2, colors):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d') 
    ax.scatter(B2[:,0],B2[:,1],B2[:,2], color=colors[:])
    #plt.show()
    plt.savefig("plot2a, titlexxx.pdf")

def plot2b(kk, losses_2):
    plt.title("Minimum loss vs k-value(number of Hidden Dimensions)") 
    plt.xlabel("Value of K")
    plt.ylabel("Minimum Loss") 
    print(f"{len(kk)=}, {len(losses_2)=}")
    plt.plot(kk[4:7],losses_2[4:7])
    plt.title("Minimum loss vs k-value(number of Hidden Dimensions)") 
    plt.xlabel("Value of K")
    plt.ylabel("Minimum Loss")
    #plt.show()
    plt.savefig("plot2b, min loss vs k")


def plot2c(x, y, z1, z2, z3, z4, z5, B2, colors):
    fig = plt.figure()
    ax = fig.add_subplot(projection='3d') 
    ax.scatter(B2[:,0],B2[:,1],B2[:,2], color=colors[:]) 
    ax.plot_surface(x,y,z1)
    ax.plot_surface(x,y,z2) 
    ax.plot_surface(x,y,z3) 
    ax.plot_surface(x,y,z4) 
    ax.plot_surface(x,y,z5)
    plt.show()
    plt.savefig("plot2c.pdf")
    
