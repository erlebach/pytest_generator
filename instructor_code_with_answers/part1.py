
#Number 1

#Import needed libraries import torch
import torch
import torch.nn as nn
import matplotlib.pyplot as plt 
import numpy as np
import torch.optim as optim 
import torch.nn.functional as F
import utils as u


def compute():

    ###   PART 1   #####################

    #Run functions and train model 
    B = u.make_binary_arrays(2)
    target = np.bitwise_xor(B[:,0],B[:,1])
    
    #Convert to torch tensor
    B = torch.from_numpy(B).float()
    target = torch.from_numpy(target).float()
    
    #declare model
    model = u.Model(B.shape[1], hidden_dim=2)
     
    #Run optimization of model and grab losses
    #losses = run_optimization(model, B, target, error_fn=lambda x: skewed_error(x,alpha=0), n_iter=1000)
    losses = u.run_optimization(model, B, target, n_iter=1000)
    
    #Plot losses in log-log plot 
    iterations=list(range(0,1000)) 
    
    u.plot1a(iterations, losses)
    
    # Run on training set to see if working correctly 
    y1,outputs_prob=model(B) 
    outputs_prob=outputs_prob.detach().numpy()
    
    #Convert prob to classes 
    outputs_class=[]
    for i in range(0,len(outputs_prob)): 
        if outputs_prob[i]>=0.5:
            outputs_class.append(1) 
        else:
            outputs_class.append(0)
    
    
    #Grab model parameters
    W1_np = model.W1.detach().numpy() 
    W2_np = model.W2.detach().numpy()
    
    #Grab model biases 
    bias1_np=model.bias1.detach().numpy() 
    bias2_np=model.bias2.detach().numpy()
    
    # Output paramters
    print("W1: {}".format(W1_np))
    print("W2: {}".format(W2_np))
    
    
    ###   PART 2   #####################

    #Create grid of points 
    nx, ny = (100, 100)
    x = np.linspace(-0.5, 1.5, nx)
    y = np.linspace(-0.5, 1.5, ny)
    xv, yv = np.meshgrid(x, y)
    
    #Place grid into a set of points 
    test=np.column_stack((xv.ravel(),yv.ravel()))
    
    #convert to torch tensor
    test= torch.from_numpy(test).float()
    
    #Run these points through trained model 
    y1_t,test_outputs_prob=model(test)
    
    #array of colors 
    colors=[]
    for i in range(0, len(test_outputs_prob)): 
        if test_outputs_prob[i]<0.5:
            colors.append('cyan') 
        else:
            colors.append('violet')
    
    
    u.plot1b(test, colors)
    
    
    ###   PART 3   #####################

    #Create equation for the line 1
    def make_explicit_line_equation(w,b,x): 
        y=(-b-x*w[0,0])/w[0,1]
        return y
    
    #Create equation for the line 2
    def make_explicit_line_equation2(w,b,x): 
        y=(-b-x*w[1,0])/w[1,1]
        return y
    
    #Create line 1
    x_line=x = np.linspace(-1, 2, 10) 
    y_line=make_explicit_line_equation(W1_np, bias1_np[0], x_line)
     
    #y_line=make_explicit_line_equation(W1_np, 0, x_line)
    
    #Create line
    y_line2=make_explicit_line_equation2(W1_np, bias1_np[0], x_line) 
    #y_line2=make_explicit_line_equation2(W1_np, 0, x_line)
    
    #Plot outputs from model with line 
    u.plot1c(x, y_line, test, colors)

    ###   PART D   #####################
    
    #Create colors for plots 
    colors2=[]
    for i in range(0, len(y1_t)):
        if y1_t[i,0]<0 and y1_t[i,1]<0: 
            colors2.append('cyan')
        else:
            colors2.append('violet') 
    
    colors3=[]
    for i in range(0, len(y1)):
        if y1[i,0]<0 and y1[i,1]<0: 
            colors3.append('blue')
        else:
            colors3.append('green')
    
    #Plot hidden layer
    y1_tt= y1_t.detach().numpy() 
    y1_np=y1.detach().numpy() 
     
    u.plot1d(y1_tt, y1_np, colors, colors3)

    ###   PART E   #####################
    
    y1_s=torch.sigmoid(y1) 
    y1_ts=torch.sigmoid(y1_t)
    
    #Convert to tensor
    y1_tts= y1_ts.detach().numpy() 
    y1_nps=y1_s.detach().numpy()
    
    u.plot1e(x, y_line, y_line2, y1_tts, y1_nps, colors2, colors3)

    ###   PART F   #####################

    # Plot the results of part A-D with a bias. 

#----------------------------------------------------------------------
if __name__ == '__main__':
    compute()
