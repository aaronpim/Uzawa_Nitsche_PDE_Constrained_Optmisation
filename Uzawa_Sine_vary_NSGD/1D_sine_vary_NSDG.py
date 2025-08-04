##############################################################################################
###### Packages to Import ######
##############################################################################################
# Standard Maths
import numpy as np
import math
from math import pi, sqrt, exp
# Neural Network
import torch
import torch.nn as nn
import torch.optim as optim
# Plotting & saving data
import os
import csv
import matplotlib.cm as cm
from matplotlib import pyplot as plt
import matplotlib.dates as mdates
from matplotlib.ticker import AutoMinorLocator
# Misc
import time
##############################################################################################
###### Standard functions related to the Neural Network ######
##############################################################################################
# Tensor to Vector
def ten2vec(M):
        v = [1] * len(M.shape)
        v[0] = M.size
        return M.reshape(v)
# Vector to Tensor with a target shape of tar        
def vec2ten(*args):
        M = args[0]
        if len(args)==2:
                return M.reshape(tar.shape)
        elif len(args)==1:
                N = int(sqrt(M.size(dim=0)))
                return M.reshape((N,N))
# Parameter defaults
def para_def(alpha = 1e-3, rho = 0.25*1e-3, EpUp = 10, NoUp = 100, h_n = 40, printyn = True):
        return (alpha, rho, EpUp, NoUp, h_n, printyn)
# Cartesian Grid 
def Domains(N, xStart = 0., xEnd = 1.):
        # 1-D Domains
        x1 = np.linspace(xStart, xEnd, N)
        x = torch.from_numpy(x1).requires_grad_(requires_grad = True).type(torch.float32)
        return torch.unsqueeze(x,-1)
# Trapezoidal Rule integral
def Int1D(fun):
        N = fun.size(dim=0)
        return torch.sum(0.5*(fun[0:N-1] + fun[1:N]))
##############################################################################################
###### Code for Plots and Saving ######
##############################################################################################
def plot_loss(loss_history, path, filename, para, labels = ['Total', 'Target', 'State', 'Control', 'Adjoint']):
        fig, ax = plt.subplots(layout='constrained')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        for i in range(len(loss_history[0])):  # Loop over each component
                # Extract and plot each component
                component_loss = [loss[i] for loss in loss_history]
                plt.semilogy(list(range(len(component_loss))), component_loss, label=labels[i],  linewidth=0.5)
        def epochs2updates(x):
                return x / para[2]
        def updates2epochs(x):
                return x * para[2]
        secax = ax.secondary_xaxis('top', functions=(epochs2updates, updates2epochs))
        secax.set_xlabel('Update No.') 
        plt.legend()  
        plt.savefig(f"{path}/{filename}.pdf", format='pdf')
        plt.close()
# Plot error        
def plot_error(para, err_his, path, filename):
        fig, ax = plt.subplots(layout='constrained')
        ax.set_xlabel('Update No.')
        ax.set_ylabel('Error')
        labels = ['State', 'Control']
        for i in range(len(err_his[0])):  # Loop over each component
                # Extract and plot each component
                component_err = [err[i] for err in err_his]
                ax.semilogy(list(range(len(component_err))), component_err, label=labels[i],  linewidth=0.5)         
        def epochs2updates(x):
                return x * para[2]
        def updates2epochs(x):
                return x / para[2]
        secax = ax.secondary_xaxis('top', functions=(epochs2updates, updates2epochs))
        secax.set_xlabel('Epoch')
        plt.legend()
        plt.savefig(f"{path}/{filename}.pdf", format='pdf')
        plt.close()
# Convert a torch vector to a numpy square matrix
def sqre(x):
        return x.reshape([int(sqrt(x.size()[0])),int(sqrt(x.size()[0]))]).detach().numpy()
# Plot the solution as a contour series of contour
def colour_plot(fun,x,y,path,filename, cont_max = 20):  
        x = sqre(x)
        y = sqre(y)
        f = sqre(fun)
        ncont = min(len(fun),cont_max)
        plt.contourf(x,y,f, ncont, cmap = 'plasma') 
        plt.colorbar()
        plt.contour(x,y,f, ncont, colors=['#FEFBEA'], linewidths = [0.25])       
        plt.savefig(f"{path}/{filename}.pdf", format='pdf')
        plt.close()
# My save
def mysave(f, path, filename, domain = True):
        if domain and torch.is_tensor(f):
                f = f.detach().numpy()
        elif torch.is_tensor(f):
                f = f.detach().numpy()
        np.savetxt(f"{path}/{filename}.csv", f, delimiter=',')
def plot_line(x, u, uexact, f, fexact, path,filename):
        plt.figure()
        plt.plot(x.detach().numpy(), uexact.detach().numpy(), '--', color='orange', label='$exact$', alpha=0.5)  
        plt.plot(x.detach().numpy(), u.detach().numpy(), 'o', color='orange', label='$u_\\theta$', alpha=0.5)  
        plt.plot(x.detach().numpy(), fexact.detach().numpy(), 'r--', label='$fexact$', alpha=0.5)  
        plt.plot(x.detach().numpy(), f.detach().numpy(), 'o', color='red', label='$f_\\theta$', alpha=0.5)  
        plt.grid(True)  # Turns on the grid
        plt.legend(loc='upper right')
        plt.savefig(f"{path}/{filename}.pdf", format='pdf')
        plt.close()        
##############################################################################################
###### Training of the Neural Network ######
##############################################################################################
def geo_train(target, para, x, exact = torch.tensor([float('nan')]), learning_rate = 1e-3,  device = torch.device("cpu")):       
        ###### Activation Function ######
        class Swish(nn.Module):
                def __init__(self, inplace=True):
                        super(Swish, self).__init__()
                        self.inplace = inplace
                        
                def forward(self, x):
                        if self.inplace:
                                x.mul_(torch.sigmoid(x))
                                return x
                        else:
                                return x * torch.sigmoid(x)
        ###### Neural Network ######                
        class generatorNN(nn.Module):
                def __init__(self, input_n, h_n):
                        super(generatorNN, self).__init__()
                        self.log_learning_rate = nn.Parameter(torch.log(torch.tensor(1e-3)))
        
                        # Shared layers for both u and f
                        self.shared_layers = nn.Sequential(
                                nn.Linear(input_n,h_n),
                                #nn.ReLU(),
                                Swish(),
                                nn.Linear(h_n,h_n),
                                #nn.ReLU(),
                                Swish(),
                                nn.Linear(h_n,h_n),
                                #nn.ReLU(),
                                Swish(),
                        )
                        # Separate layers for u
                        self.u_layers = nn.Sequential(
                                nn.Linear(h_n, h_n),
                                Swish(),
                                nn.Linear(h_n, 1),
                        )

                        # Separate layers for f
                        self.f_layers = nn.Sequential(
                                nn.Linear(h_n, h_n),
                                Swish(),
                                nn.Linear(h_n, 1),
                        )
                        # Separate output layers for u and f
                        self.u_output_layer = nn.Linear(h_n, 1)
                        self.f_output_layer = nn.Linear(h_n, 1)
                        
                def forward(self, x):
                        shared_output = self.shared_layers(x)

                        # Compute u and f using their respective layers
                        u = self.u_layers(shared_output)
                        f = self.f_layers(shared_output)
                        
                        u = u * x * (1 - x)
                        f = f * x * (1 - x)
                        return u, f
        ###### Initialize the neural network using a standard method ######
        def init_normal(m):
                if type(m) == nn.Linear:
                        nn.init.kaiming_normal_(m.weight)
        ############################################################
        
        
        def generatorCriterion(x, z, target, para):
                u, f = generator_NN(x)  # Unpack both u and f
                dx = ((torch.max(x) - torch.min(x))/ sqrt(x.size(0) - 1)).item()
                # Compute Laplacian
                ux  = torch.autograd.grad(u,  x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
                uxx = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]     
                # Convert target to a PyTorch tensor and ensure it's on the same device as x
                target = target.clone().detach()
                z = z.clone().detach()
                
                # First term: 0.5 * Norm{u - target}^2
                term1 = 0.5 * dx * Int1D((u - target)**2)
                
                # Second term: alpha/2 * Norm{f}^2
                term2_1 = para[0] / 4 * dx * Int1D((uxx)**2)
                term2_2 = para[0] / 4 * dx * Int1D((f)**2)
                
                # Third term: -∫ (u' * z' + f * z) dx approximation
                # Assuming discrete summation approximates the integral well
                term3 = dx * Int1D((uxx+f)*z)       

                return term1 + term2_1 + term2_2 + term3, term1, term2_1, term2_2, term3, u, f, uxx
        ############################################################
        dx = (torch.max(x) - torch.min(x))/ (x.size(0)- 1)
        tic = time.time()
        z = 0*x  # Assuming z is initialized based on x  
        loss_history = []
        err_history = []
        input_n = 1
        ############################################################
        # use the modules apply function to recursively apply the initialization
        generator_NN = generatorNN(input_n, para[4]).to(device)
        generator_NN.apply(init_normal)
        generatorOptimiser = optim.Adam(generator_NN.parameters(), lr=learning_rate, betas = (0.9,0.99),eps = 10**-15)
        ############################################################
        for epoch in range(para[2]*para[3]):
                generator_NN.zero_grad()
                total_loss, Target, Laplace, Control, Adjoint, u, f, Au = generatorCriterion(x, z, target, para)                                	
                total_loss.backward()
                generatorOptimiser.step()

                # Append the losses as a list
                loss_history.append([total_loss.item(), Target.item(), Laplace.item(), Control.item(), Adjoint.item()])
                if epoch % para[2] == 0:
                        if para[5]:
                                print(f'Epoch: {epoch} \tTotal Loss: {total_loss.item()}, '
                                      f'Target loss: {Target.item()}, '
                                      f'Laplace loss: {Laplace.item()}, '
                                      f'Control loss: {Control.item()}, '
                                      f'Adjoint loss: {Adjoint.item()}, ')
                                #colour_plot(u,x,y,path,f'State_Epoch_{epoch}')
                                #colour_plot(f,x,y,path,f'Control_Epoch_{epoch}')
                                #colour_plot(z,x,y,path,f'Adjoint_Epoch_{epoch}')
                        state_error = torch.sqrt(Int1D(dx*(u-exact[0])**2))
                        contr_error = torch.sqrt(Int1D(dx*(f-exact[1])**2))
                        err_history.append([state_error.item(), contr_error.item()])
                        if para[5]: print(f'State Error: ',err_history[-1][0],f'\tControl Error: ',err_history[-1][1])
                        z = z + para[1] * (Au + f)
        toc = time.time()
        elapseTime = toc - tic
        u, f = generator_NN(x)              
        return generator_NN, u, f, err_history, loss_history, elapseTime
##############################################################################################
###### Main Code ######
##############################################################################################
device = torch.device("cpu")
logalpha = 4
npts = 201
repeats = 1
EpUp_values = list(range(1,10))+list(range(10,101,30))
for epup in EpUp_values:
        alpha = 10**(-logalpha)
        para = para_def(alpha, 0.25*alpha, EpUp = epup, NoUp = 500)
        
        path = f'./Vary_alpha_{logalpha}_nPts_{npts}_EpUp_{epup}'
        if not os.path.isdir(path):
                os.makedirs(path)
        x = Domains(npts)
        ##############################################################################################
        alpha = 10**(-logalpha)
        target = (1 + alpha * pi**4)*torch.sin(pi*x)
        exact_u= torch.sin(pi*x)
        exact_f= (pi**2) * torch.sin(pi*x)
        ##############################################################################################
        u = 0
        f = 0
        err_history  = 0
        loss_history = 0
        for i in range(0,repeats):
                gNN, ui, fi, err_historyi, loss_historyi, elapseTime = geo_train(target, para, x, exact = [exact_u,exact_f]) 
                u += ui/repeats
                f += fi/repeats
                err_history  += np.log(np.array(err_historyi))/repeats
                loss_history += np.array(loss_historyi)/repeats
        err_history = np.exp(err_history)
        plot_line(x, u, exact_u, f, exact_f, path, f'State_Control')
        plot_error(para, err_history, path, f'Error')
        #plot_loss(loss_history, path, f'Loss', para)
        mysave(u, path, f'State')
        mysave(f, path, f'Control')
        mysave(err_history, path, f'Error',False)
        #mysave(loss_history, path, f'Loss',False)
