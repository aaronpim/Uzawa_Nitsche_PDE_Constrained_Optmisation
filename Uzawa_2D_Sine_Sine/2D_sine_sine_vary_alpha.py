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
def para_def(alpha = 1e-3, rho = 1e-5, EpUp = 20, NoUp = 200, h_n = 10, printyn = True):
        return (alpha, rho, EpUp, NoUp, h_n, printyn)
# Cartesian Grid 
def Domains(N, xStart = 0., xEnd = 1., yStart = 0., yEnd = 1.):
        # 1-D Domains
        x1 = np.linspace(xStart, xEnd, N)
        y1 = np.linspace(yStart, yEnd, N)
        # 2-D Domains as Matricies
        x2, y2 = np.meshgrid(x1, y1)
        # 2-D Domains as vectors
        x = torch.from_numpy(ten2vec(x2)).requires_grad_(requires_grad = True)
        y = torch.from_numpy(ten2vec(y2)).requires_grad_(requires_grad = True)
        return x.type(torch.float), y.type(torch.float)
# Trapezoidal Rule integral
def Int2D(fun):
        fun = vec2ten(fun)
        N = fun.size(dim=0)
        return torch.sum(0.25*(fun[0:N-1,0:N-1] + fun[1:N,0:N-1] + fun[0:N-1,1:N] + fun[1:N,1:N]))
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
        plt.contourf(x,y,f, ncont, cmap = 'Greys') 
        plt.colorbar()
        plt.contour(x,y,f, ncont, colors=['#FEFBEA'], linewidths = [0.25])       
        plt.savefig(f"{path}/{filename}.pdf", format='pdf')
        plt.close()
# My save
def mysave(f, path, filename, domain = True):
        if domain and torch.is_tensor(f):
                f = sqre(f)
        elif torch.is_tensor(f):
                f = f.detach().numpy()
        np.savetxt(f"{path}/{filename}.csv", f, delimiter=',')
        
##############################################################################################
###### Training of the Neural Network ######
##############################################################################################
def geo_train(target, para, x, y, exact = torch.tensor([float('nan')]), learning_rate = 1e-3,  device = torch.device("cpu")):       
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
        
                        ###### Seperate input layers for both u and f ######
                        self.x_u_layers = nn.Sequential(
                                nn.Linear(1,h_n), Swish(),
                                nn.Linear(h_n,h_n),     Swish(),
                                nn.Linear(h_n,h_n),     Swish(),
                        )
                        self.y_u_layers = nn.Sequential(
                                nn.Linear(1,h_n), Swish(),
                                nn.Linear(h_n,h_n),     Swish(),
                                nn.Linear(h_n,h_n),     Swish(),
                        )
                        self.x_f_layers = nn.Sequential(
                                nn.Linear(1,h_n), Swish(),
                                nn.Linear(h_n,h_n),     Swish(),
                                nn.Linear(h_n,h_n),     Swish(),
                        )
                        self.y_f_layers = nn.Sequential(
                                nn.Linear(1,h_n), Swish(),
                                nn.Linear(h_n,h_n),     Swish(),
                                nn.Linear(h_n,h_n),     Swish(),
                        )
                        
                        ###### Separate output layers for u and f ######
                        self.u_output_layer = nn.Bilinear(h_n,h_n,1)
                        self.f_output_layer = nn.Bilinear(h_n,h_n,1)
                        
                def forward(self, x, y):
                        # Compute transformed x and y variables
                        xu = self.x_u_layers(x)
                        yu = self.y_u_layers(y)
                        xf = self.x_f_layers(x)
                        yf = self.y_f_layers(y)
                        u  = self.u_output_layer(xu,yu)
                        f  = self.f_output_layer(xf,yf)
                        output1 = u * x * (1 - x) * y * (1 - y)
                        output2 = f 
                        return output1, output2
        ###### Initialize the neural network using a standard method ######
        def init_normal(m):
                if type(m) == nn.Linear:
                        nn.init.kaiming_normal_(m.weight)
        ############################################################
        
        
        def generatorCriterion(x, y, z, target, para):
                u, f = generator_NN(x,y)  # Unpack both u and f
                dxdy = ((torch.max(x) - torch.min(x))*(torch.max(y) - torch.min(y))/ ((sqrt(x.size(0)) - 1)**2)).item()
                # Compute Laplacian
                ux  = torch.autograd.grad(u,  x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
                uxx = torch.autograd.grad(ux, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
                uy  = torch.autograd.grad(u,  y, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
                uyy = torch.autograd.grad(uy, y, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
                Au  = uxx + uyy       

                # Convert target to a PyTorch tensor and ensure it's on the same device as x
                target = target.clone().detach()
                z = z.clone().detach()
                
                # First term: 0.5 * Norm{u - target}^2
                term1 = 0.5 * dxdy * Int2D((u - target)**2)
                
                # Second term: alpha/2 * Norm{f}^2
                term2_1 = para[0] / 4 * dxdy * Int2D((Au)**2)
                term2_2 = para[0] / 4 * dxdy * Int2D((f)**2)
                
                # Third term: -∫ (u' * z' + f * z) dx approximation
                # Assuming discrete summation approximates the integral well
                term3 = dxdy * Int2D((Au-f)*z)       

                return term1 + term2_1 + term2_2 + term3, term1, term2_1, term2_2, term3, u, f, Au
        ############################################################
        dxdy = (torch.max(x) - torch.min(x))*(torch.max(y) - torch.min(y))/ ((sqrt(x.size(0)) - 1)**2)
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
                total_loss, Target, Laplace, Control, Adjoint, u, f, Au = generatorCriterion(x,y,z, target, para)                                	
                total_loss.backward()
                generatorOptimiser.step()

                # Append the losses as a list
                
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
                        state_error = torch.sqrt(Int2D(dxdy*(u-exact[0])**2))
                        contr_error = torch.sqrt(Int2D(dxdy*(f-exact[1])**2))
                        err_history.append([state_error.item(), contr_error.item()])
                        loss_history.append([total_loss.item(), Target.item(), Laplace.item(), Control.item(), Adjoint.item()])
                        if para[5]: print(f'State Error: ',err_history[-1][0],f'\tControl Error: ',err_history[-1][1])
                        z = z + para[1] * (Au - f)
        toc = time.time()
        elapseTime = toc - tic
        u, f = generator_NN(x,y)              
        return generator_NN, u, f, err_history, loss_history, elapseTime
##############################################################################################
###### Main Code ######
##############################################################################################


# para = para_def(alpha = 1e-3, rho = 1e-5, EpUp = 10, NoUp = 20)
#    = (alpha = 1e-3, rho = 0.01*alpha, EpUp = 20, NoUp = 200, h_n = 10, printyn = True)
loga_values = range(0,10)
npts = 30
repeats = 1
for logalpha in loga_values:
        alpha = 10**(-logalpha)
        para = para_def(alpha, 0.25*alpha, EpUp = 40, NoUp = 500, h_n = 40)
        path = f'./Vary_alpha_{logalpha}_nPts_{npts}'
        if not os.path.isdir(path):
                os.makedirs(path)
        x, y = Domains(npts)
        target = (1 + 4*para[0]*pi**4)*torch.sin(pi*x)*torch.sin(pi*y)
        exact_u  = torch.sin(pi*x)*torch.sin(pi*y)
        exact_f  = -2*(pi**2)*exact_u
        u = 0
        f = 0
        err_history  = 0
        loss_history = 0
        for i in range(0,repeats):
                gNN, ui, fi, err_historyi, loss_historyi, elapseTime = geo_train(target, para, x, y, exact = [exact_u,exact_f]) 
                u += ui/repeats
                f += fi/repeats
                err_history  += np.log(err_historyi)/repeats
                loss_history += np.array(loss_historyi)
        err_history = np.exp(err_history)
        
        colour_plot(u,x,y,path,f'State')
        colour_plot(exact_u,x,y,path,f'State_Exact')
        colour_plot(u-exact_u,x,y,path,f'State_Error')
        colour_plot(f,x,y,path,f'Control')
        colour_plot(exact_f,x,y,path,f'Control_Exact')
        colour_plot(f-exact_f,x,y,path,f'Control_Error')  
        
        plot_error(para, err_history, path, f'Error')
        plot_loss(loss_history, path, f'Loss', para)
        mysave(u, path, f'State')
        mysave(f, path, f'Control')
        mysave(err_history, path, f'Error',False)
        mysave(loss_history, path, f'Loss',False)
