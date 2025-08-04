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
# Parameter defaults
##############################################################################################
def para_def(alpha = 1e-3, rho = 0.25*1e-3, EpUp = 10, NoUp = 100, h_n = 40, printyn = True, epsilon = 1.0):
        return (alpha, rho, EpUp, NoUp, h_n, printyn, epsilon)
##############################################################################################
# Cartesian Grid
##############################################################################################
def Domains(N, xStart = 0., xEnd = 1.):
        # 1-D Domains
        x1 = np.linspace(xStart, xEnd, N)
        x = torch.from_numpy(x1).requires_grad_(requires_grad = True).type(torch.float32)
        return torch.unsqueeze(x,-1)
##############################################################################################
device = torch.device("cpu")
loga_values = range(0,9)
npts = 201
for logalpha in loga_values:
        alpha = 10**(-logalpha)
        para = para_def(alpha, 0.25*alpha, EpUp = 10, NoUp = 500, h_n = 40, printyn = True, epsilon = 0.1)
        x = Domains(npts)
        ##############################################################################################
        alpha = 10**(-logalpha)
        exact_u   = torch.sin(pi*x)
        exact_ux  = torch.autograd.grad(exact_u,  x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        exact_uxx = torch.autograd.grad(exact_ux, x, grad_outputs=torch.ones_like(x), create_graph=True, only_inputs=True)[0]
        exact_f   = -((1/para[6])**2)*exact_u*(1-exact_u**2)-exact_uxx
        ##############################################################################################
        with open(f'./Vary_alpha_{logalpha}_nPts_{npts}_eps_{para[6]}/State.csv') as file:
                reader = csv.reader(file, quoting = csv.QUOTE_ALL, delimiter = ',')
                state = []
                for row in reader: 
                        state.append(row[0])
        state = np.array(state).astype(np.float64)
        u = exact_u.detach().numpy()
        np.savetxt(f'./Vary_alpha_{logalpha}_nPts_{npts}_eps_{para[6]}/State_pointwise.csv', state-u, delimiter=',')
        ##############################################################################################
        with open(f'./Vary_alpha_{logalpha}_nPts_{npts}_eps_{para[6]}/Control.csv') as file:
                reader = csv.reader(file, quoting = csv.QUOTE_ALL, delimiter = ',')
                control = []
                for row in reader: 
                        control.append(row[0])
        control = np.array(control).astype(np.float64)
        f = exact_f.detach().numpy()
        np.savetxt(f'./Vary_alpha_{logalpha}_nPts_{npts}_eps_{para[6]}/Control_pointwise.csv', control-f, delimiter=',')
