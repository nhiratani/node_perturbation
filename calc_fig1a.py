#
# Deep linear NP
#
# from Xavier-Glorot initialization
#
# Gaussian noise in the teacher network
#
# Neumerical simulation for trajectory generation
#
from math import *
#import os
#os.environ["MKL_NUM_THREADS"] = '4'
#os.environ["NUMEXPR_NUM_THREADS"] = '4'
#os.environ["OMP_NUM_THREADS"] = '4'

import sys
import numpy as np
from numpy import random as nrnd
from numpy import linalg as nlg

from lin_st_theor import calc_eta_star
from lin_st_model import xavier_weight_init, sample_generation, np_weight_update, calc_error_norm

T = 100000
dt = 10

def simul(Lx, Lh, Ly, sigmat, r_eta, r_winit2, ik):
    festr = 'data/gfb_lin_st_np_traj_Lx' + str(Lx) + '_Lh' + str(Lh) + '_Ly' + str(Ly) + '_st' + str(sigmat) + '_ret' + str(r_eta) + '_rw2-' + str(r_winit2) + '_T' + str(T) + '_ik' + str(ik) + '.txt'
    fwe = open(festr,'w')

    eta_th = calc_eta_star(Lx, Lh, Ly)
    eta = r_eta*eta_th
    
    A, W1, W2 = xavier_weight_init(Lx, Lh, Ly, r_winit2)
    for tidx in range(T):
        x, yt = sample_generation(A, sigmat)
        if tidx%dt == 0:
            alpha, beta, epsilon = calc_error_norm(W1, W2, A)
            fwe.write( str(tidx) + " " + str(alpha) + " " + str(beta) + " " + str(epsilon) + "\n" )
        
        W1, W2 = np_weight_update(W1, W2, eta, x, yt)

def main():
    param = sys.argv
    Lx = int(param[1]) # input layer size
    Lh = int(param[2]) # hidden layer size
    Ly = int(param[3]) # output layer size
    sigmat = float(param[4]) #teacher noise amplitude (s.d.)
    
    r_eta = float(param[5]) #relative learning rate
    r_winit2 = float(param[6]) #relative weight scale
    ik = int(param[7])

    simul(Lx, Lh, Ly, sigmat, r_eta, r_winit2, ik)

if __name__ == "__main__":
    main()
