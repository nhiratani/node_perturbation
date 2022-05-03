#
# Deep linear NP
#
# from Xavier-Glorot initialization
#
# Readout of learning trajectory
#
from math import *

import sys
import numpy as np
from numpy import random as nrnd
from numpy import linalg as nlg

import matplotlib.pyplot as plt
from pylab import cm

from lin_st_theor import calc_eta_star, np_mf_dynamics, np_ae_dynamics

climit = 4
clrs = []
for q in range(climit):
    clrs.append( cm.viridis( (0.5+q)/float(climit) ) )

T = 100000
dt = 10
tlen = int(floor(T/dt))
ts = range(0,T,dt)

Lx = 100
Lh = 10000
Ly = 10
sigmat = 0.0

r_etas = [0.8,1.0,1.2,1.4]
r_winit2 = 1.0
ikmax = 10

eta_zero = calc_eta_star(Lx, Lh, Ly)

svfg = plt.figure()
plt.rcParams.update({'font.size': 16})

for ridx in range(len(r_etas)):
    r_eta = r_etas[ridx]
    
    # readout of the simulation results
    a_s = np.zeros((tlen)); betas = np.zeros((tlen)); epsilons = np.zeros((tlen))
    for ik in range(ikmax):
        festr = 'data/gfb_lin_st_np_traj_Lx' + str(Lx) + '_Lh' + str(Lh) + '_Ly' + str(Ly) + '_st' + str(sigmat) + '_ret' + str(r_eta) + '_rw2-' + str(r_winit2) + '_T' + str(T) + '_ik' + str(ik) + '.txt'

        lidx = 0
        for line in open(festr, 'r'):
            ltmps = line[:-1].split(" ")
            if len(ltmps) == 4:
                a_s[lidx] += float(ltmps[1])/( np.cbrt(float(Lh))*ikmax )
                betas[lidx] += float(ltmps[2])/float(ikmax)
                epsilons[lidx] += float(ltmps[3])/float(ikmax)
            else:
                print(r_winit, ik, ltmps)
            lidx += 1
    plt.plot(ts, epsilons, color=clrs[ridx], lw=2.5)

    # prediction from the two-variable mf dynamics
    v_seqs = np_ae_dynamics(Lx, Lh, Ly, sigmat, r_eta*eta_zero, T, dt)
    plt.plot( ts, v_seqs[1], c=clrs[ridx], ls='--', lw=2.5)

plt.loglog()
plt.xlim(10, T)
plt.ylim(0.0001, 1.0)
plt.show()

svfg.savefig('gfb_lin_st_np_traj_Lx' + str(Lx) + '_Lh' + str(Lh) + '_Ly' + str(Ly) + '_st' + str(sigmat)+ '_rw2-' + str(r_winit2) + '_T' + str(T) + '_ikm' + str(ikmax) + '.pdf')

