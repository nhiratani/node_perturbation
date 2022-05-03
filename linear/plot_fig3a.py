#
# Deep linear NP
#
# from Xavier-Glorot initialization
#
# Readout of the trajectory in the presence of teacher noise
#
from math import *

import sys
import numpy as np
from numpy import random as nrnd
from numpy import linalg as nlg

import matplotlib.pyplot as plt
from pylab import cm

from lin_st_theor import calc_eta_star, calc_tauc_epsilonc

climit = 4
clrs = []
for q in range(climit):
    clrs.append( cm.viridis( (0.5+q)/float(climit) ) )

Lx = 100
Lh = 10000
Ly = 10
sigmat = 0.32
sigmat2 = sigmat*sigmat

r_etas = [0.2, 0.6, 1.0, 1.4]
r_winit2 = 1.0
ikmax = 10

eta_zero = calc_eta_star(Lx, Lh, Ly)

plt.rcParams.update({'font.size': 16})
svfg = plt.figure()
for ridx in range(len(r_etas)):
    r_eta = r_etas[ridx]
    if r_eta > 0.67:
        T = 100000; dt = 10
    else:
        T = 1000000; dt = 100
    tlen = int(floor(T/dt))

    ts = range(0,T,dt)
    mean_epsilon = np.zeros((tlen))
    for ik in range(ikmax):
        festr = 'data/gfb_lin_st_np_traj_noise_Lx' + str(Lx) + '_Lh' + str(Lh) + '_Ly' + str(Ly) + '_st' + str(sigmat) + '_ret' + str(r_eta) + '_rw2-' + str(r_winit2) + '_T' + str(T) + '_ik' + str(ik) + '.txt'

        a_s = np.zeros((tlen)); betas = np.zeros((tlen)); epsilons = np.zeros((tlen))
        lidx = 0
        for line in open(festr, 'r'):
            ltmps = line[:-1].split(" ")
            if len(ltmps) == 4:
                a_s[lidx] += float(ltmps[1])/np.cbrt(float(Lh))
                betas[lidx] += float(ltmps[2])
                epsilons[lidx] += float(ltmps[3])
                mean_epsilon[lidx] += float(ltmps[3])/float(ikmax)
            else:
                print(r_winit, ik, ltmps)
            lidx += 1
        #plot individual trajectory
        plt.plot(a_s, epsilons, color=clrs[ridx], lw=0.5)

    beta_z = 2.0*Lh/(Lh+Ly)
    c_z = r_eta*eta_zero*np.cbrt(float(Lh))*Lx*Ly*(1.0 + beta_z)
    xs = np.arange(0.05, 3.0, 0.001)
    ys = []
    for x in xs:
        if 2.0 > c_z*x:
            ys.append( c_z*x*sigmat2/(Lx*(2.0 - c_z*x)) )
    #plot nullcline
    plt.plot(xs[:len(ys)], ys, lw=2.5, ls='--', color=clrs[ridx])

plt.loglog()
plt.xlim(0.05, 3.0)
plt.ylim(0.00001, 0.5)
plt.show()

svfg.savefig('lin_st_np_noise_traj_Lx' + str(Lx) + '_Lh' + str(Lh) + '_Ly' + str(Ly) + '_st' + str(sigmat) + '_rw2-' + str(r_winit2) + '_ikm' + str(ikmax) + '.pdf')
