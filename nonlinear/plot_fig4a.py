#
# Nonlinear NP and SGD in SARCOS
#
# Readout of the cosine similarity
#
from math import *

import sys
import numpy as np
from numpy import random as nrnd
from numpy import linalg as nlg
from scipy import ndimage as scind

import matplotlib.pyplot as plt

clrs = ['C3', 'C4']

K = 2
widths = [50, 100, 200, 400, 800, 1600]
batch_size = 100
learning_rate = 0.0001
nepoch = 10000
ikmax = 5
dt = 1
dS = int(floor( 44484/batch_size ))

wlen = len(widths)

err_ths = [50.0, 5.0] #error levels at which the similarity is estimated
cos_sims = np.zeros((len(err_ths), 2, wlen))
cos_sim_cnts = np.zeros((len(err_ths), wlen))

for widx in range( wlen ):
    width = widths[widx]
    nrecord = int(floor( nepoch/dt ))

    for sidx in range(ikmax):
        seed = sidx
        festr = 'data/gfb_mlp_sarcos_sgd_np_sim_K' + str(K) + '_w' + str(width) + '_B' + str(batch_size) + '_lr' + str(learning_rate) + '_nep' + str(nepoch) + '_sd' + str(seed) + '.txt'
        lidx = 0
        for line in open(festr, 'r'):
            ltmps = line[:-1].split(" ")
            ertmp = float(ltmps[1]) #test error
            for eridx in range( len(err_ths) ):
                if 0.9*err_ths[eridx] < ertmp and ertmp < 1.1*err_ths[eridx]:
                    cos_sims[eridx][0][widx] += float(ltmps[3])
                    cos_sims[eridx][1][widx] += float(ltmps[4])
                    cos_sim_cnts[eridx][widx] += 1.0
            lidx += 1

for eridx in range(len(err_ths)):
    for widx in range(wlen):
        for k in range(2):
            cos_sims[eridx][k][widx] = cos_sims[eridx][k][widx]/cos_sim_cnts[eridx][widx]

plt.rcParams.update({'font.size': 16})
svfg = plt.figure()

for k in range(2):
    plt.plot(widths, cos_sims[0][k], 'o-', ls='--', c=clrs[k])
    plt.plot(widths, cos_sims[1][k], 'o-', c=clrs[k])

plt.loglog()
plt.xticks([50, 100, 200, 400, 800, 1600], [50, 100, 200, 400, 800, 1600])
plt.yticks([])
plt.yticks([0.03, 0.05, 0.1, 0.2], [0.03, 0.05, 0.1, 0.2])
plt.show()

svfg.savefig('mlp_sarcos_sgd_np_sim_K' + str(K) + '_w' + str(widths[0]) + '-' + str(widths[1]) + '_B' + str(batch_size) + '_lr' + str(learning_rate) + '.pdf')
