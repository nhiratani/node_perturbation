#
# Deep linear NP
#
# from Xavier-Glorot initialization
#
# Readout of the time at which crash starts
#
from math import *

import sys
import numpy as np

import matplotlib.pyplot as plt
from pylab import cm

climit = 3
clrs = []
for q in range(climit):
    clrs.append( cm.viridis( (0.5+q)/float(climit) ) )

lss = ['--', '-']

K = 2
widths = [400, 800, 1600]
batch_size = 100
lrs = [0.000036,0.000024, 0.000018]
nepochs = [3000, 30000]
wreg_types = [0, 2]
w1_scale = 10.0
ikmax = 3#5
dt = 10

plt.rcParams.update({'font.size': 16})
svfg = plt.figure()

for wdidx in range( len(widths) ):
    width = widths[wdidx]
    lr = lrs[wdidx]
    
    for wsidx in range(len(wreg_types)):
        wreg_type = wreg_types[wsidx]
        nepoch = nepochs[wsidx]
        nrec = int(floor( nepoch/dt + 0.001 ))
        
        mers = np.zeros((nrec))
        for sidx in range(ikmax):
            seed = sidx
            ers = np.zeros((nrec))
            festr = 'data/gfb_mlp_sarcos_np_wreg_K' + str(K) + '_w' + str(width) + '_B' + str(batch_size) + '_lr' + str(lr) + '_nep' + str(nepoch) + '_wrt' + str(wreg_type) + '_w1s' + str(w1_scale) + '_sd' + str(seed) + '.txt'
            lidx = 0
            for line in open(festr, 'r'):
                ltmps = line[:-1].split(" ")
                ers[lidx] += float(ltmps[1])
                mers[lidx] += float(ltmps[1])/float(ikmax)
                lidx += 1
            if lidx < nrec:
                for tidx in range(lidx, nrec):
                    ers[tidx] += 30000.0
            plt.plot(range(1, nepoch+1, dt), ers, c=clrs[wdidx], ls=lss[wsidx])

plt.loglog()
plt.ylim(1.0, 10000)
plt.xlim(1, 30000)
plt.show()

svfg.savefig('gfb_mlp_sarcos_np_wreg_learning_curves_K' + str(K) + '_w' + str(widths[0]) + '_' + str(widths[-1]) + '_B' + str(batch_size) + '_nep' + str(nepoch) + '_w1s' + str(w1_scale) + '_ikm' + str(ikmax) + '.pdf')
