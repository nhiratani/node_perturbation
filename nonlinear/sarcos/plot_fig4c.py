#
# Deep linear NP
#
# from Xavier-Glorot initialization
#
# Readout of the time at which crash starts
#
# Measure mean of the bottom/crash time
#
from math import *

import sys
import numpy as np
from scipy import stats as scist

import matplotlib.pyplot as plt

K = 2
width = 800
batch_size = 100
lrs = [0.0000032, 0.0000056, 0.00001, 0.000018, 0.000032, 0.000056, 0.0001, 0.00018]
nepochs = [30000, 30000, 10000, 10000, 1000, 1000, 1000, 1000]
ikmax = 5
dt = 10
dS = int(floor( 44484/batch_size ))

loidx = 0

lrlen = len(lrs)
t_bottoms = -1.0*np.ones( (lrlen) ); t_crashes = -1.0*np.ones( (lrlen) );
for lridx in range( lrlen ):
    learning_rate = lrs[lridx]
    nepoch = nepochs[lridx]
    nrecord = int(floor( nepoch/dt ))

    #readout the mean trajectory
    ts = []; ers = []; er_cnts = []
    for sidx in range(ikmax):
        seed = sidx
        festr = 'data/gfb_mlp_sarcos_np_ttime_K' + str(K) + '_w' + str(width) + '_B' + str(batch_size) + '_lr' + str(learning_rate) + '_nep' + str(nepoch) + '_sd' + str(seed) + '.txt'
        lidx = 0
        for line in open(festr, 'r'):
            ltmps = line[:-1].split(" ")
            if sidx == 0 or len(ers) <= lidx:
                ers.append(0.0); er_cnts.append(0.0)
                if nepoch > 1000:
                    ts.append( float(ltmps[0]) )
                else:
                    ts.append( float(ltmps[0]) + float(ltmps[1])/float(dS) )
            if nepoch > 1000:
                ers[lidx] += float(ltmps[1])
            else:
                ers[lidx] += float(ltmps[2])
            er_cnts[lidx] += 1.0
            lidx += 1

    for tidx in range( len(er_cnts) ):
        if er_cnts[tidx] > 0.5:
            ers[tidx] = ers[tidx]/er_cnts[tidx]
        else:
            ers[tidx] = 1000000
    ers = np.array(ers)

    #detect the point at which the error is minimized
    tbidx = np.argmin( np.ma.masked_invalid(ers) )
    t_bottoms[lridx] = ts[tbidx]

    #detect the point at which the error surpasses the inital error
    for tidx in range(10, len(ers)):
        if ers[tidx] > 1.1*ers[0] or np.isnan(ers[tidx]):
            t_crashes[lridx] = ts[tidx]; break
    if t_crashes[lridx] < 0.0 and len(ers) < nrecord:
        t_crashes[lridx] = len(ers)*dt

plt.rcParams.update({'font.size': 16})
svfg = plt.figure()

plt.plot(lrs, t_bottoms, 'o-', c='k')
plt.plot(lrs[1:], t_crashes[1:], 'o-', c='gray')

plt.loglog()
plt.show()

svfg.savefig('mlp_sarcos_np_learn_time2_K' + str(K) + '_w' + str(width) + '_B' + str(batch_size) + '_lr' + str(lrs[0]) + '-' + str(lrs[-1]) + '.pdf')
