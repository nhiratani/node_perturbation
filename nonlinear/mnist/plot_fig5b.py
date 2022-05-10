#
# MMIST
# Estimation of the minimum training time
# under NP, Reinforce, and SGD
#
from math import *
import matplotlib.pyplot as plt
import numpy as np
from pylab import cm
import scipy.stats as scist

clrs = ['C1', 'C2', 'C0']

perf_th = 97.5 #the target performance level

K = 2
width_scalings = ['100', '200', '400', '800', '1600', '3200']
widths = [100, 200, 400, 800, 1600, 3200]
batch_size = 1000

learning_rates = [[[0.000018, 0.000032, 0.000056, 0.0001, 0.00018], #NP
                   [0.000018, 0.000032, 0.000056, 0.0001, 0.00018],
                   [0.00001, 0.000018, 0.000032, 0.000056, 0.0001],
                   [0.0000056, 0.00001, 0.000018, 0.000032, 0.000056],
                   [0.0000032, 0.0000056, 0.00001, 0.000018, 0.000032],
                   [0.0000056, 0.00001, 0.000018]],
                  [[0.000032, 0.000056, 0.0001, 0.00018, 0.00032], #Reinforce
                   [0.000056, 0.0001, 0.00018, 0.00032, 0.00056],
                   [0.0001, 0.00018, 0.00032, 0.00056, 0.001],
                   [0.0001, 0.00018, 0.00032, 0.00056, 0.001],
                   [0.0001, 0.00018, 0.00032, 0.00056, 0.001],
                   [0.0001, 0.00018, 0.00032, 0.00056, 0.001]],
                  [[0.001, 0.0018, 0.0032, 0.0056, 0.01], #SGD
                   [0.001, 0.0018, 0.0032, 0.0056, 0.01],
                   [0.001, 0.0018, 0.0032, 0.0056, 0.01],
                   [0.001, 0.0018, 0.0032, 0.0056, 0.01],
                   [0.001, 0.0018, 0.0032, 0.0056, 0.01],
                   [0.001, 0.0018, 0.0032, 0.0056, 0.01]]]

nepochs = [10000, 3000, 100]
lrules = [0,1,2]
ikmax = 5
dt = 10

plt.rcParams.update({'font.size': 16})
svfg = plt.figure()

for llidx in range(len(lrules)):
    nepoch = nepochs[llidx]
    
    training_times = []
    for wsidx in range(len(width_scalings)):
        training_times.append( nepoch*np.ones( len(learning_rates[llidx][wsidx]) ) )

    for wsidx in range(len(width_scalings)):
        width_scaling = width_scalings[wsidx]
        width = widths[wsidx]
        for lridx in range( len(learning_rates[llidx][wsidx]) ):
            learning_rate = learning_rates[llidx][wsidx][lridx]
            
            ts = []; perfs = []
            for sidx in range(ikmax):
                festr = 'data/gfb_mlp_mnist_np_rf_sgd_K' + str(K) + '_w' + str(width) + '_B' + str(batch_size) + '_lr' + str(learning_rate) + '_nep' + str(nepoch) + '_lrl' + str(llidx) + '_sd' + str(sidx) + '.txt'
                
                lidx = 0
                for line in open(festr, 'r'):
                    ltmps = line[:-1].split(" ")
                    if sidx == 0 and lidx%dt == 0:
                        ts.append( 1+int(ltmps[0]) )
                        perfs.append( 0.0 )
                    perfs[ int(floor(lidx/dt)) ] += float(ltmps[1])/float(dt*ikmax)
                    lidx += 1
            
            #detect the first time point at which the performance surpasses the threshold
            for tidx in range(1, len(perfs)):
                if perfs[tidx-1] < perf_th and perf_th <= perfs[tidx]:
                    training_times[wsidx][lridx] = tidx*dt + dt/2.0; break;

    min_training_times = []
    for wsidx in range(len(width_scalings)):
        min_training_times.append( min(training_times[wsidx]) )

    plt.plot(widths, min_training_times, 'o-', c=clrs[llidx])
plt.loglog()
plt.show()

svfg.savefig('fig_gfb_mlp_mnist_np_rf_sgd_min_time_K' + str(K) + '_w' + str(widths[0]) + '_' + str(widths[-1]) + '_B' + str(batch_size) + '_ikm' + str(ikmax) + '.pdf')




