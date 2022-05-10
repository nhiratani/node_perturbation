#
# SARCOS
#
# SGD and NP
#
# K+1 layers
#
# Xavier-Glorot initialization
#
# Shuffled mini-batch
#
# Calculate the cosine similarity between NP and SGD
#
import sys

import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io as sciio
from math import *
import time

#print("Num gpus available:", len(tf.config.experimental.list_physical_devices('GPU')) )

def simul(K, width, batch_size, learning_rate, nepoch, seed):
    tf.disable_eager_execution()
    np.set_printoptions(precision=6, suppress=True)
    
    #initialization
    np.random.seed(seed)
    tf.set_random_seed(seed)
    
    #training data loading
    fname_train = "../dataset/sarcos_inv.mat"; dict_train_tmp = {}
    data_train_tmp = sciio.loadmat(fname_train, mdict=dict_train_tmp)
    train_data = data_train_tmp['sarcos_inv']
    x_train = train_data[:,:21]
    y_train = train_data[:,21:]
    
    shuffler = np.random.permutation(len(y_train))
    x_train = x_train[shuffler]
    y_train = y_train[shuffler]
    n_train = len(x_train)
    
    #data loading
    fname_test = "../dataset/sarcos_inv_test.mat"; dict_test_tmp = {}
    data_test_tmp = sciio.loadmat(fname_test, mdict=dict_test_tmp)
    test_data = data_test_tmp['sarcos_inv_test']
    x_test = test_data[:,:21]
    y_test = test_data[:,21:]
    
    #parameters
    Lx = 21; Ly = 7
    sigma = 0.0001
    lnr = learning_rate/float(batch_size)

    #Architecture specification
    Lhs = []; Lhs.append(Lx)
    for k in range(1,K):
        Lhs.append( width )
    Lhs.append(Ly)
    
    x = tf.placeholder(dtype=tf.float32, shape=[None, Lx])
    y = tf.placeholder(dtype=tf.float32, shape=[None, Ly])
    pred = tf.placeholder(dtype=tf.float32, shape=[None, Ly])
    pred_np = tf.placeholder(dtype=tf.float32, shape=[None, Ly])
    
    #network configuration
    initializer_hl = tf.initializers.glorot_normal()
    with tf.name_scope('variables'):
        ws = []; bs = []
        for k in range(K):
            ws.append( tf.Variable( initializer_hl( shape=([Lhs[k+1], Lhs[k]]) ) ) )
            bs.append( tf.Variable( initializer_hl( shape=([Lhs[k+1]]) ) ) )

    sigos = [] #initial weight variance
    for k in range(K):
        sigos.append( sqrt( 2.0/(Lhs[k] + Lhs[k+1]) ) )

    with tf.name_scope('noise'):
        xis = []
        for k in range(K):
            xis.append( tf.random_normal( shape=[batch_size, Lhs[k+1]], mean=0, stddev=sigma, dtype=tf.float32 ) )
    
    with tf.name_scope('batch-prediction'):
        hs = []; hs.append(x)
        for k in range(1,K):
            hs.append( tf.nn.relu( tf.add( tf.matmul( hs[k-1], tf.transpose(ws[k-1]) ), bs[k-1] ) ) )
        #linear readout
        pred = tf.add( tf.matmul( hs[K-1], tf.transpose(ws[K-1]) ), bs[K-1] )

    # calculate sgd and node-perturbation updates
    with tf.name_scope('cosine_similarity'):
        err = pred - y

        derrs = []
        derrs.append( err )
        for k in range(1,K): #caution: it goes backwards
            derrs.append( tf.multiply(tf.tensordot(derrs[k-1], ws[K-k], axes=1), tf.sign(hs[K-k])) )

        #sgd updates
        delta_ws = []; delta_bs = []
        for k in range(K):
            delta_ws.append( tf.tensordot(derrs[K-1-k], hs[k], axes=[[0],[0]]) )
            delta_bs.append( tf.reduce_sum(derrs[K-1-k], 0) )

        hs_np = []; hs_np.append(x)
        for k in range(1,K):
            hs_np.append( tf.nn.relu( tf.add( tf.add( tf.matmul(hs_np[k-1], tf.transpose(ws[k-1])), bs[k-1] ), xis[k-1]) ) )
        pred_np = tf.add( tf.add( tf.matmul(hs_np[K-1], tf.transpose(ws[K-1])), bs[K-1] ), xis[K-1])

        error = tf.reduce_sum( tf.multiply(pred-y, pred-y), 1 )
        error_np = tf.reduce_sum( tf.multiply(pred_np - y, pred_np-y), 1 )
        derror = (error_np - error)/(2.0*sigma*sigma)

        #NP updates
        delta_ws_np = []; delta_bs_np = []
        for k in range(K):
            xihtmp = tf.einsum( 'ki,kj->kij', xis[k], hs[k] )
            delta_ws_np.append( tf.tensordot( derror, xihtmp, axes=[[0],[0]] ) )
            delta_bs_np.append( tf.tensordot( derror, xis[k], axes=[[0],[0]] ) )

        #layer-wise cosine similarity
        np_norms = []; sgd_norms = []
        np_sgd_corrs = []; cos_sims = []
        for k in range(K):
            sgd_norms.append( tf.cast( tf.reduce_mean(tf.multiply(delta_ws[k], delta_ws[k])), tf.float32) )
            np_norms.append( tf.cast( tf.reduce_mean(tf.multiply(delta_ws_np[k], delta_ws_np[k])), tf.float32) )
            np_sgd_corrs.append( tf.cast( tf.reduce_mean(tf.multiply(delta_ws[k], delta_ws_np[k])), tf.float32) )
            cos_sims.append( tf.divide(np_sgd_corrs[k], tf.sqrt( tf.multiply(sgd_norms[k], np_norms[k]) ) ) )

    #SGD weight update
    with tf.name_scope('update'):
        update_ws = []; update_bs = []
        for k in range(K):
            update_ws.append( tf.assign( ws[k], tf.add(ws[k], -lnr*delta_ws[k]) ) )
            update_bs.append( tf.assign( bs[k], tf.add(bs[k], -lnr*delta_bs[k]) ) )

    accuracy = 100.0*tf.reduce_mean( tf.cast(tf.equal( tf.argmax(pred,1), tf.argmax(y,1) ), tf.float32) )
    base_error = tf.reduce_mean( tf.multiply(pred-y, pred-y) )

    #file out
    festr = 'data/gfb_mlp_sarcos_sgd_np_sim_K' + str(K) + '_w' + str(width) + '_B' + str(batch_size) + '_lr' + str(learning_rate) + '_nep' + str(nepoch) + '_sd' + str(seed) + '.txt'
    fwe = open(festr,'w')
    
    #session execution
    sess = tf.InteractiveSession()
    init = tf.global_variables_initializer()
    sess.run(init)

    base_dict = {};
    base_dict[x] = x_test; base_dict[y] = y_test;

    train_dicts = []
    for j in range( int(floor(n_train/batch_size)) ):
        train_dicts.append({})
        train_dicts[j][x] = x_train[j*batch_size:(j+1)*batch_size]
        train_dicts[j][y] = y_train[j*batch_size:(j+1)*batch_size]

    for i in range(nepoch):
        perftmps = sess.run([base_error], feed_dict=base_dict)
        b_error = perftmps[0]
        cos_sim_tmps = np.zeros((K))
        
        train_error = 0.0
        for j in range( int(floor(n_train/batch_size)) ):
            perftmps = sess.run([base_error, update_ws, update_bs, cos_sims], feed_dict = train_dicts[j])
            train_error += perftmps[0]/float(n_train/batch_size)
            
            for k in range(K):
                cos_sim_tmps[k] += perftmps[3][k]/floor(n_train/batch_size)
        
        fwe.write(str(i) + " " + str(b_error) + " " + str(train_error) + " " + str(cos_sim_tmps[0]) + " " + str(cos_sim_tmps[1]) + "\n"); fwe.flush()
        if np.isnan(b_error):
            break
    sess.close();

def main():
    param = sys.argv
    K = int(param[1])
    width = int(param[2])
    batch_size = int(param[3])
    learning_rate = float(param[4])
    nepoch = int(param[5])
    seed = int(param[6])
    
    simul(K, width, batch_size, learning_rate, nepoch, seed)

if __name__ == "__main__":
    main()

