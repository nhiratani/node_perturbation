#
# MNIST
#
# NP, Reinforce, SGD
#
# K+1 layers
#
# Xavier-Glorot initialization
#
import sys

import tensorflow.compat.v1 as tf
import numpy as np
import scipy.io as sciio
from math import *
import time

#print("Num gpus available:", len(tf.config.experimental.list_physical_devices('GPU')) )

def simul(K, width, batch_size, learning_rate, nepoch, lrule, seed):
    tf.disable_eager_execution()
    np.set_printoptions(precision=6, suppress=True)
    
    #initialization
    np.random.seed(seed)
    tf.set_random_seed(seed)
    
    #data loading
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()
    train_images, test_images = train_images/255.0, test_images/255.0
    
    #preprosessing
    with tf.Session() as sess:
        y_train = sess.run( tf.one_hot(train_labels, 10) )
        y_test = sess.run( tf.one_hot(test_labels, 10) )
    
    x_train = []; x_test = []
    for i in range( len(train_images) ):
        x_train.append( np.ndarray.flatten(train_images[i]) )
    for i in range( len(test_images) ):
        x_test.append( np.ndarray.flatten(test_images[i]) )
    
    #parameters
    Lx = 784; Ly = 10
    sigma = 0.0001#0.000001
    n_train = len(x_train)
    lnr = learning_rate#/float(batch_size)

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
        pred = tf.nn.softmax( tf.add( tf.matmul( hs[K-1], tf.transpose(ws[K-1]) ), bs[K-1] ) )

    #node perturbation update
    with tf.name_scope('delta_np'):
        hs_np = []; hs_np.append(x)
        for k in range(1,K):
            hs_np.append( tf.nn.relu( tf.add( tf.add( tf.matmul(hs_np[k-1], tf.transpose(ws[k-1])), bs[k-1] ), xis[k-1]) ) )
        pred_np = tf.nn.softmax( tf.add( tf.add( tf.matmul(hs_np[K-1], tf.transpose(ws[K-1])), bs[K-1] ), xis[K-1]) )
        
        error = tf.reduce_sum( tf.multiply(pred-y, pred-y), 1 )
        error_np = tf.reduce_sum( tf.multiply(pred_np - y, pred_np-y), 1 )
        derror = (error_np - error)/(2.0*sigma*sigma)
        
        delta_ws_np = []; delta_bs_np = []
        for k in range(K):
            xihtmp = tf.einsum( 'ki,kj->kij', xis[k], hs[k] )
            delta_ws_np.append( tf.tensordot( derror, xihtmp, axes=[[0],[0]] ) )
            delta_bs_np.append( tf.tensordot( derror, xis[k], axes=[[0],[0]] ) )

        update_ws_np = []; update_bs_np = []
        for k in range(K):
            update_ws_np.append( tf.assign( ws[k], tf.add(ws[k], -lnr*delta_ws_np[k]) ) )
            update_bs_np.append( tf.assign( bs[k], tf.add(bs[k], -lnr*delta_bs_np[k]) ) )
                
    #Reinforcement learning update
    with tf.name_scope('delta_rb'):
        #np error (for the last layer)
        hs_np = []; hs_np.append(x)
        for k in range(1,K):
            hs_np.append( tf.nn.relu( tf.add( tf.add( tf.matmul(hs_np[k-1], tf.transpose(ws[k-1])), bs[k-1] ), xis[k-1]) ) )
        pred_np = tf.nn.softmax( tf.add( tf.add( tf.matmul(hs_np[K-1], tf.transpose(ws[K-1])), bs[K-1] ), xis[K-1]) )
        
        error = tf.reduce_sum( tf.multiply(pred-y, pred-y), 1 )
        error_np = tf.reduce_sum( tf.multiply(pred_np - y, pred_np-y), 1 )
        derror = (error_np - error)/(2.0*sigma*sigma)
        
        #back-prop error (for the rest of layers)
        derrs = []
        derrs.append( tf.einsum( 'i,ij->ij', derror, xis[K-1]) )
        for k in range(1,K): #caution: it goes backwards
            derrs.append( tf.multiply(tf.tensordot(derrs[k-1], ws[K-k], axes=1), tf.sign(hs[K-k])) )
        
        delta_ws_rf = []; delta_bs_rf = []
        for k in range(K):
            if k == K-1:
                xihtmp = tf.einsum( 'ki,kj->kij', xis[k], hs[k] )
                delta_ws_rf.append( tf.tensordot( derror, xihtmp, axes=[[0],[0]] ) )
                delta_bs_rf.append( tf.tensordot( derror, xis[k], axes=[[0],[0]] ) )
            else:
                delta_ws_rf.append( tf.tensordot(derrs[K-1-k], hs[k], axes=[[0],[0]]) )
                delta_bs_rf.append( tf.reduce_sum(derrs[K-1-k], 0) )

        update_ws_rf = []; update_bs_rf = []
        for k in range(K):
            update_ws_rf.append( tf.assign( ws[k], tf.add(ws[k], -lnr*delta_ws_rf[k]) ) )
            update_bs_rf.append( tf.assign( bs[k], tf.add(bs[k], -lnr*delta_bs_rf[k]) ) )

    #sgd update
    with tf.name_scope('delta_sgd'):
        err = pred - y
        errtmp = err - tf.einsum('i,j->ij', tf.reduce_sum(tf.multiply(err, pred), 1), tf.ones([Ly], tf.float32) )
        
        derrs = []
        derrs.append( tf.multiply(errtmp, pred) )
        for k in range(1,K): #caution: it goes backwards
            derrs.append( tf.multiply(tf.tensordot(derrs[k-1], ws[K-k], axes=1), tf.sign(hs[K-k])) )
        
        delta_ws_sgd = []; delta_bs_sgd = []
        for k in range(K):
            delta_ws_sgd.append( tf.tensordot(derrs[K-1-k], hs[k], axes=[[0],[0]]) )
            delta_bs_sgd.append( tf.reduce_sum(derrs[K-1-k], 0) )

        update_ws_sgd = []; update_bs_sgd = []
        for k in range(K):
            update_ws_sgd.append( tf.assign( ws[k], tf.add(ws[k], -lnr*delta_ws_sgd[k]) ) )
            update_bs_sgd.append( tf.assign( bs[k], tf.add(bs[k], -lnr*delta_bs_sgd[k]) ) )

    #weight norm
    with tf.name_scope('weight_norm'):
        wnorms = []
        for k in range(K):
            wnorms.append( tf.cast( tf.reduce_sum(tf.multiply(ws[k], ws[k])), tf.float32) )

    accuracy = 100.0*tf.reduce_mean( tf.cast(tf.equal( tf.argmax(pred,1), tf.argmax(y,1) ), tf.float32) )
    base_error = tf.reduce_mean( tf.multiply(pred-y, pred-y) )

    #file out
    festr = 'data/gfb_mlp_mnist_np_rf_sgd_K' + str(K) + '_w' + str(width) + '_B' + str(batch_size) + '_lr' + str(learning_rate) + '_nep' + str(nepoch) + '_lrl' + str(lrule) + '_sd' + str(seed) + '.txt'
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
        perftmps = sess.run([accuracy, base_error, wnorms], feed_dict=base_dict)
        test_accuracy = perftmps[0]; b_error = perftmps[1]
        wtmps = []
        for k in range(K):
            wtmps.append( perftmps[2][k] )

        train_accuracy = 0.0; train_error = 0.0
        for j in range( int(floor(n_train/batch_size)) ):
            if lrule == 0: #NP
                perftmps = sess.run([accuracy, base_error, update_ws_np, update_bs_np], feed_dict = train_dicts[j])
            if lrule == 1: #Reinforce
                perftmps = sess.run([accuracy, base_error, update_ws_rf, update_bs_rf], feed_dict = train_dicts[j])
            if lrule == 2: #SGD
                perftmps = sess.run([accuracy, base_error, update_ws_sgd, update_bs_sgd], feed_dict = train_dicts[j])
            
            train_accuracy += perftmps[0]/float(n_train/batch_size)
            train_error += perftmps[1]/float(n_train/batch_size)

        fwe.write(str(i) + " " + str(test_accuracy) + " " + str(b_error) + " " + str(train_accuracy) + " " + str(train_error) + " " + str(wtmps[0]) + " " + str(wtmps[1]) + "\n"); fwe.flush()
        if np.isnan(b_error):
            break
    sess.close();

def main():
    param = sys.argv
    K = int(param[1]) #depth
    width = int(param[2])
    batch_size = int(param[3])
    learning_rate = float(param[4])
    nepoch = int(param[5])
    lrule = int(param[6]) #learning rule (0: NP, 1: Reinforce, 2: SGD)
    seed = int(param[7])
    
    simul(K, width, batch_size, learning_rate, nepoch, lrule, seed)

if __name__ == "__main__":
    main()

