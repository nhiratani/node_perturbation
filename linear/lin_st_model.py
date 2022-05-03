#
# Deep linear NP
#
# from Xavier-Glorot initialization
#
# Model description
#
from math import *
import sys
import numpy as np
from numpy import random as nrnd
from numpy import linalg as nlg

#Xavier-glorot weight initialization with scaling factor r_winit2
def xavier_weight_init(Lx, Lh, Ly, r_winit2):
    A = nrnd.normal(0.0, 1.0/sqrt( 0.5*(Lx+Ly) ), (Ly,Lx))
    
    W1 = nrnd.normal(0.0, sqrt( 2.0*r_winit2/(Lx + Lh) ), (Lh,Lx))
    W2 = nrnd.normal(0.0, sqrt( 2.0/(Lh + Ly) ), (Ly,Lh))

    return A, W1, W2

#Generating a sample from the teacher network
def sample_generation(A, sigmat):
    Lx = len(A[0]); Ly = len(A)
    x = nrnd.normal(0.0,1.0, (Lx));
    yt = np.dot(A,x)
    if sigmat > 0.0:
        yt += nrnd.normal(0.0, sigmat, (Ly))
    return x, yt

#weight update with NP
def np_weight_update(W1, W2, eta, x, yt):
    sigmap = 0.000001
    Lh = len(W2[0]); Ly = len(W2)
    
    #unperturbed pathway
    hs = np.dot(W1,x)
    ys = np.dot(W2,hs)
    etmp = 0.5*np.dot(ys-yt, ys-yt)

    #perturbed pathway
    hxi = nrnd.normal(0.0, 1.0, (Lh))
    yxi = nrnd.normal(0.0, 1.0, (Ly))
    hp = hs + sigmap*hxi
    yp = np.dot(W2, hp) + sigmap*yxi
    eptmp = 0.5*np.dot(yp-yt, yp-yt)
        
    detmp = eptmp - etmp
    W1 = W1 - (eta*detmp/sigmap)*np.outer( hxi, np.transpose(x) )
    W2 = W2 - (eta*detmp/sigmap)*np.outer( yxi, np.transpose(hs) )

    return W1, W2

#Weight update with SGD
def sgd_weight_update(W1sgd, W2sgd, eta, x, yt):
    hs_sgd = np.dot(W1sgd, x)
    ys_sgd = np.dot(W2sgd, hs_sgd)
    W1sgd_new = W1sgd - eta*np.outer(np.dot(np.transpose(W2sgd), ys_sgd-yt), x)
    W2sgd_new = W2sgd - eta*np.outer(ys_sgd-yt, np.dot(W1sgd, x))
    return W1sgd_new, W2sgd_new

#weight regularization
def weight_reg(wtmp, wnorm):
    wtmp_norm = np.sqrt( np.mean(np.multiply(wtmp, wtmp), axis=1) )
    ztmp = np.outer(np.divide(wnorm, wtmp_norm), np.ones(len(wtmp[0])))
    return np.multiply(ztmp, wtmp)

#Calculate the order_parameters
def calc_error_norm(W1, W2, A):
    Lx = len(A[0]); Ly = len(A)
    alpha = np.sum(np.multiply(W1, W1))/float(Lx)
    beta = np.sum(np.multiply(W2, W2))/float(Ly)
    epsilon = ( nlg.norm(np.dot(W2, W1) - A)**2 )/float(Lx*Ly)

    return alpha, beta, epsilon

#Calculate the planar dimensionality
def calc_pl_dim(W1):
    Ctmp = np.dot( np.transpose(W1), W1 )
    trtmp = np.trace(Ctmp)
    return trtmp*trtmp/np.trace( np.dot(Ctmp, Ctmp) )

#weight update with NP, under an adaptive learning rate
def np_ada_weight_update(W1, W2, eta, x, yt, gm, n):
    sigmap = 0.000001
    Lh = len(W2[0]); Ly = len(W2)
    eta_a = eta*(n**(-gm))
    
    hs = np.dot(W1,x)
    ys = np.dot(W2,hs)
    etmp = 0.5*np.dot(ys-yt, ys-yt)
    
    hxi = nrnd.normal(0.0, 1.0, (Lh))
    yxi = nrnd.normal(0.0, 1.0, (Ly))
    hp = hs + sigmap*hxi
    yp = np.dot(W2, hp) + sigmap*yxi
    eptmp = 0.5*np.dot(yp-yt, yp-yt)
    
    detmp = eptmp - etmp
    W1 = W1 - (eta_a*detmp/sigmap)*np.outer( hxi, np.transpose(x) )
    W2 = W2 - (eta_a*detmp/sigmap)*np.outer( yxi, np.transpose(hs) )
    
    return W1, W2

if __name__ == "__main__":
    main()
