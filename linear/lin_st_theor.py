#
# Deep linear NP and SGD
#
# Mean-field theory
#
#
from math import *
import numpy as np

#estimate the optimal learning rate under SGD
def calc_sgd_eta_opt(Lx, Lh, Ly):
    alpha_z = 2.0*Lh/(Lx+Lh)
    beta_z = 2.0*Lh/(Lh+Ly)
    return 1.0/( Lx*(alpha_z + beta_z) )

#estimate the critical learning rate by solving a root
def calc_eta_star(Lx, Lh, Ly):
    delta_c = 0.0001
    alpha_z = 2.0*Lh/(Lx+Lh)
    beta_z = 2.0*Lh/(Lh+Ly)
    epsilon_z = 4.0*Lh/((Lx + Lh)*(Ly + Lh)) + 2.0/(Lx + Ly)
    
    a_z = alpha_z/np.cbrt(float(Lh))
    c_zs = np.roots([(epsilon_z - a_z*a_z*a_z/3.0), a_z*a_z, 0.0, -4.0/3.0])
    c_z = -1.0
    if len(c_zs) == 3:
        #Take the largest real root
        for q in range(3):
            if np.isreal(c_zs[q]) and np.real(c_zs[q]) > c_z:
                c_z = (1.0 - delta_c)*np.real(c_zs[q])
    else:
        print(c_zs)

    epsilon_tot = epsilon_z - a_z*a_z/c_z + a_z*a_z*a_z/3.0
    eta_z = c_z/(Lx*Ly*(1.0 + beta_z))
    eta_star = eta_z/np.cbrt( float(Lh) )
                                 
    return eta_star

#Two-variable description of NP dynamics (by a and epsilon)
def np_ae_dynamics(Lx, Lh, Ly, sigmat, eta, T, dt):
    tlen = int(floor(T/dt))
    
    alpha = 2.0*Lh/(Lx+Lh)
    beta_z = 2.0*Lh/(Lh+Ly)
    epsilon_z = 4.0*Lh/((Lx + Lh)*(Ly + Lh)) + 2.0/(Lx + Ly)

    a_z = alpha/np.cbrt(float(Lh))
    eta_z = eta*np.cbrt(float(Lh))
    c_z = eta_z*Lx*Ly*(1.0 + beta_z)
    sigmat2x = sigmat*sigmat/float(Lx)
    
    a = a_z; epsilon = epsilon_z
    v_seqs = np.zeros((2, tlen))
    for t in range(T):
        da = c_z*(epsilon + sigmat2x)
        depsilon = -2.0*a*epsilon + c_z*a*a*(epsilon + sigmat2x)
        
        a += eta_z*da; epsilon += eta_z*depsilon
        
        if t%dt == 0:
            tidx = int(floor(t/dt + 0.0001))
            v_seqs[0][tidx] = a
            v_seqs[1][tidx] = epsilon

    return v_seqs

#Four-variable description of NP dynamics
def np_mf_dynamics(Lx, Lh, Ly, sigmat, eta, T, dt):
    tlen = int(floor(T/dt))
    
    alpha = 2.0*Lh/(Lx+Lh)
    beta = 2.0*Lh/(Lh+Ly)
    epsilon = 4.0*Lh/((Lx + Lh)*(Ly + Lh)) + 2.0/(Lx + Ly)
    phi = -4.0*Lh/((Lx + Lh)*(Ly + Lh))
    
    psi_z = 2.0/(Lx+Ly)
    sigmat2 = sigmat*sigmat
    
    v_seqs = np.zeros((4, tlen))
    for t in range(T):
        lxe_tmp = Lx*epsilon + sigmat2
        noise_tmp = eta*eta*( Ly*(1+beta)*(alpha*alpha + beta) + 4*alpha*beta )*lxe_tmp
        psi = psi_z - (epsilon + 2.0*phi)
        
        da = 2.0*eta*Ly*phi + eta*eta*Lh*Ly*(1+beta)*lxe_tmp
        db = 2.0*eta*Lx*phi + eta*eta*Lx*Ly*alpha*(1+beta)*lxe_tmp
        de = -2*eta*(alpha+beta)*epsilon - 4*eta*eta*phi*(Lx*Ly*epsilon + sigmat2) + noise_tmp
        dphi = eta*(alpha + beta)*(epsilon - phi) + 2*eta*eta*(Lx*Ly*phi*(epsilon - phi) - sigmat2*(psi-phi) ) - noise_tmp
        
        alpha += da
        beta += db
        epsilon += de
        phi += dphi
        
        if t%dt == 0:
            tidx = int(floor(t/dt + 0.0001))
            v_seqs[0][tidx] = alpha
            v_seqs[1][tidx] = beta
            v_seqs[2][tidx] = epsilon
            v_seqs[3][tidx] = phi

    return v_seqs

# return the minimum iteration and the error value at which the error starts to go up
# NP dynamics is calculated by the two-variable model
def calc_tauc_epsilonc(Lx, Lh, Ly, sigmat, eta, T):
    beta_zero = 2.0*Lh/float(Lh+Ly)
    a_zero = ( 2.0*Lh/float(Lh+Lx) )/np.cbrt( float(Lh) )
    epsilon_zero = 4.0*Lh/float( (Lx+Lh)*(Ly+Lh) ) + 2.0/float(Lx + Ly)
    epsilon_t = sigmat*sigmat/float(Lx)
    
    etah = eta*np.cbrt(Lh) # conversion from eta to eta_h
    co = Lx*Ly*(1.0+beta_zero)
    a = a_zero; epsilon = epsilon_zero

    tauc = np.nan; epsilonc = np.nan
    for t in range(T):
        da = etah*etah*co*(epsilon + epsilon_t)
        de = etah*(-2.0*a*epsilon + etah*co*a*a*(epsilon + epsilon_t))
        a += da
        epsilon += de
        if de > 0.0:
            tauc = t; epsilonc = epsilon
            break;

    return tauc, epsilonc

if __name__ == "__main__":
    main()
