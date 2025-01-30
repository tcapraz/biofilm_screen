import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


k = 3

A = np.zeros((90,90))
A[0:30,0:30] =1
A[30:60,30:60]  = 1
A[60:90, 60:90] =1
B = np.zeros((90,90))
B[0:30,0:30] =1
B[30:60,30:60]  = 1
B[60:90, 60:90] =1
T = np.stack([A,B])
#T = T.reshape((A.shape+tuple([2])))



Z = np.random.dirichlet(np.ones(3), size=100)
Z =np.ones((90,3))*0.25
Z[0:30,0] =0.5
Z[30:60,1] =0.5
Z[60:90,2] =0.5

def compute_pi(Z,A):
    return np.sum(Z,axis=0)/A.shape[1]

def compute_x_kk(Z,A):
    x_kk = np.zeros((Z.shape[1], A.shape[0]))
    for c in range(Z.shape[1]):
        for m in range(A.shape[0]):
            for i in range(A.shape[1]):
                for j in range(A.shape[2]):
                    x_kk[c,m] += Z[i,c]*Z[j,c] * A[m,i,j]
    return x_kk

def compute_x_k(Z,A):
    x_i = np.sum(A,axis=2)
    return np.dot(x_i,Z).T

def compute_gamma_kk(x_kk, x_k):
    return x_kk/x_k**2

def compute_gamma(A, x_kk, x_k):
    Nb = np.sum(np.sum(A, axis=1),axis=1)
    gamma = (Nb- np.sum(x_kk, axis=0))/(Nb**2 - np.sum(x_k**2,axis=0))
    
    return gamma

def compute_Z(pi,Z,A, gamma_kk, gamma):
    z_ = []
    
    g= []
    for k in range(Z.shape[1]):
        g.append(np.zeros((90,90)))
        for b in range(A.shape[0]):
            for i in range(A.shape[1]):
                for j in range(A.shape[2]): 
                    log = np.log(gamma_kk[k,b]/gamma[b])
                    g[k][i,j] = g[k][i,j] + A[b,i,j] *log

    exp = []
    for k in range(Z.shape[1]): 
        exp.append(np.zeros(90))
        for j in range(g[0].shape[1]):    
            exp[k] = exp[k] + Z[j,k]*g[k][:,j]
        
    z_ = []
    for i in range(len(exp)):
        z_.append(pi[i]*np.exp(exp[i]*0.5))
    z_ = np.stack(z_).T
    #g = np.dot(A.T, log.T)
   #  exp  = np.einsum("ijk,jk ->ik", g, Z)
   # # exp = exp.astype(np.float128)
   #  z_ = pi*np.exp(out*0.5)
    return (z_.T/np.sum(z_, axis=1)).T
    

def compute_loss(Z,pi, gamma_kk, gamma, x_k, A):
    term1 = np.sum(Z*np.log(pi))
    Nb = np.sum(np.sum(A, axis=1),axis=1)

    term2 = 0
    for i in range(T.shape[0]):
        log = np.log(gamma_kk[:,i]/gamma[i])
        term2 += np.sum(log*x_kk[:,i] - (x_k[:,i]**2) * (gamma_kk[:,i] - gamma[i]))
    
    Nb = np.sum(np.sum(A, axis=1),axis=1)
    term3 = np.sum(Nb*np.log(gamma) - Nb*gamma)
    
    entropy= -np.sum(Z*np.log(Z))
    return term1 +term2*0.5 + term3 +entropy    

it = 30

for i in range(it):
    
    pi = compute_pi(Z,T)
    x_k = compute_x_k(Z,T)
    x_kk = compute_x_kk(Z,T)
    gamma_kk = compute_gamma_kk(x_kk,x_k)
    gamma = compute_gamma(T,x_kk,x_k)

    Z = compute_Z(pi, Z,T, gamma_kk, gamma)
    
    
    loss = compute_loss(Z,pi, gamma_kk, gamma, x_k, T)
    print(loss)

np.argmax(Z, axis=1)
