import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import laplacian
import scipy
from sklearn.datasets import make_blobs, make_circles
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.metrics import adjusted_rand_score


dataA, labelA = make_blobs(n_samples=500,n_features= 2,centers=5, shuffle=False)
#dataA, labelA = make_circles(factor=0.5, shuffle=False)
plt.scatter(dataA[:,0], dataA[:,1] , c = labelA)
#dataB, labelB = make_circles(factor=0.8, shuffle=False)
dataB, labelB = make_blobs(n_samples=500,n_features= 2,centers=5, shuffle=False)

dataC, labelC = make_blobs(n_samples=500,n_features= 2,centers=5, shuffle=True)

plt.scatter(dataB[:,0], dataB[:,1] , c = labelB)

A = 1/kneighbors_graph(dataA, 5, mode='distance', include_self=False, n_jobs=8).todense()
A[A==np.inf]=0
B = 1/kneighbors_graph(dataB, 5, mode='distance', include_self=False, n_jobs=8).todense()
B[B==np.inf]=0
C = 1/kneighbors_graph(dataC, 5, mode='distance', include_self=False, n_jobs=8).todense()
C[C==np.inf]=0

#T = T.reshape((A.shape+tuple([2])))

Da = np.diag(np.sum(np.array(A), axis=0))
Db = np.diag(np.sum(np.array(B), axis=0))
Dc = np.diag(np.sum(np.array(C), axis=0))


La = Da - A
Lb = Db - B
Lc = Dc - B

# La = laplacian(A, normed=True)
# Lb = laplacian(B, normed=True)
# Lc = laplacian(C, normed=True)

eigvecA = np.linalg.eig(La)[1].real
eigvalA = np.linalg.eig(La)[0].real
eigvecA = eigvecA[:,eigvalA>0]
eigvalA = eigvalA[eigvalA>0]
inda = np.argsort(abs(eigvalA))[:5]


eigvecB = np.linalg.eig(Lb)[1].real
eigvalB = np.linalg.eig(Lb)[0].real
eigvecB = eigvecB[:,eigvalB>0]
eigvalB = eigvalB[eigvalB>0]

indb = np.argsort(abs(eigvalB))[:5]


eigvecC = np.linalg.eig(Lc)[1].real
eigvalC = np.linalg.eig(Lc)[0].real
eigvecC = eigvecC[:,eigvalC>0]
eigvalC = eigvalC[eigvalC>0]
indc = np.argsort(abs(eigvalC))[:5]

eigvalA[inda]
eigvalB[indb]
eigvalC[indc]

L  = La + Lb + Lc - (np.dot(eigvecA[:,inda], eigvecA[:,inda].T) + np.dot(eigvecB[:,indb], eigvecB[:,indb].T) +np.dot(eigvecC[:,indc], eigvecC[:,indc].T))


eigvec = np.linalg.eig(L)[1].real
eigval = np.linalg.eig(L)[0].real
eigvec = eigvec[:,eigval>0]
eigval= eigval[eigval>0]
ind = np.argsort(abs(eigval))[:5]

eigval[ind]


plt.matshow(eigvec[:,ind])

v = eigvec[:,ind]


v_norm = (v.T/np.sqrt((np.sum(np.power(v,2), axis=1)))).T

km = KMeans(5)

pred = km.fit_predict(v_norm)

ari = adjusted_rand_score(labelA, pred)

plt.scatter(dataB[:,0], dataB[:,1] , c = pred)

