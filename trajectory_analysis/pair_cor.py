import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.sparse.csgraph import laplacian
import scipy
from sklearn.datasets import make_blobs, make_circles
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sklearn.metrics import adjusted_rand_score
from itertools import chain, combinations

def powerset(iterable):
    "powerset([1,2,3]) --> () (1,) (2,) (3,) (1,2) (1,3) (2,3) (1,2,3)"
    s = list(iterable)
    return chain.from_iterable(combinations(s, r) for r in range(9,len(s)+1))


data = []
for i in range(10):
    
    data.append(make_blobs(n_samples=20,n_features= 2,centers=2, shuffle=False)[0])

for i in range(3):
    data.append(make_blobs(n_samples=20,n_features= 2,centers=2, shuffle=True)[0])


cube = np.stack(data)

cube1d = cube[:,:,0]

comb = [i for i in powerset(range(cube.shape[0]))]

pairwise = [i for i in combinations([i for i in range(cube.shape[1])], r=2)]


# pairs = {}
# for x,i in enumerate(pairwise):
#     pairs[i] = []
#     for j in comb:
#         c = cube[j, :, ]
#         c = c[:,i]
#         cor1 = np.corrcoef(c.T)[0,1]
#         c = cube[j, :, 1]
#         c = c[:,i]
#         cor2 = np.corrcoef(c.T)[0,1]
        
#         pairs[i].append(np.mean([cor1,cor2]))
#     print(x)


cors = []
for i in comb:
    varc = []
    for j in range(cube.shape[2]):
        c =cube[i,:,j]
        cor = np.corrcoef(c.T)
        varc.append(cor[np.triu_indices(20, k =1)])
    cors.append(np.mean(np.vstack(varc), axis=0))


pairs = {}

for i in range(190):
    pairs[i] = np.array([j[i] for j in cors])


best_cond= []
for i in pairs:
    best_cond.append(comb[np.argmax(pairs[i])])

v1 = cube1d[:,0]
v2 = cube1d[:,1]


summands = (np.mean(v1) - v1) * (np.mean(v2) - v2)
#summands[np.argpartition(summands, -14)[-14:]]

norm = np.sqrt(np.sum(np.square(np.mean(v1) - v1)) * np.sqrt(np.sum(np.square(np.mean(v2) - v2))))

np.sum(summands)/norm
