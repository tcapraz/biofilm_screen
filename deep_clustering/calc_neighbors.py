import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

features  = pd.read_csv("featuresSimCLR.csv", header=None)

NN = NearestNeighbors(n_neighbors =20)

NN.fit(features)
neighbors = NN.kneighbors(return_distance=False)

np.savetxt("kneighbors.csv", neighbors)