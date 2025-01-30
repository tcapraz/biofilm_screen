from skimage.io import imread
import mahotas.features as mht
from skimage.feature import greycomatrix, greycoprops
from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
from skimage.measure import label, regionprops, regionprops_table 
import pandas as pd
import numpy as np
import os
from pathlib import Path
import cv2 
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
import string
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import umap
from PIL import Image, ImageStat
import math
import sys
from itertools import product, combinations
import matplotlib.cm as cm
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sknetwork.clustering import Louvain
from scanpy.pp import neighbors
import anndata as ad
import scipy
import scipy.stats as stats
from matplotlib import cm




# Scatter with images instead of points
def imscatter(x, y, ax, imageData, zoom):
    imags = []
    for i in range(len(x)):
        x0, y0 = x[i], y[i]
        # Convert to image
        #img = imageData[i]*255.
        #img = img.astype(np.uint8).reshape([imageSize,imageSize])
        #img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
        # Note: OpenCV uses BGR and plt uses RGB
        #image = OffsetImage(img, zoom=zoom)
        imag = OffsetImage(imageData[i], zoom=zoom)
        ab = AnnotationBbox(imag, (x0, y0), xycoords='data', frameon=False)
        imags.append(ax.add_artist(ab))
    
    ax.update_datalim(np.column_stack([x, y]))
    ax.autoscale()
    
filepaths = []
for root, dirs, f in os.walk("../data/all_data"):
    for name in f:
        if name.endswith((".jpg")):
            filepaths.append(root +"/"+ name)
    
testfilepaths = []
for root, dirs, f in os.walk("../data/test/"):
    for name in f:
        if name.endswith((".jpg")):
            testfilepaths.append(root +"/"+ name)
            
anno = pd.read_csv("../data/annotationsM1-M10.txt", delim_whitespace=True, header=None)
anno = anno.drop_duplicates()
anno.index = anno[1]
names = [Path(i).stem for i in testfilepaths]
anno = anno.loc[names,:]

data = pd.read_csv("embeddings/vae_giovanni_features_params1.csv", index_col=0)


meta_data = pd.read_csv("../metadata/meta_data.csv")
meta_data.index = meta_data["filename"]


data =  data.loc[:,~np.isnan(data.astype(float)).any(axis=0)]
day4 = meta_data[meta_data["day"] =="Day4"]
data_day4 = data.loc[day4.index,:]



testdata = data.loc[names,:]
#data = data.iloc[0:50000,:]
meta_data = meta_data.loc[data.index,:]



conditions  = [np.where(day4["cond"] == i)[0] for i  in np.unique(day4["cond"])]

condition_data = []
strains = []
for i in conditions:
    cd = data_day4.iloc[i,:]
    cd.index = day4["genotype"][i]
    strains.append(day4["genotype"][i])
    condition_data.append(cd)

consensus_strains = set.intersection(*map(set,strains))
consensus_strains =  list(consensus_strains)
for i in range(len(condition_data)):
    cd = condition_data[i].loc[consensus_strains,:]
    condition_data[i]  =  cd[~cd.index.duplicated(keep='first')]
    
cube = np.stack(condition_data)

cube1d = cube[:,:,0]

#################### correlation approach ####################################
# pairwise = combinations([i for i in range(cube1d.shape[1])], r=2)

# v1 = cube1d[:,0]
# v2 = cube1d[:,1]


# summands = (np.mean(v1) - v1) * (np.mean(v2) - v2)
# summands[np.argpartition(summands, -14)[-14:]]

# norm = np.sqrt(np.sum(np.square(np.mean(v1) - v1)) * np.sqrt(np.sum(np.square(np.mean(v2) - v2))))

# np.sum(summands)/norm

pairs = np.array([(i,j) for i,j in zip(np.triu_indices(cube.shape[1], k =1)[0],np.triu_indices(cube.shape[1], k =1)[1]) ])
varc = []
for j in range(cube.shape[2]):
    c =cube[:,:,j]
    cor = np.corrcoef(c.T)
    varc.append(cor[np.triu_indices(cube.shape[1], k =1)])

cors = np.quantile(np.vstack(varc), 0.95, axis=0)

cor_pairs = pairs[cors > 0.8]
cor_strains = [(consensus_strains[i[0]], consensus_strains[i[1]]) for i in cor_pairs]

s = cor_strains[12]

s1 = day4[day4["genotype"] == s[0]]
s2 = day4[day4["genotype"] == s[1]]

s1 = s1.drop_duplicates("cond")
s2 = s2.drop_duplicates("cond")

aligned = s1.merge(s2, how='outer', indicator=True, on="cond")

images = []

for i,j in zip(aligned["filename_x"], aligned["filename_y"]):
    images.append((imread("../data/all_data_bg/data/"+i+".jpg"), imread("../data/all_data_bg/data/"+j+".jpg")))

for i in images:
    fig, ax = plt.subplots(1,2) 
    
    ax[0].imshow(i[0])
    ax[1].imshow(i[1])
    
    
#################### graph intersection approach ##############################
graphs = []
for i in range(cube.shape[0]):
    graphs.append(np.array(kneighbors_graph(cube[i,::], 30, mode='connectivity', include_self=False, n_jobs=8).todense()))

graph_cube = np.stack(graphs)

graph_sum = np.sum(graph_cube, axis=0)

pairs = np.where(graph_sum >=5)

cond_graph = np.zeros((365,365))
cond_graph[pairs] =1

louvain = Louvain()
l = louvain.fit_transform(cond_graph)


km = KMeans(10)
l = km.fit_predict(graph_sum)        

pca = PCA(2)
x_pca = pca.fit_transform(graph_sum)
plt.scatter(x_pca[:,0],x_pca[:,1], c=l)

plt.matshow(cond_graph)

import networkx as nx
G=nx.Graph()
pairs_tuple = [(i,j) for i,j in zip(pairs[0], pairs[1])]
G.add_edges_from(pairs_tuple)

max_cliques = list(nx.find_cliques(G))

#c4 = np.array(list(consensus_strains))[np.where(l == 2)[0]]

c4 = np.array(list(consensus_strains))[max_cliques[0]]
idx = []
for i in c4:
    idx.append(np.where(day4["genotype"] ==i)[0])
    
c4_data = data_day4.iloc[np.hstack(idx),:]
c4_meta = day4.iloc[np.hstack(idx),:]

pca = PCA(10)
scaler = StandardScaler()
data_sc = scaler.fit_transform(c4_data )
x = pca.fit_transform(data_sc)

tsne = TSNE(n_components=2, n_jobs=8)
x_tsne= tsne.fit_transform(x)

plt.scatter(x_tsne[:,0], x_tsne[:,1], s=10, c  = c4_meta["cond"])
plt.xlabel("TSNE 1")
plt.ylabel("TSNE 2")
plt.savefig("condition_traj.png", dpi=500)
plt.show()
plt.close()

cluster_images = []
for m in c4_data.index:
    cluster_images.append(imread("../data/all_data/data/" + m+".jpg"))
    
fig, ax = plt.subplots()
imscatter(x_tsne[:,0], x_tsne[:,1], imageData=cluster_images, ax=ax, zoom=0.09)
plt.xlabel("TSNE 1")
plt.ylabel("TSNE 2")
plt.savefig("condition_traj_imgs.png", dpi=500)

#################### tensor decomposition #####################################
import tensorly as tl
from tensorly.decomposition import parafac
tensor = tl.tensor(cube)
factors = parafac(tensor, rank=2)

plt.scatter(factors.factors[1][:,0],factors.factors[1][:,1])
