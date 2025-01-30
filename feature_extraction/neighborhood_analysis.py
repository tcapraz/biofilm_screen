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
from itertools import product
import matplotlib.cm as cm
from sklearn.neighbors import NearestNeighbors, kneighbors_graph

from sknetwork.clustering import Louvain
import anndata as ad
from scanpy.pp import neighbors
import anndata as ad
import scipy
import scipy.stats as stats
from matplotlib import cm
import networkx as nx
import community as community_louvain
from sklearn.linear_model import LinearRegression


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

########################## read data ##########################################
filepaths = []
for root, dirs, f in os.walk("../data"):
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

data = pd.read_csv("extracted_features.csv", index_col=0)
todrop = ["convex_image", "coords", "filled_image", "image",  "intensity_image", "slice", ]
data = data.drop(todrop, axis=1)

meta_data = pd.read_csv("../metadata/meta_data.csv")
meta_data.index = meta_data["filename"]


data =  data.loc[:,~np.isnan(data.astype(float)).any(axis=0)]
day4 = meta_data[meta_data["day"] =="Day4"]
data_day4 = data.loc[day4.index,:]



testdata = data.loc[names,:]
data = data.iloc[0:50000,:]
meta_data = meta_data.loc[data.index,:]


###################### dim reduction ###########################################

pca = PCA(n_components =50)
scaler = StandardScaler()
data_sc = scaler.fit_transform(data)
x_pca = pca.fit_transform(data_sc)

tsne = TSNE(n_components=2, n_jobs=8)
x_tsne = tsne.fit_transform(x_pca)


um = umap.UMAP()
um  = um.fit(x_pca)
x_um = um.fit_transform(x_pca)

# get genotype label
genotype_map  = {g:i for i,g in enumerate(meta_data["genotype"].unique())}
geno_label = [genotype_map[i] for i in meta_data["genotype"]]

geno_cond_label = (meta_data["cond"] + np.array(geno_label).astype(str)).values

plt.scatter(x_um[:,0], x_um[:,1], s=0.1, c = geno_label)

plt.scatter(x_um[:,0], x_um[:,1], s=0.1, c = geno_cond_label)

plt.scatter(x_um[:,0], x_um[:,1], s=0.1, c  = meta_data["cond"])

########################### neighborhood analysis ##########################
# according to 
# Reshef et al. 2022 Co-varying neighborhood analysis identifies cell populations associated with phenotypes 
# of interest from single-cell transcriptomics
#
var = pd.DataFrame(index = data.columns)
obs = pd.DataFrame(meta_data["cond"])
adata = ad.AnnData(data.values, obs=obs, var=var)
neighbors(adata, n_neighbors=10, n_pcs=30)

S = pd.get_dummies(geno_cond_label)
C =S.sum(axis=0)
a = adata.obsp["connectivities"]

def random_walk(a,s, nsteps=10):
    colsums = np.array(a.sum(axis=0)).flatten() + 1
    for i in range(nsteps):
        s  = a.dot(s/colsums[:,None])+ s/colsums[:,None]
        yield s

def R(A, B):
        return ((A - A.mean(axis=0))*(B - B.mean(axis=0))).mean(axis=0) \
            / A.std(axis=0) / B.std(axis=0)
            
prevmedkurt = np.inf
old_s = np.zeros(S.shape)    
nsteps=30
# walk nsteps stop if lowering of kurtosis is lower than 3 
for i, s in enumerate(random_walk(a, S.values, nsteps=nsteps)):
      medkurt = np.median(stats.kurtosis(s/C.values, axis=1))
      R2 = R(s, old_s)**2
      old_s = s
      print('\tmedian kurtosis:', medkurt+3)
      print('\t20th percentile R2(t,t-1):', np.percentile(R2, 20))
     
      if prevmedkurt - medkurt < 3 and i+1 >= 3:
          print('stopping after', i+1, 'steps')
          break
      else:
          prevmedkurt = medkurt
# normalize s
snorm = (s / C.values).T

pca = PCA(n_components =50)
pca = pca.fit(snorm)

loadings = pca.components_
nam_pcs = pca.fit_transform(snorm)

plt.scatter(nam_pcs[:,0], nam_pcs[:,1], s = 2)


plt.scatter(x_um[:,0], x_um[:,1], s=0.1, c = loadings[3,:], cmap = "seismic")



plt.scatter(x_um[:,0], x_um[:,1], s=0.1, c = np.dot ( S, nam_pcs[:,4]), cmap = "seismic")



graph = kneighbors_graph(x_pca, 5, mode='connectivity', include_self=True, n_jobs=8)
louvain = Louvain()
l = louvain.fit_transform(graph)
plt.scatter(x_um[:,0], x_um[:,1], s=0.1, c =l)


import re
import statsmodels.api as sm

conds = [re.search(r"C\d{2}",i)[0] for i in S.columns]

X = pd.get_dummies(S.columns)
X = sm.add_constant(X)
results = sm.OLS(nam_pcs[:,0], X).fit()
    
results.summary()


############### cluster matching approach ####################################
# we want to know whether there are groups of strains that covary across conditions
# for this we first need to find groups of strains in the same conditions which cluster together in morphology space
#   - cluster strains in each condition 
#   - match strains in clusters between conditions
# this gives us strains that co-vary across conditions, however we don't whether 
# they actually change their morphology (they could also have the same morphology in all conditions, 
# which is a not very interesting result)

conditions = {}
condition_strains = {}
for i in np.unique(meta_data["cond"]):
    conditions[i] = data.iloc[np.where(meta_data["cond"] == i)[0],:]
    strains = meta_data["genotype"][np.where(meta_data["cond"] == i)[0]].values
    condition_strains[i] = [genotype_map[i] for i in strains]


cond_pca ={}
cond_embeddings = {}
cond_graphs = {}
cond_clusters = {}

for i in conditions.keys():
    pca = PCA(n_components =50)
    scaler = StandardScaler()
    sc = scaler.fit_transform(conditions[i])
    cond_pca[i] = pca.fit_transform(sc)
    
    um = umap.UMAP()
    cond_embeddings[i] = um.fit_transform(cond_pca[i])

    
    cond_graphs[i] = kneighbors_graph(cond_pca[i], 10, mode='connectivity', include_self=True, n_jobs=8)
    louvain = Louvain()
    cond_clusters[i] = louvain.fit_transform(cond_graphs[i])

for i in conditions.keys():
    plt.scatter(cond_embeddings[i][:,0],cond_embeddings[i][:,1], c = cond_clusters[i])
    plt.show()
    plt.close()
    
    plt.scatter(cond_embeddings[i][:,0],cond_embeddings[i][:,1], s = 4 , c = condition_strains[i])
    plt.show()
    plt.close()


