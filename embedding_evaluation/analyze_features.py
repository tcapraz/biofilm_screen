from skimage.io import imread

from skimage.util import img_as_ubyte
from skimage.color import rgb2gray
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
from scanpy.pp import neighbors
import anndata as ad
import scipy
import scipy.stats as stats
from matplotlib import cm
import matplotlib


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



data = pd.read_csv("../feature_extraction/extracted_features_total.csv", index_col=0)
todrop = ["convex_image", "coords", "filled_image", "image",  "intensity_image", "slice"]
data = data.drop(todrop, axis=1)
data =  data.loc[:,~np.isnan(data.astype(float)).any(axis=0)]

data_polar = pd.read_csv("../feature_extraction/extracted_features_polar.csv", index_col=0)
todrop = ["convex_image",  "filled_image",  "intensity_image"]
data_polar = data_polar.drop(todrop, axis=1)
data_polar =  data_polar.loc[:,~np.isnan(data_polar.astype(float)).any(axis=0)]

vae = pd.read_csv("vae_clahe.csv", index_col=0)
vae = vae.loc[data.index,:]

subsample_idx = np.random.choice(range(len(data)), size=5000, replace=False)
data = data.iloc[subsample_idx, :]
data_polar = data_polar.iloc[subsample_idx, :]
vae = vae.iloc[subsample_idx, :]
data  = pd.concat([data,data_polar,vae], axis=1)
data = vae
images = []
for i in data.index:
    im = imread("../data/all_data_bg_png/data/"+i+".png")
    result = im.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    mask = im[:,:,0].copy()
    mask[mask>0] = 255
    result[:, :, 3] = mask
    images.append(result)
images = np.array(images)


pca = PCA(n_components = 50)
scaler = StandardScaler()
data_sc = scaler.fit_transform(data)
data_pca = pca.fit_transform(data_sc)


loadings = pca.components_
idx1 = (-abs(loadings[0,:])).argsort()[:30]
data.columns[idx1]
idx2 = (-abs(loadings[1,:])).argsort()[:20]
data.columns[idx2]

# plt.scatter(data_pca[:,0], data_pca[:,1], s = 0.4, c=data["weighted_moments-0-3-0"])


# plt.scatter(data.loc[:,"area"], data.loc[:,"dissimilarity5"], s = 0.4, c=data["contrast5"])

# fig, ax = plt.subplots()
# imscatter(data.loc[:,"area"], data.loc[:,"correlation6"], imageData=images , ax=ax, zoom=0.04)
# plt.xlabel("area")
# plt.ylabel("correlation6")
# ax.set_aspect(1./ax.get_data_ratio())
# plt.savefig("featuremap.png", dpi=400)
# plt.show()
# plt.close()

fig, ax = plt.subplots()
imscatter(data_pca[:,0], data_pca[:,1], imageData=images , ax=ax, zoom=0.04)
plt.xlabel("PC1")
plt.ylabel("PC2")
ax.set_aspect(1./ax.get_data_ratio())
plt.savefig("PCmap_clahe.png", dpi=400)
plt.show()
plt.close()

# idx = (data["weighted_moments_normalized-0-2-0"]).argsort()[:10]

# for i in idx:
#     plt.imshow(images[i,:,:,:])
#     plt.show()
#     plt.close()

tsne = TSNE(n_components=2, n_jobs=8)
data_tsne= tsne.fit_transform(data_pca)


data_um = umap.UMAP(n_neighbors=5, random_state=42).fit(data_sc)

fig, ax = plt.subplots()
imscatter(data_um.embedding_[:,0], data_um.embedding_[:,1], imageData=images , ax=ax, zoom=0.04)
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
ax.set_aspect(1./ax.get_data_ratio())
plt.savefig("UMAP_clahe.png", dpi=400)
plt.show()
plt.close()

