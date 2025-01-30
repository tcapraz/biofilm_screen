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

def brightness_mean_cv(img):
    mask = np.zeros((160,160))
    mask[0:10,:] = 1
    mask[150:160,:] = 1
    mask[:,0:10] = 1
    mask[:,150:160] = 1
    mask = mask.astype(bool)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(gray[mask])
 


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

data = pd.read_csv("extracted_features_total.csv", index_col=0)
todrop = ["convex_image", "coords", "filled_image", "image",  "intensity_image", "slice", ]
data = data.drop(todrop, axis=1)

meta_data = pd.read_csv("../metadata/meta_data.csv")
meta_data.index = meta_data["filename"]


data =  data.loc[:,~np.isnan(data.astype(float)).any(axis=0)]
day4 = meta_data[meta_data["day"] =="Day4"]
data_day4 = data.loc[day4.index,:]



testdata = data.loc[names,:]
data = data.iloc[0:10000,:]
meta_data = meta_data.loc[data.index,:]


#######################################################################

pca = PCA(n_components = 50)
scaler = StandardScaler()
testdata = scaler.fit_transform(testdata)
x_test = pca.fit_transform(testdata)

tsne = TSNE(n_components=2, n_jobs=8)
x_test  = tsne.fit_transform(x_test )
label_map = {"M" + str(i):i for i in range(1,11)}

label = [label_map[i] for i in anno[0]]

plt.scatter(x_test [:,0], x_test [:,1],  c = label)
plt.xlabel("TSNE 1")
plt.ylabel("TSNE 2")
plt.savefig("tsne_embedding_test_labeled.png", dpi=500)


testimages = []
for i in testfilepaths:
    testimages.append(imread(i))

fig, ax = plt.subplots()
imscatter(x_test [:, 0], x_test [:, 1], imageData=testimages, ax=ax, zoom=0.09)
plt.xlabel("TSNE 1")
plt.ylabel("TSNE 2")
#plt.savefig("tsne_extracted_feat_test.png", dpi=500)

#########################################################################
    
b_mean = []

images = []
for i in meta_data["filename"]:
    images.append(imread("../data/all_data_bg/data/"+i+".jpg"))

images = []
for i in meta_data["filename"]:
    im = imread("../data/all_data_bg/data/"+i+".jpg")
    result = im.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    mask = im[:,:,0].copy()
    mask[mask>20] = 255
    result[:, :, 3] = mask
    images.append(result)


for i in images:
    b_mean.append(brightness_mean_cv(i))



pca = PCA(n_components =50)
scaler = StandardScaler()
data_sc = scaler.fit_transform(data)
x = pca.fit_transform(data_sc)

tsne = TSNE(n_components=2, n_jobs=8)
x_tsne = tsne.fit_transform(x)
#np.savetxt("tsne_embedding_full.csv",x_tsne)
x_tsne = np.loadtxt("tsne_embedding_full.csv")

plt.scatter(x_tsne [:, 0], x_tsne [:, 1], s = 0.1)
plt.xlabel("TSNE 1")
plt.ylabel("TSNE 2")
#plt.savefig("tsne_extracted_feat_full.png", dpi=500)


fig, ax = plt.subplots()
imscatter(x_tsne [0:10000, 0], x_tsne [0:10000, 1], imageData=images, ax=ax, zoom=0.04)
plt.xlabel("TSNE 1")
plt.ylabel("TSNE 2")
plt.savefig("tsne_extracted_feat_images.png", dpi=500)


label = np.zeros(meta_data.shape[0])
label[meta_data["day"].values == "Day5"] = 1
#plt.scatter(x_tsne[:,0], x_tsne[:,1], s=0.1, c  = label)


plt.scatter(x_tsne[:,0], x_tsne[:,1], s=0.1, c  = meta_data["cond"])
plt.xlabel("TSNE 1")
plt.ylabel("TSNE 2")
#plt.savefig("condition.png", dpi=500)

genotype_map  = {g:i for i,g in enumerate(meta_data["genotype"].unique())}
geno_label = [genotype_map[i] for i in meta_data["genotype"]]

plt.scatter(x_tsne[:,0], x_tsne[:,1], s=0.1, c = geno_label)

plt.scatter(x_tsne[:,0], x_tsne[:,1], s=0.1, c = b_mean, cmap="binary")
plt.xlabel("TSNE 1")
plt.ylabel("TSNE 2")
#plt.savefig("brightness.png", dpi=500)


day4["repid"] = day4["genotype"] + day4["cond"]

rep_map = {r:i for i,r in enumerate(np.unique(day4["repid"]))}
rep_label = [rep_map[i] for i in day4["repid"]]



single_class = np.unique(rep_label)[np.unique(rep_label, return_counts=True)[1] >1]
single_class_idx = []

for i in single_class:
    single_class_idx.append(np.where(rep_label == i)[0])

rep_label = np.array(rep_label)[np.hstack(single_class_idx)]

rep_data = data_day4.iloc[np.hstack(single_class_idx),:]

rep_x = x[np.hstack(single_class_idx),:]

rep1_indices = []
rep2_indices = []

for i in np.unique(rep_label):
    rep1_indices.append(np.where(rep_label == i)[0][0])
    rep2_indices.append(np.where(rep_label == i)[0][1])

rep1_data = rep_data.iloc[rep1_indices,:]
rep2_data = rep_data.iloc[rep2_indices,:]
rep1_x = rep_x[rep1_indices,:]
rep2_x = rep_x[rep2_indices,:]

cor = np.corrcoef(rep1_data.values.flatten(), rep2_data.values.flatten())



plt.scatter(rep1_data, rep2_data, s =0.2)
plt.plot([0,2e16], [0,2e16], c = "black", linestyle="--")
plt.xlabel("Replicate 1")
plt.ylabel("Replicate 2")
plt.text(1.6e16, 2e16, "r = 0.5")
plt.tight_layout()
plt.savefig("replicate_agreement.png", dpi=400)
plt.show()
plt.close()





graph = kneighbors_graph(x, 10, mode='connectivity', include_self=True, n_jobs=8)
louvain = Louvain()
l = louvain.fit_transform(graph)

kelly_colors = ['#F2F3F4', '#222222', '#F3C300', '#875692', '#F38400', '#A1CAF1', '#BE0032', '#C2B280', '#848482', '#008856', '#E68FAC', '#0067A5', '#F99379', '#604E97', '#F6A600', '#B3446C', '#DCD300', '#882D17', '#8DB600', '#654522', '#E25822', '#2B3D26']
kelly_cmap = matplotlib.colors.ListedColormap(kelly_colors , name='kelly_cmap')

palette36 =  ["#808080","#556b2f","#228b22","#800000","#483d8b","#008080",
"#b8860b","#000080", "#32cd32", "#7f007f","#8fbc8f","#b03060", "#d2b48c",
"#ff0000", "#00ced1", "#ff8c00", "#ffff00",  "#00ff00",  "#8a2be2",  "#dc143c",
"#00bfff", "#0000ff", "#f08080",  "#adff2f", "#da70d6",  "#ff7f50", "#ff00ff",
"#f0e68c", "#6495ed", "#dda0dd", "#add8e6", "#ff1493",  "#7b68ee", "#98fb98",
"#7fffd4", "#ffc0cb"
]
cmap36 = matplotlib.colors.ListedColormap(palette36, name='36_cmap')

plt.scatter(x_tsne[:,0], x_tsne[:,1], s=0.1, c  =l, cmap=kelly_cmap)
plt.xlabel("TSNE 1")
plt.ylabel("TSNE 2")
plt.savefig("allclusters.png", dpi=500)
plt.show()
plt.close()




# km = KMeans(50, n_jobs=8)

# l = km.fit_predict(x_tsne)
# np.unique(l, return_counts=True)
# plt.scatter(x_tsne[:,0], x_tsne[:,1], s=0.1, c = l)

# images = images[0:50000]




# for i in np.unique(l):
#     cl = i
#     cluster_names = data.index[l == cl]
#     cluster_images = []
#     for i in cluster_names:
#         cluster_images.append(imread("../data/all_data/" + i+".jpg"))
    
    
#     # tsne = TSNE(n_components=2, n_jobs=8)
#     # x_tsne_cluster = tsne.fit_transform(x[l==cl,:])
#     labs = np.array(l)
#     labs[labs != cl] = 100
#     labs[labs == cl] = 200
#     plt.scatter(x_tsne[:,0],x_tsne[:,1], s=0.1 ,c = labs)
#     plt.show()
#     plt.close()
#     fig, ax = plt.subplots()
#     imscatter(x_tsne[l == cl,0], x_tsne[l == cl,1], imageData=cluster_images, ax=ax, zoom=0.09)
#     plt.show()
#     plt.close()

single_class = np.unique(rep_label)[np.unique(rep_label, return_counts=True)[1] >1]
single_class_idx = []

for i in single_class:
    single_class_idx.append(np.where(rep_label == i)[0])

rep_label = np.array(rep_label)[np.hstack(single_class_idx)]
data_day4 = data_day4.iloc[np.hstack(single_class_idx),:]

cors = []
for i in np.unique(rep_label):
    cormat = np.corrcoef(data_day4.values[np.where(rep_label == i)[0],:])
    cors.append(np.mean(cormat[~np.eye(cormat.shape[0],dtype=bool)]))

plt.matshow(data_day4.iloc[0:1000,:].T.corr())
plt.show()

pca = PCA(n_components =50)
scaler = StandardScaler()
data_day4_sc = scaler.fit_transform(data_day4)
x_day4 = pca.fit_transform(data_day4_sc)

tsne = TSNE(n_components=2, n_jobs=8)
x_tsne_day4 = tsne.fit_transform(x_day4)

um = umap.UMAP()
um  = um.fit(x)
x_um = um.fit_transform(x)

plt.scatter(x_um[:,0], x_um[:,1], s=0.1, c = geno_label)

plt.scatter(x_um[:,0], x_um[:,1], s=0.1, c  = meta_data["cond"])


# nbrs = NearestNeighbors(n_neighbors=100, algorithm='ball_tree', n_jobs=8).fit(data)
# distances, indices = nbrs.kneighbors(x_day4)    


# rep_occurrence = {i:np.sum(rep_label==i) for i in rep_label}


# frac = []

# for i in range(indices.shape[0]):
#     nn = indices[i,:]
#     reps = np.array(rep_label)[nn]
#     sample = reps[0]
#     other = reps[1:]
    
#     frac.append(np.sum(other==sample)/rep_occurrence[sample])
    

# np.mean(frac)
