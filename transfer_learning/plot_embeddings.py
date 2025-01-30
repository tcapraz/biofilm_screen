import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import  image_dataset_from_directory
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.applications import VGG19,NASNetLarge,ResNet50,MobileNetV2
from sklearn.metrics import roc_curve, confusion_matrix, auc, precision_recall_curve, f1_score, roc_auc_score
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
import umap
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from skimage.io import imread
import cv2 
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sknetwork.clustering import Louvain
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

embeddings = pd.read_csv("embedding_bin.csv", index_col=0)
pred = pd.read_csv("y_pred_bin.csv", index_col=0)
assert all(pred.index == embeddings.index)

pca = PCA(n_components=50)
scaler = StandardScaler()
embeddings = scaler.fit_transform(embeddings)

y = np.zeros(pred.shape)
y[pred >= 0.5] = 1
y = y.flatten()

PCs=  pca.fit_transform(embeddings)


scatter =plt.scatter(PCs[:,0], PCs[:,1],s=5, c=y)
plt.legend(*scatter.legend_elements(),
                     title="Classes")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.savefig("PCA_bin.png", dpi=400)
plt.show()
plt.close()

b_names = pred.index[y==1]
b_names = [i.split("/")[1].split(".")[0] for i in b_names]
pd.Series(b_names).to_csv("biofilm_image_names.csv", index=False, header=None)
b_images = []
for i in b_names:
    im = imread("../data/all_data_bg_png/data/"+i+".png")
    result = im.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    mask = im[:,:,0].copy()
    mask[mask>0] = 255
    result[:, :, 3] = mask
    b_images.append(result)

b_idx = np.where(y==1)[0]
fig, ax = plt.subplots()
imscatter(PCs[b_idx,0], PCs[b_idx,1], imageData=b_images , ax=ax, zoom=0.04)
plt.xlabel("TSNE 1")
plt.ylabel("TSNE 2")
ax.set_aspect(1./ax.get_data_ratio())
plt.savefig("pca_bin_embedding_biofilms.png", dpi=500)
plt.show()
plt.close()


nb_names = pred.index[y==0]
nb_names = [i.split("/")[1].split(".")[0] for i in nb_names][1:10000]
nb_images = []
for i in nb_names:
    im = imread("../data/all_data_bg_png/data/"+i+".png")
    result = im.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    mask = im[:,:,0].copy()
    mask[mask>0] = 255
    result[:, :, 3] = mask
    nb_images.append(result)

nb_idx = np.where(y==0)[0][1:10000]
fig, ax = plt.subplots()
imscatter(PCs[nb_idx,0], PCs[nb_idx,1], imageData=nb_images , ax=ax, zoom=0.04)
plt.xlabel("TSNE 1")
plt.ylabel("TSNE 2")
ax.set_aspect(1./ax.get_data_ratio())
plt.savefig("pca_bin_embedding_smooth.png", dpi=500)
plt.show()
plt.close()

subset = np.hstack((b_idx, nb_idx))

subset_data = embeddings[subset,:]
pca = PCA(n_components=50)
pca.fit(subset_data .T)
subset_PCs= pca.components_.T



um = umap.UMAP(n_neighbors=5, random_state=42).fit(subset_PCs)

fig, ax = plt.subplots()
imscatter(um.embedding_[:,0], um.embedding_[:,1], imageData=b_images+ nb_images, ax=ax, zoom=0.04)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP 2")
ax.set_aspect(1./ax.get_data_ratio())
plt.savefig("UMAP_bin_embedding_smooth_biofilm.png", dpi=500)
plt.show()
plt.close()

subset_y = y[subset].astype(object)
subset_y[subset_y == 1] = "smooth"
subset_y[subset_y == 2] = "biofilm"


fig, ax = plt.subplots()
plt.scatter(um.embedding_[1:len(b_idx), 0], um.embedding_[1:len(b_idx), 1], s= 5,alpha=0.2, label="biofilm")
plt.scatter(um.embedding_[len(b_idx): len(b_idx)+ len(nb_idx), 0], um.embedding_[len(b_idx): len(b_idx)+ len(nb_idx), 1],alpha=0.2, s= 5, label="smooth")
plt.legend( title="Classes")
plt.xlabel("UMAP1")
plt.ylabel("UMAP2")
ax.set_aspect(1./ax.get_data_ratio())
plt.savefig("UMAP_bin.png", dpi=500)
plt.show()
plt.close()



tsne = TSNE(n_components=2, n_jobs=8)
tsne_em = tsne.fit_transform(subset_PCs)

fig, ax = plt.subplots()
plt.scatter(tsne_em[1:len(b_idx), 0], tsne_em[1:len(b_idx), 1], s= 5,alpha=0.2, label="biofilm")
plt.scatter(tsne_em[len(b_idx): len(b_idx)+ len(nb_idx), 0], tsne_em[len(b_idx): len(b_idx)+ len(nb_idx), 1], s= 5,alpha=0.2, label="smooth")
plt.legend( title="Classes")
plt.xlabel("TSNE1")
plt.ylabel("TSNE2")
ax.set_aspect(1./ax.get_data_ratio())
plt.savefig("TSNE_bin.png", dpi=500)
plt.show()
plt.close()


fig, ax = plt.subplots()
imscatter(tsne_em[:,0], tsne_em[:,1], imageData=b_images+ nb_images, ax=ax, zoom=0.04)
plt.xlabel("TSNE1")
plt.ylabel("TSNE2")
ax.set_aspect(1./ax.get_data_ratio())
plt.savefig("TSNE_bin_embedding_smooth_biofilm.png", dpi=500)
plt.show()
plt.close()

######################### use embeddings from vae and feature extraction#######
  
fdata = pd.read_csv("../feature_extraction/extracted_features_total.csv", index_col=0)
todrop = ["convex_image", "coords", "filled_image", "image",  "intensity_image", "slice", ]
fdata = fdata.drop(todrop, axis=1)
fdata = fdata.loc[:,fdata.columns[np.var(fdata) !=0]]

data = pd.read_csv("../autoencoders/embeddings/vae_clustering.csv", index_col=0)
#data =  pd.read_csv("embedding_full_pca.csv", index_col=0)
#data.index = [i.split("/")[1].split(".")[0] for i in data.index]

#data.index = [i[2:-3] for i in data.index]

fdata = fdata.loc[data.index,:]
#data = pd.concat((data,fdata), axis=1)
# subset to selected features
feat = pd.read_csv("../autoencoders/selected_feature100.csv", index_col=0)
fdata = fdata.loc[:,feat["x"]]

#data = pd.concat((data,fdata), axis=1)
data =  data.loc[:,~np.isnan(data.astype(float)).any(axis=0)]


meta_data = pd.read_csv("../metadata/meta_data.csv")
meta_data.index = meta_data["filename"]


b_data = data.loc[b_names,:]


pca = PCA(n_components = 50)
scaler = StandardScaler()
b_data_sc = scaler.fit_transform(b_data)
b_data_pca = pca.fit_transform(b_data_sc)


tsne = TSNE(n_components=2, n_jobs=8)
b_data_tsne= tsne.fit_transform(b_data_pca)


b_um = umap.UMAP(n_neighbors=5, random_state=42).fit(b_data_pca)


fig, ax = plt.subplots()
imscatter(b_data_tsne[:,0], b_data_tsne[:,1], imageData=b_images , ax=ax, zoom=0.04)
plt.xlabel("TSNE 1")
plt.ylabel("TSNE 2")
ax.set_aspect(1./ax.get_data_ratio())
plt.savefig("TSNE_VAE_polar_biofilms.png", dpi=500)
plt.show()
plt.close()

fig, ax = plt.subplots()
imscatter(b_um.embedding_[:,0], b_um.embedding_[:,1], imageData=b_images , ax=ax, zoom=0.04)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP2")
ax.set_aspect(1./ax.get_data_ratio())
plt.savefig("UMAP_VAE_polar_biofilms.png", dpi=500)
plt.show()
plt.close()




graph = kneighbors_graph(b_data_pca, 15, mode='connectivity', include_self=True, n_jobs=8)
louvain = Louvain()
l = louvain.fit_transform(graph)


kelly_colors = ['#222222', '#F3C300', '#875692', '#F38400', '#A1CAF1', '#BE0032', '#C2B280', '#848482', '#008856', '#E68FAC', '#0067A5', '#F99379', '#604E97', '#F6A600', '#B3446C', '#DCD300', '#882D17', '#8DB600', '#654522', '#E25822', '#2B3D26']
kelly_cmap = matplotlib.colors.ListedColormap(kelly_colors , name='kelly_cmap')

palette36 =  ["#808080","#556b2f","#228b22","#800000","#483d8b","#008080",
"#b8860b","#000080", "#32cd32", "#7f007f","#8fbc8f","#b03060", "#d2b48c",
"#ff0000", "#00ced1", "#ff8c00", "#ffff00",  "#00ff00",  "#8a2be2",  "#dc143c",
"#00bfff", "#0000ff", "#f08080",  "#adff2f", "#da70d6",  "#ff7f50", "#ff00ff",
"#f0e68c", "#6495ed", "#dda0dd", "#add8e6", "#ff1493",  "#7b68ee", "#98fb98",
"#7fffd4", "#ffc0cb"
]
cmap36 = matplotlib.colors.ListedColormap(palette36, name='36_cmap')


fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].scatter(b_data_tsne[:,0], b_data_tsne[:,1], s=5,alpha=0.5, c  = l, cmap=kelly_cmap)
ax[0].set_xlabel("UMAP 1")
ax[0].set_ylabel("UMAP 2")
ax[0].set(aspect=1)
#ax[0].set_aspect(1./ax[0].get_data_ratio())
imscatter(b_data_tsne[:,0], b_data_tsne[:,1], imageData=b_images , ax=ax[1], zoom=0.04)
ax[1].set_xlabel("UMAP 1")
ax[1].set_ylabel("UMAP 2")
#ax[1].set_aspect(1./ax[1].get_data_ratio())
ax[1].set(aspect=1)
plt.tight_layout()
plt.savefig("TSNE_biofilm_clustering2048.png", dpi=500)
plt.show()
plt.close()


fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].scatter(b_um.embedding_[:,0], b_um.embedding_[:,1], s=5,alpha=0.5, c  = l, cmap=kelly_cmap)
ax[0].set_xlabel("UMAP 1")
ax[0].set_ylabel("UMAP 2")
ax[0].set(aspect=1)
#ax[0].set_aspect(1./ax[0].get_data_ratio())
imscatter(b_um.embedding_[:,0], b_um.embedding_[:,1], imageData=b_images , ax=ax[1], zoom=0.04)
ax[1].set_xlabel("UMAP 1")
ax[1].set_ylabel("UMAP 2")
#ax[1].set_aspect(1./ax[1].get_data_ratio())
ax[1].set(aspect=1)
plt.tight_layout()
plt.savefig("UMAP_biofilm_clustering2048.png", dpi=500)
plt.show()
plt.close()

###################### pilot integration with condition C22 ###################


meta_data = meta_data.loc[b_data.index,:]

day5 = meta_data[meta_data["day"] =="Day5"]
data_day5 = data.loc[day5.index,:]

conditions  = [np.where(day5["cond"] == i)[0] for i  in np.unique(day5["cond"])]

condition_data = []
condition_names = []
strains = []
nt_nums = []
for i in conditions:
    cd = data_day5.iloc[i,:]
    cd.index = day5["genotype"][i]
    strains.append(day5["genotype"][i])
    nt_nums.append(day5["nt_num"][i])
    condition_data.append(cd)
    condition_names.append(data_day5.index[i])

cond = np.where(np.unique(day5["cond"]) == "C22")[0][0]
C22_data = condition_data[cond]
nt = nt_nums[cond]

cluster_images = []
for m in condition_names[cond]:
   # cluster_images.append(imread("../data/all_data/data/" + m+".jpg"))
    im = imread("../data/all_data_bg_png/data/"+m+".png")
    result = im.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    mask = im[:,:,0].copy()
    mask[mask>0] = 255
    result[:, :, 3] = mask
    cluster_images.append(result)



disruption = pd.read_csv("../phylo_data/disruption_scores.tsv", sep="\t", index_col=0)
disruption = disruption.fillna(0)

biofilm_genes = pd.read_csv("../phylo_data/biofilm_genes_uniprot.csv", sep="\t")

inter = np.intersect1d(biofilm_genes["Entry"], disruption.index)

inter_strains = np.intersect1d(nt, disruption.columns)
biofilm_disrupt = disruption.loc[inter,inter_strains]

km = KMeans(3)
l = km.fit_predict(biofilm_disrupt.T)

pca = PCA(n_components=2)
PC = pca.fit_transform(biofilm_disrupt.T)

plt.scatter(PC[:,0], PC[:,1], c =l)
cormat = np.corrcoef(biofilm_disrupt.T)
cormat[np.isnan(cormat)] = 0


import seaborn as sns
g = sns.clustermap(cormat)
new_row = np.array(g.dendrogram_row.reordered_ind)

cormat = cormat[new_row,:]
cormat = cormat[:,new_row]

plt.matshow(cormat)
aggclust = AgglomerativeClustering(9)
l = aggclust.fit_predict(cormat)


C22_data.index = nt

cluster_images = np.array(cluster_images)[~C22_data.index.duplicated(keep='first')]
C22_data =C22_data[~C22_data.index.duplicated(keep='first')]
indices, = np.in1d(inter_strains, C22_data.index).nonzero()
cluster_images =cluster_images[indices]
C22_data = C22_data.loc[inter_strains,:]

pca = PCA(n_components = 50)
scaler = StandardScaler()
C22_data_sc = scaler.fit_transform(C22_data)
C22_data_pca = pca.fit_transform(C22_data_sc)


tsne = TSNE(n_components=2, n_jobs=8)
C22_data_tsne= tsne.fit_transform(C22_data_pca)


C22_um = umap.UMAP(n_neighbors=5, random_state=42).fit(C22_data_pca)


graph = kneighbors_graph(C22_data_pca, 5, mode='connectivity', include_self=True, n_jobs=8)
louvain = Louvain()
l = louvain.fit_transform(graph)



fig, ax = plt.subplots()
imscatter(C22_data_pca[:,0], C22_data_pca[:,1], imageData=cluster_images  , ax=ax, zoom=0.1)
plt.xlabel("PC 1")
plt.ylabel("PC 2")
ax.set(aspect=1)
plt.savefig("PCA_biofilm_C22_vae2048.png", dpi=500)
plt.show()
plt.close()

fig, ax = plt.subplots()
imscatter(C22_data_tsne[:,0], C22_data_tsne[:,1], imageData=cluster_images  , ax=ax, zoom=0.1)
plt.xlabel("TSNE 1")
plt.ylabel("TSNE 2")
ax.set(aspect=1)
plt.savefig("TSNE_biofilm_C22_vae2048.png", dpi=500)
plt.show()
plt.close()

fig, ax = plt.subplots()
imscatter(C22_um.embedding_[:,0],C22_um.embedding_[:,1], imageData=cluster_images  , ax=ax, zoom=0.1)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP2")
ax.set(aspect=1)
plt.savefig("UMAP_biofilm_C22_vae2048.png", dpi=500)
plt.show()
plt.close()



fig, ax = plt.subplots(nrows=1, ncols=2)
ax[0].scatter(C22_um.embedding_[:,0],C22_um.embedding_[:,1], s=10,alpha=1, c  = l, cmap=kelly_cmap)
ax[0].set_xlabel("UMAP 1")
ax[0].set_ylabel("UMAP 2")
ax[0].set(aspect=1)
imscatter(C22_um.embedding_[:,0],C22_um.embedding_[:,1], imageData=np.array(cluster_images) , ax=ax[1], zoom=0.1)
ax[1].set_xlabel("UMAP 1")
ax[1].set_ylabel("UMAP 2")
#ax[1].set_aspect(1./ax[1].get_data_ratio())
ax[1].set(aspect=1)
#plt.tight_layout()
plt.savefig("UMAP_biofilm_C22_clustering2048.png", dpi=500)
plt.show()
plt.close()


