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
    
#######################INPUT PARAMS############################
PATH = "../autoencoders/embeddings/multiview_vae.csv"



data = pd.read_csv(PATH, index_col=0)
data =  data.loc[:,~np.isnan(data.astype(float)).any(axis=0)]

biofilm_samples = pd.read_csv("../autoencoders/biofilm_image_names.csv",  header=None).values.flatten()


meta_data = pd.read_csv("../metadata/meta_data.csv")
meta_data.index = meta_data["filename"]


subsample_idx = np.random.choice(range(len(data)), size=5000, replace=False)
data_sub = data.iloc[subsample_idx, :]

images_sub = []
for i in data_sub.index:
    im = imread("../data/all_data_bg_png/data/"+i+".png")
    result = im.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    mask = im[:,:,0].copy()
    mask[mask>0] = 255
    result[:, :, 3] = mask
    images_sub.append(result)
    

pca = PCA(n_components = 25)
scaler = StandardScaler()
data_sub_sc = scaler.fit_transform(data_sub)
data_sub_pca = pca.fit_transform(data_sub_sc)


tsne = TSNE(n_components=2, n_jobs=8)
data_sub_tsne= tsne.fit_transform(data_sub_pca)


data_sub_um = umap.UMAP(n_neighbors=5, random_state=42).fit(data_sub_pca)



fig, ax = plt.subplots()
imscatter(data_sub_tsne[:,0], data_sub_tsne[:,1], imageData=images_sub , ax=ax, zoom=0.04)
plt.xlabel("TSNE 1")
plt.ylabel("TSNE 2")
ax.set_aspect(1./ax.get_data_ratio())
plt.savefig("TSNE.png", dpi=500)
plt.show()
plt.close()

fig, ax = plt.subplots()
imscatter(data_sub_um.embedding_[:,0], data_sub_um.embedding_[:,1], imageData=images_sub, ax=ax, zoom=0.04)
plt.xlabel("UMAP 1")
plt.ylabel("UMAP2")
ax.set_aspect(1./ax.get_data_ratio())
plt.savefig("UMAP.png", dpi=500)
plt.show()
plt.close()
