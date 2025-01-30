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
#data = data.iloc[0:10000,:]
meta_data = meta_data.loc[data.index,:]


images = []
for i in meta_data["filename"]:
    im = imread("../data/all_data_bg/data/"+i+".jpg")
    result = im.copy()
    result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
    mask = im[:,:,0].copy()
    mask[mask>20] = 255
    result[:, :, 3] = mask
    images.append(result)
    
cols = data.columns[0:27]

for ix,i in enumerate(cols):
    idx_low = data.iloc[:,ix].argsort()[0:10]
    idx_high = data.iloc[:,ix].argsort()[-10:]
    d = data.iloc[np.hstack([idx_low,idx_high]),ix]
    
    indicator = np.zeros(20)
    indicator[10:] = 1
    plotdf = pd.DataFrame(np.vstack( (d, indicator)).T ,index=d.index)
    imgs = []
    for i in plotdf.index:
        imgs.append(imread("../data/all_data/data/"+i+".jpg"))
    
    fig, ax = plt.subplots()
    imscatter(plotdf.iloc[:, 0], plotdf.iloc[:, 1], imageData=imgs, ax=ax, zoom=0.3)
    plt.show()
    plt.close()

images = []
for i in meta_data["filename"]:
    images.append(rgb2gray(imread("../data/all_data_bg/data/"+i+".jpg")))

var = np.array([np.var(i[:,80],) for i in images])

idx_low = var.argsort()[0:10]
idx_high = var.argsort()[-10:]
    
for i in idx_high:
    plt.imshow(images[i])
    plt.show()
    plt.close()
