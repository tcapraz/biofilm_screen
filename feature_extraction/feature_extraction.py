import mahotas.features as mht

# import the necessary packages
import numpy as np
import argparse
import cv2
import matplotlib.pyplot as plt
import os
from PIL import Image
from sklearn.decomposition import PCA
import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import  image_dataset_from_directory
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.applications import VGG19
from sklearn.metrics import roc_curve, confusion_matrix, auc, precision_recall_curve, f1_score, roc_auc_score
from skimage.util import img_as_ubyte
from skimage.feature import graycomatrix, graycoprops, daisy,multiscale_basic_features

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

datagen = ImageDataGenerator(rescale=1/255)

data_generator = datagen.flow_from_directory('classification/data/', classes=['M1', "M2", "M3", "M4", "M5"],
                                             target_size=(100,100), color_mode="grayscale", 
                                             class_mode="input", batch_size=1, shuffle=False)
data_list = []
batch_index = 0

while batch_index <= data_generator.batch_index:
    data = data_generator.next()
    data_list.append(data[0].reshape(100,100))
    batch_index = batch_index + 1
    
X = np.asarray(data_list)
y = data_generator.classes


props = ['contrast', 'dissimilarity',
          'homogeneity',
          'ASM', 'energy', 'correlation']
features = []
for i in range(X.shape[0]):
    f = mht.haralick(img_as_ubyte(X[i,:,:])).ravel()
    fz = mht.zernike_moments(img_as_ubyte(X[i,:,:]), radius=50)
    #fl = mht.lbp(img_as_ubyte(X[i,:,:]), radius=50, points=10)
    glcm = graycomatrix(img_as_ubyte(X[0,:,:]), distances=[3, 5, 7],angles=[0, np.pi/4, np.pi/2, 3*np.pi/4])
    
    allprops=[]
    for j in props:
        allprops.append(graycoprops(glcm,j).ravel())
    allprops= np.hstack(allprops)
    features.append(np.hstack([f, fz, allprops]))

data = np.vstack(features)
scale = StandardScaler()
data = scale.fit_transform(data)
data_vanilla = X.reshape(X.shape[0], 10000)

pca = PCA(n_components=50)
pcs = pca.fit_transform(data)
km = KMeans(init = "random", n_clusters=5)
km = km.fit(pcs)
labels_pred= km.predict(pcs)
ari = adjusted_rand_score(y, labels_pred)

scatter =plt.scatter(pcs[:,0], pcs[:,1],s=5, c=y)

X_train, X_test, y_train, y_test = train_test_split(pcs, y, test_size=0.25, random_state=42)

logit = LogisticRegression(max_iter=1000)

logit =logit.fit(X_train,y_train)

accuracy_score(logit.predict(X_test), y_test)

################### experiments###########################
from skimage import transform
from skimage.color import rgb2gray
from skimage.feature import match_descriptors, plot_matches, SIFT

img1 = img_as_ubyte(X[0,:,:])
img2 = transform.rotate(img1, 180)
tform = transform.AffineTransform(scale=(1.3, 1.1), rotation=0.5,
                                  translation=(0, -200))
img3 = transform.warp(img1, tform)

descriptor_extractor = SIFT()

descriptor_extractor.detect_and_extract(img1)
keypoints1 = descriptor_extractor.keypoints
descriptors1 = descriptor_extractor.descriptors

descriptor_extractor.detect_and_extract(img2)
keypoints2 = descriptor_extractor.keypoints
descriptors2 = descriptor_extractor.descriptors


matches12 = match_descriptors(descriptors1, descriptors2, max_ratio=0.6,
                              cross_check=True)

f = daisy(img_as_ubyte(X[0,:,:])).ravel()
f = multiscale_basic_features(img_as_ubyte(X[0,:,:]))
