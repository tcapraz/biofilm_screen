from skimage.io import imread
import mahotas.features as mht
from skimage.feature import graycomatrix, graycoprops
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
import skimage.draw as draw
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split


filepaths = []
for root, dirs, f in os.walk("../data/test/"):
    for name in f:
        if name.endswith((".jpg")):
            filepaths.append(root +"/"+ name)
            
images = []
for i in filepaths:
      images.append(imread(i))


colony_px =[]
background_px = []


def remove_isolated_pixels( image):
    image  = image.astype(np.uint8)
    connectivity = 8

    output = cv2.connectedComponentsWithStats(image, 8, cv2.CV_32S)

    num_stats = output[0]
    labels = output[1]
    stats = output[2]

    new_image = image.copy()

    for l in range(num_stats):
        if stats[l,cv2.CC_STAT_AREA] <= 5000:
            new_image[labels == l] = 0
    plt.imshow(new_image)
    return new_image

for i in images:
    plt.imshow(i)
    plt.show()
    plt.close()
    gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
        
    edge = cv2.Canny(gray,50,50)

    circles = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT,3,120, minRadius = 30, maxRadius=120)
    cx = int(circles[0,0,1]) 
    cy = int(circles[0,0,0]) 
    r= int(circles[0,0,2]) -10
    indices = draw.disk((cy,cx),r)
    mask = np.zeros((160,160))
    mask[indices] = 1
    mask[0:10,:] = 2
    mask[150:160,:] = 2
    mask[:,0:10] = 2
    mask[:,150:160] = 2
    
    plt.imshow(mask)
    plt.show()
    plt.close()
    
    mask = mask.flatten()
    unclass_idx = np.where(mask ==0 )[0]
    X_idx  = np.where(mask !=0 )[0]
    data = np.reshape(i, (25600,3))
    X = data[X_idx,:]
    y = mask[X_idx]
    unclass = data[unclass_idx,:]
    rf =    RandomForestClassifier()
    X_train, X_test, y_train, y_test = train_test_split(X,y, test_size=0.2)
    rf.fit(X_train,y_train)
    rf.score(X_test,y_test)
    rf =    RandomForestClassifier()
    rf.fit(X,y)
    unclass_y  = rf.predict(unclass)
    mask[unclass_idx] = unclass_y 
    
    mask = np.reshape(mask, (160,160))
    mask[mask==2] = 0
    plt.imshow(mask)
    plt.show()
    plt.close()
    
    
    dilated = cv2.dilate(mask, None, iterations=1)
    plt.imshow(dilated)
    plt.show()
    plt.close()
    mask = remove_isolated_pixels(dilated)
    plt.imshow(mask)
    plt.show()
    plt.close()


    mask[mask==1] = 255
    i = cv2.cvtColor(i, cv2.COLOR_BGR2BGRA)
    i[:, :, 3] = mask
    plt.imshow(i)
    plt.show()
    plt.close()