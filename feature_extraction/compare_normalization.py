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
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from PIL import Image, ImageStat
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import operator


mask = np.zeros((160,160))
mask[0:10,:] = 1
mask[150:160,:] = 1
mask[:,0:10] = 1
mask[:,150:160] = 1
mask = mask.astype(bool)
def brightness_mean( im_file ):
   im = Image.open(im_file).convert('L')
   stat = ImageStat.Stat(im)
   return stat.mean[0]

def brightness_mean_cv(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return np.mean(img[mask])


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
for root, dirs, f in os.walk("data/test/"):
    for name in f:
        if name.endswith((".jpg")):
            filepaths.append(root +"/"+ name)
            
            

# anno = pd.read_csv("data/annotationsM1-M10.txt", delim_whitespace=True, header=None)
# anno = anno.drop_duplicates()
# anno.index = anno[1]
names = [Path(i).stem for i in filepaths]
# anno = anno.loc[names,:]


masks = []

# looking at the first image
i = 0
images = []
images_norm = []
images_hist_equal = []
images_clahe = []
images_scaled = []

for i in range(len(filepaths)):
    image_path = filepaths[i]
    
    img_path = image_path                                                                                                                                                                             
    img = cv2.imread(img_path) 
    gimg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    n_img = np.zeros((160,160,3))
    norm_img = cv2.normalize(img,  n_img, 0, 255, cv2.NORM_MINMAX)
    norm_img = cv2.cvtColor(norm_img, cv2.COLOR_BGR2GRAY)
    img_scaled = (img-np.mean(img)) / np.std(img) * 32 + 128
    img_scaled = np.array(img_scaled, dtype=np.uint8)
    img_scaled = cv2.cvtColor(img_scaled, cv2.COLOR_BGR2GRAY)

    img_hist= cv2.equalizeHist(gimg)
    
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
    img_clahe = clahe.apply(gimg)
    images.append(gimg)
    images_norm.append(norm_img)
    images_hist_equal.append(img_hist)
    images_clahe.append(img_clahe)
    images_scaled.append(img_scaled)
        
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    mask = np.zeros(img.shape[:2],np.uint8)
    rect = (1,1,160,160)
    m = cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)[0]
    m[m==2] =0
    m[m==1] =0
    m[m!=0] =1

    label_img = label(m)
    label_img[label_img!=0] =1
    masks.append(label_img)



    
supported =[]
unsupported = []
props = regionprops(masks[0], images[0])
for prop in props[0]:
    try:
        supported.append(prop)
    except NotImplementedError:
        unsupported.append(prop)

exclude = [18, 24,25,26,27, 151]

features = []
features_norm = []
features_hist_equal = []
features_clahe = []
features_scaled = []

for l,i in zip(masks,images):
    features.append( pd.DataFrame(regionprops_table(l, i, properties= supported)))
for l,i in zip(masks,images_norm):
    features_norm.append( pd.DataFrame(regionprops_table(l, i, properties= supported)))
for l,i in zip(masks,images_hist_equal):
    features_hist_equal.append( pd.DataFrame(regionprops_table(l, i, properties= supported)))
for l,i in zip(masks,images_clahe):
    features_clahe.append( pd.DataFrame(regionprops_table(l, i, properties= supported)))
for l,i in zip(masks,images_scaled):
    features_scaled.append( pd.DataFrame(regionprops_table(l, i, properties= supported)))

feature_names = features[0].columns

features = np.vstack(features)
features_norm = np.vstack(features_norm)
features_clahe = np.vstack(features_clahe)
features_scaled = np.vstack(features_scaled)
features_hist_equal = np.vstack(features_hist_equal)


globprops = ['contrast', 'dissimilarity',
          'homogeneity',
          'ASM', 'energy', 'correlation']




feature_names2 =["haralick"+str(i) for i in range(1,53)] + ["zernike" +str(i) for i in range(1,26)]

feature_names2 = feature_names2 + [j + str(i) for j in globprops for i in range(1,13)]

def calc_glob_feat(X):
    feat  =[]
    for i in range(len(X)):
        f = mht.haralick(img_as_ubyte(X[i])).ravel()
        fz = mht.zernike_moments(img_as_ubyte(X[i]), radius=50)
        #fl = mht.lbp(img_as_ubyte(X[i,:,:]), radius=50, points=10)
        glcm = greycomatrix(img_as_ubyte(X[i]), distances=[3, 5, 7],angles=[0, np.pi/4, np.pi/2, 3*np.pi/4])
        
        allprops=[]
        for j in globprops:
            allprops.append(greycoprops(glcm,j).ravel())
        allprops= np.hstack(allprops)
        feat.append(np.hstack([f, fz, allprops]))
    return np.vstack(feat)


features2 = calc_glob_feat(images)
features2_norm = calc_glob_feat(images_norm)
features2_scaled = calc_glob_feat(images_scaled)
features2_clahe = calc_glob_feat(images_clahe)
features2_hist_equal = calc_glob_feat(images_hist_equal)


data = np.hstack([features, features2])
data_norm = np.hstack([features_norm, features2_norm])
data_scaled = np.hstack([features_scaled, features2_scaled])
data_clahe = np.hstack([features_clahe, features2_clahe])
data_hist_equal = np.hstack([features_hist_equal, features2_hist_equal])

def process_df(data):
    
    df = pd.DataFrame(data)
    colnames = feature_names.tolist() + feature_names2
    df.columns = colnames
    df.index = names
    
    data  = df
    todrop = ["convex_image", "coords", "filled_image", "image",  "intensity_image", "slice", ]
    data = data.drop(todrop, axis=1)
    data =  data.loc[:,~np.isnan(data.astype(float)).any(axis=0)]
    return data

data = process_df(data)
data_norm = process_df(data_norm)
data_scaled = process_df(data_scaled)
data_clahe =process_df(data_clahe)
data_hist_equal = process_df(data_hist_equal)
    


d = [data,data_norm,data_scaled,data_clahe, data_hist_equal]


images = []
for i in filepaths:
                                                                                                                                                                      
    img = imread(i) 
    images.append(img)
b_mean = []

    
for i in images:
    b_mean.append(brightness_mean_cv(i))

for i in d:
    pca = PCA(n_components = 50)
    scaler = StandardScaler()
    i = scaler.fit_transform(i)
    x = pca.fit_transform(i)
    
    tsne = TSNE(n_components=2, n_jobs=8)
    x = tsne.fit_transform(x)



    plt.scatter(x[:,0], x[:,1],c = b_mean, cmap=cm.Blues)
    plt.show()
    plt.close()
    fig, ax = plt.subplots()
    imscatter(x[:, 0], x[:, 1], imageData=images, ax=ax, zoom=0.09)
    plt.show()
    plt.close()


# for i in images:
#     plt.imshow(i)
#     plt.show()
#     plt.close()
