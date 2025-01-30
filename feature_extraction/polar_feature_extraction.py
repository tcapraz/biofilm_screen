import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from skimage.io import imread
from skimage.transform import warp_polar
import skimage
import cv2 
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
import skimage
from matplotlib.offsetbox import OffsetImage, AnnotationBbox


b_names = pd.read_csv("biofilm_image_names.csv", header=None)
b_images = []
for i in b_names[0]:
    im = imread("../data/all_data/data/"+i+".jpg")
    b_images.append(im)


polar1 = warp_polar(b_images[1], radius=60, channel_axis=-1)
plt.imshow(b_images[1])
plt.imshow(polar1)


polar2 = warp_polar(b_images[2], radius=80, channel_axis=-1)
plt.imshow(b_images[2])
plt.imshow(polar2)


polar_imgs = []
for i in b_images:
    gray = cv2.cvtColor(i, cv2.COLOR_BGR2GRAY)
    # plt.imshow(gray)
    # plt.show()
    # plt.close()
    edge = cv2.Canny(gray,50,50)
    # plt.imshow(edge)
    # plt.show()
    # plt.close()
    circles = cv2.HoughCircles(edge, cv2.HOUGH_GRADIENT,3,120, minRadius = 10, maxRadius=120)
    
    #rr, cc = skimage.draw.circle_perimeter(int(circles[0][0][0]), int(circles[0][0][1]), int(circles[0][0][2])+10)
    # copy = i.copy()
    # copy[rr,cc,:]=1
    # plt.imshow(copy)
    # plt.show()
    # plt.close()
    if circles is not None:
        polar = warp_polar(i, center = (int(circles[0][0][0]), int(circles[0][0][1])), radius=int(circles[0][0][2])+10, channel_axis=-1)
    else:
        polar = warp_polar(i, radius=80, channel_axis=-1)
    # plt.imshow(polar)
    # plt.show()
    # plt.close()
    polar_imgs.append(polar)
    
supported =[]
unsupported = []
label_image = skimage.measure.label(b_images[0] != 0, connectivity=b_images[0].ndim)

props = regionprops(label_image, b_images[0])
for prop in props[0]:
    try:
        props[0][prop]
        supported.append(prop)
    except NotImplementedError:
        unsupported.append(prop)


features = []

for i in polar_imgs:
    label_image =  skimage.measure.label(i != 0, connectivity=i.ndim)
    features.append( pd.DataFrame(regionprops_table(label_image, i, properties= supported)))



feature_names = features[0].columns
features = np.vstack(features)

df1 = pd.DataFrame(features)
colnames = feature_names.tolist() 
df1.columns = colnames
df1.index = b_names[0]



globprops = ['contrast', 'dissimilarity',
          'homogeneity',
          'ASM', 'energy', 'correlation']
features2 = []


feature_names2 =["haralick"+str(i) for i in range(1,53)] + ["zernike" +str(i) for i in range(1,26)]

feature_names2 = feature_names2 + [j + str(i) for j in globprops for i in range(1,13)]



X = []

for i in polar_imgs:
    X.append(rgb2gray(i))
    
for i in range(len(X)):
    f = mht.haralick(img_as_ubyte(X[i])).ravel()
    fz = mht.zernike_moments(img_as_ubyte(X[i]), radius=50)
    #fl = mht.lbp(img_as_ubyte(X[i,:,:]), radius=50, points=10)
    glcm = greycomatrix(img_as_ubyte(X[i]), distances=[3, 5, 7],angles=[0, np.pi/4, np.pi/2, 3*np.pi/4])
    
    allprops=[]
    for j in globprops:
        allprops.append(greycoprops(glcm,j).ravel())
    allprops= np.hstack(allprops)
    features2.append(np.hstack([f, fz, allprops]))

features2 = np.vstack(features2)

data = np.hstack([features, features2])

df = pd.DataFrame(data)
colnames = feature_names.tolist() + feature_names2
df.columns = colnames
df.index = b_names[0]
#

todrop = [ "coords",  "image",   "slice" , "image_convex", "image_filled", "image_intensity"]
df = df.drop(todrop, axis=1)
df = df.loc[:,df.columns[np.var(df) !=0]]

df = df.loc[:,~np.any(df.isna(), axis=0)]

df.to_csv("extracted_features_polar.csv")


###########evaluate 
# b_data = df.values

# pca = PCA(n_components = 10)
# scaler = StandardScaler()
# b_data_sc = scaler.fit_transform(b_data)
# b_data_pca = pca.fit_transform(b_data_sc)


# tsne = TSNE(n_components=2, n_jobs=8)
# b_data_tsne= tsne.fit_transform(b_data_pca)

# def imscatter(x, y, ax, imageData, zoom):
#     imags = []
#     for i in range(len(x)):
#         x0, y0 = x[i], y[i]
#         # Convert to image
#         #img = imageData[i]*255.
#         #img = img.astype(np.uint8).reshape([imageSize,imageSize])
#         #img = cv2.cvtColor(img,cv2.COLOR_GRAY2RGB)
#         # Note: OpenCV uses BGR and plt uses RGB
#         #image = OffsetImage(img, zoom=zoom)
#         imag = OffsetImage(imageData[i], zoom=zoom)
#         ab = AnnotationBbox(imag, (x0, y0), xycoords='data', frameon=False)
#         imags.append(ax.add_artist(ab))
    
#     ax.update_datalim(np.column_stack([x, y]))
#     ax.autoscale()

# imgs = []
# for i in b_names[0][0:40]:
#     im = imread("../data/all_data_bg_png/data/"+i+".png")
#     result = im.copy()
#     result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
#     mask = im[:,:,0].copy()
#     mask[mask>0] = 255
#     result[:, :, 3] = mask
#     imgs.append(result)


# fig, ax = plt.subplots()
# imscatter(b_data_pca[:,0], b_data_pca[:,1], imageData=imgs , ax=ax, zoom=0.2)
# plt.xlabel("TSNE 1")
# plt.ylabel("TSNE 2")
# ax.set_aspect(1./ax.get_data_ratio())
# #plt.savefig("pca_bin_embedding_smooth.png", dpi=500)
# plt.show()
# plt.close()

