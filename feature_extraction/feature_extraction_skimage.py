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
for root, dirs, f in os.walk("data/all_data_bg_png/"):
    for name in f:
        if name.endswith(("png")):
            filepaths.append(root +"/"+ name)
            
            

# anno = pd.read_csv("data/annotationsM1-M10.txt", delim_whitespace=True, header=None)
# anno = anno.drop_duplicates()
# anno.index = anno[1]
names = [Path(i).stem for i in filepaths]
# anno = anno.loc[names,:]



# looking at the first image
# i = 0
# for i in range(len(filepaths)):
#     image_path = filepaths[i]
    
#     img_path = image_path                                                                                                                                                                             
#     img = cv2.imread(img_path) 
    
#     bgdModel = np.zeros((1,65),np.float64)
#     fgdModel = np.zeros((1,65),np.float64)
#     mask = np.zeros(img.shape[:2],np.uint8)
#     rect = (1,1,160,160)
#     m = cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)[0]
#     m[m==2] =0
#     m[m==1] =0
#     m[m!=0] =1

#     label_img = label(m)
#     label_img[label_img!=0] =1
#     masks.append(label_img)






images = []
for i in filepaths:
     images.append(rgb2gray(imread(i)))
     


supported =[]
unsupported = []
label_image = skimage.measure.label(images[0] != 0, connectivity=images[0].ndim)

props = regionprops(label_image, images[0])
for prop in props[0]:
    try:
        props[0][prop]
        supported.append(prop)
    except NotImplementedError:
        unsupported.append(prop)

#exclude = [18, 24,25,26,27, 151]

features = []

for i in images:
    label_image = skimage.measure.label(i != 0, connectivity=i.ndim)
    label_image[label_image !=0] = 1
    features.append( pd.DataFrame(regionprops_table(label_image, i, properties= supported)))

feature_names = features[0].columns
features = np.vstack(features)

df1 = pd.DataFrame(features)
colnames = feature_names.tolist() 
df1.columns = colnames
df1.index = names
df1.to_csv("extracted_features_regional.csv")

#features = np.delete(features, exclude, 1)
#features =  features[:,~np.isnan(features.astype(float)).any(axis=0)]


globprops = ['contrast', 'dissimilarity',
          'homogeneity',
          'ASM', 'energy', 'correlation']
features2 = []


feature_names2 =["haralick"+str(i) for i in range(1,53)] + ["zernike" +str(i) for i in range(1,26)]

feature_names2 = feature_names2 + [j + str(i) for j in globprops for i in range(1,13)]

X = images
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
df.index = names
df.to_csv("extracted_features_total.csv")

# pca = PCA(n_components = 50)
# scaler = StandardScaler()
# data = scaler.fit_transform(data)
# x = pca.fit_transform(data)

# tsne = TSNE(n_components=2)
# x = tsne.fit_transform(x)
# label_map = {"M" + str(i):i for i in range(1,11)}

# label = [label_map[i] for i in anno[0]]

# plt.scatter(x[:,0], x[:,1], c = label)

# images = []
# for i,p in enumerate(filepaths):
#     im = imread(p)
#     result = im.copy()
#     result = cv2.cvtColor(result, cv2.COLOR_BGR2BGRA)
#     mask = masks[i]
#     mask[mask==1] = 255
#     result[:, :, 3] = mask
#     images.append(result)
     
# fig, ax = plt.subplots()
# imscatter(x[:, 0], x[:, 1], imageData=images, ax=ax, zoom=0.09)
# plt.xlabel("TSNE 1")
# plt.ylabel("TSNE 2")
# plt.savefig("tsne_extracted_feat.png", dpi=500)
