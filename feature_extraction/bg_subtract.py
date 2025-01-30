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
import skimage.draw as draw
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import skimage.draw as draw
from PIL import Image

filepaths = []
for root, dirs, f in os.walk("../data/test/"):
    for name in f:
        if name.endswith((".jpg")):
            filepaths.append(root +"/"+ name)
            
images = []
for i in filepaths:
      images.append(imread(i))

#outdir = "bg_subtract"

# mask = np.zeros((160,160))
# mask[0:10,:] = 1
# mask[150:160,:] = 1
# mask[:,0:10] = 1
# mask[:,150:160] = 1

# background_px = images[0][mask==1]

# background_img = []
# for i in range(3):
#     background_img.append(np.random.choice(background_px[:,i], (160,160)))
# background_img =np.stack(background_img, axis=2)
# plt.imshow(background_img)
# plt.imshow(images[0])
# plt.imshow(images[0]-background_img)
# plt.imshow(cv2.absdiff(images[0], background_img))


# segmentor = SelfiSegmentation()
# img_Out = segmentor.removeBG(images[0], (255,255,255), threshold=0.4)
# plt.imshow(img_Out)

names = [Path(i).stem for i in filepaths]

for i,name in zip(images, names):
    img = i.copy()
    img =  cv2.convertScaleAbs(img, alpha=1.2, beta=0)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)
    mask = np.zeros(img.shape[:2],np.uint8)
    indices = draw.disk((80,80),75)
    mask[indices] =3
    indices_colony = draw.disk((80,80),20)
    mask[indices_colony ] = 1
    rect = (5,5,155,155)
    m = cv2.grabCut(img,mask,rect,bgdModel,fgdModel,50,cv2.GC_INIT_WITH_MASK)[0]
    plt.imshow(img)
    plt.show()
    plt.close()
    outputMask = np.where((m == cv2.GC_BGD) | (m == cv2.GC_PR_BGD), 0, 1)
    
    out_img = img.copy()
    out_img[outputMask == 0] = 0
    # plt.imshow(out_img)
    # plt.show()
    # plt.close()
    
    newmask = np.zeros(img.shape[:2],np.uint8)
    indices = draw.disk((80,80),75)
    newmask[indices] =3   
    
    newmask[outputMask==1] = 1
    m = cv2.grabCut(img,newmask,rect,bgdModel,fgdModel,50,cv2.GC_INIT_WITH_MASK)[0]
    outputMask = np.where((m == cv2.GC_BGD) | (m == cv2.GC_PR_BGD), 0, 1)

    out_img = img.copy()
    out_img[outputMask == 0] = 0
    plt.imshow(out_img)
    plt.show()
    plt.close()
    # im = Image.fromarray(out_img)
    # im.save(os.path.join(outdir, name+  ".jpg"))

