import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import cv2

from torchvision import datasets, transforms

import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import  silhouette_score, adjusted_rand_score
from sklearn.preprocessing import StandardScaler

test = pd.read_csv("testfeatures_pretrained_norm.csv", header=None)
train = pd.read_csv("features_pretrained_norm.csv", header=None)


testpath ="../data/test"
trainpath ="../circle_cropping/out_new/train"

transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.ConvertImageDtype(torch.float),transforms.Resize(96),
                               ])

testdataset = datasets.ImageFolder(testpath, transform=transform)
traindataset = datasets.ImageFolder(trainpath, transform=transform)

test_loader = torch.utils.data.DataLoader(testdataset, batch_size=250, shuffle=False)
dataiter = iter(test_loader)
testimages, testlabels = dataiter.next()

train_loader = torch.utils.data.DataLoader(traindataset, batch_size=30000, shuffle=False)
dataiter = iter(train_loader)
trainimages, _= dataiter.next()

test = StandardScaler().fit_transform(test)
pca = PCA(n_components=2)
pcs = pca.fit_transform(test)
scatter = plt.scatter(pcs[:,0], pcs[:,1],s = 5, c=testlabels)
plt.legend(*scatter.legend_elements())


train = StandardScaler().fit_transform(train)

pca = PCA(n_components=20)

pcs = pca.fit_transform(train)
scatter =plt.scatter(pcs[:,0], pcs[:,1], s=5)
km = KMeans(10)
labels = km.fit_predict(pcs)
plt.scatter(pcs[:,0], pcs[:,1], s=5, c = labels)

class0 = trainimages[labels==1]
for i in range(10):
    plt.imshow(class0[i,:,:,:].permute(1, 2, 0)  )
    plt.show()
    plt.close()

img = class0[0,:,:,:]
img = img.detach().numpy()
dst = cv2.blur(img, (50, 50)) 
plt.imshow(dst[0,:,:])
avg_hist = img.mean()
ffc = (img/dst)*avg_hist
plt.imshow(ffc[2,:,:])
plt.imshow(img[0,:,:])

l_ffc = img - dst + avg_hist
plt.imshow(l_ffc[2,:,:])



