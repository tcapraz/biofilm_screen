import numpy as np
import pandas as pd
from itertools import product

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets, transforms

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
    

trainpath ="circle_cropping/out_new/train"
testpath ="classification/data"

means= [0.49248388, 0.3756774 , 0.3482648 ]
std = [0.18929635, 0.21403217, 0.18945132]
transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.ConvertImageDtype(torch.float),transforms.Resize(96)])



traindataset = datasets.ImageFolder(trainpath, transform=transform)
testdataset = datasets.ImageFolder(testpath, transform=transform)

train_loader = torch.utils.data.DataLoader(traindataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(testdataset, batch_size=200, shuffle=False)


torch.set_num_threads(4)
# define the NN architecture
class ConvAutoencoder(nn.Module):
    def __init__(self, in_dim=96, pools =2, filters1=16, filters2=4, z_dim=50):
        super(ConvAutoencoder, self).__init__()
        ## encoder layers ##
        # conv layer (depth from 3 --> 16), 3x3 kernels
        self.conv1 = nn.Conv2d(3, filters1, 3, padding=1)  
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(filters1, filters2, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(pools, pools)
        # fully connected layer to get to final latent dim
        
        fc_dim = ((in_dim/(pools**2))**2)*filters2

        
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(int(fc_dim), int(z_dim))
        
        ## decoder layers ##
        ## a kernel of 2 and a stride of 2 will increase the spatial dims by 2
        self.t_fc = nn.Linear(int(z_dim), int(fc_dim))
        self.unflatten = nn.Unflatten(1, (filters2, int(in_dim/(pools)**2),int(in_dim/(pools**2))))
        self.t_conv1 = nn.ConvTranspose2d(filters2, filters1, pools, stride=pools)
        self.t_conv2 = nn.ConvTranspose2d(filters1, 3, pools, stride=pools)

    def forward(self, x):
        p = False
        ## encode ##
        # add hidden layers with relu activation function
        # and maxpooling after
        if p: print(x.shape)
        x = F.relu(self.conv1(x))
        if p: print(x.shape)
        x = self.pool(x)
        if p: print(x.shape)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        if p: print(x.shape)
        x = self.pool(x)  
        if p: print(x.shape)
        # compressed representation
        x = self.flatten(x)
        if p: print(x.shape)
        x = F.relu(self.fc(x))
        if p: print(x.shape)

        ## decode ##
        x =self.t_fc(x)
        if p: print(x.shape)
        x = self.unflatten(x)
        # add transpose conv layers, with relu activation function
        x = F.relu(self.t_conv1(x))
        if p: print(x.shape)
        # output layer (with sigmoid for scaling from 0 to 1)
        x = torch.sigmoid(self.t_conv2(x))
        if p: print(x.shape)
        return x
    
    def encode(self, x):
        
        x = F.relu(self.conv1(x))
       # print(x.shape)
        x = self.pool(x)
        #print(x.shape)
        # add second hidden layer
        x = F.relu(self.conv2(x))
        #print(x.shape)
        x = self.pool(x)
        # compressed representation
        x = self.flatten(x)
        x = F.relu(self.fc(x))
        return x

# hyperparam optimization
filters1 = [16, 32]
filters2 = [2]
pools = [2,4]
z_dim = [25,50,75]
params = [filters1, filters2, pools, z_dim]

param_prod = product(*params)
param_dict = [{k:v  for k, v in zip(["filters1", "filters2", "pools", "z_dim"], p)} for p in param_prod]

res = {"Train loss": [], "ARI": []}
res_idx = [str(i) for i in param_dict]

# run each of the parameter combinations
for par in param_dict:
    # initialize the NN
    model = ConvAutoencoder(in_dim=96, pools=par["pools"], filters1=par["filters1"], 
                            filters2=par["filters2"], z_dim=par["z_dim"])
    print(model)
    
    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    # number of epochs to train the model
    n_epochs = 100
    
    for epoch in range(1, n_epochs+1):
        # monitor training loss
        train_loss = 0.0
        
        ###################
        # train the model #
        ###################
        for data in train_loader:
            # _ stands in for labels, here
            # no need to flatten images
            images, _ = data
            # clear the gradients of all optimized variables
            optimizer.zero_grad()
            # forward pass: compute predicted outputs by passing inputs to the model
            outputs = model(images)
            # calculate the loss
            loss = criterion(outputs, images)
            # backward pass: compute gradient of the loss with respect to model parameters
            loss.backward()
            # perform a single optimization step (parameter update)
            optimizer.step()
            # update running training loss
            train_loss += loss.item()*images.size(0)
                
        # print avg training statistics 
        train_loss = train_loss/len(train_loader)
        if epoch % 1 == 0:
            print('Epoch: {} \tTraining Loss: {:.6f}'.format(
                epoch, 
                train_loss
                ))
            
    res["Train loss"].append(train_loss)

    # benchmark with annotated images
    
    dataiter = iter(test_loader)
    images, labels = dataiter.next()
    
    
    # get latent representation
    with torch.no_grad():
        img_enc = model.encode(images)
        
    
    # show a couple of images
    # for i in range(10):
    #     fig, (ax1, ax2) = plt.subplots(1, 2)
    #     ax1.imshow(images[i,0,:,:])
    #     ax2.imshow(img_dec[i,0,:,:])
    

    
    #img_enc = img_enc.reshape((200, 2304))
    
    # pca = PCA(n_components=2)
    # pcs = pca.fit_transform(img_enc)
    # plt.scatter(pcs[:,0], pcs[:,1], c=labels)
    
    pca = PCA(n_components=10)
    pcs = pca.fit_transform(img_enc)
    km = KMeans(n_clusters=5)
    labels_pred= km.fit_predict(pcs)
    ari = adjusted_rand_score(labels, labels_pred)
    res["ARI"].append(ari)
    
df = pd.DataFrame.from_dict(res)
df.index = res_idx
df.to_csv("res.csv")
"""
# from stack overflow: how to use SSIM loss in pytorch
from piqa import SSIM

class SSIMLoss(SSIM):
    def forward(self, x, y):
        return 1. - super().forward(x, y)

criterion = SSIMLoss() # .cuda() if you need GPU support


loss = criterion(x, y)

"""