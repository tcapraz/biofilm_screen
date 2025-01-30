import argparse

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
#import visdom
from torchvision import datasets, transforms
import torch.nn.functional as F
import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam


from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import sys

sys.stdout.flush()


torch.set_num_threads(8)

on_gpu=True

if on_gpu:  
    dev = "cuda:0" 
else:  
    dev = "cpu"

# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(z|x)
class Encoder(nn.Module):
    def __init__(self, z_dim, in_dim=96, filters=[32, 64, 128]):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, filters[0], 5, stride = 2, padding=0)
        self.bn1 = nn.BatchNorm2d(filters[0])
        
        self.conv2 = nn.Conv2d(filters[0], filters[1], 5, stride = 2, padding=0)
        self.bn2 = nn.BatchNorm2d(filters[1])
        
        self.conv3 = nn.Conv2d(filters[1], filters[2], 3, stride = 2, padding=0)
        self.bn3 = nn.BatchNorm2d(filters[2])
        
        #fc_dim = 12800
        fc_dim = 41472
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(int(fc_dim), int(z_dim))
        
        # setup the three linear transformations used96
        # self.fc1 = nn.Linear(2500, hidden_dim)
        # self.fc21 = nn.Linear(hidden_dim, z_dim)
        # self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.relu = nn.ReLU()
    
    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        #x = x.reshape(-1, 2500)
        # then compute the hidden units
        x = self.relu(self.conv1(x))
        # print(x.shape)
        x = self.bn1(x)
        x = self.relu(self.conv2(x))
        # print(x.shape)
        x = self.bn2(x)
        x = self.relu(self.conv3(x))
        # print(x.shape)
        x = self.bn3(x)
        x = self.flatten(x)
        # print(x.shape)
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc1(x)
        return z_loc


# define the PyTorch module that parameterizes the
# observation likelihood p(x|z)
class Decoder(nn.Module):
    def __init__(self, z_dim, in_dim=96,filters=[32, 64, 128]):
        super().__init__()
        # setup the two linear transformations used
        # self.fc1 = nn.Linear(z_dim, hidden_dim)
        # self.fc21 = nn.Linear(hidden_dim, 2500)
        #fc_dim = 18432 #filters[2]*int(in_dim/8)*int(in_dim/8)
        fc_dim = 41472
        self.t_fc = nn.Linear(int(z_dim), int(fc_dim))
        self.relu = nn.ReLU()
        # use dim after conv3 in encoder
        self.unflatten = nn.Unflatten(1, (filters[2], 18, 18))
        self.t_conv1 = nn.ConvTranspose2d(filters[2], filters[1], 3, stride=2, padding=0)
        self.bn4 = nn.BatchNorm2d(filters[1])
        
        self.t_conv2 = nn.ConvTranspose2d(filters[1], filters[0], 5, stride=2, padding=0, output_padding=1)
        self.bn5 = nn.BatchNorm2d(filters[0])
        
        self.t_conv3 = nn.ConvTranspose2d(filters[0], 3 , 5, stride=2, padding=0, output_padding=1)
        self.flatten = nn.Flatten()
        # setup the non-linearities
        self.relu = nn.ReLU()
    
    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        #hidden = self.softplus(self.fc1(z))
        x = self.relu(self.t_fc(z))
        #print(x.shape)
        x = self.unflatten(x)
        #print(x.shape)
        
        # add transpose conv layers, with relu activation function
        x = self.relu(self.t_conv1(x))
        #print(x.shape)
        x = self.bn4(x)
        
        x = self.relu(self.t_conv2(x))
        #print(x.shape)
        x = self.bn5(x)
        
        #x = F.pad(x, (0,-1,0,-1), mode='constant')
        
        # output layer (with sigmoid for scaling from 0 to 1)
        loc_img = torch.sigmoid((self.t_conv3(x)))
        #print(loc_img.shape)
        
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        #loc_img = torch.sigmoid(self.fc21(x))
        return loc_img


# define a PyTorch module for the VAE
class AE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, z_dim=50, use_cuda=False):
        super().__init__()
        # create the encoder and decoder networks
        self.encoder = Encoder(z_dim)
        self.decoder = Decoder(z_dim)
        
        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim
    
    def forward(self,x):
        encoded = self.encoder(x)
        decoded = self.decoder(encoded)
        return decoded
    
    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z = self.encoder(x)
        # sample in latent space
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        loc_img = torch.reshape(loc_img, (3,160,160))
        return loc_img
    
    def get_latent_rep(self, x):
        
        z_loc = self.encoder(x)

        return z_loc
 


trainpath ="../data/alldata_png/"



transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.ConvertImageDtype(torch.float)])

traindataset = datasets.ImageFolder(trainpath, transform=transform)



valsize = 1
train_set, val_set = torch.utils.data.random_split(traindataset, [len(traindataset)-valsize, valsize])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=False)


# setup the VAE
ae = AE(use_cuda=on_gpu)
#vae = torch.load("model_png.pth")
# setup the optimizer
 
# Validation using MSE Loss function
loss_function = torch.nn.MSELoss()
 
# Using an Adam Optimizer with lr = 0.1
optimizer = torch.optim.Adam(ae.parameters(), lr = 0.1, weight_decay = 1e-8)

# setup visdom for visualization
print("Starting optimization",flush=True)

train_elbo = []
test_elbo = []
# training loop
for epoch in range(150):
    # initialize loss accumulator
    epoch_loss = 0.0
    val_loss = 0.0
    # do a training epoch over each mini-batch x returned
    # by the data loader
    i = 0
    for x, _ in train_loader:
        
        reconstructed = ae(x.to(dev, non_blocking=True))

        loss = loss_function(reconstructed,x.to(dev, non_blocking=True))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        epoch_loss += loss

        i += 1
    for x, _ in val_loader:
        reconstructed = ae(x.to(dev, non_blocking=True))
        loss = loss_function(reconstructed,x.to(dev, non_blocking=True))        
        val_loss += loss
    # report training diagnostics
    normalizer_train = len(train_loader)
    normalizer_val = len(val_loader)

    total_epoch_loss_train = epoch_loss / normalizer_train
    total_epoch_loss_val = val_loss / normalizer_val
    train_elbo.append(total_epoch_loss_train)
    print(
        "[epoch %03d]  average training loss: %.4f \n average validation loss: %.4f"
        % (epoch, total_epoch_loss_train,total_epoch_loss_val), flush=True
    )
    



def compute_features(model, train_loader, N =3000,  batch=128, only_params = True):
    for i, (data, _)  in enumerate(train_loader):
        with torch.no_grad():
            data = data.to(dev, non_blocking=True)
            aux = model.get_latent_rep(data)
            aux = aux.to("cpu")
            if i == 0:
                features = np.zeros((N, aux.shape[1])).astype('float32')
            
            if i < len(train_loader) - 1:
                features[i * batch: (i + 1) * batch] = aux
            else:
            # special treatment for final batch
                features[i * batch:] = aux
    return features

torch.save(ae, "ae_fullsize.pth")

test_loader = torch.utils.data.DataLoader(traindataset, batch_size=128, shuffle=False)


features = compute_features(ae,test_loader,  N=len(traindataset), batch = 128)

paths = [i[0] for i in test_loader.dataset.samples]
names = [Path(i).stem for i in paths]

out = pd.DataFrame(features, index=names)
out.to_csv("ae_fullsize.csv")


# train_loader = torch.utils.data.DataLoader(traindataset, batch_size=len(traindataset), shuffle=False)
# images,_ = next(iter(train_loader))



# pca = PCA(n_components =10)
# scaler = StandardScaler()
# data_sc = scaler.fit_transform(out)
# # x = pca.fit_transform(data_sc)

# tsne = TSNE(n_components=2, n_jobs=8)
# x_tsne= tsne.fit_transform(data_sc)

# out_tsne = pd.DataFrame(x_tsne, index=names)
# out.to_csv("tsne_fullsize.csv")

# # images = np.transpose(images.numpy(), axes=[0,2,3,1])

# fig, ax = plt.subplots()
# imscatter(x_tsne [:, 0], x_tsne [:, 1], imageData=images, ax=ax, zoom=0.09)


# pca = PCA(n_components=20)
# pcs = pca.fit_transform(enc_expectations)
# km = KMeans(n_clusters=9)
# labels_pred= km.fit_predict(pcs)
# ari = adjusted_rand_score(labels, labels_pred)
# print("ARI for expectations of latent representations:", ari)


# pca = PCA(n_components=20)
# pcs = pca.fit_transform(enc)
# km = KMeans(n_clusters=9)
# labels_pred= km.fit_predict(pcs)
# ari = adjusted_rand_score(labels, labels_pred)
# print("ARI for sampled latent variables:", ari)
