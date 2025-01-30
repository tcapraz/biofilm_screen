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



torch.set_num_threads(8)

on_gpu=False

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
        
        fc_dim = 12800
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(int(fc_dim), int(z_dim))
        self.fc2 = nn.Linear(int(fc_dim), int(z_dim))
        
        # setup the three linear transformations used
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
        #print(x.shape)
        x = self.bn1(x)
        x = self.relu(self.conv2(x))
        #print(x.shape)
        x = self.bn2(x)
        x = self.relu(self.conv3(x))
        #print(x.shape)
        x = self.bn3(x)
        x = self.flatten(x)
        #print(x.shape)
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc1(x)
        z_scale = torch.exp(self.fc2(x))
        return z_loc, z_scale


# define the PyTorch module that parameterizes the
# observation likelihood p(x|z)
class Decoder(nn.Module):
    def __init__(self, z_dim, in_dim=96,filters=[32, 64, 128]):
        super().__init__()
        # setup the two linear transformations used
        # self.fc1 = nn.Linear(z_dim, hidden_dim)
        # self.fc21 = nn.Linear(hidden_dim, 2500)
        fc_dim = 18432 #filters[2]*int(in_dim/8)*int(in_dim/8)
        self.t_fc = nn.Linear(int(z_dim), int(fc_dim))
        self.relu = nn.ReLU()
        self.unflatten = nn.Unflatten(1, (filters[2], 12, 12))
        self.t_conv1 = nn.ConvTranspose2d(filters[2], filters[1], 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(filters[1])
        
        self.t_conv2 = nn.ConvTranspose2d(filters[1], filters[0], 5, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(filters[0])
        
        self.t_conv3 = nn.ConvTranspose2d(filters[0], 3 , 5, stride=2, padding=1, output_padding=1)
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
        loc_img = self.flatten(loc_img)
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        #loc_img = torch.sigmoid(self.fc21(x))
        return loc_img


# define a PyTorch module for the VAE
class VAE(nn.Module):
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
    
    # define the model p(x|z)p(z)
    def model(self, x):
        # register PyTorch module `decoder` with Pyro
        pyro.module("decoder", self.decoder)
        #sigma = pyro.sample('σ', dist.HalfCauchy(1.))
        
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = torch.zeros(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            z_scale = torch.ones(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img = self.decoder.forward(z)
            # score against actual images (with relaxed Bernoulli values)
            sigmas = torch.ones(x.shape[0],3*96*96, dtype=x.dtype, device=x.device)*0.1
            pyro.sample(
                "obs",
                dist.Normal(loc_img, sigmas, validate_args=False).to_event(1),
                obs=x.reshape(-1, 3*96*96),
            )
            # return the loc so we can visualize it later
            return loc_img
    
    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            # print(z_loc)
            
            # print(z_scale)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
    
    # define a helper function for reconstructing images
    def reconstruct_img(self, x):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z)
        loc_img = torch.reshape(loc_img, (3,96,96))
        return loc_img
    
    def get_latent_rep(self, x, only_params=True):
        
        z_loc, z_scale = self.encoder(x)
        if only_params == True:
            return z_loc
        else:
            z = dist.Normal(z_loc, z_scale).sample()
            return z

def compute_features(model, train_loader, N =3000,  batch=128, only_params = True):
    for i, (data, _)  in enumerate(train_loader):
        with torch.no_grad():
            data = data.to(dev, non_blocking=True)
            aux = model.get_latent_rep(data, only_params=only_params)
            aux = aux.to("cpu")
            if i == 0:
                features = np.zeros((N, aux.shape[1])).astype('float32')
            
            if i < len(train_loader) - 1:
                features[i * batch: (i + 1) * batch] = aux
            else:
            # special treatment for final batch
                features[i * batch:] = aux
    return features

# clear param store
pyro.clear_param_store()

trainpath ="../data/alldata_png/"
vae = torch.load("model_holdout_png.pth",map_location=dev)
outname = "holdout_features.csv"


transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.ConvertImageDtype(torch.float),transforms.Resize(96)])


dataset = datasets.ImageFolder(trainpath, transform=transform)

loader = torch.utils.data.DataLoader(dataset, batch_size=128, shuffle=False)

features = compute_features(vae,loader,  N=len(dataset), batch = 128)

paths = [i[0] for i in loader.dataset.samples]
names = [Path(i).stem for i in paths]

out = pd.DataFrame(features, index=names)
out.to_csv(outname)
