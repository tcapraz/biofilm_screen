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
from itertools import product

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from torch.utils.data import Dataset

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import sys
import scipy.stats as stats
import matplotlib.gridspec as gridspec
sys.stdout.flush()
from PIL import Image




class AugmentedDataset(Dataset):
    def __init__(self, dataset):
        super(AugmentedDataset, self).__init__()
        transform = dataset.transform
        dataset.transform = None
        self.dataset = dataset
        
        if isinstance(transform, dict):
            self.image_transform = transform['standard']
            self.cropped_transform = transform['crop']

        else:
            self.image_transform = transform
            self.cropped_transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample,_ = self.dataset.samples[index]
        image = Image.open(sample)
        out = dict()
        out['image'] = self.image_transform(image)
        out['image_cropped'] = self.cropped_transform(image)

        return out


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
    

torch.set_num_threads(8)

on_gpu=False

if on_gpu:  
    dev = "cuda:0" 
else:  
    dev = "cpu"

# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(z|x)
class Encoder(nn.Module):
    def __init__(self, z_dim,  filters=[32, 64, 128]):
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
        self.fc2 = nn.Linear(int(fc_dim), int(z_dim))
        
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
        z_scale = torch.exp(self.fc2(x))
        return z_loc, z_scale

class Encoder_crop(nn.Module):
    def __init__(self, z_dim,  filters=[32, 64, 128]):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, filters[0], 5, stride = 2, padding=0)
        self.bn1 = nn.BatchNorm2d(filters[0])
        
        self.conv2 = nn.Conv2d(filters[0], filters[1], 5, stride = 2, padding=0)
        self.bn2 = nn.BatchNorm2d(filters[1])
        
        self.conv3 = nn.Conv2d(filters[1], filters[2], 3, stride = 2, padding=0)
        self.bn3 = nn.BatchNorm2d(filters[2])
        
        #fc_dim = 12800
        fc_dim = 2048
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(int(fc_dim), int(z_dim))
        self.fc2 = nn.Linear(int(fc_dim), int(z_dim))
        
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
        self.encoder = Encoder(int(z_dim/2))
        self.encoder_crop = Encoder_crop(int(z_dim/2))

        self.decoder = Decoder(z_dim)
        
        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim
    
    # define the model p(x|z)p(z)
    def model(self, x, x_crop):
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
            sigmas = torch.ones(x.shape[0],3*160*160, dtype=x.dtype, device=x.device)*0.1
            pyro.sample(
                "obs",
                dist.Normal(loc_img, sigmas, validate_args=False).to_event(1),
                obs=x.reshape(-1, 3*160*160),
            )
            # return the loc so we can visualize it later
            return loc_img
    
    # define the guide (i.e. variational distribution) q(z|x)
    def guide(self, x, x_crop):
        # register PyTorch module `encoder` with Pyro
        pyro.module("encoder", self.encoder)
        pyro.module("encoder", self.encoder_crop)

        with pyro.plate("data", x.shape[0]):
            # use the encoder to get the parameters used to define q(z|x)
            z_loc, z_scale = self.encoder.forward(x)
            z_loc_crop, z_scale_crop = self.encoder_crop.forward(x_crop)
            # print(z_loc.shape)

            # print(z_scale_crop.shape)
            z_loc = torch.cat((z_loc,z_loc_crop), dim=-1)
            z_scale = torch.cat((z_scale,z_scale_crop), dim=-1)

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
        loc_img = torch.reshape(loc_img, (3,360,80))
        return loc_img
    
    def get_latent_rep(self, x,x_cropped):
        
        z_loc, z_scale = self.encoder(x)
        z_loc_crop, z_scale_cropped = self.encoder_crop(x_cropped)
        z_loc = torch.cat((z_loc,z_loc_crop), dim=-1)
        return z_loc



pyro.clear_param_store()

trainpath ="../data/all_data/"

holdout1 = pd.read_csv("../data/biofilm_annotations-export.csv")
holdout2 = pd.read_csv("../data/biofilm-set-2-export.csv")
holdout = pd.concat((holdout1,holdout2))
holdout = holdout[["image","label"]]
holdout_paths = trainpath + "data/" + holdout["image"]
holdout_paths = [(i, 0) for i in holdout_paths]

biofilm_names = pd.read_csv("biofilm_image_names.csv", header=None)



transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.ConvertImageDtype(torch.float)])
transform_crop = transforms.Compose([transforms.ToTensor(), 
                                transforms.ConvertImageDtype(torch.float), transforms.CenterCrop(50)])


transform_dict  = {"crop": transform_crop, "standard": transform}


traindataset = datasets.ImageFolder(trainpath, transform=transform_dict)
traindataset = AugmentedDataset(traindataset)

    
train_loader = torch.utils.data.DataLoader(traindataset, batch_size=128, shuffle=False)

vae = torch.load("model_multiview.pth",map_location=torch.device('cpu'))


def reconstruct_img(vae, x):
    # encode image x
    z_loc, z_scale = vae.encoder(x["image"])
    z_loc_crop, z_scale_crop = vae.encoder_crop(x["image_cropped"])
    # sample in latent space
    z_loc = torch.cat((z_loc,z_loc_crop), dim=-1)
    z_scale = torch.cat((z_scale,z_scale_crop), dim=-1)
    z = dist.Normal(z_loc, z_scale).sample()
    # decode the image (note we don't sample in image space)
    loc_img = vae.decoder(z)
    loc_img = torch.reshape(loc_img,x["image"].shape)
    return loc_img

rec_imgs = []
original_imgs = []
for x in train_loader:
    
    # do ELBO gradient and accumulate loss
    original_imgs.append(x["image"].numpy())
    rec_imgs.append(reconstruct_img(vae,x).detach().numpy())
    
rec_imgs  = np.vstack(rec_imgs)    
original_imgs =  np.vstack(original_imgs)    


plt.imshow(rec_imgs[1,:,:,:].T)
#plt.savefig("reconstructed.png", dpi=400)
plt.imshow(original_imgs[1,:,:,:].T)
#plt.savefig("original.png", dpi=400)


LF = 50
fvalues = stats.norm.ppf(np.linspace(0.01,0.99, 10))

sampled_data = []
sampled_imgs = []
for i in range(LF):
    mat = np.zeros((10,LF))
    mat[:,i] = fvalues
    sampled_data.append(mat)
    loc_img = vae.decoder(torch.tensor(mat, dtype=torch.float32))
    loc_img = torch.reshape(loc_img,(10,3,96,96)).detach().numpy()
    sampled_imgs.append(loc_img)
    
fig,ax = plt.subplots(10,50,constrained_layout=True,figsize=(25,5))
# plt.figure(figsize = (10,50))
# gs1 = gridspec.GridSpec(10,50)
# gs1.update(wspace=0, hspace=0) 
for i in range(10):
    for j in range(50):
        # ax1 = plt.subplot(gs1[i*10+j])
        ax[i,j].imshow(sampled_imgs[j][i,:,:,:].T)
        ax[i,j].axes.xaxis.set_visible(False)
        ax[i,j].axes.yaxis.set_visible(False)
        # ax1.imshow(sampled_imgs[j][i,:,:,:].T)
        # ax1.axes.xaxis.set_visible(False)
        # ax1.axes.yaxis.set_visible(False)
        ax[i,j].axis('off')
plt.savefig("latent_space_reconstruction_multiview.png", dpi=500)


def compute_features(model, train_loader, N =3000,  batch=128, only_params = True):
    for i,data  in enumerate(train_loader):
        with torch.no_grad():
            x = data["image"]
            x_cropped = data["image_cropped"]
            x = x.to(dev, non_blocking=True)
            x_cropped = x_cropped.to(dev, non_blocking=True)
            aux = model.get_latent_rep(x, x_cropped)
            aux = aux.to("cpu")
            if i == 0:
                features = np.zeros((N, aux.shape[1])).astype('float32')
            
            if i < len(train_loader) - 1:
                features[i * batch: (i + 1) * batch] = aux
            else:
            # special treatment for final batch
                features[i * batch:] = aux
    return features


features = compute_features(vae,train_loader,  N=len(traindataset), batch = 128)

pca = PCA(n_components=2)
scaler = StandardScaler()
features  = scaler.fit_transform(features)


PCs=pca.fit_transform(features)
W = pca.components_
W_ = np.linalg.inv(W)

LF = 50
fvalues_pca1 = np.percentile(PCs[:,0], np.linspace(0,100, 20))*pca.explained_variance_ratio_[0]
fvalues_pca2 = np.percentile(PCs[:,1], np.linspace(100,0, 20))*pca.explained_variance_ratio_[1]
grid = [i for i in product(fvalues_pca2, fvalues_pca1)]

    
X_ = np.stack(grid)
X_[:,[0,1]] = X_[:,[1,0]]
PC_LF = X_@W



pca_img = vae.decoder(torch.tensor(PC_LF, dtype=torch.float32))
pca_img = torch.reshape(pca_img,(20,20,3,96,96)).detach().numpy()

fig,ax = plt.subplots(20,20,constrained_layout=True,figsize=(10,10))
# plt.figure(figsize = (10,50))
# gs1 = gridspec.GridSpec(10,50)
# gs1.update(wspace=0, hspace=0) 
for i in range(20):
    for j in range(20):
        # ax1 = plt.subplot(gs1[i*10+j])
        ax[i,j].imshow(pca_img[j][i,:,:,:].T)
        ax[i,j].axes.xaxis.set_visible(False)
        ax[i,j].axes.yaxis.set_visible(False)
        # ax1.imshow(sampled_imgs[j][i,:,:,:].T)
        # ax1.axes.xaxis.set_visible(False)
        # ax1.axes.yaxis.set_visible(False)
        ax[i,j].axis('off')
plt.savefig("pca_latent_space_reconstruction_multiview.png", dpi=500)


mean_img = vae.decoder(torch.tensor(np.zeros((1,50)), dtype=torch.float32))
mean_img = torch.reshape(mean_img,(3,96,96)).detach().numpy().T
plt.imshow(mean_img)
