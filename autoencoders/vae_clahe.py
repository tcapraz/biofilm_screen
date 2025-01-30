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
            sigmas = torch.ones(x.shape[0],3*160*160, dtype=x.dtype, device=x.device)*0.1
            pyro.sample(
                "obs",
                dist.Normal(loc_img, sigmas, validate_args=False).to_event(1),
                obs=x.reshape(-1, 3*160*160),
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
        loc_img = torch.reshape(loc_img, (3,360,80))
        return loc_img
    
    def get_latent_rep(self, x, only_params=True):
        
        z_loc, z_scale = self.encoder(x)
        if only_params == True:
            return z_loc
        else:
            z = dist.Normal(z_loc, z_scale).sample()
            return z


# clear param store
pyro.clear_param_store()

trainpath ="../data/clahe/"

holdout = pd.read_csv("../data/biofilm_annotations-export.csv")
holdout = holdout[["image","label"]]
holdout_paths = trainpath + "data/" + holdout["image"]
holdout_paths = [(i, 0) for i in holdout_paths]


transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.ConvertImageDtype(torch.float)])

fulldataset = datasets.ImageFolder(trainpath, transform=transform)
traindataset = datasets.ImageFolder(trainpath, transform=transform)

for i in holdout_paths:
    traindataset.imgs.remove(i)

valsize = 1000
train_set, val_set = torch.utils.data.random_split(traindataset, [len(traindataset)-valsize, valsize])

train_loader = torch.utils.data.DataLoader(train_set, batch_size=128, shuffle=True)
val_loader = torch.utils.data.DataLoader(val_set, batch_size=128, shuffle=False)


# setup the VAE
vae = VAE(use_cuda=on_gpu)
#vae = torch.load("model_png.pth")
# setup the optimizer
adam_args = {"lr": 1.0e-5}
optimizer = Adam(adam_args)

# setup the inference algorithm
elbo = Trace_ELBO()
svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)


# setup visdom for visualization
print("Starting optimization",flush=True)

train_elbo = []
test_elbo = []
# training loop
for epoch in range(100):
    # initialize loss accumulator
    epoch_loss = 0.0
    val_loss = 0.0
    # do a training epoch over each mini-batch x returned
    # by the data loader
    i = 0
    for x, _ in train_loader:
        
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(x.to(dev, non_blocking=True))
       # print(epoch_loss/((i+1)*128), flush=True)
        i += 1
    for x, _ in val_loader:
        val_loss += svi.evaluate_loss(x.to(dev, non_blocking=True))
    # report training diagnostics
    normalizer_train = len(train_loader.dataset)
    normalizer_val = len(val_loader.dataset)

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

torch.save(vae, "model_clahe.pth")

test_loader = torch.utils.data.DataLoader(fulldataset, batch_size=128, shuffle=False)


features_params = compute_features(vae,test_loader,  N=len(fulldataset), batch = 128)

paths = [i[0] for i in test_loader.dataset.samples]
names = [Path(i).stem for i in paths]

out = pd.DataFrame(features_params, index=names)
out.to_csv("vae_clahe.csv")


# train_loader = torch.utils.data.DataLoader(traindataset, batch_size=len(traindataset), shuffle=False)
# images,_ = next(iter(train_loader))



# pca = PCA(n_components =10)
#scaler = StandardScaler()
#data_sc = scaler.fit_transform(out)
# x = pca.fit_transform(data_sc)

#tsne = TSNE(n_components=2, n_jobs=8)
#x_tsne= tsne.fit_transform(data_sc)

#out_tsne = pd.DataFrame(x_tsne, index=names)
#out.to_csv("tsne_fullsize.csv")

# images = np.transpose(images.numpy(), axes=[0,2,3,1])

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
