
import argparse

import numpy as np
import torch
import torch.nn as nn
#import visdom
from torchvision import datasets, transforms

import pyro
import pyro.distributions as dist
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO
from pyro.optim import Adam


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score

torch.set_num_threads(8)

# define the PyTorch module that parameterizes the
# diagonal gaussian distribution q(z|x)
class Encoder(nn.Module):
    def __init__(self, z_dim, in_dim=96, filters1=16, filters2=4, pools=2):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, filters1, 3, padding=1)  
        # conv layer (depth from 16 --> 4), 3x3 kernels
        self.conv2 = nn.Conv2d(filters1, filters2, 3, padding=1)
        # pooling layer to reduce x-y dims by two; kernel and stride of 2
        self.pool = nn.MaxPool2d(pools, pools)
        
        
        fc_dim = ((in_dim/(pools**2))**2)*filters2
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(int(fc_dim), int(z_dim))
        self.fc2 = nn.Linear(int(fc_dim), int(z_dim))

        # setup the three linear transformations used
        # self.fc1 = nn.Linear(2500, hidden_dim)
        # self.fc21 = nn.Linear(hidden_dim, z_dim)
        # self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, x):
        # define the forward computation on the image x
        # first shape the mini-batch to have pixels in the rightmost dimension
        #x = x.reshape(-1, 2500)
        # then compute the hidden units
        x = self.softplus(self.conv1(x))
        x = self.pool(x)
        x = self.softplus(self.conv2(x))
        x = self.pool(x)
        x = self.flatten(x)
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc1(x)
        z_scale = torch.exp(self.fc2(x))
        return z_loc, z_scale


# define the PyTorch module that parameterizes the
# observation likelihood p(x|z)
class Decoder(nn.Module):
    def __init__(self, z_dim, in_dim=96, filters1=16, filters2=4, pools=2):
        super().__init__()
        # setup the two linear transformations used
        # self.fc1 = nn.Linear(z_dim, hidden_dim)
        # self.fc21 = nn.Linear(hidden_dim, 2500)
        fc_dim = ((in_dim/(pools**2))**2)*filters2
        self.t_fc = nn.Linear(int(z_dim), int(fc_dim))
        self.flatten= nn.Flatten()
        self.unflatten = nn.Unflatten(1, (filters2, int(in_dim/(pools)**2),int(in_dim/(pools**2))))
        self.t_conv1 = nn.ConvTranspose2d(filters2, filters1, 2, stride=2)
        self.t_conv2 = nn.ConvTranspose2d(filters1, 3, 2, stride=2)
        # setup the non-linearities
        self.softplus = nn.Softplus()

    def forward(self, z):
        # define the forward computation on the latent z
        # first compute the hidden units
        #hidden = self.softplus(self.fc1(z))
        x = self.softplus(self.t_fc(z))
        
        x = self.unflatten(x)
        # add transpose conv layers, with relu activation function
        x = self.softplus(self.t_conv1(x))

        # output layer (with sigmoid for scaling from 0 to 1)
        loc_img = torch.sigmoid((self.t_conv2(x)))
        
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
        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = torch.zeros(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            z_scale = torch.ones(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            # decode the latent code z
            loc_img = self.decoder.forward(z)
            # score against actual images (with relaxed Bernoulli values)
            pyro.sample(
                "obs",
                dist.Bernoulli(loc_img, validate_args=False).to_event(1),
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
            return z_loc,  z_scale
        else:
            z = dist.Normal(z_loc, z_scale).sample()
            return z
    
    
# clear param store
pyro.clear_param_store()

trainpath ="data/train/"
testpath ="data/test/"

transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.ConvertImageDtype(torch.float),transforms.Resize(96)])


traindataset = datasets.ImageFolder(trainpath, transform=transform)
testdataset = datasets.ImageFolder(testpath, transform=transform)

train_loader = torch.utils.data.DataLoader(traindataset, batch_size=128, shuffle=True)
test_loader = torch.utils.data.DataLoader(testdataset, batch_size=200, shuffle=False)


# setup the VAE
vae = VAE(use_cuda=False)

# setup the optimizer
adam_args = {"lr": 1.0e-3}
optimizer = Adam(adam_args)

# setup the inference algorithm
elbo = Trace_ELBO()
svi = SVI(vae.model, vae.guide, optimizer, loss=elbo)

# setup visdom for visualization


train_elbo = []
test_elbo = []
# training loop
for epoch in range(100):
    # initialize loss accumulator
    epoch_loss = 0.0
    # do a training epoch over each mini-batch x returned
    # by the data loader
    for x, _ in train_loader:
 
        # do ELBO gradient and accumulate loss
        epoch_loss += svi.step(x)

    # report training diagnostics
    normalizer_train = len(train_loader.dataset)
    total_epoch_loss_train = epoch_loss / normalizer_train
    train_elbo.append(total_epoch_loss_train)
    print(
        "[epoch %03d]  average training loss: %.4f"
        % (epoch, total_epoch_loss_train)
    )

    # if epoch % 5 == 0:
    #     # initialize loss accumulator
    #     test_loss = 0.0
    #     # compute the loss over the entire test set
    #     for i, (x, _) in enumerate(test_loader):
    #         # if on GPU put mini-batch into CUDA memory

    #         # compute ELBO estimate and accumulate loss
    #         test_loss += svi.evaluate_loss(x)

    #         # pick three random test images from the first mini-batch and
    #         # visualize how well we're reconstructing them
          
    #     # report test diagnostics
    #     normalizer_test = len(test_loader.dataset)
    #     total_epoch_loss_test = test_loss / normalizer_test
    #     test_elbo.append(total_epoch_loss_test)
    #     print(
    #         "[epoch %03d]  average test loss: %.4f" % (epoch, total_epoch_loss_test)
    #     )



dataiter = iter(test_loader)
images, labels = dataiter.next()


# get latent representation
with torch.no_grad():
    enc_expectations,_ = vae.get_latent_rep(images)
    enc = vae.get_latent_rep(images, only_params=False)

# plot a few reconstructed test images:
# import matplotlib.pyplot as plt
# decoded = vae.reconstruct_img(images)
    
# decoded =torch.reshape(decoded , (200, 3,96,96))
# decoded = decoded.detach().numpy()
# plt.imshow(decoded[3,0,:,:])


pca = PCA(n_components=10)
pcs = pca.fit_transform(enc_expectations)
km = KMeans(n_clusters=5)
labels_pred= km.fit_predict(pcs)
ari = adjusted_rand_score(labels, labels_pred)
print("ARI for expectations of latent representations:", ari)

    
pca = PCA(n_components=10)
pcs = pca.fit_transform(enc)
km = KMeans(n_clusters=5)
labels_pred= km.fit_predict(pcs)
ari = adjusted_rand_score(labels, labels_pred)
print("ARI for sampled latent variables:", ari)

