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
from pyro.infer import SVI, JitTrace_ELBO, Trace_ELBO, TraceEnum_ELBO,config_enumerate
from pyro.optim import Adam
from pyro.distributions.util import broadcast_shape

from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score, accuracy_score, confusion_matrix
from sklearn.model_selection import train_test_split, StratifiedKFold

from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
import matplotlib
matplotlib.use('Agg')

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
class Encoder_y(nn.Module):
    def __init__(self, z_dim, y_dim, in_dim=96, filters=[32, 64, 128]):
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
        self.fc2 = nn.Linear(int(z_dim), int(y_dim))
        
        # setup the three linear transformations used
        # self.fc1 = nn.Linear(2500, hidden_dim)
        # self.fc21 = nn.Linear(hidden_dim, z_dim)
        # self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.relu = nn.ReLU()
        self.softmax = nn.Softmax()
    
    def forward(self, x, y=None):
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
        z_loc = self.relu(self.fc1(x))

        alpha =  self.softmax(self.fc2(z_loc))
        return alpha

      
class Encoder_z(nn.Module):
    def __init__(self, z_dim,y_dim, in_dim=96, filters=[32, 64, 128]):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, filters[0], 5, stride = 2, padding=0)
        self.bn1 = nn.BatchNorm2d(filters[0])
        
        self.conv2 = nn.Conv2d(filters[0], filters[1], 5, stride = 2, padding=0)
        self.bn2 = nn.BatchNorm2d(filters[1])
        
        self.conv3 = nn.Conv2d(filters[1], filters[2], 3, stride = 2, padding=0)
        self.bn3 = nn.BatchNorm2d(filters[2])
        
        fc_dim = 12800
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(int(fc_dim)+int(y_dim), int(z_dim))
        self.fc2 = nn.Linear(int(fc_dim)+int(y_dim), int(z_dim))
        
        # setup the three linear transformations used
        # self.fc1 = nn.Linear(2500, hidden_dim)
        # self.fc21 = nn.Linear(hidden_dim, z_dim)
        # self.fc22 = nn.Linear(hidden_dim, z_dim)
        # setup the non-linearities
        self.relu = nn.ReLU()
    
    def forward(self, x, y):
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
        # print(x.shape)
        # print(y.shape)
        inputs = [x,y]
        shape = broadcast_shape(*[s.shape[:-1] for s in inputs]) + (-1,)
        inputs = [s.expand(shape) for s in inputs]
        x = torch.cat(inputs, dim=-1)
        #print(x.shape)
        # then return a mean vector and a (positive) square root covariance
        # each of size batch_size x z_dim
        z_loc = self.fc1(x)
        z_scale = torch.exp(self.fc2(x))
        return z_loc, z_scale

# define the PyTorch module that parameterizes the
# observation likelihood p(x|z)
class Decoder(nn.Module):
    def __init__(self, z_dim,y_dim, in_dim=96,filters=[32, 64, 128]):
        super().__init__()
        # setup the two linear transformations used
        # self.fc1 = nn.Linear(z_dim, hidden_dim)
        # self.fc21 = nn.Linear(hidden_dim, 2500)
        fc_dim = 18432 #filters[2]*int(in_dim/8)*int(in_dim/8)
        self.t_fc = nn.Linear(int(z_dim)+int(y_dim), int(fc_dim))
        self.relu = nn.ReLU()
        
        self.unflatten_latent = nn.Unflatten(2, (filters[2], 12, 12))
        self.unflatten_supervised = nn.Unflatten(1, (filters[2], 12, 12))

        self.t_conv1 = nn.ConvTranspose2d(filters[2], filters[1], 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(filters[1])
        
        self.t_conv2 = nn.ConvTranspose2d(filters[1], filters[0], 5, stride=2, padding=1)
        self.bn5 = nn.BatchNorm2d(filters[0])
        
        self.t_conv3 = nn.ConvTranspose2d(filters[0], 3 , 5, stride=2, padding=1, output_padding=1)
        self.flatten = nn.Flatten()
        # setup the non-linearities
        self.relu = nn.ReLU()
        self.inchannel = filters[2]
        self.y_dim = y_dim
        
    def forward(self, z,y):


        inputs = [z,y]
        shape = broadcast_shape(*[s.shape[:-1] for s in inputs]) + (-1,)
        inputs = [s.expand(shape) for s in inputs]
        z = torch.cat(inputs, dim=-1)
        # define the forward computation on the latent z
        # first compute the hidden units
        #hidden = self.softplus(self.fc1(z))
        x = self.relu(self.t_fc(z))
        
        #print(x.shape)
        if (len(y.shape)) == 2:
            batch_s = x.shape[0]
            x = self.unflatten_supervised(x)
        else:
            batch_s = x.shape[1]
            x = self.unflatten_latent(x)
            x = x.reshape((x.shape[0]*x.shape[1], self.inchannel, 12, 12))
          
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
        if (len(y.shape)) == 2:
            loc_img = loc_img.reshape((batch_s, loc_img.shape[1]*loc_img.shape[2]*loc_img.shape[3]))
        else:
            loc_img = loc_img.reshape((self.y_dim,batch_s, loc_img.shape[1]*loc_img.shape[2]*loc_img.shape[3]))
        #print(loc_img.shape)
        #loc_img = self.flatten(loc_img)
        # return the parameter for the output Bernoulli
        # each is of size batch_size x 784
        #loc_img = torch.sigmoid(self.fc21(x))
        return loc_img


# define a PyTorch module for the VAE
class VAE(nn.Module):
    # by default our latent space is 50-dimensional
    # and we use 400 hidden units
    def __init__(self, z_dim=50,y_dim=6, use_cuda=False,aux_loss_multiplier =46):
        super().__init__()
        # create the encoder and decoder networks
        #self.encoder_y = Encoder_y(y_dim)
        self.encoder_y = Encoder_y(z_dim, y_dim)
        self.encoder_z = Encoder_z(z_dim, y_dim)

        self.decoder = Decoder(z_dim, y_dim)
        
        if use_cuda:
            # calling cuda() here will put all the parameters of
            # the encoder and decoder networks into gpu memory
            self.cuda()
        self.use_cuda = use_cuda
        self.z_dim = z_dim
        self.y_dim = y_dim
        self.aux_loss_multiplier = aux_loss_multiplier
    # define the model p(x|z)p(z)
    def model(self, x, y = None):
        pyro.module("ss_vae", self)
        # register PyTorch module `decoder` with Pyro
        #pyro.module("decoder", self.decoder)
        #sigma = pyro.sample('σ', dist.HalfCauchy(1.))


        with pyro.plate("data", x.shape[0]):
            # setup hyperparameters for prior p(z)
            z_loc = torch.zeros(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            z_scale = torch.ones(x.shape[0], self.z_dim, dtype=x.dtype, device=x.device)
            # sample from prior (value will be sampled by guide when computing the ELBO)
            z = pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            
            alpha_prior = x.new_ones([x.shape[0], self.y_dim]) / (1.0 * self.y_dim)
            y = pyro.sample("y", dist.OneHotCategorical(alpha_prior), obs=y)
            
            # decode the latent code z
            #print(z.shape)
            loc_img = self.decoder.forward(z,y)
            #print(loc_img.shape)
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
    def guide(self, x, y = None):
        # register PyTorch module `encoder` with Pyro
        #pyro.module("encoder", self.encoder)
       
        with pyro.plate("data", x.shape[0]):
            if y is None:
                alpha = self.encoder_y(x)
                y = pyro.sample("y", dist.OneHotCategorical(alpha))
            # use the encoder to get the parameters used to define q(z|x)
            # print(y.shape)
            # print(x.shape)
            z_loc, z_scale = self.encoder_z.forward(x,y)
            # print(z_loc)

            
            # print(z_scale)
            # sample the latent code z
            pyro.sample("latent", dist.Normal(z_loc, z_scale).to_event(1))
            
    def model_classify(self, xs, ys=None):
        """
        this model is used to add an auxiliary (supervised) loss as described in the
        Kingma et al., "Semi-Supervised Learning with Deep Generative Models".
        """
        # register all pytorch (sub)modules with pyro
        pyro.module("ss_vae", self)

        # inform Pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate("data"):
            # this here is the extra term to yield an auxiliary loss that we do gradient descent on
            if ys is not None:
                alpha = self.encoder_y.forward(xs)
                with pyro.poutine.scale(scale=self.aux_loss_multiplier):
                    pyro.sample("y_aux", dist.OneHotCategorical(alpha), obs=ys)

    def guide_classify(self, xs, ys=None):
        """
        dummy guide function to accompany model_classify in inference
        """
        pass
    
    # define a helper function for reconstructing images
    def reconstruct_img(self, x,y):
        # encode image x
        z_loc, z_scale = self.encoder(x)
        # sample in latent space
        z = dist.Normal(z_loc, z_scale).sample()
        # decode the image (note we don't sample in image space)
        loc_img = self.decoder(z,y)
        loc_img = torch.reshape(loc_img, (3,96,96))
        return loc_img
    
    def get_latent_rep(self, x, only_params=True):
        
        z_loc, z_scale = self.encoder(x)
        if only_params == True:
            return z_loc
        else:
            z = dist.Normal(z_loc, z_scale).sample()
            return z
        
    def classify(self,x):
        proba = self.encoder_y(x)
        return proba
    
def compute_class_proba(model, loader, N =3000,  batch=128):
    for i, (data, _)  in enumerate(loader):
        with torch.no_grad():
            data = data.to(dev, non_blocking=True)
            aux = model.encoder_y.forward(data)
            aux = aux.to("cpu")
            if i == 0:
                proba = np.zeros((N, aux.shape[1])).astype('float32')
            
            if i < len(loader) - 1:
                proba[i * batch: (i + 1) * batch] = aux
            else:
            # special treatment for final batch
                proba[i * batch:] = aux
    return proba

# clear param store
pyro.clear_param_store()

trainpath ="../data/alldata_png/"
supervised_path = "../data/annotated/"
holdout1 = pd.read_csv("../data/biofilm_annotations-export.csv")
holdout2 = pd.read_csv("../data/biofilm-set-2-export.csv")
holdout = pd.concat((holdout1,holdout2))

holdout = holdout[["image","label"]]
holdout_paths = trainpath + "data/" + holdout["image"]
holdout_paths  =[".."+ i.split(".")[-2] +".png" for i in holdout_paths]
holdout_paths = [(i, 0) for i in holdout_paths]
train_labels= holdout["label"]
labelmap = { i:j for i,j in zip(np.unique(train_labels), range(len(np.unique(train_labels))))}
inv_labelmap = { j:i for i,j in zip(np.unique(train_labels), range(len(np.unique(train_labels))))}

labels = [labelmap[i] for i in train_labels]


transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.ConvertImageDtype(torch.float),transforms.Resize(96)])

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.ConvertImageDtype(torch.float),transforms.Resize(96)])


unsupervised_dataset = datasets.ImageFolder(trainpath, transform=transform)
supervised_data = datasets.ImageFolder(supervised_path, transform=transform)


# remove supervised samples from unsupervised set
for i in holdout_paths:
    unsupervised_dataset.imgs.remove(i)


# setup the VAE


# setup visdom for visualization
print("Starting optimization",flush=True)

accs = []
y_scores = []
y_true = []
y_pred = []
for train_idx, test_idx in kf.split(supervised_data, supervised_data.targets):
    train_elbo = []
    test_elbo = []
   
    train_X = torch.utils.data.Subset(supervised_data, train_idx)
    test_X= torch.utils.data.Subset(supervised_data, test_idx)
    train_y = np.array(supervised_data.targets)[train_idx]
    test_y = np.array(supervised_data.targets)[test_idx]


    unsupervised_loader = torch.utils.data.DataLoader(unsupervised_dataset, batch_size=32, shuffle=False)
    train_loader = torch.utils.data.DataLoader(train_X, batch_size=32, shuffle=False)
    test_loader =  torch.utils.data.DataLoader(test_X, batch_size=32, shuffle=False)
    
    pyro.clear_param_store()
    vae = VAE(use_cuda=True)
    # training loop
    adam_args = {"lr": 1.0e-5}
    optimizer = Adam(adam_args)

    elbo = Trace_ELBO()
    svi = SVI(vae.model, config_enumerate(vae.guide), optimizer, loss=TraceEnum_ELBO(max_plate_nesting=1))

    svi_aux = SVI(vae.model_classify, config_enumerate(vae.guide_classify), optimizer, loss=TraceEnum_ELBO(max_plate_nesting=1))

    for epoch in range(150):
        # initialize loss accumulator
        epoch_loss = 0.0
        supervised_loss = 0.0
        aux_loss = 0.0
        # do a training epoch over each mini-batch x returned
        # by the data loader
        i = 0
        for x, _ in unsupervised_loader:
            
            # do ELBO gradient and accumulate loss
            epoch_loss += svi.step(x.cuda())
           # print(epoch_loss/((i+1)*128), flush=True)
            i += 1
        for x, y in train_loader:
            y = F.one_hot(y,num_classes=6)
            supervised_loss += svi.step(x.cuda(),y.cuda())
            aux_loss += svi_aux.step(x.cuda(),y.cuda())
        # report training diagnostics
        normalizer_train = len(unsupervised_loader.dataset)
        normalizer_supervised = len(train_loader.dataset)

        total_epoch_loss_train = epoch_loss / normalizer_train
        total_epoch_loss_val = supervised_loss / normalizer_supervised 
        train_elbo.append(total_epoch_loss_train)
        print(
            "[epoch %03d]  average training loss: %.4f \n average supervised loss: %.4f"
            % (epoch, total_epoch_loss_train,total_epoch_loss_val), flush=True
        )
        
    pred_proba = compute_class_proba(vae, test_loader, N = len(test_X), batch=32)
    y_scores.append(pred_proba)
    pred_class = np.argmax(pred_proba, axis=1)
    y_pred.append(pred_class)
    acc = accuracy_score(test_y,pred_class)
    accs.append(acc)
    y_true.append(test_y)
    print("Training accuracy:", acc)



y_scores_all = np.concatenate(y_scores)
y_true_all = np.concatenate(y_true)        
y_true_class = [inv_labelmap[i] for i in y_true_all]

y_pred_all = np.concatenate(y_pred)
y_pred_class = [inv_labelmap[i] for i in y_pred_all]


def plot_confmatrix(y_pred, y_true, title):
    classes = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.rcParams["figure.figsize"] = [20, 20]
    plt.rcParams["font.size"] = 26
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = ".2f"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.savefig(title, dpi=400)

    plt.show()
    plt.close()
    plt.rcdefaults()
    return 0

plot_confmatrix(y_pred_class, y_true_class, title="cm_ssvae_aux_150.png")



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



# fulldataset = datasets.ImageFolder(trainpath, transform=transform)


# full_loader = torch.utils.data.DataLoader(fulldataset, batch_size=128, shuffle=False)


# features = compute_features(vae,full_loader,  N=len(fulldataset), batch = 128)

# paths = [i[0] for i in train_loader.dataset.samples]
# names = [Path(i).stem for i in paths]

# out = pd.DataFrame(features, index=names)
# out.to_csv("ssvae_features.csv")






# train_loader = torch.utils.data.DataLoader(fulldataset, batch_size=128, shuffle=False)


# features_params = compute_features(vae,train_loader,  N=len(fulldataset), batch = 128)
# #features_sampled = compute_features(vae,train_loader,  N=len(traindataset), batch = 128, only_params=False)

# paths = [i[0] for i in train_loader.dataset.samples]
# names = [Path(i).stem for i in paths]

# out = pd.DataFrame(features_params, index=names)
# out.to_csv("vae_png.csv")


# # train_loader = torch.utils.data.DataLoader(traindataset, batch_size=len(traindataset), shuffle=False)
# # images,_ = next(iter(train_loader))



# # pca = PCA(n_components =10)
# scaler = StandardScaler()
# data_sc = scaler.fit_transform(out)
# # x = pca.fit_transform(data_sc)

# tsne = TSNE(n_components=2, n_jobs=8)
# x_tsne= tsne.fit_transform(data_sc)

# out_tsne = pd.DataFrame(x_tsne, index=names)
# out.to_csv("tsne_png.csv")

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
