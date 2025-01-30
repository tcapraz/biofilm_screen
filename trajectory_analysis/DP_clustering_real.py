
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
from sklearn.cluster import KMeans
import string
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
import matplotlib.pyplot as plt
import umap
from PIL import Image, ImageStat
import math
import sys
from itertools import product, combinations
import matplotlib.cm as cm
from sklearn.neighbors import NearestNeighbors, kneighbors_graph
from sknetwork.clustering import Louvain
from scanpy.pp import neighbors
import anndata as ad
import scipy
import scipy.stats as stats
from matplotlib import cm
import pyro
from pyro.distributions import *
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.optim import Adam
import pyro.distributions as dist
from tqdm import tqdm
import torch
import torch.nn.functional as F
from torch.distributions import constraints

def mix_weights(beta):
    beta1m_cumprod = (1 - beta).cumprod(-1)
    return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)

def model(data):
    p = data.shape[1]
    N = data.shape[0]

    with pyro.plate("beta_plate", T-1):
        beta = pyro.sample("beta", Beta(1, alpha))

    with pyro.plate("mu_plate", T):
        mu = pyro.sample("mu", MultivariateNormal(torch.zeros(p), torch.eye(p)))
 
    with pyro.plate("data", N):
        z = pyro.sample("z", Categorical(mix_weights(beta)))
        pyro.sample("obs", MultivariateNormal(mu[z], torch.eye(p)), obs=data)


def guide(data):
    p = data.shape[1]
    N = data.shape[0]
    kappa = pyro.param('kappa', lambda: Uniform(0, 2).sample([T-1]), constraint=constraints.positive)
    tau = pyro.param('tau', lambda: MultivariateNormal(torch.zeros(p), 3 * torch.eye(p)).sample([T]))
    
    phi = pyro.param('phi', lambda: Dirichlet(1/T * torch.ones(T)).sample([N]), constraint=constraints.simplex)
    sigma = pyro.param("sigma", torch.eye(p).unsqueeze(0).repeat(T,1,1),constraint=constraints.positive_definite  )


    
    with pyro.plate("beta_plate", T-1):
        q_beta = pyro.sample("beta", Beta(torch.ones(T-1), kappa))
      

    with pyro.plate("mu_plate", T):
        q_mu = pyro.sample("mu", MultivariateNormal(tau,sigma))
        
        
        
    with pyro.plate("data", N):
        z = pyro.sample("z", Categorical(phi))
        
def train(num_iterations):
    pyro.clear_param_store()
    for j in tqdm(range(num_iterations)):
        loss = svi.step(tdata)
        losses.append(loss)

def truncate(alpha, centers, weights):
    threshold = alpha**-1 / 100.
    true_centers = centers[weights > threshold]
    true_weights = weights[weights > threshold] / torch.sum(weights[weights > threshold])
    return true_centers, true_weights


data = pd.read_csv("embeddings/vae_giovanni_features_params_png10.csv", index_col=0)


meta_data = pd.read_csv("../metadata/meta_data.csv")
meta_data.index = meta_data["filename"]


data =  data.loc[:,~np.isnan(data.astype(float)).any(axis=0)]
day4 = meta_data[meta_data["day"] =="Day4"]
data_day4 = data.loc[day4.index,:]



meta_data = meta_data.loc[data.index,:]



conditions  = [np.where(day4["cond"] == i)[0] for i  in np.unique(day4["cond"])]

condition_data = []
strains = []
for i in conditions:
    cd = data_day4.iloc[i,:]
    cd.index = day4["genotype"][i]
    strains.append(day4["genotype"][i])
    condition_data.append(cd)

consensus_strains = set.intersection(*map(set,strains))
consensus_strains =  list(consensus_strains)
for i in range(len(condition_data)):
    cd = condition_data[i].loc[consensus_strains,:]
    condition_data[i]  =  cd[~cd.index.duplicated(keep='first')]
    
cube = np.stack(condition_data)

strain_data = np.hstack(condition_data)

tdata = torch.tensor(strain_data)

T = 20
optim = Adam({"lr": 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO())
losses = []


pyro.clear_param_store()
alpha = 0.1
train(1000)