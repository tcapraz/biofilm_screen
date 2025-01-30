import pyro
import torch
import pyro.distributions as dist
from pyro.infer.autoguide import AutoDiagonalNormal, AutoNormal
from pyro.infer import SVI, Trace_ELBO, TraceEnum_ELBO, config_enumerate, infer_discrete
import pyro.poutine as poutine
from pyro.infer import Predictive
import pandas as pd
import numpy as np
import scipy.stats as stats
import matplotlib.pyplot as plt
import string
from itertools import product
from scipy.stats import poisson, norm
from pyro.infer import MCMC, NUTS
import torch.nn.functional as f
from pyro.optim import Adam
import os
from sklearn.datasets import make_blobs, make_circles
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

#from sklearn.preprocessing import StandardScaler

def pFA_model(data1,data2, num_factors=5, device=torch.device('cpu')):
    """
    Defines the probabilistic model of PCA (see above).
    """
    num_samples1, num_features1 = data1.shape
    num_samples2, num_features2 = data2.shape
    # num_samples3, num_features3 = data3.shape
    # num_samples, num_cond = data.shape
    # with pyro.plate('latent_dim', num_factors, dim=-1):
    #     with pyro.plate('coefficient_plate1', num_features1, dim=-2):
    #         lambdas1 = pyro.sample("lambdas1", dist.HalfCauchy(torch.ones(1)))
    #         tau1 = pyro.sample("tau1", dist.HalfCauchy(torch.ones(1)))
    #         #p1 = pyro.sample('p1', dist.Beta(0.5,0.5))
    #     with pyro.plate('coefficient_plate2', num_features2, dim=-2):
    #         #p2 = pyro.sample('p2', dist.Beta(0.5,0.5))
    #         lambdas2 = pyro.sample("lambdas2", dist.HalfCauchy(torch.ones(1)))
    #         tau2 = pyro.sample("tau2", dist.HalfCauchy(torch.ones(1)))
            
    
    data = torch.cat([data1.T,data2.T]).T

    σ = pyro.param("σ", torch.ones(1, device=device), constraint=pyro.distributions.constraints.positive)
    
    
    W1 = pyro.param("W1", torch.zeros((num_factors, num_features1), device=device))
    W2 = pyro.param("W2", torch.zeros((num_factors, num_features2), device=device))
    # W3 = pyro.param("W3", torch.zeros((num_factors, num_features3), device=device))
    
    # W = pyro.param("W", torch.zeros((num_factors, num_cond), device=device))
    # with pyro.plate("factors", num_factors):
    #     W1 = pyro.sample("W1", dist.Normal(torch.zeros(num_features1, device=device), torch.ones(num_features1, device=device)).to_event(1))
    #     W2 = pyro.sample("W2", dist.Normal(torch.zeros(num_features2, device=device), torch.ones(num_features2, device=device)).to_event(1))
        


    with pyro.plate("data", num_samples1):
        z = pyro.sample("z", dist.Normal(torch.zeros(num_factors, device=device), torch.ones(num_factors, device=device)).to_event(1))
        # print(z.shape)
        x1 = W1.T @ z.T
        x2 = W2.T @ z.T         
        # x3 = W3.T @ z.T 
        # x = (W.T @ z.T).T
        x = torch.cat([x1,x2]).T
        pyro.deterministic('mean', x)
        # print(x.shape)
        # print(data.shape)
        pyro.sample("obs", dist.Normal(x, σ).to_event(1), obs=data)

    
    
    
def pFA_guide(data1,data2,  num_factors=5, device=torch.device('cpu')):

    
    # num_samples1, num_features1 = data1.shape
    # num_samples2, num_features2 = data2.shape
    # num_samples3, num_features3 = data3.shape
    num_samples, num_cond = data1.shape

    # if num_factors is None:
    #     num_factors = num_features1
    
            
    z_loc = pyro.param('z_loc', torch.zeros((num_samples, num_factors), device=device))
    z_scale = pyro.param('z_scale', torch.ones((num_samples, num_factors), device=device), constraint=pyro.distributions.constraints.positive)
    
    # W1_loc = pyro.param('W_loc', torch.zeros( (num_factors,num_features1), device=device))
    # W1_scale = pyro.param('W_scale', torch.ones((num_factors,num_features1), device=device), constraint=pyro.distributions.constraints.positive)
    # W2_loc = pyro.param('W_loc', torch.zeros( (num_factors,num_features2), device=device))
    # W2_scale = pyro.param('W_scale', torch.ones((num_factors,num_features2), device=device), constraint=pyro.distributions.constraints.positive)
    
    # with pyro.plate("factors", num_factors):
    #     W1 = pyro.sample("W1", dist.Normal(W1_loc, W1_scale).to_event(1))
    #     W2 = pyro.sample("W2", dist.Normal(W2_loc, W2_scale).to_event(1))
        


    with pyro.plate("data", num_samples):    
        pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))



cluster1_1 = np.random.normal(loc=100.0, scale=1.0, size=20)
random1_1 = np.random.uniform(low=-3.0, high=3.0, size=20)
cluster1_2 = np.random.normal(loc=50.0, scale=1.0, size=20)
random1_2 = np.random.uniform(low=-3.0, high=3.0, size=20)
data1_1 = np.concatenate((cluster1_1,random1_1))
data1_2 = np.concatenate((cluster1_2,random1_2))

cluster2_1 = np.random.normal(loc=0.0, scale=1.0, size=20)
random2_1 = np.random.normal(loc=0.0, scale=100.0, size=20)
cluster2_2 = np.random.normal(loc=10.0, scale=1.0, size=20)
random2_2 = np.random.normal(loc=0.0, scale=100.0, size=20)
data2_1 = np.concatenate((cluster2_1,random2_1))
data2_2 = np.concatenate((cluster2_2,random2_2))

# cluster3 = np.random.normal(loc=6, scale=1.0, size=20)
# random3 = np.random.uniform(low=-3.0, high=3.0, size=20)
data3_1 = np.random.normal(loc=0.0, scale=100.0, size=40)
data3_2 =np.random.normal(loc=0.0, scale=100.0, size=40)

data4_1 = np.random.normal(loc=0.0, scale=100.0, size=40)
data4_2 =np.random.normal(loc=0.0, scale=100.0, size=40)

# cluster4 = np.random.normal(loc=8, scale=1.0, size=20)
# random4 = np.random.uniform(low=-3.0, high=3.0, size=20)
# data4 = np.concatenate((random4,cluster4))

# cluster5 = np.random.normal(loc=-2, scale=1.0, size=20)
# random5 = np.random.uniform(low=-3.0, high=3.0, size=20)
# data5 = np.concatenate((random5,cluster5))

# data6 = np.random.uniform(low=-3.0, high=3.0, size=40)
# data7 = np.random.uniform(low=-3.0, high=3.0, size=40)


# data = torch.tensor(np.vstack([data1,data2,data3,data4, data5, data6, data7]).T)
data1 = torch.tensor(np.vstack([data1_1,data2_1,data3_1,data4_1]).T)
data2 = torch.tensor(np.vstack([data1_2,data2_2,data3_2,data4_2]).T)
data = torch.cat((data1.T,data2.T)).T

label = np.zeros(data1.shape[0])
label[0:20] = 1

# dataA, labelA = make_blobs(n_samples=50,n_features= 1,centers=5, shuffle=False)
# #dataA, labelA = make_circles(factor=0.5, shuffle=False)
# plt.scatter(dataA[:,0], dataA[:,1] , c = labelA)
# plt.show()
# plt.close()
# #dataB, labelB = make_circles(factor=0.8, shuffle=False)
# dataB1, labelB1 = make_blobs(n_samples=20,n_features= 1,centers=2, shuffle=False)
# dataB2, labelB2 = make_blobs(n_samples=30,n_features= 1,centers=3, shuffle=True)
# dataB = np.concatenate((dataB1,dataB2))
# labelB = np.concatenate((labelB1,labelB2))

# plt.scatter(dataB[:,0], dataB[:,1] , c = labelA)
# plt.show()
# plt.close()
# dataC, labelC = make_blobs(n_samples=500,n_features= 2,centers=5, shuffle=True)
# plt.scatter(dataC[:,0], dataC[:,1] , c = labelA)
# plt.show()
# plt.close()
# data =np.concatenate([dataA.T, dataB.T,dataC.T])

pyro.clear_param_store()

adam_params = {"lr": 0.005, "betas": (0.95, 0.999)}
optimizer = Adam(adam_params)

svi = SVI(pFA_model, pFA_guide, optimizer, loss=Trace_ELBO())
# data1 = torch.tensor(dataA)
# data2 = torch.tensor(dataB)
# data3 = torch.tensor(dataC)


n_steps = 5000


# do gradient steps
for step in range(n_steps):
    loss = svi.step(data1,data2)
    if step%1000 == 0:
        print(loss)



predictive = Predictive(pFA_model, guide=pFA_guide, num_samples=1000,return_sites=("mean", "z"))
samples = predictive(data1,data2)
y_pred =   np.mean(samples["mean"].numpy(), axis=0)

plt.scatter(data, y_pred)
plt.show()
plt.close()


z = np.mean(samples["z"].numpy(), axis=0)


scaler = StandardScaler()
data = scaler.fit_transform(data)
pca= PCA(2)
PC = pca.fit_transform(data)

plt.scatter(PC[:,0], PC[:,1], c = label)
plt.show()
plt.close()

plt.scatter(z[:,0], z[:,1], c = label)
plt.show()
plt.close()


W =pyro.get_param_store()["W1"].detach().numpy()
