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
from sklearn.preprocessing import StandardScaler

pyro.clear_param_store()


def ppca_model(data1,data2 , num_factors=2, device=torch.device('cpu')):
    """
    Defines the probabilistic model of PCA (see above).
    """
    num_samples1, num_features1 = data1.shape
    num_samples2, num_features2 = data2.shape

        
    data = torch.cat([data1.T,data2.T]).T

    σ = pyro.param("σ", torch.ones(1, device=device), constraint=pyro.distributions.constraints.positive)
    W1 = pyro.param("W1", torch.zeros((num_factors, num_features1), device=device))
    W2 = pyro.param("W2", torch.zeros((num_factors, num_features2), device=device))
    # with pyro.plate("factors", num_factors):
    #     W = pyro.sample("W", dist.Normal(torch.zeros(num_features, device=device), torch.ones(num_features, device=device)).to_event(1))
    


    with pyro.plate("data", num_samples1):
        z = pyro.sample("z", dist.Normal(torch.zeros(num_factors, device=device), torch.ones(num_factors, device=device)).to_event(1))
        # print(z.shape)
        x1 = W1.T @ z.T 
        x2 = W2.T @ z.T 
        x = torch.cat([x1,x2]).T
        pyro.deterministic('mean', x)
        # print(x.shape)
        # print(data.shape)
        pyro.sample("obs", dist.Normal(x, σ).to_event(1), obs=data)
       
    
    
    
def ppca_guide(data,data2,  batches=None, num_factors=2, fit_mu=False, device=torch.device('cpu')):
    """
    A guide defines the variational distributions for the latent variables in the model.
    In this case this is just z.
    """
    num_samples, num_features = data.shape
    if num_factors is None:
        num_factors = num_features
    
    z_loc = pyro.param('z_loc', torch.zeros((num_samples, num_factors), device=device))
    z_scale = pyro.param('z_scale', torch.ones((num_samples, num_factors), device=device), constraint=pyro.distributions.constraints.positive)
    

    # W_loc = pyro.param('W_loc', torch.zeros( num_features, device=device))
    # W_scale = pyro.param('W_scale', torch.ones(num_features, device=device), constraint=pyro.distributions.constraints.positive)
    
    # with pyro.plate("factors", num_factors):
    #     W = pyro.sample("W", dist.Normal(W_loc.T,  W_scale.T).to_event(1))

    with pyro.plate("data", num_samples):    
        pyro.sample("z", dist.Normal(z_loc, z_scale).to_event(1))
        

# sim data
# X = torch.normal(mean=torch.zeros((1000,2)))
# W = torch.normal(mean=torch.ones((2,50))*5)
# W_norm = f.normalize(abs(W), p=1, dim=1)


# # sim data
# W2 = torch.normal(mean=torch.zeros((2,50)))
# W_norm2 = f.normalize(abs(W2), p=1, dim=1)


data1_cluster1 = torch.normal(mean=torch.ones((200,50))*0)
data1_cluster2 = torch.normal(mean=torch.ones((200,50))*5)
data1_cluster3 = torch.normal(mean=torch.ones((200,50))*-2)
data1 = torch.cat([data1_cluster1,data1_cluster2,data1_cluster3])

f1 = torch.normal(torch.zeros(1))
f2 = torch.normal(torch.ones(1)*2)


labels = [0 for i in range(200)] + [1 for i in range(200)] + [2 for i in range(200)]

data2 = torch.cat([data1_cluster1*f1,data1_cluster2*f2, torch.normal(mean=torch.ones((200,50))*20)])
plt.scatter(data1.numpy()[:,0], data1.numpy()[:,1], c =labels)
plt.scatter(data2.numpy()[:,0], data2.numpy()[:,1], c = labels)

scaler = StandardScaler()
data1 = torch.tensor(scaler.fit_transform(data1))

data2 = torch.tensor(scaler.fit_transform(data2))

        
trace = poutine.trace(ppca_model).get_trace(data1,data2)
#trace.compute_log_prob()  # optional, but allows printing of log_prob shapes
print(trace.format_shapes())

Y = torch.cat([data1.T,data2.T])

adam_params = {"lr": 0.005, "betas": (0.95, 0.999)}
optimizer = Adam(adam_params)

svi = SVI(ppca_model, ppca_guide, optimizer, loss=Trace_ELBO())


n_steps = 5000


# do gradient steps
for step in range(n_steps):
    loss = svi.step(data1,data2)
    print(loss)



predictive = Predictive(ppca_model, guide=ppca_guide, num_samples=1000,return_sites=("mean", "z"))
samples = predictive(data1,data2)
y_pred =   np.mean(samples["mean"].numpy(), axis=0)
z_pred =   np.mean(samples["z"].numpy(), axis=0)


plt.scatter(Y.T.numpy()[:,40], y_pred[:,40], s =0.2, c = labels)
plt.show()
plt.close()

plt.scatter(z_pred[:,0], z_pred[:,1], s =0.2, c = labels)

