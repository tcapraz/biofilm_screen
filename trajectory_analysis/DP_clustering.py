import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

import torch
import torch.nn.functional as F
from torch.distributions import constraints

import pyro
from pyro.distributions import *
from pyro.infer import Predictive, SVI, Trace_ELBO
from pyro.optim import Adam
import pyro.distributions as dist



def mix_weights(beta):
    beta1m_cumprod = (1 - beta).cumprod(-1)
    return F.pad(beta, (0, 1), value=1) * F.pad(beta1m_cumprod, (1, 0), value=1)


def model(data):
    p = data.shape[1]
    

    
    with pyro.plate("beta_plate", T-1):
        beta = pyro.sample("beta", Beta(1, alpha))

    with pyro.plate("mu_plate", T):
        mu = pyro.sample("mu", MultivariateNormal(torch.zeros(2), torch.eye(p)))
 
    with pyro.plate("data", N):
        z = pyro.sample("z", Categorical(mix_weights(beta)))
        pyro.sample("obs", MultivariateNormal(mu[z], torch.eye(2)), obs=data)


def guide(data):
    p = data.shape[1]

    kappa = pyro.param('kappa', lambda: Uniform(0, 2).sample([T-1]), constraint=constraints.positive)
    tau = pyro.param('tau', lambda: MultivariateNormal(torch.zeros(2), 3 * torch.eye(2)).sample([T]))
    
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
        loss = svi.step(data)
        losses.append(loss)

def truncate(alpha, centers,sigma, phi,weights):
    threshold = alpha**-1 / 100.
    true_centers = centers[weights > threshold]
    true_weights = weights[weights > threshold] / torch.sum(weights[weights > threshold])
    true_sigma = sigma[weights > threshold]
    true_phi = phi[:,weights > threshold]
    return true_centers, true_sigma,true_phi, true_weights

data = torch.cat((MultivariateNormal(-8 * torch.ones(2), torch.eye(2)).sample([50]),
                  MultivariateNormal(8 * torch.ones(2), torch.eye(2)).sample([50]),
                  MultivariateNormal(torch.tensor([-0.5, 1]), torch.eye(2)).sample([50])))

plt.scatter(data[:, 0], data[:, 1])
plt.title("Data Samples from Mixture of 3 Gaussians")
plt.show()
N = data.shape[0]

T = 6
optim = Adam({"lr": 0.05})
svi = SVI(model, guide, optim, loss=Trace_ELBO())
losses = []


pyro.clear_param_store()
alpha = 1
train(2000)
alpha=0.08
# We make a point-estimate of our model parameters using the posterior means of tau and phi for the centers and weights
Bayes_Centers_01, Bayes_sigma,bayes_phi, Bayes_Weights_01 = truncate(alpha, pyro.param("tau").detach(),pyro.param("sigma").detach(), pyro.param("phi").detach(),torch.mean(pyro.param("phi").detach(), dim=0))


cmap = {0:"blue", 1:"red", 2:"black", 3:"grey", 4:"green", 5:"lime"}
labels = np.argmax(bayes_phi.numpy(), axis=1)
labels = [cmap[i] for i in labels]
pyro.param("sigma").detach()
pyro.param("tau").detach()

plt.figure(figsize=(15, 5))
plt.subplot(1, 2, 1)
plt.scatter(data[:, 0], data[:, 1], c=labels, s=10)
plt.scatter(Bayes_Centers_01[:, 0], Bayes_Centers_01[:, 1], c=list(cmap.values())[0:3], s=100, marker="x")


# alpha = 1.5
# train(1000)

# # We make a point-estimate of our model parameters using the posterior means of tau and phi for the centers and weights
# Bayes_Centers_15, Bayes_Weights_15 = truncate(alpha, pyro.param("tau").detach(), torch.mean(pyro.param("phi").detach(), dim=0))


# plt.subplot(1, 2, 2)
# plt.scatter(data[:, 0], data[:, 1], color="blue")
# plt.scatter(Bayes_Centers_15[:, 0], Bayes_Centers_15[:, 1], color="red")
# plt.tight_layout()
# plt.show()