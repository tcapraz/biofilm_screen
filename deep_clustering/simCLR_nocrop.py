import pandas as pd
import torch
import torch.nn as nn
import numpy as np
import collections
import sys
from torch.functional import F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision.models as models
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import  silhouette_score, adjusted_rand_score

from PIL import Image

"""
Code adapted from https://github.com/wvangansbeke/Unsupervised-Classification.
(implementation of SCAN)
Method from https://arxiv.org/abs/2002.05709
"""
sys.stdout.flush()

print(torch.cuda.memory_summary(device=None, abbreviated=False))
print("Start!")

# simCLR transforms
transform = transforms.Compose([
            transforms.Resize(224),
            transforms.RandomHorizontalFlip(0.5),
            transforms.RandomApply([
                transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            ], p=1),
            transforms.RandomGrayscale(p=0.2),
            transforms.ToTensor(),
            transforms.ConvertImageDtype(torch.float)
            
        ])
transform_standard = transforms.Compose([transforms.ToTensor(), 
                                transforms.ConvertImageDtype(torch.float),transforms.Resize(224)])

transform_dict  = {"standard": transform_standard, "augment": transform}

on_gpu =True
class AugmentedDataset(Dataset):
    def __init__(self, dataset):
        super(AugmentedDataset, self).__init__()
        transform = dataset.transform
        dataset.transform = None
        self.dataset = dataset
        
        if isinstance(transform, dict):
            self.image_transform = transform['standard']
            self.augmentation_transform = transform['augment']

        else:
            self.image_transform = transform
            self.augmentation_transform = transform

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        sample,_ = self.dataset.samples[index]
        image = Image.open(sample)
        out = dict()
        out['image'] = self.image_transform(image)
        out['image_augmented'] = self.augmentation_transform(image)

        return out

if on_gpu:  
    dev = "cuda:0" 
else:  
    dev = "cpu"

trainpath ="../data/alldata_png"
#testpath ="../data/test"

traindataset = datasets.ImageFolder(trainpath, transform=transform_dict)
#testdataset = datasets.ImageFolder(testpath, transform=transform_standard)

augdataset = AugmentedDataset(traindataset)

train_loader = torch.utils.data.DataLoader(augdataset, batch_size=128, shuffle=True, pin_memory=True,num_workers=8, drop_last=True)
#test_loader = torch.utils.data.DataLoader(testdataset, batch_size=212, shuffle=True, pin_memory=True, drop_last=True)
# test plot images
# for i, batch in enumerate(train_loader):
#     images = batch['image']
#     images_augmented = batch['image_augmented']
#     plt.imshow(np.transpose(images_augmented.squeeze().numpy(), axes=[1,2,0]))
#     plt.show()
#     plt.close()
    
    
class SimCLRLoss(nn.Module):
    # Based on the implementation of SupContrast
    def __init__(self, temperature):
        super(SimCLRLoss, self).__init__()
        self.temperature = temperature

    
    def forward(self, features):
        """
        input:
            - features: hidden feature representation of shape [b, 2, dim]
        output:
            - loss: loss computed according to SimCLR 
        """

        b, n, dim = features.size()
        assert(n == 2)
        mask = torch.eye(b, dtype=torch.float32).to(dev, non_blocking=True)

        contrast_features = torch.cat(torch.unbind(features, dim=1), dim=0)
        anchor = features[:, 0]

        # Dot product
        dot_product = torch.matmul(anchor, contrast_features.T) / self.temperature
        
        # Log-sum trick for numerical stability
        logits_max, _ = torch.max(dot_product, dim=1, keepdim=True)
        logits = dot_product - logits_max.detach()

        mask = mask.repeat(1, 2)
        logits_mask = torch.scatter(torch.ones_like(mask), 1, torch.arange(b).view(-1, 1).to(dev, non_blocking=True), 0)
        mask = mask * logits_mask

        # Log-softmax
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))
        
        # Mean log-likelihood for positive
        loss = - ((mask * log_prob).sum(1) / mask.sum(1)).mean()

        return loss


class ContrastiveModel(nn.Module):
    def __init__(self, head='mlp', features_dim=128, hidden_dim=1000):
        super(ContrastiveModel, self).__init__()
        self.backbone = models.resnet50(pretrained=True)
        self.backbone = torch.nn.Sequential(*(list(self.backbone.children())[:-1]))
        # latent dim is 2048
        self.latent_dim = 2048
        self.hidden_dim = hidden_dim
        self.head = head
 
        if head == 'linear':
            self.contrastive_head = nn.Linear(self.latent_dim, features_dim)

        elif head == 'mlp':
            self.contrastive_head = nn.Sequential(
                    nn.Linear(self.latent_dim, self.hidden_dim),
                    nn.ReLU(), nn.Linear(self.hidden_dim, features_dim))
        
        else:
            raise ValueError('Invalid head {}'.format(head))

    def forward(self, x):
        features = self.contrastive_head(self.backbone(x).squeeze())
        features = F.normalize(features, dim = 1)
        return features
    
    def get_latent(self,x):
        latent = self.backbone(x).squeeze()
        return latent
    
model = ContrastiveModel()
model.to(dev)
criterion = SimCLRLoss(0.1)
criterion.to(dev)
print(torch.cuda.memory_summary(device=None, abbreviated=False))
optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.04,
    momentum= 0.9,
    nesterov=False,
    weight_decay=10 ** -5,
    )

epochs = 100

print("Starting simCLR optimization with ", epochs, "and batch size ", train_loader.batch_size)

for e in range(epochs):
    avg_loss = 0
    for i, batch in enumerate(train_loader):
        images = batch['image']
        images_augmented = batch['image_augmented']
        b, c, h, w = images.size()
        input_ = torch.cat([images.unsqueeze(1), images_augmented.unsqueeze(1)], dim=1)
        input_ = input_.view(-1, c, h, w) 
        input_ = input_.to(dev, non_blocking=True)
        #targets = batch['target']
        #print(torch.cuda.memory_summary(device=None, abbreviated=False))
        output = model(input_).view(b, 2, -1)
        loss = criterion(output)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        avg_loss += loss.item()
    avg_loss = avg_loss/len(traindataset)
    print("Epoch ", e, " Loss: ", avg_loss, flush=True)

def compute_features(model, train_loader, N =28476,  batch=128):
    for i, (data, _)  in enumerate(train_loader):
        with torch.no_grad():
            data = data.to(dev, non_blocking=True)
            aux = model(data)
            aux = aux.to("cpu")
            if i == 0:
                features = np.zeros((N, aux.shape[1])).astype('float32')

            if i < len(train_loader) - 1:
                features[i * batch: (i + 1) * batch] = aux
            else:
            # special treatment for final batch
                features[i * batch:] = aux
    return features

def compute_latent(model, train_loader, N =28476,  batch=128):
    for i, (data, _)  in enumerate(train_loader):
        with torch.no_grad():
            data = data.to(dev, non_blocking=True)
            aux = model.get_latent(data)
            aux = aux.to("cpu")
            if i == 0:
                features = np.zeros((N, aux.shape[1])).astype('float32')

            if i < len(train_loader) - 1:
                features[i * batch: (i + 1) * batch] = aux
            else:
            # special treatment for final batch
                features[i * batch:] = aux
    return features

traindataset = datasets.ImageFolder(trainpath, transform=transform_standard)

train_loader = torch.utils.data.DataLoader(traindataset, batch_size=128)
features = compute_features(model, train_loader, len(traindataset), train_loader.batch_size)
paths = [i[0] for i in train_loader.dataset.samples]
names = [Path(i).stem for i in paths]

out = pd.DataFrame(features, index=names)
out.to_csv("featuresSimCLR_nocrop.csv")


latent = compute_latent(model, train_loader, len(traindataset), train_loader.batch_size)


out = pd.DataFrame(latent, index=names)
out.to_csv("latentSimCLR_nocrop.csv")

