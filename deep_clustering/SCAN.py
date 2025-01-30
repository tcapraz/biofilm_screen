import torch
import torch.nn as nn
import numpy as np

from torch.functional import F
from torch.utils.data import Dataset
from torchvision import datasets, transforms
import torchvision.models as models
import matplotlib.pyplot as plt

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import  silhouette_score, adjusted_rand_score

from PIL import Image

import random

import PIL, PIL.ImageOps, PIL.ImageEnhance, PIL.ImageDraw
import numpy as np
import torch
from torchvision.transforms.transforms import Compose

random_mirror = True

def ShearX(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, v, 0, 0, 1, 0))

def ShearY(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, v, 1, 0))

def Identity(img, v):
    return img

def TranslateX(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[0]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateY(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    v = v * img.size[1]
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

def TranslateXAbs(img, v):
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, v, 0, 1, 0))

def TranslateYAbs(img, v):
    if random.random() > 0.5:
        v = -v
    return img.transform(img.size, PIL.Image.AFFINE, (1, 0, 0, 0, 1, v))

def Rotate(img, v):
    if random_mirror and random.random() > 0.5:
        v = -v
    return img.rotate(v)

def AutoContrast(img, _):
    return PIL.ImageOps.autocontrast(img)

def Invert(img, _):
    return PIL.ImageOps.invert(img)

def Equalize(img, _):
    return PIL.ImageOps.equalize(img)

def Solarize(img, v):
    return PIL.ImageOps.solarize(img, v)

def Posterize(img, v):
    v = int(v)
    return PIL.ImageOps.posterize(img, v)

def Contrast(img, v):
    return PIL.ImageEnhance.Contrast(img).enhance(v)

def Color(img, v):
    return PIL.ImageEnhance.Color(img).enhance(v)

def Brightness(img, v):
    return PIL.ImageEnhance.Brightness(img).enhance(v)

def Sharpness(img, v):
    return PIL.ImageEnhance.Sharpness(img).enhance(v)

def augment_list():
    l = [
        (Identity, 0, 1),  
        (AutoContrast, 0, 1),
        (Equalize, 0, 1), 
        (Rotate, -30, 30),
        (Solarize, 0, 256),
        (Color, 0.05, 0.95),
        (Contrast, 0.05, 0.95),
        (Brightness, 0.05, 0.95),
        (Sharpness, 0.05, 0.95),
        (ShearX, -0.1, 0.1),
        (TranslateX, -0.1, 0.1),
        (TranslateY, -0.1, 0.1),
        (Posterize, 4, 8),
        (ShearY, -0.1, 0.1),
    ]
    return l


augment_dict = {fn.__name__: (fn, v1, v2) for fn, v1, v2 in augment_list()}

class Augment:
    def __init__(self, n):
        self.n = n
        self.augment_list = augment_list()

    def __call__(self, img):
        ops = random.choices(self.augment_list, k=self.n)
        for op, minval, maxval in ops:
            val = (random.random()) * float(maxval - minval) + minval
            img = op(img, val)

        return img

def get_augment(name):
    return augment_dict[name]

def apply_augment(img, name, level):
    augment_fn, low, high = get_augment(name)
    return augment_fn(img.copy(), level * (high - low) + low)

class Cutout(object):
    def __init__(self, n_holes, length, random=False):
        self.n_holes = n_holes
        self.length = length
        self.random = random

    def __call__(self, img):
        h = img.size(1)
        w = img.size(2)
        length = random.randint(1, self.length)
        mask = np.ones((h, w), np.float32)

        for n in range(self.n_holes):
            y = np.random.randint(h)
            x = np.random.randint(w)

            y1 = np.clip(y - length // 2, 0, h)
            y2 = np.clip(y + length // 2, 0, h)
            x1 = np.clip(x - length // 2, 0, w)
            x2 = np.clip(x + length // 2, 0, w)

            mask[y1: y2, x1: x2] = 0.

        mask = torch.from_numpy(mask)
        mask = mask.expand_as(img)
        img = img * mask

        return img

class ClusteringModel(nn.Module):
    def __init__(self, nclusters=50):
        super(ClusteringModel, self).__init__()
        self.backbone = models.resnet18(pretrained=True)
        self.backbone_dim =1000
        self.cluster_head = nn.Linear(self.backbone_dim, nclusters) 

    def forward(self, x, forward_pass='default'):
        if forward_pass == 'default':
            features = self.backbone(x)
            out = self.cluster_head(features) 

        elif forward_pass == 'backbone':
            out = self.backbone(x)

        elif forward_pass == 'head':
            out = self.cluster_head(x)

        elif forward_pass == 'return_all':
            features = self.backbone(x)
            out = {'features': features, 'output': [cluster_head(features) for cluster_head in self.cluster_head]}
        
        else:
            raise ValueError('Invalid forward pass {}'.format(forward_pass))        

        return out

def entropy(x, input_as_probabilities):
    """ 
    Helper function to compute the entropy over the batch 
    input: batch w/ shape [b, num_classes]
    output: entropy value [is ideally -log(num_classes)]
    """

    if input_as_probabilities:
        x_ =  torch.clamp(x, min = 1e-8)
        b =  x_ * torch.log(x_)
    else:
        b = F.softmax(x, dim = 1) * F.log_softmax(x, dim = 1)

    if len(b.size()) == 2: # Sample-wise entropy
        return -b.sum(dim = 1).mean()
    elif len(b.size()) == 1: # Distribution-wise entropy
        return - b.sum()
    else:
        raise ValueError('Input tensor is %d-Dimensional' %(len(b.size())))


class SCANLoss(nn.Module):
    def __init__(self, entropy_weight = 2.0):
        super(SCANLoss, self).__init__()
        self.softmax = nn.Softmax(dim = 1)
        self.bce = nn.BCELoss()
        self.entropy_weight = entropy_weight # Default = 2.0

    def forward(self, anchors, neighbors):
        """
        input:
            - anchors: logits for anchor images w/ shape [b, num_classes]
            - neighbors: logits for neighbor images w/ shape [b, num_classes]
        output:
            - Loss
        """
        # Softmax
        b, n = anchors.size()
        anchors_prob = self.softmax(anchors)
        positives_prob = self.softmax(neighbors)
       
        # Similarity in output space
        similarity = torch.bmm(anchors_prob.view(b, 1, n), positives_prob.view(b, n, 1)).squeeze()
        ones = torch.ones_like(similarity)
        consistency_loss = self.bce(similarity, ones)
        
        # Entropy loss
        entropy_loss = entropy(torch.mean(anchors_prob, 0), input_as_probabilities = True)

        # Total loss
        total_loss = consistency_loss - self.entropy_weight * entropy_loss
        
        return total_loss, consistency_loss, entropy_loss




class NeighborsDataset(Dataset):
    def __init__(self, dataset, indices, transform, num_neighbors=20):
        super(NeighborsDataset, self).__init__()
       
        

        self.anchor_transform = transform
        self.neighbor_transform = transform
       
        self.dataset = dataset        
        self.indices = indices[:, :num_neighbors]
        assert(self.indices.shape[0] == len(self.dataset))

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        output = {}
        anchor = self.dataset.__getitem__(index)
        #image = Image.open(anchor)
        
        #target = anchor[0]
        
        neighbor_index = np.random.choice(self.indices[index], 1)[0]
        neighbor = self.dataset.__getitem__(neighbor_index)

        anchor = self.anchor_transform(anchor[0])
        neighbor = self.neighbor_transform(neighbor[0])

        output['anchor'] = anchor
        output['neighbor'] = neighbor
        output['possible_neighbors'] = torch.from_numpy(self.indices[index])
        #output['target'] = target
        
        return output
    
transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomResizedCrop(size=224,scale=[0.8,1.0]),
            Augment(n=4),
            transforms.ToTensor(),
            transforms.Normalize([0.4925, 0.3757, 0.3483],[0.1223, 0.1090, 0.0985]),
            Cutout(
                n_holes =1,
                length = 16,
                random = True)])

transform_standard = transforms.Compose([transforms.ToTensor(), 
                                transforms.ConvertImageDtype(torch.float),transforms.Resize(224)])
trainpath ="../autoencoder/data/train"
testpath ="../autoencoder/data/test"


tdataset = datasets.ImageFolder(trainpath)
testdataset = datasets.ImageFolder(testpath, transform=transform_standard)

indices = np.genfromtxt('kneighbors.csv', dtype=int)

traindataset = NeighborsDataset(tdataset, indices=indices, transform = transform)

train_loader = torch.utils.data.DataLoader(traindataset, batch_size=128, shuffle=True,pin_memory=True,drop_last=True)
test_loader = torch.utils.data.DataLoader(testdataset, batch_size=212, shuffle=True, pin_memory=True, drop_last=True)


  
model = ClusteringModel()
model.cuda()

criterion = SCANLoss(0.1)
criterion.cuda()

optimizer = torch.optim.SGD(
    model.parameters(),
    lr=0.04,
    momentum= 0.9,
    nesterov=False,
    weight_decay=10 ** -5,
    )

epochs = 30

for e in range(epochs):
    avg_loss = 0
    for i, batch in enumerate(train_loader):
        #print("loaded")
        # Forward pass
        anchors = batch['anchor'].cuda(non_blocking=True)
        neighbors = batch['neighbor'].cuda(non_blocking=True)

        anchors_output = model(anchors)
        neighbors_output = model(neighbors)    
        total_loss, consistency_loss, entropy_loss = criterion(anchors_output,
                                                                         neighbors_output)
        
        
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        avg_loss += total_loss.item()
    avg_loss = avg_loss/len(traindataset)
    print("Epoch ", e, " Loss: ", avg_loss, flush=True)

def compute_features(model, train_loader, N =28476,  batch=128):
    for i, (data, _)  in enumerate(train_loader):
        with torch.no_grad():
            aux = model(data, "backbone")
            if i == 0:
                features = np.zeros((N, aux.shape[1])).astype('float32')

            if i < len(train_loader) - 1:
                features[i * batch: (i + 1) * batch] = aux
            else:
            # special treatment for final batch
                features[i * batch:] = aux
    return features


model.to("cpu")

traindataset = datasets.ImageFolder(trainpath, transform=transform_standard)

train_loader = torch.utils.data.DataLoader(traindataset, batch_size=128)
features = compute_features(model, train_loader)
np.savetxt("featuresSCAN.csv", features, delimiter=",")

features_test = compute_features(model, test_loader, N=212, batch =212)
np.savetxt("featuresSCAN_test.csv", features_test, delimiter=",")

