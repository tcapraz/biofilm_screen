import torchvision.models as models

import torch
import torch.nn as nn
import numpy as np


from torchvision import datasets, transforms


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import  silhouette_score, adjusted_rand_score

on_gpu=True

if on_gpu:  
    dev = "cuda:0" 
else:  
    dev = "cpu"


class TunedModel(nn.Module):
    def __init__(self, pretrained_model, num_classes):
        super(TunedModel, self).__init__()
        self.pretrained = pretrained_model
        self.new_layers = nn.Sequential(nn.ReLU(),
                                           nn.Linear(1000, 128),
                                           nn.ReLU(),
                                           nn.Linear(128, 64))
        self.top_layer = nn.Sequential(nn.ReLU(), nn.Linear(64, num_classes), nn.Softmax(dim = 1))
    
    def forward(self, x):
        x = self.pretrained(x)
        x = self.new_layers(x)
        if self.top_layer:
            x = self.top_layer(x)
        return x
    

vgg16 = models.vgg16(pretrained=True)

for i in vgg16.features.parameters():
    i.requires_grad = False
    
model = TunedModel(vgg16, 50)

model.to(dev)


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



transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.ConvertImageDtype(torch.float),transforms.Resize(224),
								transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
                               ])



trainpath ="../data/train"
testpath ="../data/test"

traindataset = datasets.ImageFolder(trainpath, transform=transform)
testdataset = datasets.ImageFolder(testpath, transform=transform)

train_loader = torch.utils.data.DataLoader(traindataset, batch_size=128, shuffle=False)
test_loader = torch.utils.data.DataLoader(testdataset, batch_size=250, shuffle=False)
dataiter = iter(test_loader)
testimages, testlabels = dataiter.next()


num_classes=50



opt = torch.optim.SGD(
    filter(lambda x: x.requires_grad, model.parameters()),
    lr=0.05,
    momentum=0.9,
    weight_decay=10**-5,
)
criterion = nn.CrossEntropyLoss()


outer_epochs = 30
print("Starting training!")
for o_epoch in range(outer_epochs):
    
    model.top_layer = None
    # omit relu activation
    
    features_raw = compute_features(model,train_loader,  N=len(traindataset), batch = 128)
    print("Computed features!")

    pca = PCA(n_components=20)
    features = pca.fit_transform(features_raw)
    
    km = KMeans(n_clusters=50)
    labels_pred= km.fit_predict(features)
    print("Clustering finished!")

    sil_score = silhouette_score(features, labels_pred)
    print("Outer epoch ", o_epoch, " Silhouette score: ", sil_score, flush=True)
    
    label_dist = np.unique(labels_pred, return_counts=True)[1]/len(labels_pred) + 1e-10
    print("Outer epoch ", o_epoch, " label_dist: ", flush=True)
    print(label_dist, flush=True)

    entropy = -np.sum(label_dist* np.log10(label_dist))
    
    print("Outer epoch ", o_epoch, " Entropy: ", entropy, flush=True)
    
    # evaluate on test set with known labels
    test_features = compute_features(model, test_loader, N=len(testdataset), batch = 212)
    pca = PCA(n_components=20)
    pcs = pca.fit_transform(test_features)
    km_test = KMeans(n_clusters=9)
    labels_pred_test= km.fit_predict(pcs)
    ari = adjusted_rand_score(testlabels, labels_pred_test)
    print("Outer epoch ", o_epoch, " ARI: ", ari, flush=True)
    

    traindataset.targets = labels_pred
    traindataset.classes = list(set(labels_pred))
    traindataset.class_to_idx = {label: idx for idx, label in enumerate(set(labels_pred))}
    newimgs = []
    for i, l in zip(traindataset.imgs, labels_pred):
        newimgs.append((i[0], l))
    traindataset.samples = newimgs
    
    train_loader = torch.utils.data.DataLoader(traindataset, batch_size=128, shuffle=False)
    
    # set last fully connected layer
   
    model.top_layer = nn.Sequential(nn.ReLU(), nn.Linear(64, num_classes), nn.Softmax(dim = 1))
    
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=0.05,
        weight_decay=10 ** -5,
        )
    
    
    model.to(dev)

    train_loss=0
    for input_tensor, target in train_loader:
        input_tensor = input_tensor.to(dev, non_blocking=True)
        target = target.long().to(dev, non_blocking=True)
        output = model(input_tensor)
        loss = criterion(output, target)

        # compute gradient and do SGD step
        opt.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        optimizer_tl.step()
        train_loss += loss.item()*input_tensor.size(0)
                
    # print avg training statistics 
    train_loss = train_loss/len(train_loader)
    print("outer_epoch epoch ", o_epoch, " loss: ", train_loss, flush=True)
    
model.top_layer = None
  
features_raw = compute_features(model,train_loader,  N=len(traindataset), batch = 128)
pca = PCA(n_components=20)
features = pca.fit_transform(features_raw)

km = KMeans(n_clusters=50)
labels_pred= km.fit_predict(features)

sil_score = silhouette_score(features, labels_pred)
print("Outer epoch ", o_epoch, " Silhouette score: ", sil_score, flush=True)

label_dist = np.unique(labels_pred, return_counts=True)[1]/len(labels_pred) + 1e-10
print("Outer epoch ", o_epoch, " label_dist: ", flush=True)
print(label_dist, flush=True)

np.savetxt("features_pretrained_norm.csv", features_raw, delimiter=",")

testfeatures_raw = compute_features(model,test_loader,  N=len(testdataset), batch = 212)
pca = PCA(n_components=20)
features = pca.fit_transform(testfeatures_raw)

km = KMeans(n_clusters=9)
labels_pred= km.fit_predict(features)

np.savetxt("testfeatures_pretrained_norm.csv", testfeatures_raw, delimiter=",")



