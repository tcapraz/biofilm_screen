import torch
import torch.nn as nn
import numpy as np


from torchvision import datasets, transforms


from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import  silhouette_score, adjusted_rand_score

torch.set_num_threads(64)

class CNN(nn.Module):
    def __init__(self, in_channels=3, filters1=16,filters2=32, z_dim=64, num_classes=50):
        super(CNN, self).__init__()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.conv1 = nn.Conv2d(in_channels, filters1, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(filters1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(filters1, filters2, kernel_size=3, padding=2, dilation=2)
        self.bn2 = nn.BatchNorm2d(filters2)
        self.relu2 = nn.ReLU(inplace=True)
        self.dense1 = nn.Conv2d(filters2, filters2, kernel_size=1, padding=0)
        self.bn3 = nn.BatchNorm2d(filters2)
        self.relu3 = nn.ReLU(inplace=True)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        self.dense2 = nn.Conv2d(in_channels + filters2, in_channels + filters2, kernel_size=1, padding=0)
        self.bn4 = nn.BatchNorm2d(in_channels + filters2)
        self.relu4 = nn.ReLU(inplace=True)
        self.flatten = nn.Flatten()
        self.fc = nn.Linear(35*48*48, 128)
        self.relu5 = nn.ReLU(inplace=True)

        self.classifier = nn.Sequential(
            nn.Linear(128, z_dim),
            nn.ReLU(True),
        )
        self.top_layer = nn.Linear(z_dim, num_classes)
    
    def forward(self, x):
        y = self.pool1(x)
       # print(y.shape)
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu2(out)
        out = self.dense1(out)
        out = self.bn3(out)
        out = self.relu3(out)
        out = self.pool2(out)
        #print(out.shape)
        out = torch.cat((y, out), 1)
       # print(out.shape)
        out = self.dense2(out)
        out = self.bn4(out)
        out = self.relu4(out)
       # print(out.shape)
        out = self.flatten(out)
        out = self.fc(out)
        out = self.relu5(out)
        out = self.classifier(out)
        if self.top_layer:
            out = self.top_layer(out)
        return out

def compute_features(model, train_loader, N =28476,  batch=128):
    for i, (data, _)  in enumerate(train_loader):
        with torch.no_grad():
            aux = model(data)
            if i == 0:
                features = np.zeros((N, aux.shape[1])).astype('float32')

            if i < len(train_loader) - 1:
                features[i * batch: (i + 1) * batch] = aux
            else:
            # special treatment for final batch
                features[i * batch:] = aux
    return features



transform = transforms.Compose([transforms.ToTensor(), 
                                transforms.ConvertImageDtype(torch.float),transforms.Resize(96)
                               ])



trainpath ="../autoencoder/data/train"
testpath ="../autoencoder/data/test"


traindataset = datasets.ImageFolder(trainpath, transform=transform)
testdataset = datasets.ImageFolder(testpath, transform=transform)

train_loader = torch.utils.data.DataLoader(traindataset, batch_size=128, shuffle=False)
test_loader = torch.utils.data.DataLoader(testdataset, batch_size=250, shuffle=False)
dataiter = iter(test_loader)
testimages, testlabels = dataiter.next()

num_classes=50

model = CNN()

opt = torch.optim.SGD(
    filter(lambda x: x.requires_grad, model.parameters()),
    lr=0.05,
    momentum=0.9,
    weight_decay=10**-5,
)
criterion = nn.CrossEntropyLoss()

outer_epochs = 30
for o_epoch in range(outer_epochs):
    
    model.top_layer = None
    # omit relu activation
    model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
    
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

    entropy = -np.sum(label_dist* np.log10(label_dist))
    
    print("Outer epoch ", o_epoch, " Entropy: ", entropy, flush=True)
    
	# evaluate on test set with known labels
    test_features = compute_features(model, test_loader, N=len(testdataset), batch = 212)
    pca = PCA(n_components=20)
    pcs = pca.fit_transform(test_features)
    km_test = KMeans(n_clusters=5)
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
    mlp = list(model.classifier.children())
    mlp.append(nn.ReLU(inplace=True))
    model.classifier = nn.Sequential(*mlp)
    model.top_layer = nn.Sequential(nn.Linear(64, num_classes), nn.Softmax(dim = 1))
    
    optimizer_tl = torch.optim.SGD(
        model.top_layer.parameters(),
        lr=0.05,
        weight_decay=10 ** -5,
        )
    
    train_loss=0
    for input_tensor, target in train_loader:
        output = model(input_tensor)
        loss = criterion(output, target.long())
        # compute gradient and do SGD step
        opt.zero_grad()
        optimizer_tl.zero_grad()
        loss.backward()
        opt.step()
        optimizer_tl.step()
        train_loss += loss.item()*input_tensor.size(0)
                    
    # print avg training statistics 
    train_loss = train_loss/len(train_loader)
    print("Inner epoch ", o_epoch, " loss: ", train_loss, flush=True)

model.top_layer = None
      # omit relu activation
model.classifier = nn.Sequential(*list(model.classifier.children())[:-1])
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

np.savetxt("features.csv", features_raw, delimiter=",")

features_raw = compute_features(model,test_loader,  N=len(testdataset), batch = 212)

np.savetxt("testfeatures.csv", features_raw, delimiter=",")

