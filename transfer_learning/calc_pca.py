import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

data = pd.read_csv("embedding_full.csv", index_col=0)

ix = data.index
pca = PCA(n_components=50)
scaler = StandardScaler()
data = scaler.fit_transform(data)


pca.fit(data.T)
PCs= pca.components_.T

out  = pd.DataFrame(PCs, index = ix)

out.to_csv("embedding_full_pca.csv")