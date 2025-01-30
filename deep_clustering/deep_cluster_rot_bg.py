import numpy as np
import pandas as pd
np.random.seed(10)
from time import time
import numpy as np
import keras.backend as K
from tensorflow.keras.layers import Layer, InputSpec
from keras.models import Model
from sklearn.cluster import KMeans
#import metrics
from keras.layers import Input, Add, Dense, ELU, Input, Lambda, Flatten, Reshape, \
    Activation, ZeroPadding2D, BatchNormalization, Dropout, Flatten, Conv2D, \
        Convolution2D, UpSampling2D, MaxPooling2D, AveragePooling2D, GlobalMaxPooling2D,\
            Conv2DTranspose

from pathlib import Path
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt


from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, accuracy_score,normalized_mutual_info_score
import tensorflow as tf

import os
import sys

import numpy as np
from scipy.stats import norm
from sklearn import manifold
import matplotlib.pyplot as plt
from matplotlib.offsetbox import OffsetImage, AnnotationBbox
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas

from keras.optimizers import RMSprop

import numpy as np

from skimage.io import imread
from skimage.transform import resize

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE


sys.stdout.flush()

def autoencoderConv2D_3(input_shape=(64, 64, 3), filters=[32, 64, 128, 50], latent_dim=50):
    input_img = Input(shape=input_shape)
    if input_shape[0] % 8 == 0:
        pad3 = 'same'
    else:
        pad3 = 'valid'
    x = Conv2D(filters[0], 5, strides=2, padding='same', activation='relu', name='conv1', input_shape=input_shape)(input_img)
    x = BatchNormalization(axis=3, name='BN1')(x)
    x = Conv2D(filters[1], 5, strides=2, padding='same', activation='relu', name='conv2')(x)
    x = BatchNormalization(axis=3, name='BN2')(x)
    x = Conv2D(filters[2], 3, strides=2, padding=pad3, activation='relu', name='conv3')(x)
    x = BatchNormalization(axis=3, name='BN3')(x)
    
    x = Flatten()(x)
    encoded = Dense(units=filters[3], kernel_regularizer=tf.keras.regularizers.l2(1e-8), name='embedding')(x)
    x = Dense(units=filters[2]*int(input_shape[0]/8)*int(input_shape[0]/8), activation='relu')(encoded)

    x = Reshape((int(input_shape[0]/8), int(input_shape[0]/8), filters[2]))(x)
    x = Conv2DTranspose(filters[1], 3, strides=2, padding=pad3, activation='relu', name='deconv3')(x)
    x = BatchNormalization(axis=3, name='BN4')(x)
    x = Conv2DTranspose(filters[0], 5, strides=2, padding='same', activation='relu', name='deconv2')(x)
    x = BatchNormalization(axis=3, name='BN5')(x)
    
    decoded = Conv2DTranspose(input_shape[2], 5, strides=2, padding='same',kernel_regularizer=tf.keras.regularizers.l2(1e-8), name='deconv1')(x)
    return Model(inputs=input_img, outputs=decoded, name='AE'), Model(inputs=input_img, outputs=encoded, name='encoder')


class ClusteringLayer(Layer):
    """
    Clustering layer converts input sample (feature) to soft label.

    # Example
    ```
        model.add(ClusteringLayer(n_clusters=10))
    ```
    # Arguments
        n_clusters: number of clusters.
        weights: list of Numpy array with shape `(n_clusters, n_features)` witch represents the initial cluster centers.
        alpha: degrees of freedom parameter in Student's t-distribution. Default to 1.0.
    # Input shape
        2D tensor with shape: `(n_samples, n_features)`.
    # Output shape
        2D tensor with shape: `(n_samples, n_clusters)`.
    """

    def __init__(self, n_clusters, weights=None, alpha=1.0, **kwargs):
        if 'input_shape' not in kwargs and 'input_dim' in kwargs:
            kwargs['input_shape'] = (kwargs.pop('input_dim'),)
        super(ClusteringLayer, self).__init__(**kwargs)
        self.n_clusters = n_clusters
        self.alpha = alpha
        self.initial_weights = weights
        self.input_spec = InputSpec(ndim=2)

    def build(self, input_shape):
        assert len(input_shape) == 2
        input_dim = input_shape[1]
        self.input_spec = InputSpec(dtype=K.floatx(), shape=(None, input_dim))
        self.clusters = self.add_weight(shape=(self.n_clusters, input_dim), initializer='glorot_uniform', name='clusters')
        if self.initial_weights is not None:
            self.set_weights(self.initial_weights)
            del self.initial_weights
        self.built = True

    def call(self, inputs, **kwargs):
        """ student t-distribution, as same as used in t-SNE algorithm.        
                 q_ij = 1/(1+dist(x_i, µ_j)^2), then normalize it.
                 q_ij can be interpreted as the probability of assigning sample i to cluster j.
                 (i.e., a soft assignment)
        Arguments:
            inputs: the variable containing data, shape=(n_samples, n_features)
        Return:
            q: student's t-distribution, or soft labels for each sample. shape=(n_samples, n_clusters)
        """
        q = 1.0 / (1.0 + (K.sum(K.square(K.expand_dims(inputs, axis=1) - self.clusters), axis=2) / self.alpha))
        q **= (self.alpha + 1.0) / 2.0
        q = K.transpose(K.transpose(q) / K.sum(q, axis=1)) # Make sure each sample's 10 values add up to 1.
        return q

    def compute_output_shape(self, input_shape):
        assert input_shape and len(input_shape) == 2
        return input_shape[0], self.n_clusters

    def get_config(self):
        config = {'n_clusters': self.n_clusters}
        base_config = super(ClusteringLayer, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
    
def target_distribution(q):
    weight = q ** 2 / q.sum(0)
    return (weight.T / weight.sum(1)).T


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

 
#filepaths = []
#for root, dirs, f in os.walk("../data/"):
#    for name in f:
#        if name.endswith((".jpg")):
#            filepaths.append(root +"/"+ name)

filepaths = pd.read_csv("subset.csv", header=None)
filepaths = "../data/alldata_bg_subtract/data/" + filepaths

names = [Path(i).stem for i in filepaths.values.flatten().tolist()]


images = []
for i in filepaths.values.flatten().tolist():
    images.append(resize(imread(i), (64, 64)))

x = np.stack(images)

autoencoder, encoder = autoencoderConv2D_3()
pretrain_epochs = 100
batch_size = 64


batchSize = 64
nbBatch = int(x.shape[0]/batchSize)
batchTimes = [0. for i in range(5)]
startEpoch = 1
autoencoder.compile(optimizer=RMSprop(lr=0.01), loss="mse")
autoencoder.fit(x, x, batch_size=batch_size, epochs=pretrain_epochs)
#autoencoder.save_weights('/weights_biofilm1_100_RMS_cleaned.h5')

# X = x
# z = encoder.predict(X)


# tsne = TSNE(n_components=2, n_jobs=8)
# x_tsne = tsne.fit_transform(z)

# fig, ax = plt.subplots()
# imscatter(x_tsne [:, 0], x_tsne [:, 1], imageData=images, ax=ax, zoom=0.09)


n_clusters= 50

clustering_layer = ClusteringLayer(n_clusters, name='clustering')(encoder.output)
model = Model(inputs=encoder.input, outputs=clustering_layer)
model.compile(optimizer='adam', loss='kld')
kmeans = KMeans(n_clusters=n_clusters, n_init=20)
y_pred = kmeans.fit_predict(encoder.predict(x))

y_pred_last = np.copy(y_pred)
model.get_layer(name='clustering').set_weights([kmeans.cluster_centers_])

loss = 0
index = 0
maxiter = 328000
update_interval = 140
index_array = np.arange(x.shape[0])
tol = 0.001 # tolerance threshold to stop training
for ite in range(int(maxiter)):
    if ite % update_interval == 0:
        q = model.predict(x, verbose=0)
        p = target_distribution(q)  # update the auxiliary target distribution p

        # evaluate the clustering performance
        y_pred = q.argmax(1)

        # check stop criterion
        delta_label = np.sum(y_pred != y_pred_last).astype(np.float32) / y_pred.shape[0]
        y_pred_last = np.copy(y_pred)
        if ite > 0 and delta_label < tol:
            print('delta_label ', delta_label, '< tol ', tol)
            print('Reached tolerance threshold. Stopping training.')
            break
    idx = index_array[index * batch_size: min((index+1) * batch_size, x.shape[0])]
    loss = model.train_on_batch(x=x[idx], y=p[idx])
    index = index + 1 if (index + 1) * batch_size <= x.shape[0] else 0

    print(ite)
    
z = encoder.predict(x)

z = pd.DataFrame(z, index = names)

z.to_csv("deep_clustering_features50_subset_bg.csv", header=None)

cluster_labels = pd.DataFrame(y_pred, index=names)
cluster_labels.to_csv("deep_cluster_labels50_subset_bg.csv", header=None)
# tsne = TSNE(n_components=2, n_jobs=8)
# x_tsne = tsne.fit_transform(z)

# fig, ax = plt.subplots()
# imscatter(x_tsne [:, 0], x_tsne [:, 1], imageData=images, ax=ax, zoom=0.09)
