import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import  image_dataset_from_directory
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.applications import VGG19
from sklearn.metrics import roc_curve, confusion_matrix, auc, precision_recall_curve, f1_score, roc_auc_score
from keras.applications.vgg19 import preprocess_input
import pandas as pd
####### transfer learning ########
datagen = ImageDataGenerator(rescale=1/255)

data_generator = datagen.flow_from_directory('../data/annotated2_clahe', classes=["discard",  "struct1",  "struct3", "struct4", "struct6",  "struct8", "struct9", "struct10", "struct11"],target_size=(224,224), color_mode="rgb", class_mode="input", batch_size=1, shuffle=False)
data_list = []
batch_index = 0

while batch_index <= data_generator.batch_index:
    data = data_generator.next()
    data_list.append(data[0].reshape(224,224,3))
    batch_index = batch_index + 1

# now, data_array is the numeric data of whole images
X = np.asarray(data_list)
y = data_generator.classes

inp = keras.layers.Input(shape=(224, 224, 3), name='image_input')
vgg_model = VGG19(input_tensor = inp, weights='imagenet', include_top=False)
vgg_model.trainable = False
flat1 = layers.Flatten(name="fl_last")(vgg_model.layers[-1].output)
x = keras.layers.Dense(128, activation='relu', name='fc1x')(flat1)
x = keras.layers.Dense(64, activation='relu', name='fc2x')(x)
outputs = keras.layers.Dense(9, activation='softmax', name='predictionsx')(x)
model = keras.Model(inputs=vgg_model.inputs, outputs=outputs)
model.compile(optimizer='adam',  loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
model.fit(X,y, epochs=15)

model.save("VGG19_trained_multi9")


test_generator = datagen.flow_from_directory('../data/clahe_', classes=["data"],target_size=(224,224), color_mode="rgb", class_mode="input", batch_size=512, shuffle=False)
files = test_generator.filenames
y_pred_list = []
batch_index = 0

while batch_index <= test_generator.batch_index:
    data = test_generator.next()
    y_pred_list.append(model.predict(data[0]))
    batch_index = batch_index + 1

out = pd.DataFrame(np.vstack(y_pred_list), index=files)
out.to_csv("y_pred_multi9.csv")
