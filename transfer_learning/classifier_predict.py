import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import  image_dataset_from_directory
from sklearn.model_selection import train_test_split, StratifiedKFold
from tensorflow.keras.applications import VGG19
from sklearn.metrics import roc_curve, confusion_matrix, auc, precision_recall_curve, f1_score, roc_auc_score
import pandas as pd
from keras.applications.vgg19 import preprocess_input
####### transfer learning ########
datagen = ImageDataGenerator(rescale=1/255)

data_generator = datagen.flow_from_directory('../data/annotated', classes=["discard", "smooth", "structured1", "structured2", "structured3", "structured4"],target_size=(196,196), color_mode="rgb", class_mode="input", batch_size=1, shuffle=False)
data_list = []
batch_index = 0

while batch_index <= data_generator.batch_index:
    data = data_generator.next()
    data = preprocess_input(data[0])
    data_list.append(data.reshape(196,196,3))
    batch_index = batch_index + 1

# now, data_array is the numeric data of whole images
X = np.asarray(data_list)
y = data_generator.classes



# inp = keras.layers.Input(shape=(196, 196, 3), name='image_input')

# vgg_model = VGG19(input_tensor = inp,weights='imagenet', include_top=False)
# vgg_model.trainable = False
# flat1 = layers.Flatten()(vgg_model.layers[-1].output)
# x = keras.layers.Dense(128, activation='relu', name='fc1')(flat1)
# x = keras.layers.Dense(64, activation='relu', name='fc2')(x)
# outputs = keras.layers.Dense(6, activation='softmax', name='predictions')(x)
# #new_model = keras.models.Model(inputs=inp, outputs=x)
# model = keras.Model(inputs=vgg_model.inputs, outputs=outputs)
# model.compile(optimizer='adam',  loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])

# model.fit(X,y, epochs=10)
#model.save('VGG19_trained')
model = keras.models.load_model("VGG_model_bin")

test_generator = datagen.flow_from_directory('../data/alldata', classes=["good"],target_size=(196,196), color_mode="rgb", class_mode="input", batch_size=512, shuffle=False)
files = test_generator.filenames
y_pred_list = []
batch_index = 0

while batch_index <= test_generator.batch_index:
    data = test_generator.next()
    data = preprocess_input(data[0])
    y_pred_list.append(model.predict(data))
    batch_index = batch_index + 1

out = pd.DataFrame(np.vstack(y_pred_list), index=files)
out.to_csv("y_pred_bin.csv")

weights = model.get_layer('fc2').get_weights()
outputs = keras.layers.Dense(64, name='fc2')(model.layers[-3].output)
model = keras.Model(inputs=model.inputs, outputs=outputs)
model.get_layer("fc2").set_weights(weights)

test_generator = datagen.flow_from_directory('../data/alldata', classes=["good"],target_size=(196,196), color_mode="rgb", class_mode="input", batch_size=512, shuffle=False)
files = test_generator.filenames
y_pred_list = []
batch_index = 0

while batch_index <= test_generator.batch_index:
    data = test_generator.next()
    data = preprocess_input(data[0])
    y_pred_list.append(model.predict(data))
    batch_index = batch_index + 1

out = pd.DataFrame(np.vstack(y_pred_list), index=files)
out.to_csv("embedding_bin.csv")



inp = keras.layers.Input(shape=(196, 196, 3), name='image_input')

vgg_model = VGG19(input_tensor = inp,weights='imagenet', include_top=False)
vgg_model.trainable = False



test_generator = datagen.flow_from_directory('../data/alldata', classes=["good"],target_size=(196,196), color_mode="rgb", class_mode="input", batch_size=1, shuffle=False)
files = test_generator.filenames
y_pred_list = []
batch_index = 0

while batch_index <= test_generator.batch_index:
    data = test_generator.next()
    data = preprocess_input(data[0])
    y_pred_list.append(np.array(vgg_model.predict(data)).flatten())
    batch_index = batch_index + 1

out = pd.DataFrame(np.vstack(y_pred_list), index=files)
out.to_csv("embedding_full.csv")
