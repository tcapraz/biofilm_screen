import pandas as pd
import numpy as np
from pathlib import Path
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from skimage.io import imread
import matplotlib.pyplot as plt

datagen = ImageDataGenerator(rescale=1/255)

pred = pd.read_csv("y_pred_multi9.csv", index_col=0)
files = [Path(i).stem for i in pred.index]
pred.index = files
meta_data = pd.read_csv("../metadata/meta_data.csv")
meta_data.index = meta_data["filename"]

biofilm = pd.read_csv("biofilm_image_names.csv", header=None)
pred = pred.loc[biofilm[0]]
pred_class = np.argmax(pred.values, axis=1)

data_generator = datagen.flow_from_directory('../data/annotated2_clahe', classes=["discard",  "struct1",  "struct3", "struct4", "struct6",  "struct8", "struct9", "struct10", "struct11"],target_size=(224,224), color_mode="rgb", class_mode="input", batch_size=1, shuffle=False)
classes = data_generator.class_indices
classes = {v: k for k, v in classes.items()}
pred_class = pd.Series([classes[i] for i in pred_class], index=pred.index)


for i in np.unique(pred_class):
    f = pred_class.index[np.where(pred_class==i)]
    for j in range(5):
        im = imread("../data/all_data_bg_png/data/"+f[j]+".png")
        plt.imshow(im)
        plt.title(i)
        plt.show()
        plt.close()

meta_data_pred = meta_data

meta_data_pred["pred_biofilm"] = "non_structure"
meta_data_pred["pred_biofilm"].loc[pred_class.index] = pred_class
meta_data_pred.to_csv("structure_predictions9.csv", index=False)
