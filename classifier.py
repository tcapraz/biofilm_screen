import tensorflow.keras as keras
from tensorflow.keras import layers
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras import datasets, layers, models
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import  image_dataset_from_directory
from sklearn.model_selection import train_test_split
from tensorflow.keras.applications import VGG19
from sklearn.metrics import roc_curve

######## untrained cnn #########
# n_trials = 10
# accs = []
# for i in range(n_trials): 
    
#     model = models.Sequential()
#     model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(196, 196, 1)))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(layers.MaxPooling2D((2, 2)))
#     model.add(layers.Conv2D(64, (3, 3), activation='relu'))
#     model.add(layers.Flatten())
#     model.add(layers.Dense(64, activation='relu'))
#     model.add(layers.Dense(3,activation="softmax"))
#     model.compile(optimizer='adam',
#                   loss=tf.keras.losses.SparseCategoricalCrossentropy(),
#                   metrics=['accuracy'])

#     datagen = ImageDataGenerator(rescale=1/255)
    
#     data_generator = datagen.flow_from_directory('./data/', classes=['M1', "M2", "M3", "M4", "M5"],target_size=(196,196), color_mode="grayscale", class_mode="input", batch_size=1, shuffle=False)
#     data_list = []
#     batch_index = 0
    
#     while batch_index <= data_generator.batch_index:
#         data = data_generator.next()
#         data_list.append(data[0].reshape(196,196,1))
#         batch_index = batch_index + 1
    
#     # now, data_array is the numeric data of whole images
#     X = np.asarray(data_list)
#     y = data_generator.classes
#     (X_train, X_test, y_train, y_test) = train_test_split(X, y, stratify=y)
    
#     model.fit(X_train,y_train, epochs=10)
    
#     results = model.evaluate(X_test, y_test, batch_size=1)
#     print("test loss, test acc:", results)
#     accs.append(results[1])

####### transfer learning ########
n_trials = 10
accs = []
for i in range(n_trials): 
        
    inp = keras.layers.Input(shape=(196, 196, 3), name='image_input')
    
    vgg_model = VGG19(input_tensor = inp,weights='imagenet', include_top=False)
    vgg_model.trainable = False
    flat1 = layers.Flatten()(vgg_model.layers[-1].output)
    x = keras.layers.Dense(128, activation='relu', name='fc1')(flat1)
    x = keras.layers.Dense(64, activation='relu', name='fc2')(x)
    outputs = keras.layers.Dense(5, activation='softmax', name='predictions')(x)
    #new_model = keras.models.Model(inputs=inp, outputs=x)
    model = keras.Model(inputs=vgg_model.inputs, outputs=outputs)
    model.compile(optimizer='adam',  loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    datagen = ImageDataGenerator(rescale=1/255)
        
    data_generator = datagen.flow_from_directory('./data/', classes=['M1', "M2", "M3", "M4", "M5"],target_size=(196,196), color_mode="rgb", class_mode="input", batch_size=1, shuffle=False)
    data_list = []
    batch_index = 0
    
    while batch_index <= data_generator.batch_index:
        data = data_generator.next()
        data_list.append(data[0].reshape(196,196,3))
        batch_index = batch_index + 1
    
    # now, data_array is the numeric data of whole images
    X = np.asarray(data_list)
    y = data_generator.classes
    (X_train, X_test, y_train, y_test) = train_test_split(X, y, stratify=y)
    
    model.fit(X_train,y_train, epochs=10)
    
    results = model.evaluate(X_test, y_test, batch_size=1)
    accs.append(results[1])

# got 93% mean accuracy with 4% std 
# 109_Day1_P5_x11_y1 duplicate in M5
classes = data_generator.class_indices
classes = {v: k for k, v in classes.items()}
fpr_all = []
tpr_all = []
for i in range(len(classes)):
    y_scores = model.predict(X_test)
    y_class = y_test.copy()
    if i ==0:
        # need some other random number to convert class 0 to true class (class 1)
        y_class[y_class!=i] = 333
        y_class[y_class==i] = 1
        y_class[y_class==333] = 0
    else:
        y_class[y_class!=i] = 0
        y_class[y_class==i] = 1
    y_scores_class = y_scores[:,i]
    fpr, tpr,_ = roc_curve(y_class, y_scores_class)
    fpr_all.append(fpr)
    tpr_all.append(tpr)

for i in range(len(fpr_all)):
    plt.plot(fpr_all[i],tpr_all[i], label=classes[i])
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.show()
plt.close()
