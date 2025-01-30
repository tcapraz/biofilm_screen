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
from tensorflow.keras.applications import VGG19, InceptionV3
from sklearn.metrics import roc_curve, confusion_matrix, auc, precision_recall_curve, f1_score, roc_auc_score
from tensorflow.keras.applications.inception_v3 import preprocess_input

####### transfer learning ########
datagen = ImageDataGenerator(rescale=1/255)

data_generator = datagen.flow_from_directory('../data/annotated2_clahe', classes=["discard",  "struct1",  "struct3", "struct4", "struct5", "struct6", "struct7", "struct8", "struct9", "struct10", "struct11", "struct12"],target_size=(299,299), color_mode="rgb", class_mode="input", batch_size=1, shuffle=False)
data_list = []
batch_index = 0

while batch_index <= data_generator.batch_index:
    data = data_generator.next()
    data = preprocess_input(data[0].reshape(299,299,3))
    data_list.append(data)
    batch_index = batch_index + 1

# now, data_array is the numeric data of whole images
X = np.asarray(data_list)
y = data_generator.classes

kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

accs = []
y_scores = []
y_true = []
y_pred = []
for train, test in kf.split(X,y):
    #inp = keras.layers.Input(shape=(160, 160, 3), name='image_input')
    
    vgg_model = InceptionV3(weights='imagenet', include_top=True)
    vgg_model.trainable = False
    flat1 = layers.Flatten(name="fl_last")(vgg_model.layers[-2].output)
    x = keras.layers.Dense(128, activation='relu', name='fc1x')(flat1)
    x = keras.layers.Dense(64, activation='relu', name='fc2x')(x)
    outputs = keras.layers.Dense(12, activation='softmax', name='predictions')(x)
    #new_model = keras.models.Model(inputs=inp, outputs=x)
    model = keras.Model(inputs=vgg_model.inputs, outputs=outputs)
    model.compile(optimizer='adam',  loss=tf.keras.losses.SparseCategoricalCrossentropy(), metrics=['accuracy'])
    X_train = X[train,:,:,:]
    X_test =  X[test,:,:,:]
    y_train = y[train]
    y_test = y[test]
    model.fit(X_train,y_train, epochs=20)
    
    results = model.evaluate(X_test, y_test, batch_size=32)
    accs.append(results[1])
    
    y_score = model.predict(X_test)
    y_scores.append(y_score)
    y_true.append(y_test)
    y_classes = y_score.argmax(axis=-1)
    y_pred.append(y_classes)
        



y_scores_all = np.concatenate(y_scores)
y_true_all = np.concatenate(y_true)
# got 93% mean accuracy with 4% std 
# 109_Day1_P5_x11_y1 duplicate in M5
classes = data_generator.class_indices
classes = {v: k for k, v in classes.items()}
y_true_all = np.concatenate(y_true)
y_true_class = [classes[i] for i in y_true_all]

y_pred_all = np.concatenate(y_pred)
y_pred_class = [classes[i] for i in y_pred_all]

f1_score(y_true_class, y_pred_class, average="micro")
print("accuracy: ", np.mean(accs))

fpr_all = []
tpr_all = []
auc_all = []
prec_all = []
rec_all =[]
pr_auc_all=[]

for i in range(len(classes)):
    #y_scores = model.predict(X_test)
    y_class = y_true_all.copy()
    if i ==0:
        # need some other random number to convert class 0 to true class (class 1)
        y_class[y_class!=i] = 333
        y_class[y_class==i] = 1
        y_class[y_class==333] = 0
    else:
        y_class[y_class!=i] = 0
        y_class[y_class==i] = 1
    y_scores_class = y_scores_all[:,i]
    fpr, tpr,_ = roc_curve(y_class, y_scores_class)
    prec, rec,_ = precision_recall_curve(y_class, y_scores_class)
    rec_all.append(rec)
    prec_all.append(prec)
    auc_all.append(auc(fpr,tpr)) 
    pr_auc_all.append(auc(rec, prec))
    fpr_all.append(fpr)
    tpr_all.append(tpr)

for i in range(len(fpr_all)):
    plt.plot(fpr_all[i],tpr_all[i], label=classes[i]+" (AUC: "+ "{:0.2f}".format(auc_all[i]) + ")")
plt.xlabel("FPR")
plt.ylabel("TPR")
plt.legend()
plt.savefig("roc2_clahe_incep.png", dpi=400)
plt.show()
plt.close()


for i in range(len(fpr_all)):
    plt.plot(rec_all[i],prec_all[i], label=classes[i]+" (AUC: "+ "{:0.2f}".format(pr_auc_all[i]) + ")")
plt.ylabel("Precision")
plt.xlabel("Recall")
plt.legend()
plt.savefig("precision_recall2_clahe_incep.png", dpi=400)
plt.show()
plt.close()

def plot_confmatrix(y_pred, y_true, title):
    classes = np.unique(y_true)
    cm = confusion_matrix(y_true, y_pred, labels=classes)
    cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    plt.rcParams["figure.figsize"] = [20, 20]
    plt.rcParams["font.size"] = 26
    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    # We want to show all ticks...
    ax.set(
        xticks=np.arange(cm.shape[1]),
        yticks=np.arange(cm.shape[0]),
        # ... and label them with the respective list entries
        xticklabels=classes,
        yticklabels=classes,
        ylabel="True label",
        xlabel="Predicted label",
    )
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right", rotation_mode="anchor")
    fmt = ".2f"
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j,
                i,
                format(cm[i, j], fmt),
                ha="center",
                va="center",
                color="white" if cm[i, j] > thresh else "black",
            )
    plt.colorbar(im, fraction=0.046, pad=0.04)
    fig.tight_layout()
    plt.savefig(title, dpi=400)

    plt.show()
    plt.close()
    plt.rcdefaults()
    return 0

plot_confmatrix(y_pred_class, y_true_class, title="cm_transfer2_clahe_incep.png")
