from keras.models import Sequential
from keras.layers import LSTM, Dense, AveragePooling2D, Masking
from keras.optimizers import Adam, SGD, RMSprop
from sklearn import preprocessing
from sklearn.preprocessing import LabelEncoder, OneHotEncoder
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_curve, auc
from keras import utils as np_utils
from scipy import interp
from itertools import cycle
import tensorflow as tf
from tensorflow.keras import layers
from keras.utils.vis_utils import plot_model
import sys
import seaborn as sn


data_dim = 75
num_classes = 3
timesteps = 180
epochs = 30
batch_size = 60
adam_lr = 0.0001
n_classes = 3

keypoints = pd.read_csv('keypoints_new.csv')
print('Keypoints dataframe shape = ', keypoints.shape)
#print(keypoints)

list_dfs = []
label_df = []
for v in keypoints['MovementID'].unique().tolist():
    new_df = keypoints[keypoints['MovementID'] == v].drop(columns=['Frame #','MovementID','Label']).as_matrix()
    list_dfs.append(new_df)
    #print(v)
    test_df = keypoints[keypoints['MovementID'] == v].iloc[:,76].iloc[0]
    label_df.append(test_df)
    #print(test_df)
    #print(list_dfs)
    #sys.exit()
    #print('MovementID =', v)

list_dfs = np.array(list_dfs)
label_df = np.array(label_df)
print(list_dfs.shape)
print(label_df.shape)

print('Amount of different videos: ', len(list_dfs))
print('Amount of frames for video 1: ', len(list_dfs[0]))
print('Amount of frames for video 2: ', len(list_dfs[1]))
print('etc...')

# Create a function called "chunks" with two arguments, l and n:
def chunks(l, n):
    # For item i in a range that is a length of l,
    for i in range(0, len(l), n):
        # Create an index range for l of n items:
        yield l[i:i+n]

#Cut the list_dfs into chunks based on timesteps
timestep_dfs = []
for i in range(len(list_dfs)):
    df_timesteps = list(chunks(list_dfs[i], timesteps))
    list_bla = [label_df[i] for _ in range(len(df_timesteps))]
    list_bla = np.array(list_bla)
    #print('Length of list_bla: ', len(list_bla)) - various based on movementID (4-10)
    #print('Length of list_dfs: ', len(list_dfs)) - 413
    #print(len(df_timesteps)
    for y in range(len(df_timesteps)):
        timestep_dfs.append(np.array([df_timesteps[y],list_bla[y]]))

timestep_dfs = np.array(timestep_dfs)
print('timestep_dfs[0]: ', timestep_dfs[0])
print('timestep_dfs[0] shape: ', timestep_dfs.shape)
    
print('Amount of different chunks: ', len(timestep_dfs))
print('Amount of timesteps: ',timesteps)
print('Amount of frames for video 1: ', len(list_dfs[0]))
#print('Amount of lists within first MovementID in chunks based on timesteps: ', len(timestep_dfs[0]))
print('Amount of data in first list first MovementID, should be = timesteps: ', len(timestep_dfs[0][0]))
print('Length of last list in a MovementID: ', len(timestep_dfs[0][-1]))

#print(timestep_dfs[0][0])

#Padding of zeros to the remaining list:
for k in range(len(timestep_dfs)):
    if len(timestep_dfs[k][0]) < timesteps:
        zeros_list = []
        zeros_list = [0] * (len(timestep_dfs[k][0][0]))
        #zeros_list.append(timestep_dfs[k][0][0][-1])
        zeros_list = np.array(zeros_list)
        #print(zeros_list)
        #print('Length of zeros_list: ', len(zeros_list))
        #print('Shape of zeros_list: ', zeros_list.shape)
        #print('Length of last timestep: ', len(timestep_dfs[k][0]))
        #print('Shape of last timestep: ', timestep_dfs[k][0].shape)
        for t in range(len(timestep_dfs[k][0]),timesteps):
            timestep_dfs[k][0] = np.vstack([timestep_dfs[k][0], zeros_list])
        #print(timestep_dfs[k][0].shape)

print('New length of last list in a MovementID: ', len(timestep_dfs[9][0]))
print('Test of random MovementID: ', timestep_dfs[100])
#print('Test of random MovementIDs last timestep shape: ', timestep_dfs[100][-1].shape

X = []
y = []
for p in range(len(timestep_dfs)):
    X.append(timestep_dfs[p][0])
    y.append(timestep_dfs[p][1])

#print(X)
print('Shape of X = ', np.array(X).shape)
#print(y)
print('Shape of y = ', np.array(y).shape)

#________________________________________

le = LabelEncoder()
y = le.fit_transform(y)
ohe = OneHotEncoder(categorical_features = [0])
y = ohe.fit_transform(y[:,None]).toarray()
#print(y)
print('Shape of y after One-Hot Encoder: ', y.shape)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=0, stratify=y)
print('X_train shape = ', np.array(X_train).shape)
print('X_test shape = ', np.array(X_test).shape)
print('y_train shape = ', np.array(y_train).shape)
print('y_test shape = ', np.array(y_test).shape)

X_train = np.array(X_train)
X_test = np.array(X_test)
#print(type(X_train))
#print(type(y_train))
#print(type(X_test))
#print(type(y_test))

y_train = np.reshape(y_train,(-1,3))
y_test = np.reshape(y_test,(-1,3))

print(y_train.shape)
print(y_test.shape)

# expected input data shape: (batch_size, timesteps, data_dim)
model = Sequential()
model.add(Masking(mask_value=0, input_shape=(timesteps, data_dim)))
model.add(LSTM(132, return_sequences=True))
model.add(LSTM(132, return_sequences=True))
model.add(LSTM(132, return_sequences=True))
model.add(LSTM(132, return_sequences=False))
model.add(Dense(64, activation='linear'))
model.add(Dense(3, activation='softmax'))

model.summary()

#plot_model(model, to_file='model_plot_3LSTM.png', show_shapes=True, show_layer_names=True)

model.compile(loss='categorical_crossentropy',
              optimizer = Adam(lr=adam_lr),
              metrics=['accuracy'])

history = model.fit(X_train, y_train,
                    batch_size, epochs=epochs,
                    validation_data=(X_test, y_test))

#Evaluation and such
y_pred = history.model.predict(X_test, verbose=0)
print(y_pred.shape)

#Reduce to 1D
#y_pred = y_pred[:, 0]
#print('Reduced to 1D: ', y_pred.shape)

print('y_test:')
print(y_test)
print('y_pred:')
print(y_pred)
#print(y_pred.shape)

y_pred_new = np.argmax(y_pred, axis=1)
y_test_new = np.argmax(y_test, axis=1)

print('y_test_new:')
print(y_test_new)
print('y_pred_new:')
print(y_pred_new)

#Accuracy
scores = model.evaluate(X_test, y_test, verbose=0)
print("Accuracy of the model: %.2f%%" % (scores[1]*100))

#Confution Matrix
labels = ['Fondu','Jete','Plie']
cm = confusion_matrix(y_test_new, y_pred_new)
print(cm)

fig1 = plt.figure()
sn.heatmap(cm, annot=True, cmap='BuPu', xticklabels=labels, yticklabels=labels, linewidth=0.5)
#plt.matshow(cm)
plt.title('Confusion Matrix')
#plt.colorbar()
plt.ylabel('True Label')
plt.xlabel('Predicated Label')
plt.savefig('Results/confusion_matrix_testTestLowTS.jpg')

recall = np.diag(cm) / np.sum(cm, axis=1)

print("Recall of the model: ")
print(recall)
print('Mean recall: %.2f%%' % np.mean(recall*100))

precision = np.diag(cm) / np.sum(cm, axis=0)
print("Precision of the model:")
print(precision)
print('Mean precision: ', np.mean(precision*100))


# list all data in history
#print(history.history.keys())
# summarize history for accuracy
fig2 = plt.figure()
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.ylim(ymax = 1, ymin = 0)
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Results/HistoryAccuracy_testTestLowTS.jpg')
# summarize history for loss
fig3 = plt.figure()
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.ylim(ymax = 1, ymin = 0)
plt.legend(['train', 'test'], loc='upper left')
plt.savefig('Results/HistoryLoss_testTestLowTS.jpg')

# AUC plot:
# Compute ROC curve and ROC area for each class

lw = 2

fpr = dict()
tpr = dict()
roc_auc = dict()
for i in range(n_classes):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_pred[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])

fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_pred.ravel())
roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])

all_fpr = np.unique(np.concatenate([fpr[i] for i in range(n_classes)]))

# Then interpolate all ROC curves at this points
mean_tpr = np.zeros_like(all_fpr)
for i in range(n_classes):
    mean_tpr += interp(all_fpr, fpr[i], tpr[i])

# Finally average it and compute AUC
mean_tpr /= n_classes

fpr["macro"] = all_fpr
tpr["macro"] = mean_tpr
roc_auc["macro"] = auc(fpr["macro"], tpr["macro"])

# Plot all ROC curves
fig3 = plt.figure()
plt.plot(fpr["micro"], tpr["micro"],
         label='micro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["micro"]),
         color='deeppink', linestyle=':', linewidth=4)

plt.plot(fpr["macro"], tpr["macro"],
         label='macro-average ROC curve (area = {0:0.2f})'
               ''.format(roc_auc["macro"]),
         color='navy', linestyle=':', linewidth=4)

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])
for i, color in zip(range(n_classes), colors):
    plt.plot(fpr[i], tpr[i], color=color, lw=lw,
             label='ROC curve of class {0} (area = {1:0.2f})'
             ''.format(i, roc_auc[i]))

plt.plot([0, 1], [0, 1], 'k--', lw=lw)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver operating characteristic')
plt.legend(loc="lower right")
plt.savefig('Results/AUC_testTestLowTS.jpg')