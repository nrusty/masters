import pandas as pd
import numpy as np
import tensorflow as tf
import matplotlib.pylab as plt

from keras import optimizers, losses, activations, models
from keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from keras.layers import Dense, Input, Dropout, Convolution1D, MaxPool1D, GlobalMaxPool1D, GlobalAveragePooling1D, \
    concatenate
from sklearn.metrics import accuracy_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from tensorflow import keras

patient = '001'
df_1 = pd.read_csv("C:/Users/nrust/PycharmProjects/d1namo/pivot_%s_pos.csv" % patient, header=None)

df_2 = pd.read_csv("C:/Users/nrust/PycharmProjects/d1namo/pivot_%s_neg.csv" % patient, header=None)
#print(df_1[174])
#df_1 = df_1[df_1[174] == 1]
#df_2 = df_2[df_2[174] == 0]
#print(len(df_1), len(df_2))
#breakpoint()
df = pd.concat([df_1, df_2])

df_1 = df[df[174] == 1]
df_2 = df[df[174] == 0]
print(len(df_1), len(df_2))

df = pd.concat([df_1, df_2[:2808]])
#df = df.dropna()
#df = df.iloc[1:]

#df = pd.read_csv("C:/Users/nrust/PycharmProjects/d1namo/pivot_%s_partial.csv" % patient, header=None)

def plot_beat():
    i = 50
    plt.plot(df_1.iloc[i])
    plt.plot(df_2.iloc[i])
    plt.show()
    breakpoint()
    return

df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[174])


Y = np.array(df_train[174].values).astype(np.int8)
X = np.array(df_train[list(range(174))].values)[..., np.newaxis]

Y_test = np.array(df_test[174].values).astype(np.int8)
X_test = np.array(df_test[list(range(174))].values)[..., np.newaxis]


def get_model():
    nclass = 1
    inp = Input(shape=(174, 1))
    img_1 = Convolution1D(32, kernel_size=5, activation=activations.relu, padding="valid")(inp)
    img_1 = Convolution1D(32, kernel_size=5, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(64, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(64, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(64, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(64, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = MaxPool1D(pool_size=2)(img_1)
    img_1 = Dropout(rate=0.1)(img_1)
    img_1 = Convolution1D(512, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = Convolution1D(512, kernel_size=3, activation=activations.relu, padding="valid")(img_1)
    img_1 = GlobalMaxPool1D()(img_1)
    img_1 = Dropout(rate=0.2)(img_1)

    dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
    dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
    dense_1 = Dense(nclass, activation=activations.sigmoid, name="dense_3_ptbdb")(dense_1)

    model = models.Model(inputs=inp, outputs=dense_1)
    opt = optimizers.Adam(0.001)
    #opt = optimizers.SGD(lr=0.00051)

    model.compile(optimizer=opt, loss=losses.binary_crossentropy, metrics='acc') #tf.keras.metrics.Precision()
    model.summary()
    return model

model = keras.models.load_model("baseline_cnn007_v2_ptbdb.h5")
file_path = "baseline_cnn007_v2_ptbdb.h5"
#checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
#early = EarlyStopping(monitor='val_acc', mode="max", patience=15, verbose=1)
#redonplat = ReduceLROnPlateau(monitor='val_acc', mode="max", patience=10, verbose=2)
#callbacks_list = [checkpoint, early, redonplat]  # early


#model.fit(X, Y, epochs=200, verbose=2, callbacks=callbacks_list, validation_split=0.1)
model.load_weights(file_path)

pred_train = model.predict(X)
#pred_train = (pred_train>0.5).astype(np.int8)
#print(pred_train)

pred_test = model.predict(X_test)
pred_test = (pred_test>0.5).astype(np.int8)

f1 = f1_score(Y_test, pred_test)

print("Test f1 score : %s "% f1)

acc = accuracy_score(Y_test, pred_test)

print("Test accuracy score : %s "% acc)

print('#######   Validation   #######')
report_dnn_val = classification_report(Y_test, pred_test) #, target_names=['not hypo', 'hypo']
CM_val = confusion_matrix(Y_test, pred_test)
print(report_dnn_val)
print(CM_val)