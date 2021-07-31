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
from sklearn.model_selection import StratifiedKFold


# plot a histogram of each variable in the dataset
def plot_variable_distributions(trainX):
    xaxis = None
    for i in range(trainX.shape[1]):
        ax = plt.subplot(trainX.shape[1], 1, i + 1, sharex=xaxis)
        ax.set_xlim(-1, 1)
        if i == 0:
            xaxis = ax
        plt.hist(trainX[:, i], bins=100)
    plt.show()


patient = '002'
df_1 = pd.read_csv("C:/Users/nrust/PycharmProjects/d1namo/pivot_%s_v4_pos.csv" % patient, header=None)
# print(df_1.describe())

df_2 = pd.read_csv("C:/Users/nrust/PycharmProjects/d1namo/pivot_%s_v4_neg.csv" % patient, header=None)
# print(df_1[174])
# df_1 = df_1[df_1[174] == 1]
# df_2 = df_2[df_2[174] == 0]
# print(len(df_1), len(df_2))
# breakpoint()

# df = df_2
df = pd.concat([df_1, df_2])

df_1 = df[df[174] == 1]
df_2 = df[df[174] == 0]
# print(len(df_1), len(df_2))

size_acc = min(len(df_1), len(df_2))

# shuffle the DataFrame rows
df_1 = df_1.sample(frac=1, random_state=42)
df_2 = df_2.sample(frac=1, random_state=42)
# UNCOMMENT
df = pd.concat([df_1[:size_acc], df_2[:size_acc]])
# df = pd.concat([df_1[:], df_2[:]])

df = df.dropna(axis=0)

# print(df.describe())
df0 = df[2:]

X = np.array(df0)
y = np.array(df0[174].astype(int))
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=1337)
kf.get_n_splits(X, y)

# print(kf)
acc_list, prc_list, sen_list = [], [], []
i = 0

for train_index, test_index in kf.split(X, y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    # print(list(test_index))
    a = df0.iloc[list(train_index)]
    i = i + 1
    # X_train, X_test = X[train_index], X[test_index]
    # y_train, y_test = y[train_index], y[test_index]

    df = a


    # df = df.dropna()

    # df = pd.read_csv("C:/Users/nrust/PycharmProjects/d1namo/pivot_%s_partial.csv" % patient, header=None)

    def plot_beat():
        i = 50
        plt.plot(df_1.iloc[i])
        plt.plot(df_2.iloc[i])
        plt.show()
        breakpoint()
        return


    df_train, df_test = df0.iloc[list(train_index)], df0.iloc[list(test_index)]
    # df_train, df_test = train_test_split(df, test_size=0.2, random_state=1337, stratify=df[174])
    print(len(df_train), len(df_test))

    Y = np.array(df_train[174].values).astype(np.int8)
    X = np.array(df_train[list(range(174))].values)[..., np.newaxis]

    Y_test = np.array(df_test[174].values).astype(np.int8)
    X_test = np.array(df_test[list(range(174))].values)[..., np.newaxis]


    def get_model():
        nclass = 1
        inp = Input(shape=(174, 1))
        img_1 = Convolution1D(32, kernel_size=5, activation=activations.relu)(inp)
        img_1 = Convolution1D(32, kernel_size=5, activation=activations.relu)(img_1)
        img_1 = MaxPool1D(pool_size=2)(img_1)
        img_1 = Dropout(rate=0.1)(img_1)
        img_1 = Convolution1D(64, kernel_size=3, activation=activations.relu)(img_1)
        img_1 = Convolution1D(64, kernel_size=3, activation=activations.relu)(img_1)
        img_1 = MaxPool1D(pool_size=2)(img_1)
        img_1 = Dropout(rate=0.1)(img_1)
        img_1 = Convolution1D(64, kernel_size=3, activation=activations.relu)(img_1)
        img_1 = Convolution1D(64, kernel_size=3, activation=activations.relu)(img_1)
        img_1 = MaxPool1D(pool_size=2)(img_1)
        img_1 = Dropout(rate=0.1)(img_1)
        img_1 = Convolution1D(512, kernel_size=3, activation=activations.relu)(img_1)
        img_1 = Convolution1D(512, kernel_size=3, activation=activations.relu)(img_1)
        img_1 = GlobalMaxPool1D()(img_1)
        img_1 = Dropout(rate=0.2)(img_1)

        dense_1 = Dense(64, activation=activations.relu, name="dense_1")(img_1)
        dense_1 = Dense(64, activation=activations.relu, name="dense_2")(dense_1)
        dense_1 = Dense(nclass, activation=activations.sigmoid, name="dense_3_ptbdb")(dense_1)

        model = models.Model(inputs=inp, outputs=dense_1)
        opt = optimizers.Adam(0.001)
        # opt = optimizers.SGD(lr=0.00051)

        model.compile(optimizer=opt, loss=losses.binary_crossentropy,
                      metrics='acc')  # tf.keras.metrics.Precision() 'acc'
        model.summary()
        return model


    model = get_model()
    file_path = "baseline_cnn%s__kfold%s.h5" % (patient, i)
    checkpoint = ModelCheckpoint(file_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')  # 'val_acc'
    early = EarlyStopping(monitor='val_acc', mode="max", patience=15, verbose=1)
    redonplat = ReduceLROnPlateau(monitor='val_acc', mode="max", patience=10, verbose=2)
    callbacks_list = [checkpoint, early, redonplat]  # early

    model.fit(X, Y, epochs=200, verbose=2, callbacks=callbacks_list, validation_split=0.2)
    model.load_weights(file_path)

    pred_train = model.predict(X)
    # pred_train = (pred_train>0.5).astype(np.int8)
    # print(pred_train)

    pred_test = model.predict(X_test)
    pred_test = (pred_test > 0.5).astype(np.int8)

    f1 = f1_score(Y_test, pred_test)

    print("Test f1 score : %s " % f1)

    acc = accuracy_score(Y_test, pred_test)

    print("Test accuracy score : %s " % acc)

    print('#######   Validation   #######')
    report_dnn_val = classification_report(Y_test, pred_test, output_dict=True)  # , target_names=['not hypo', 'hypo']
    CM_val = confusion_matrix(Y_test, pred_test)
    print(report_dnn_val)
    print(CM_val)

    acc = report_dnn_val.get('accuracy')
    prc = report_dnn_val.get('macro avg', {}).get('precision')
    sen = report_dnn_val.get('macro avg', {}).get('recall')
    acc_list = acc_list + [acc]
    prc_list = prc_list + [prc]
    sen_list = sen_list + [sen]
print('accuracy', acc_list)
print('precision', prc_list)
print('sensitivity', sen_list)

print(np.mean(acc_list), np.std(acc_list))
print(np.mean(prc_list), np.std(prc_list))
print(np.mean(sen_list), np.std(sen_list))