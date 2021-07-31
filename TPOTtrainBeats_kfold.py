from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.utils import resample
import torch
from tpot.config import classifier_config_nn
from tpot.builtins import nn as nn

from sklearn.impute import SimpleImputer

from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import SGD
from keras.utils import np_utils
from keras.wrappers.scikit_learn import KerasClassifier

from sklearn.metrics import confusion_matrix, classification_report
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import make_pipeline
from tpot.export_utils import set_param_recursive
import matplotlib.pylab as plt

import argparse

import tensorflow as tf
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.tree import export_graphviz


def scale_data(df):
    X = np.zeros((len(df.index), len(df.columns)))
    print(len(df.index), len(df.columns))
    X[:, :len(df.columns) - 2] = df.iloc[:, :len(df.columns) - 2]
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X[:, len(df.columns) - 1] = tpot_data.iloc[:, len(df.columns) - 1]
    dfX = pd.DataFrame(X, columns=tpot_data.columns.values)
    return dfX


tpot_data = pd.read_csv('C:/Users/nrust/Downloads/ECG_single_002_partial2.csv', sep=',',
                        usecols=[i for i in range(1, 15)],
                        dtype=np.float64, engine='python')

tpot_data['target'] = pd.read_csv('C:/Users/nrust/Downloads/ECG_single_002_partial2.csv', sep=',', usecols=['target'],
                                  dtype=np.float64, engine='python')
tpot_data = tpot_data.replace([np.inf, -np.inf], 0).dropna(axis=0)

# tpot_data = tpot_data.drop(columns=['T_amp', 'R_amp', 'P_amp'])

# ORIGINALLY USED SCALE_DATA FUNCTION
dfX = tpot_data

#df = tpot_data
#print(df.shape[1])
#xaxis = None
#for i in range(df.shape[1]):
    #ax = plt.subplot(df.shape[1], 1, i + 1, sharex=xaxis) #df.shape[1] for all features
    #ax.set_xlim(-1, 1)
    #if i == 0:
        #xaxis = ax
#    plt.hist(df.iloc[i], bins=100)
#    plt.show()
#breakpoint()

dfX = (dfX-dfX.min())/(dfX.max()-dfX.min())


#dfX['R_amp'] = (dfX['R_amp']-dfX['R_amp'].min())/(dfX['R_amp'].max()-dfX['R_amp'].min())
#dfX['P_amp'] = (dfX['P_amp']-dfX['P_amp'].min())/(dfX['P_amp'].max()-dfX['P_amp'].min())
#dfX['T_amp'] = (dfX['T_amp']-dfX['T_amp'].min())/(dfX['T_amp'].max()-dfX['T_amp'].min())

# print(dfX[dfX['target'] == 1])

# START getting balanced data
df_1 = dfX[dfX['target'] == 1]
df_2 = dfX[dfX['target'] == 0]

df_1.plot.box()
df_2.plot.box()


with pd.option_context('display.max_columns', 40):
    print(df_1.describe())
    print(df_2.describe())


# shuffle the DataFrame rows
df_1 = df_1.sample(frac=1, random_state=42)
df_2 = df_2.sample(frac=1, random_state=42)
df = pd.concat([df_1[:min(len(df_1), len(df_2))], df_2[:min(len(df_1), len(df_2))]])
#df = pd.concat([df_1, df_2])

print(len(df[df['target'] == 0]), len(df[df['target'] == 1]))


# END getting balanced data
df = df[['QTc','target']]

X = np.array(df)
y = np.array(df['target'].astype(int))
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)
kf.get_n_splits(X, y)

acc_list, prc_list, sen_list = [], [], []
i = 0

for train_index, test_index in kf.split(X, y):
    # print("TRAIN:", train_index, "TEST:", test_index)
    # print(list(test_index))
    Xi = df.iloc[list(train_index)]
    yi = pd.DataFrame(df['target'].iloc[list(train_index)], columns=['target'])

    Vi = df.iloc[list(test_index)]
    Vyi = pd.DataFrame(df['target'].iloc[list(test_index)], columns=['target'])

    i = i + 1
    # if i == 3:
    #    breakpoint()

    #X_train, X_val, y_train, y_val = Xi, Vi, yi, Vyi
    #train_test_split(Xi.drop(columns=['target']), yi, test_size=0.2, random_state=10, stratify=yi)

    #dfX2 = pd.concat([X_train, y_train], axis=1, sort=False)
    #validate = pd.concat([X_val, y_val], axis=1, sort=False)

    dfX2 = Xi
    validate = Vi

    X_train, X_test, y_train, y_test = train_test_split(dfX2.drop(columns=['target']), dfX2['target'], test_size=0.2,
                                                        stratify=dfX2['target'])

    train = pd.concat([X_train, y_train], axis=1, sort=False)
    test = pd.concat([X_test, y_test], axis=1, sort=False)

    # train, validate, test = np.split(dfX.sample(frac=1, random_state=42), [int(.75 * len(tpot_data)), int(.85 * len(tpot_data))])
    # print(train.shape[0], validate.shape[0], test.shape[0])

    # print(train[train['target'] == 1])

    # start treat undersampled data
    # concatenate our training data back together

    print('Before \n', train['target'].value_counts())

    # separate minority and majority classes
    not_fraud = train[train['target'] == 0]
    fraud = train[train['target'] == 1]
    # fraud2 = tpot_data[tpot_data['target'] == 2]

    # upsample minority
    fraud_upsampled = resample(fraud,
                               replace=True,  # sample with replacement
                               n_samples=len(not_fraud),  # match number in majority class
                               random_state=42)  # reproducible results

    # combine majority and upsampled minority
    upsampled = pd.concat([not_fraud, fraud_upsampled])
    # print(upsampled.describe())

    # check new class counts
    print('After \n', upsampled['target'].value_counts())

    # trying logistic regression again with the balanced dataset
    y_train = np.array(upsampled['target'])

    X_train = np.array(upsampled.iloc[:, :train.shape[1] - 1])

    #print('train data\n', X_train[1])

    # print(train['target'].value_counts())
    # end of undersampled

    X_test = np.array(test.iloc[:, :train.shape[1] - 1])
    y_test = np.array(test['target'])

    X_validate = np.array(validate.iloc[:, :train.shape[1] - 1])
    y_validate = np.array(validate['target'])

    imputer = SimpleImputer(strategy="median")
    imputer.fit(X_train)
    X_train = imputer.transform(X_train)
    X_test = imputer.transform(X_test)
    X_validate = imputer.transform(X_validate)

    # Average CV score on the training set was: 0.7010039899271261
    # tpot = make_pipeline(
    #    PCA(iterated_power=10, svd_solver="randomized"),
    #    RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.55, min_samples_leaf=1, min_samples_split=19, n_estimators=100)
    # )

    # LinearSVC(KNeighborsClassifier(
    #    RandomForestClassifier(PCA(input_matrix, iterated_power=1, svd_solver=randomized), bootstrap=False,
    #                           criterion=entropy, max_features=0.25, min_samples_leaf=8, min_samples_split=5,
    #                           n_estimators=100), n_neighbors=10, p=1, weights=distance), C=5.0, dual=True,
    #          loss=squared_hinge, penalty=l2, tol=0.001)

    # Fix random state for all the steps in exported pipeline

    # tpot = TPOTClassifier(verbosity=2, max_time_mins=2)

    tpot = RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.55, min_samples_leaf=1,
                                  min_samples_split=19, n_estimators=100)

    tpot.fit(X_train, y_train)

    tree = tpot.estimators_[5]
    features = Xi.drop(columns=['target'])

    # Export as dot file
    export_graphviz(tree,
                    out_file='tree002.dot',
                    feature_names=list(features.columns.values),
                    class_names=['normal', 'hypo'],
                    rounded=True, proportion=False,
                    precision=2, filled=True)

    print(tpot.score(X_test, y_test))

    # tpot.export('tpot_001_%s.py' % i)

    target_names = ['normal', 'hypo']  # , 'hyper'

    # inv_predictions = onehot_encoder.inverse_transform([np.argmax(predictions[:, 1])])
    # inv_predictions = label_encoder.inverse_transform([np.argmax(predictions[0, :])])

    results = tpot.predict(X_test)

    report_dnn = classification_report(y_test, results, target_names=target_names)
    CM = confusion_matrix(y_test, results)
    print(report_dnn)
    print(CM)
    print('#######   Validation   #######')
    validation = tpot.predict(X_validate)
    report_dnn_val = classification_report(y_validate, validation, target_names=target_names, output_dict=True)
    CM_val = confusion_matrix(y_validate, validation)
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
