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


import argparse

import tensorflow as tf
from sklearn.neural_network import MLPClassifier

def scale_data(df):
    X = np.zeros((len(df.index), len(df.columns)))
    print(len(df.index), len(df.columns))
    X[:, :len(df.columns) - 2] = df.iloc[:, :len(df.columns) - 2]
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X[:, len(df.columns) - 1] = tpot_data.iloc[:, len(df.columns) - 1]
    dfX = pd.DataFrame(X, columns=tpot_data.columns.values)
    return dfX

tpot_data = pd.read_csv('C:/Users/nrust/Downloads/dataset_001.csv', sep=',', usecols=[i for i in range(1,64)],
                        dtype=np.float64, engine='python')
tpot_data = tpot_data.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
dfX = scale_data(tpot_data)

X_train, X_val, y_train, y_val = train_test_split(dfX.drop(columns=['target']), dfX['target'], test_size=0.2,
                                                  random_state=10, stratify=dfX['target'])

dfX2 = pd.concat([X_train, y_train], axis=1, sort=False)
validate = pd.concat([X_val, y_val], axis=1, sort=False)

X_train, X_test, y_train, y_test = train_test_split(dfX2.drop(columns=['target']), dfX2['target'], test_size=0.33,
                                                    stratify=dfX2['target'])

train = pd.concat([X_train, y_train], axis=1, sort=False)
test = pd.concat([X_test, y_test], axis=1, sort=False)

#train, validate, test = np.split(dfX.sample(frac=1, random_state=42), [int(.75 * len(tpot_data)), int(.85 * len(tpot_data))])
print(train.shape[0], validate.shape[0], test.shape[0])


# start treat undersampled data
# concatenate our training data back together

print('Before \n', train['target'].value_counts())

# separate minority and majority classes
not_fraud = train[train['target'] == 0]
fraud = train[train['target'] == 1]
#fraud2 = tpot_data[tpot_data['target'] == 2]

# upsample minority
fraud_upsampled = resample(fraud,
                           replace=True,  # sample with replacement
                           n_samples=len(not_fraud),  # match number in majority class
                           random_state=42)  # reproducible results


# combine majority and upsampled minority
upsampled = pd.concat([not_fraud, fraud_upsampled])

# check new class counts
print('After \n',upsampled['target'].value_counts())


# trying logistic regression again with the balanced dataset
y_train = np.array(upsampled['target'])
X_train = np.array(upsampled.iloc[:, :58])

#print(train['target'].value_counts())
# end of undersampled

X_test = np.array(test.iloc[:, :58])
y_test = np.array(test['target'])

X_validate = np.array(validate.iloc[:, :58])
y_validate = np.array(validate['target'])

imputer = SimpleImputer(strategy="median")
imputer.fit(X_train)
X_train = imputer.transform(X_train)
X_test = imputer.transform(X_test)
X_validate = imputer.transform(X_validate)

"""
# integer encode
label_encoder = LabelEncoder()
y_train = label_encoder.fit_transform(y_train)
y_test = label_encoder.fit_transform(y_test)
y_validate = label_encoder.fit_transform(y_validate)
"""
# binary encode
onehot_encoder = OneHotEncoder(sparse=False)
y_train = y_train.reshape(len(y_train), 1)
y_train = onehot_encoder.fit_transform(y_train)

y_test = y_test.reshape(len(y_test), 1)
y_test = onehot_encoder.fit_transform(y_test)

y_validate = y_validate.reshape(len(y_validate), 1)
y_validate = onehot_encoder.fit_transform(y_validate)



# fix random seed for reproducibility
seed = 42
np.random.seed(seed)

# Create DL model
model = Sequential()
model.add(Dense(70, input_dim=58, activation='relu'))
model.add(Dense(32, activation='relu'))
model.add(Dense(2, activation="sigmoid")) #"softmax"

# DNN model
model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.0051), metrics=[tf.keras.metrics.Precision()])
#binary_crossentropy
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=500, batch_size=50, shuffle=True)
predictions = model.predict(X_test, batch_size=80)
target_names = ['normal', 'hypo'] #, 'hyper'

# summarize the first 5 cases
#for i in range(5):
#    print('A => %d (expected %d)' % (predictions[i], y_test[i]))
print(len(predictions))

#inv_predictions = onehot_encoder.inverse_transform([np.argmax(predictions[:, 1])])
#inv_predictions = label_encoder.inverse_transform([np.argmax(predictions[0, :])])

report_dnn = classification_report(y_test.argmax(axis=1), predictions.argmax(axis=1), target_names=target_names)
CM = confusion_matrix(y_test.argmax(axis=1), predictions.argmax(axis=1))
print(report_dnn)
print(CM)
print('#######   Validation   #######')
validation = model.predict(X_validate, batch_size=80)
report_dnn_val = classification_report(y_validate.argmax(axis=1), validation.argmax(axis=1), target_names=target_names)
CM_val = confusion_matrix(y_validate.argmax(axis=1), validation.argmax(axis=1))
print(report_dnn_val)
print(CM_val)


#tpot = TPOTClassifier(generations=5, population_size=20, cv=5,
                                   # random_state=42, verbosity=2)
#tpot = PytorchMLPClassifier(verbose=True, num_epochs=10)
#tpot.fit(X_train, y_train)
#print(tpot.score(X_test, y_test))

#tpot.export('tpot_updateNN5001_pipeline.py')

"""
GradientBoostingClassifier(input_matrix, learning_rate=1.0, max_depth=7, max_features=0.25, min_samples_leaf=14, min_samples_split=20, n_estimators=100, subsample=0.55)
DecisionTreeClassifier(MaxAbsScaler(XGBClassifier(input_matrix, learning_rate=0.01, max_depth=3, min_child_weight=2, n_estimators=100, n_jobs=1, subsample=0.9500000000000001, verbosity=0)), criterion=entropy, max_depth=6, min_samples_leaf=6, min_samples_split=9)
XGBClassifier(FastICA(input_matrix, tol=0.75), learning_rate=0.5, max_depth=8, min_child_weight=7, n_estimators=100, n_jobs=1, subsample=0.6500000000000001, verbosity=0)
GradientBoostingClassifier(PolynomialFeatures(input_matrix, degree=2, include_bias=False, interaction_only=False), learning_rate=0.5, max_depth=5, max_features=0.05, min_samples_leaf=1, min_samples_split=14, n_estimators=100, subsample=0.4)
GradientBoostingClassifier(LogisticRegression(PCA(input_matrix, iterated_power=6, svd_solver=randomized), C=5.0, dual=False, penalty=l2), learning_rate=0.1, max_depth=4, max_features=0.25, min_samples_leaf=10, min_samples_split=3, n_estimators=100, subsample=0.7000000000000001)
ExtraTreesClassifier(input_matrix, bootstrap=False, criterion=entropy, max_features=0.25, min_samples_leaf=4, min_samples_split=18, n_estimators=100)
MLPClassifier(StandardScaler(input_matrix), alpha=0.0001, learning_rate_init=0.001)


"""
