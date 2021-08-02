from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.utils import resample
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier

from sklearn.ensemble import GradientBoostingClassifier, ExtraTreesClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import make_pipeline
from tpot.export_utils import set_param_recursive
from sklearn.decomposition import FastICA
from sklearn.tree import export_graphviz
from sklearn import tree
from sklearn.metrics import confusion_matrix, classification_report

#iris = load_iris()
#iris.data[0:5], iris.target

tpot_data = pd.read_csv('C:/Users/nrust/Downloads/dataset_008.csv', sep=',', usecols=[i for i in range(1,64)],
                        dtype=np.float64, engine='python')

tpot_data = tpot_data.replace([np.inf, -np.inf], np.nan).dropna(axis=1)

fractions = np.array([0.8, 0.2])
# shuffle your input
tpot_data = tpot_data.sample(frac=1, random_state=1337)
# split into 3 parts
tpot_data, val_data = np.array_split(tpot_data, (fractions[:-1].cumsum() * len(tpot_data)).astype(int))

Xval = np.array(val_data.iloc[:, :val_data.shape[1]-1])
yval = np.array(val_data.iloc[:, val_data.shape[1]-1])
#print(yval)
#print(val_data.iloc[1, :val_data.shape[1]-1])


tpot_target = tpot_data['target']


X_train, X_test, y_train, y_test = train_test_split(np.array(tpot_data.iloc[:, :tpot_data.shape[1]-1]), np.array(tpot_data['target']),
                                                    train_size=0.70, test_size=0.30, random_state=3)

# start treat undersampled data
# concatenate our training data back together
X = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train, columns=['target'])], axis=1)

# separate minority and majority classes
not_hypo = X[X['target'] == 0]
hypo = X[X['target'] == 1]

# upsample minority
hypo_upsampled = resample(hypo,
                          replace=True, # sample with replacement
                          n_samples=len(not_hypo), # match number in majority class
                          random_state=27) # reproducible results


# combine majority and upsampled minority
upsampled = pd.concat([not_hypo, hypo_upsampled])

# check new class counts
#print(upsampled['target'].value_counts())


# trying logistic regression again with the balanced dataset
#y_train = np.array(upsampled['target'])
#X_train = np.array(upsampled.iloc[:, :upsampled.shape[1]-1])

#end of undersampled

#print(X_train.shape, X_test.shape, y_train.shape, y_test.shape)

tpot = TPOTClassifier(verbosity=2, generations=5, scoring='f1_weighted') # max_time_mins=2

#config_dict='TPOT cuML'

#tpot = make_pipeline(
#    FastICA(tol=0.2),
#    ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.8, min_samples_leaf=9,
#                         min_samples_split=17, n_estimators=100)
#)

tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))

tpot.export('tpot_TEMPORARY008_up.py')


# Classification report
target_names = ['not hypo', 'hypo']

results = tpot.predict(Xval)

report_dnn = classification_report(yval, results, target_names=target_names)
CM = confusion_matrix(yval, results)
print(report_dnn)
print(CM)


"""
features = tpot_data.drop(columns=['target'])

# Export tree as txt file
with open("fruit_classifier1.txt", "w") as f:
    f = tree.export_graphviz(tpot[1].estimators_[1],
                             feature_names=list(features.columns.values),
                             class_names=['not hypo', 'hypo'],
                             rounded=True, proportion=False,
                             filled=True,
                             out_file=f)

#
"""
