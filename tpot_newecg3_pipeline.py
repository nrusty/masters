import numpy as np
import pandas as pd
from sklearn.decomposition import FastICA
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('C:/Users/nrust/Downloads/ECGHRV3.csv', sep=',', dtype=np.float64, engine='python')
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

imputer = SimpleImputer(strategy="median")
imputer.fit(training_features)
training_features = imputer.transform(training_features)
testing_features = imputer.transform(testing_features)

# Average CV score on the training set was: 0.9366057838660578
exported_pipeline = make_pipeline(
    FastICA(tol=0.0),
    ExtraTreesClassifier(bootstrap=False, criterion="gini", max_features=0.9500000000000001, min_samples_leaf=7, min_samples_split=10, n_estimators=100)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

CM = confusion_matrix(results, testing_target)
print(CM)
target_names = ['normal', 'hypo', 'hyper']
print(classification_report(testing_target, results, target_names=target_names))
