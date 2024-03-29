import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('C:/Users/nrust/Downloads/breathing_feat.csv', sep=',', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

imputer = SimpleImputer(strategy="median")
imputer.fit(training_features)
training_features = imputer.transform(training_features)
testing_features = imputer.transform(testing_features)

# Average CV score on the training set was: 1.0
exported_pipeline = GaussianNB()

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

confusion_matrix(results, testing_target)
target_names = ['normal', 'hypo']
print(classification_report(testing_target, results, target_names=target_names))

print(testing_target.iloc[:])
print(results)
