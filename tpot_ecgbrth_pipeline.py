import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('C:/Users/nrust/Downloads/hist_target3.csv', sep=',', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.8594268476621417
exported_pipeline = DecisionTreeClassifier(criterion="gini", max_depth=1, min_samples_leaf=1, min_samples_split=10)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

confusion_matrix(results, testing_target)
target_names = ['normal', 'hypo']
print(classification_report(testing_target, results, target_names=target_names))