import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator, ZeroCount
from sklearn.preprocessing import FunctionTransformer
from copy import copy
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('C:/Users/nrust/Downloads/hist_target3.csv', sep=',', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.8515837104072398
exported_pipeline = make_pipeline(
    make_union(
        FunctionTransformer(copy),
        FunctionTransformer(copy)
    ),
    ZeroCount(),
    MLPClassifier(alpha=0.1, learning_rate_init=0.1)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

confusion_matrix(results, testing_target)
target_names = ['normal', 'hypo']
print(classification_report(testing_target, results, target_names=target_names))