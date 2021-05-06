import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from sklearn.svm import LinearSVC
from tpot.builtins import StackingEstimator
from sklearn.impute import SimpleImputer

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('C:/Users/nrust/Downloads/ECGHRV3.csv', sep=',', dtype=np.float64, engine='python')
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=42)

imputer = SimpleImputer(strategy="median")
imputer.fit(training_features)
training_features = imputer.transform(training_features)
testing_features = imputer.transform(testing_features)

# Average CV score on the training set was: 0.9877627715795654
exported_pipeline = make_pipeline(
    StackingEstimator(estimator=LinearSVC(C=25.0, dual=True, loss="hinge", penalty="l2", tol=0.001)),
    GradientBoostingClassifier(learning_rate=1.0, max_depth=7, max_features=0.6000000000000001, min_samples_leaf=8, min_samples_split=3, n_estimators=100, subsample=0.8500000000000001)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

confusion_matrix(results, testing_target)
target_names = ['normal', 'hypo', 'hyper']
print(classification_report(testing_target, results, target_names=target_names))
