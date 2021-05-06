import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

# NOTE: Make sure that the outcome column is labeled 'target' in the data file
tpot_data = pd.read_csv('C:/Users/nrust/Downloads/hist3_s3422.csv', sep=',', dtype=np.float64)
features = tpot_data.drop('target', axis=1)
training_features, testing_features, training_target, testing_target = \
            train_test_split(features, tpot_data['target'], random_state=None)

# Average CV score on the training set was: 0.9610596781085317
exported_pipeline = make_pipeline(
    PCA(iterated_power=4, svd_solver="randomized"),
    GradientBoostingClassifier(learning_rate=0.1, max_depth=6, max_features=0.05, min_samples_leaf=11, min_samples_split=8, n_estimators=100, subsample=0.4)
)

exported_pipeline.fit(training_features, training_target)
results = exported_pipeline.predict(testing_features)

print(confusion_matrix(results, testing_target))
target_names = ['normal', 'hypo']
print(classification_report(testing_target, results, target_names=target_names))
