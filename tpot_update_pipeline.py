import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
import torch

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

CM_t = 0
results_t = []
val_target_t = []

tpot_data = pd.read_csv('C:/Users/nrust/Downloads/ECGHRV.csv', sep=',', dtype=np.float64, engine='python')
features = tpot_data.drop('target', axis=1)

#training_features, testing_features, training_target, testing_target = \
            #train_test_split(features, tpot_data['target'], random_state=None)
train, validate, test = np.split(tpot_data.sample(frac=1), [int(.8 * len(tpot_data)), int(.8 * len(tpot_data))])
X = train

# separate minority and majority classes
not_fraud = train[train['target']==0]
fraud = train[train['target']==1]

print('Normal samples: ', len(not_fraud))
print('Hypo samples: ', len(fraud))

# upsample minority
fraud_upsampled = resample(fraud,
                          replace=True, # sample with replacement
                          n_samples=len(not_fraud), # match number in majority class
                          random_state=27) # reproducible results


# combine majority and upsampled minority
upsampled = pd.concat([not_fraud, fraud_upsampled])

# check new class counts
#print(upsampled['target'].value_counts())

# trying logistic regression again with the balanced dataset
y_train = np.array(upsampled['target'])
X_train = np.array(upsampled.iloc[:, :62])

#end of undersampled

X_test = test.iloc[:, :62]
#print(X_test.describe())
y_test = test['target']
#print(y_test.describe())

X_train.shape, X_test.shape, y_train.shape, y_test.shape

imputer = SimpleImputer(strategy="median")
imputer.fit(X_train)
training_features = imputer.transform(X_train)
testing_features = imputer.transform(X_test)

# Average CV score on the training set was: 0.9903846153846153
exported_pipeline = GradientBoostingClassifier(learning_rate=0.5, max_depth=8, max_features=0.3, min_samples_leaf=11, min_samples_split=19, n_estimators=100, subsample=0.9000000000000001)

exported_pipeline.fit(training_features, y_train)

results = exported_pipeline.predict(testing_features)

CM = confusion_matrix(results, y_test)
print(CM)

target_names = ['normal', 'hypo']  # , 'hyper'

print(classification_report(y_test, results, target_names=target_names))
