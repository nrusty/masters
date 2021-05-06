import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.kernel_approximation import RBFSampler
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.impute import SimpleImputer
from sklearn.utils import resample

from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

CM_t = 0
results_t = []
val_target_t = []

tpot_data = pd.read_csv('C:/Users/nrust/Downloads/ECGHRV_v2.csv', sep=',', dtype=np.float64, engine='python')
tpot_data = tpot_data.sample(frac=1)
tpot_target = tpot_data['target']
print(tpot_data.columns[0:62])
print(len(tpot_data))
train, validate, test = np.split(tpot_data.sample(frac=1), [int(.8 * len(tpot_data)), int(.9 * len(tpot_data))])
print(len(train))
print(len(validate))
print(len(test))

# start treat undersampled data
# concatenate our training data back together
X = train

# separate minority and majority classes
not_fraud = train[train['target']==0]
fraud = train[train['target']==1]
#fraud2 = tpot_data[tpot_data['target']==2]

# upsample minority
fraud_upsampled = resample(fraud,
                          replace=True, # sample with replacement
                          n_samples=len(not_fraud), # match number in majority class
                          random_state=27) # reproducible results


# combine majority and upsampled minority
upsampled = pd.concat([not_fraud, fraud_upsampled])

# check new class counts
print(upsampled['target'].value_counts())

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

# Average CV score on the training set was: 0.9865384615384617
exported_pipeline = make_pipeline(
    RBFSampler(gamma=0.25),
    RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.15000000000000002, min_samples_leaf=10, min_samples_split=3, n_estimators=100)
)

exported_pipeline.fit(training_features, y_train)
results = exported_pipeline.predict(testing_features)

CM = confusion_matrix(results, y_test)
print(CM)
target_names = ['normal', 'hypo']  # , 'hyper'
print(classification_report(y_test, results, target_names=target_names))
