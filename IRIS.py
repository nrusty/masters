from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from sklearn.utils import resample

#iris = load_iris()
#iris.data[0:5], iris.target

tpot_data = pd.read_csv('C:/Users/nrust/Downloads/ECGHRV.csv', sep=',', dtype=np.float64, engine='python')
tpot_target = tpot_data['target']
#tpot_target = pd.read_csv('C:/Users/nrust/Downloads/glucose001target.csv', sep=',', dtype=np.float64, engine='python')
print(tpot_data.columns[0:62])
#tpot_data['target'] = tpot_target['target']
X_train, X_test, y_train, y_test = train_test_split(np.array(tpot_data.iloc[:, :62]), np.array(tpot_data['target']),
                                                    train_size=0.70, test_size=0.30, random_state=3)

print(y_test)
# start treat undersampled data
# concatenate our training data back together
X = pd.concat([pd.DataFrame(X_train), pd.DataFrame(y_train)], axis=1)

# separate minority and majority classes
not_fraud = tpot_data[tpot_data['target']==0]
fraud = tpot_data[tpot_data['target']==1]
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

X_train.shape, X_test.shape, y_train.shape, y_test.shape


tpot = TPOTClassifier(verbosity=2, max_time_mins=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))

tpot.export('tpot_newecg6_pipeline.py')

