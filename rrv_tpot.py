from tpot import TPOTClassifier
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np

#iris = load_iris()
#iris.data[0:5], iris.target

tpot_data = pd.read_csv('C:/Users/nrust/Downloads/hist_target3.csv', sep=',', dtype=np.float64, engine='python')
print(tpot_data.columns[0:113])

X_train, X_test, y_train, y_test = train_test_split(np.array(tpot_data.iloc[:, :113]), np.array(tpot_data['target']),
                                                    train_size=0.75, test_size=0.25)
X_train.shape, X_test.shape, y_train.shape, y_test.shape


tpot = TPOTClassifier(verbosity=2, max_time_mins=2)
tpot.fit(X_train, y_train)
print(tpot.score(X_test, y_test))

tpot.export('tpot_ecgbrth_pipeline2.py')

