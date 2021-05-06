import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.tree import export_graphviz
from subprocess import call
# Display in jupyter notebook
# from IPython.display import Image


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

CM_t = 0
results_t = []
val_target_t = []

tpot_data = pd.read_csv('C:/Users/nrust/Downloads/ECGHRV_v2.csv', sep=',', dtype=np.float64, engine='python')
features = tpot_data.drop('target', axis=1)

for i in range(1, 3):
    # NOTE: Make sure that the outcome column is labeled 'target' in the data file
    print('Round', i)
    train, validate, test = np.split(tpot_data.sample(frac=1), [int(.6 * len(tpot_data)), int(.8 * len(tpot_data))])
    #print(train.describe())
    #print(test.describe())
    #print(validate.describe())

    training_features, testing_features, training_target, testing_target = \
        train_test_split(features[4:190], tpot_data['target'][4:190], random_state=None)

    #New type of split
    training_features = train.drop('target', axis=1)

    # check new class counts
    print('check original class counts')
    print(tpot_data['target'][4:190].value_counts())

    #print(testing_features.describe())
    # separate minority and majority classes
    not_fraud = tpot_data[tpot_data['target'] == 0]
    fraud = tpot_data[tpot_data['target'] == 1]
    #fraud2 = tpot_data[tpot_data['target']==2]

    # upsample minority
    fraud_upsampled = resample(fraud,
                               replace=True,  # sample with replacement
                               n_samples=len(not_fraud),  # match number in majority class
                               random_state=i)  # reproducible results 27

    # combine majority and upsampled minority
    upsampled = pd.concat([not_fraud, fraud_upsampled])

    # check new class counts
    print('check new class counts')
    print(upsampled['target'].value_counts())

    training_target = np.array(upsampled['target'])
    training_features = np.array(upsampled.iloc[:, :62])

    imputer = SimpleImputer(strategy="median")
    imputer.fit(training_features)
    training_features = imputer.transform(training_features)
    testing_features = imputer.transform(testing_features)

    # Average CV score on the training set was: 0.9739753376394598
    exported_pipeline = RandomForestClassifier(bootstrap=False, criterion="entropy", max_features=0.1,
                                               min_samples_leaf=1, min_samples_split=15, n_estimators=100)

    exported_pipeline.fit(training_features, training_target)

    tree = exported_pipeline.estimators_[5]

    # Export as dot file
    export_graphviz(tree,
                    out_file='tree.dot',
                    feature_names=list(features.columns.values),
                    class_names=['normal', 'hypo'],
                    rounded=True, proportion=False,
                    precision=2, filled=True)

    #call(['dot', '-Tpng', 'tree.dot', '-o', 'tree.png', '-Gdpi=600'])

    # Image(filename='tree.png')

    results = exported_pipeline.predict(testing_features)
    CM = confusion_matrix(results, testing_target)
    #CM_t = CM_t + CM
    #print(CM)
    target_names = ['normal', 'hypo']  # , 'hyper'
    print(classification_report(testing_target, results, target_names=target_names))

    print('\n Start of Validation\n')
    #val_features = imputer.fit(features[196:387])
    val_features = imputer.transform(features[196:380])
    val_target = np.array(tpot_data['target'][196:380])
    print('....value count....')
    print(tpot_data['target'][196:380].value_counts())
    # print(val_features)
    # print(val_target)
    results2 = exported_pipeline.predict(val_features)
    results_t = [*results_t, *results2]
    val_target_t = [*val_target_t, *val_target]
    CM = confusion_matrix(results2, val_target)
    CM_t = CM_t + CM
    print(CM)
    target_names = ['normal', 'hypo']  # , 'hyper'

    print(classification_report(val_target, results2, target_names=target_names))

    print('\n End of Validation\n')

print('\n Accumulated Validation\n')
print(CM_t)
print(classification_report(val_target_t, results_t, target_names=target_names))


print('\n Final test \n')

data_val = pd.read_csv('C:/Users/nrust/Downloads/val001.csv', sep=',', dtype=np.float64, engine='python')
features = data_val.drop('target', axis=1)
val_feat = np.array(features)
val_tgt = np.array(data_val['target'])
print(data_val['target'].value_counts())
# print(val_feat)
print(val_tgt)
results_final = exported_pipeline.predict(val_feat)
print(results_final)
CMv = confusion_matrix(results_final, val_tgt)
print(CMv)




