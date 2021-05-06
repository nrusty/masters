import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier
from sklearn.decomposition import FastICA
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from sklearn.impute import SimpleImputer
from sklearn.utils import resample
from sklearn.tree import export_graphviz
from subprocess import call
# Display in jupyter notebook
# from IPython.display import Image


from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report
#from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.pipeline import make_pipeline, make_union
from tpot.builtins import StackingEstimator
from tpot.export_utils import set_param_recursive

def scale_data(df):
    X = np.zeros((len(df.index), len(df.columns)))
    print(len(df.index), len(df.columns))
    X[:, :len(df.columns) - 2] = df.iloc[:, :len(df.columns) - 2]
    sc = StandardScaler()
    X = sc.fit_transform(X)
    X[:, len(df.columns) - 1] = tpot_data.iloc[:, len(df.columns) - 1]
    dfX = pd.DataFrame(X, columns=tpot_data.columns.values)
    return dfX

def upsamp_hypo(unbalanced):
    # separate minority and majority classes
    not_hypo = unbalanced[unbalanced['target'] == 0]
    hypo = unbalanced[unbalanced['target'] == 1]

    # upsample minority
    hypo_upsampled = resample(hypo,
                              replace=True,  # sample with replacement
                              n_samples=len(not_hypo),  # match number in majority class
                              random_state=i)  # reproducible results 27

    # combine majority and upsampled minority
    upsampled = pd.concat([not_hypo, hypo_upsampled])

    return upsampled

CM_t = 0
results_t = []
val_target_t = []

tpot_data = pd.read_csv('C:/Users/nrust/Downloads/dataset_001.csv', sep=',', usecols=[i for i in range(1,64)],
                        dtype=np.float64, engine='python')
#print(tpot_data.describe())
tpot_data = tpot_data.replace([np.inf, -np.inf], np.nan).dropna(axis=1)
#print(tpot_data.describe())
#breakpoint()
dfX = scale_data(tpot_data)

for i in range(1, 11):
    # NOTE: Make sure that the outcome column is labeled 'target' in the data file
    print('Round', i)
    #train, validate, test = np.split(dfX.sample(frac=1, random_state=9),
                                     #[int(.75 * len(tpot_data)), int(.85 * len(tpot_data))])

    X_train, X_val, y_train, y_val = train_test_split(dfX.drop(columns=['target']), dfX['target'], test_size=0.1)

    dfX = pd.concat([X_train, y_train], axis=1, sort=False)
    validate = pd.concat([X_val, y_val], axis=1, sort=False)

    X_train, X_test, y_train, y_test = train_test_split(dfX.drop(columns=['target']), dfX['target'], test_size=0.33)

    train = pd.concat([X_train, y_train], axis=1, sort=False)
    test = pd.concat([X_test, y_test], axis=1, sort=False)


    upsampled = upsamp_hypo(train)

    # check new class counts
    print('check new class counts')
    print(upsampled['target'].value_counts())

    training_target = np.array(upsampled['target'])
    training_features = np.array(upsampled.iloc[:, :58])

    testing_target = np.array(test['target'])
    testing_features = np.array(test.iloc[:, :58])

    imputer = SimpleImputer(strategy="median")
    imputer.fit(training_features)
    training_features = imputer.transform(training_features)
    testing_features = imputer.transform(testing_features)

    # Average CV score on the training set was: 1.0
    #exported_pipeline = GradientBoostingClassifier(learning_rate=1.0, max_depth=5, max_features=0.3, min_samples_leaf=8,
                                                   #min_samples_split=8, n_estimators=100, subsample=0.9000000000000001)
    exported_pipeline = make_pipeline(
        StackingEstimator(
            estimator=ExtraTreesClassifier(bootstrap=True, criterion="gini", max_features=0.5, min_samples_leaf=1,
                                           min_samples_split=6, n_estimators=100)),
        RandomForestClassifier(bootstrap=True, criterion="gini", max_features=0.2, min_samples_leaf=8,
                               min_samples_split=4, n_estimators=100)
    )

    #PytorchMLPClassifier
    exported_pipeline.fit(training_features, training_target)
    """
    
        # Average CV score on the training set was: 0.9636263736263736
    exported_pipeline = GradientBoostingClassifier(learning_rate=1.0, max_depth=5, max_features=0.7500000000000001,
                                                   min_samples_leaf=13, min_samples_split=11, n_estimators=100,
                                                   subsample=0.55)
                                                   
                                                   
    tree = exported_pipeline.estimators_[5]

    # Export as dot file
    export_graphviz(tree,
                    out_file='treen.dot',
                    feature_names=list(features.columns.values),
                    class_names=['normal', 'hypo'],
                    rounded=True, proportion=False,
                    precision=2, filled=True)

    # call(['dot', '-Tpng', 'treen.dot', '-o', 'treen.png', '-Gdpi=600'])

    # Image(filename='tree.png')
    """
    #print(testing_features[1])
    results = exported_pipeline.predict(testing_features)
    CM = confusion_matrix(results, testing_target)
    # CM_t = CM_t + CM
    # print(CM)
    target_names = ['normal', 'hypo']  # , 'hyper'
    print(classification_report(testing_target, results, target_names=target_names))

    print('\n Start of Validation\n')

    #print(validate.describe())
    val_target = np.array(validate['target'])
    val_features = np.array(validate.iloc[:, :58])
    print(val_features[1:3,-1])
    # val_features = imputer.fit(features[196:387])
    val_features = imputer.transform(val_features)
    val_target = np.array(validate['target'])
    print('....value count....')
    print(validate['target'].value_counts())
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


#print('\n Final test \n')
"""
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
"""