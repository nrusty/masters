import val as val
from tsfresh.feature_extraction import extract_features
from tsfresh.feature_extraction import settings
from tsfresh.feature_selection import select_features
from tsfresh.utilities.dataframe_functions import impute

import numpy as np
import pandas as pd

#df = pd.DataFrame({"id": ["a", "a", "b", "b"], "temperature": [1,2,3,1], "pressure": [-1, 2, -1, 7]})

df = pd.read_csv('C:/Users/nrust/Downloads/data007_tsfresh.csv')
#df = df.replace('', np.nan, inplace=True)
#df = df.dropna()


# dropping ALL duplicate values
y_temp = df.drop(columns=['Time', 'value'])
y_temp = y_temp.drop_duplicates(subset=['id'], keep='first')

df = df.drop(columns=['y'])

settings.ComprehensiveFCParameters, settings.EfficientFCParameters, settings.MinimalFCParameters

settings_minimal = settings.MinimalFCParameters()

if __name__ == '__main__':

    X_tsfresh = extract_features(df, column_id="id", default_fc_parameters=settings_minimal)
    print(X_tsfresh.head())


    del settings_minimal["length"]
    print(settings_minimal)


    X_tsfresh = extract_features(df, column_id="id", default_fc_parameters=settings_minimal)


    #X_tsfresh = impute(X_tsfresh)
    #print(len(X_tsfresh), 'Size after impute')
    #print(len(X_tsfresh.index.values))
    #y = y_temp[y_temp['id'].isin(X_tsfresh.index.values)]

    print(y_temp)
    features_filtered = select_features(X_tsfresh, np.array(y_temp['y']))
    print('Features:\n', features_filtered)