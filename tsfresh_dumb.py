from tsfresh.feature_extraction import extract_features
from tsfresh.feature_extraction import settings

import numpy as np
import pandas as pd

df = pd.DataFrame({"id": ["a", "a", "b", "b"], "temperature": [1,2,3,1], "pressure": [-1, 2, -1, 7]})

settings.ComprehensiveFCParameters, settings.EfficientFCParameters, settings.MinimalFCParameters

settings_minimal = settings.MinimalFCParameters()

if __name__ == '__main__':

    X_tsfresh = extract_features(df, column_id="id", default_fc_parameters=settings_minimal)
    print(X_tsfresh.head())


    del settings_minimal["length"]
    print(settings_minimal)


    X_tsfresh = extract_features(df, column_id="id", default_fc_parameters=settings_minimal)
    print(X_tsfresh.head())

    impute(X_tsfresh)
    features_filtered = select_features(X_tsfresh, y)