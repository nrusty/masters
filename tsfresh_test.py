import numpy as np
import pandas as pd
from tsfresh import extract_features
from tsfresh import select_features
from tsfresh import extract_relevant_features
from tsfresh.utilities.dataframe_functions import impute
import matplotlib.pylab as plt
import openpyxl
import xlrd

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd


class PCAForPandas(PCA):
    """This class is just a small wrapper around the PCA estimator of sklearn including normalization to make it
    compatible with pandas DataFrames.
    """

    def __init__(self, **kwargs):
        self._z_scaler = StandardScaler()
        super(self.__class__, self).__init__(**kwargs)

        self._X_columns = None

    def fit(self, X, y=None):
        """Normalize X and call the fit method of the base class with numpy arrays instead of pandas data frames."""

        X = self._prepare(X)

        self._z_scaler.fit(X.values, y)
        z_data = self._z_scaler.transform(X.values, y)

        return super(self.__class__, self).fit(z_data, y)

    def fit_transform(self, X, y=None):
        """Call the fit and the transform method of this class."""

        X = self._prepare(X)

        self.fit(X, y)
        return self.transform(X, y)

    def transform(self, X, y=None):
        """Normalize X and call the transform method of the base class with numpy arrays instead of pandas data frames."""

        X = self._prepare(X)

        z_data = self._z_scaler.transform(X.values, y)

        transformed_ndarray = super(self.__class__, self).transform(z_data)

        pandas_df = pd.DataFrame(transformed_ndarray)
        pandas_df.columns = ["pca_{}".format(i) for i in range(len(pandas_df.columns))]

        return pandas_df

    def _prepare(self, X):
        """Check if the data is a pandas DataFrame and sorts the column names.

        :raise AttributeError: if pandas is not a DataFrame or the columns of the new X is not compatible with the
                               columns from the previous X data
        """
        if not isinstance(X, pd.DataFrame):
            raise AttributeError("X is not a pandas DataFrame")

        X.sort_index(axis=1, inplace=True)

        if self._X_columns is not None:
            if self._X_columns != list(X.columns):
                raise AttributeError(
                    "The columns of the new X is not compatible with the columns from the previous X data")
        else:
            self._X_columns = list(X.columns)

        return X

timeseries1 = pd.read_csv('C:/Users/nrust/Downloads/D0_test/001/fltECG_2014-10-01-19-54.csv.csv', nrows=5000, index_col=None)  # 8135147, nrows=100,
timeseries1['id'] = 1
#df1 = pd.read_excel('C:/Users/nrust/Downloads/D1_test/Acc_001_10-01_m3254058.xlsx', nrows=1000, index_col=None)
#print(df1.head())
#timeseries1[['Vertical', 'Lateral', 'Sagittal']] = df1[['Vertical', 'Lateral', 'Sagittal']]


timeseries2 = pd.read_csv('C:/Users/nrust/Downloads/D0_test/001/fltECG_2014-10-04-11-29.csv.csv', nrows=5000, index_col=None)
timeseries2['id'] = 2
#df2 = pd.read_excel('C:/Users/nrust/Downloads/D0_test/Acc_001_10-04_m2292260.xlsx', nrows=1000, index_col=None)
#timeseries2[['Vertical', 'Lateral', 'Sagittal']] = df2[['Vertical', 'Lateral', 'Sagittal']]

frames = [timeseries1, timeseries2]
timeseries = pd.concat(frames, ignore_index=True)
print(timeseries1.head())
print(timeseries2.head())
print(timeseries.describe())

#timeseries['EcgWaveform'].plot()

#plt.show()

#timeseries.rename(columns={"1975": "a", "B": "c"}) = ['Time', 'EcgWaveform', 'id']

#timeseries['EcgWaveform'].plot()

#plt.show()

#print(timeseries.head())
#print(timeseries.tail())

y1 = np.arange(len(timeseries)/2)
y1.fill(103)
y2 = np.arange(len(timeseries)/2)
y2.fill(75)
a = np.concatenate((y1, y2), axis=None)
y = pd.Series([1, 2], index=[1, 2])

if __name__ == '__main__':

    extracted_features = extract_features(timeseries, column_id="id", column_sort="Time")
    print('All Features')
    print(extracted_features.head())
    #print(extracted_features.tail())
    #print(extracted_features.describe())


    #impute(extracted_features)
    print('y: ', y)
    features_filtered = select_features(extracted_features, y)
    print('Selected Features')
    print(features_filtered.head())

    #features_filtered_direct = extract_relevant_features(timeseries, y,
                                                         #column_id='id', column_sort='Time')

    #print(features_filtered_direct.head())