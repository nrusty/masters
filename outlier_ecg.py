import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import seaborn as sns
import alibi_detect
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from alibi_detect.od import OutlierSeq2Seq
from alibi_detect.utils.fetching import fetch_detector
from alibi_detect.utils.saving import save_detector, load_detector
from alibi_detect.datasets import fetch_ecg
from alibi_detect.utils.visualize import plot_roc

df = pd.read_csv(r'C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\001\sensor_data\2014_10_01-10_09_39\2014_10_01'
                 r'-10_09_39_ECG.csv', parse_dates=[0], skiprows=8135147, nrows=30002)

#df.columns = df.iloc[0]
#df = df.iloc[1:]
df['Time'] = pd.to_datetime(df['Time']).astype('datetime64[ms]')
df['EcgWaveform'] = df['EcgWaveform'].astype('int')

X_test = df['EcgWaveform'].to_numpy()

#print(len(X_test.index))
load_outlier_detector = False

filepath = 'my_path'  # change to (absolute) directory where model is downloaded
if load_outlier_detector:  # load pretrained outlier detector
    detector_type = 'outlier'
    dataset = 'ecg'
    detector_name = 'OutlierSeq2Seq'
    od = fetch_detector(filepath, detector_type, dataset, detector_name)
    filepath = os.path.join(filepath, detector_name)
else:  # define model, initialize, train and save outlier detector

    # initialize outlier detector
    od = OutlierSeq2Seq(1,
                        X_test.shape[1],  # sequence length
                        threshold=None,
                        latent_dim=40)

    # train
    od.fit(X_test,
           epochs=100,
           verbose=False)

    # save the trained outlier detector
    save_detector(od, filepath)

ecg_pred = od.seq2seq.decode_seq(X_test)[0]

i_normal = np.where(y_test == 0)[0][0]
plt.plot(ecg_pred[i_normal], label='Prediction')
plt.plot(X_test[i_normal], label='Original')
plt.title('Predicted vs. Original ECG of Inlier Class 1')
plt.legend()
plt.show()

i_outlier = np.where(y_test == 1)[0][0]
plt.plot(ecg_pred[i_outlier], label='Prediction')
plt.plot(X_test[i_outlier], label='Original')
plt.title('Predicted vs. Original ECG of Outlier')
plt.legend()
plt.show()