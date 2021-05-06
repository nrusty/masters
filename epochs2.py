import pandas as pd
import numpy as np
import os
import neurokit2 as nk
import matplotlib.pyplot as plt
# %matplotlib inline

import padasip as pa

plt.rcParams['figure.figsize'] = [15, 9]  # Bigger images

participant = '007'

sampling_rate = 250

# List all data files in Subject_001 folder
all_files = os.listdir('C:/Users/nrust/Downloads/D0_test/' + participant)
# all_files = os.listdir('C:/Users/nrust/Downloads/D1_test/' + participant)

# Create an empty directory to store your files (events)
epochs = {}
data2 = []
FeatECG = []
ECGQRS = []
HRVECG = []
result = []

ECGFeat = pd.DataFrame(FeatECG,
                       columns=['date', 'P_mean', 'P_std', 'Q_mean', 'Q_std', 'S_mean', 'S_std', 'T_mean', 'T_std',
                                'QT_mean', 'QT_std', 'ST_mean', 'ST_std'])
print(ECGFeat.head())
ECG_HRV = pd.DataFrame(HRVECG)
ECG_QRS = pd.DataFrame(ECGQRS)
# Loop through each file in the subject folder
for i, file in enumerate(all_files[4:]):
    # Read the file
    # file = file.replace('.csv.csv', '.csv')
    print(i)
    print(file)
    data = pd.read_csv('C:/Users/nrust/Downloads/D0_test/' + participant + '/' + file)
    date = data['Time'][15001]
    # print(date)
    # Add a Label column (e.g Label 1 for epoch 1)
    data['Label'] = np.full(len(data), str(i + 1))
    # Set index of data to time in seconds
    index = data.index / sampling_rate
    data = data.set_index(pd.Series(index))
    # Append the file into the dictionary
    epochs[str(i + 1)] = data

    x = data['EcgWaveform'].to_numpy()
    ecg_signal = pd.DataFrame(data=x, columns=['ECG'])
    ecg_signal = nk.ecg_clean(ecg_signal, sampling_rate=250, method="neurokit")
    ecg_signal = pd.DataFrame(ecg_signal, columns=['ECG'])
    ecg_signal['ECG'] = ecg_signal['ECG'].astype('float')
    cleaned = nk.ecg_clean(ecg_signal, sampling_rate=250)

    #    def plot_results():
    # show results
    # plt.figure(figsize=(15, 9))
    # plt.subplot(211);
    # plt.title("Adaptation")  # ;plt.xlabel("samples - k")
    # plt.ylim(-200, 200)
    # plt.plot(cleaned[20000:30000], "b", label="d - input");
    # plt.legend()
    cleaned2 = pd.DataFrame(cleaned)
    # plt.show()
    #        return

    cleaned = cleaned[1000:]
    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=250)
    _, waves_peak = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=250, show=True, method='dwt', show_type='all')
    file_rename = file.replace('flt_ECG', 'ECG')
    plt.savefig('img/%s_dwt.png' % file_rename)
    # Find peaks
    peaks, info = nk.ecg_peaks(cleaned, sampling_rate=250)

    # Compute HRV indices
    ECG_HRV1 = nk.hrv(peaks, sampling_rate=250)
    # print(ECG_HRV1)

    ECG_QT = list(np.array(waves_peak['ECG_T_Peaks']) - np.array(waves_peak['ECG_Q_Peaks']))
    ECG_ST = list(np.array(waves_peak['ECG_T_Peaks']) - np.array(waves_peak['ECG_S_Peaks']))

    QTdf = pd.DataFrame(ECG_QT)

    P_mean = np.nanmean(waves_peak['ECG_P_Peaks'])
    P_std = np.nanstd(waves_peak['ECG_P_Peaks'])
    Q_mean = np.nanmean(waves_peak['ECG_Q_Peaks'])
    Q_std = np.nanstd(waves_peak['ECG_Q_Peaks'])
    T_mean = np.nanmean(waves_peak['ECG_T_Peaks'])
    T_std = np.nanstd(waves_peak['ECG_T_Peaks'])
    S_mean = np.nanmean(waves_peak['ECG_S_Peaks'])
    S_std = np.nanstd(waves_peak['ECG_S_Peaks'])
    ST_mean = np.nanmean(ECG_ST)
    ST_std = np.nanstd(ECG_ST)
    QT_mean = np.nanmean(ECG_QT)
    QT_std = np.nanstd(ECG_QT)

    data = {'date': [date],
            'P_mean': [P_mean],
            'P_std': [P_std],
            'Q_mean': [Q_mean],
            'Q_std': [Q_std],
            'S_mean': [S_mean],
            'S_std': [S_std],
            'T_mean': [T_mean],
            'T_std': [T_std],
            'QT_mean': [QT_mean],
            'QT_std': [QT_std],
            'ST_mean': [ST_mean],
            'ST_std': [ST_std],
            }

    values = [[date, P_mean, P_std, Q_mean, Q_std, S_mean, S_std, T_mean, T_std, QT_mean, QT_std, ST_mean, ST_std]]

    df2 = pd.DataFrame(values,
                       columns=['date', 'P_mean', 'P_std', 'Q_mean', 'Q_std', 'S_mean', 'S_std', 'T_mean', 'T_std',
                                'QT_mean',
                                'QT_std', 'ST_mean', 'ST_std'])
    # print(df2)
    ECGFeat = ECGFeat.append(df2)

    # , ignore_index=True
    ECG_HRV = ECG_HRV.append(ECG_HRV1)

    # ECG_HRV.to_csv(fr'C:\Users\nrust\Downloads\D0_test\val_HRV_ECG{i}.csv', index=False)

    # hdf = pd.DataFrame(data=ECGFeat)
    # hdf.to_csv(fr'C:\Users\nrust\Downloads\D0_test\val_QRS_ECG{i}.csv', index=False)

    result = pd.concat([ECGFeat, ECG_HRV], axis=1)

    # result.to_csv(fr'C:\Users\nrust\Downloads\D0_test\009_feat\ecg_feat_all.csv', index=False)

# result.to_csv(fr'C:\Users\nrust\Downloads\D0_test\007_feat\ECG_all.csv', index=False)

# peaks, info = nk.ecg_peaks(epochs["EcgWaveform"], sampling_rate=250)

# Extract clean EDA and SCR features
# hrv_time = nk.hrv_time(peaks, sampling_rate=250, show=True)
# hrv_time
