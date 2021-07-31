import pandas as pd
import numpy as np
import os
import neurokit2 as nk
import matplotlib.pyplot as plt
import datetime
# %matplotlib inline

import padasip as pa
def plot_heartbeats(cleaned_ecg, peaks, sampling_rate=250):
    heartbeats = nk.epochs_create(cleaned_ecg, events=peaks, epochs_start=-0.3, epochs_end=0.4,
                                  sampling_rate=sampling_rate)
    heartbeats = nk.epochs_to_df(heartbeats)
    return heartbeats

plt.rcParams['figure.figsize'] = [15, 9]  # Bigger images

participant = '002'

sampling_rate = 250

# List all data files in Subject_001 folder
all_files = os.listdir('C:/Users/nrust/Downloads/D0_test/' + participant)

# Create an empty directory to store your files (events)
epochs = {}
data2 = []
FeatECG = []
ECGQRS = []
HRVECG = []
result = []

#ECGFeat = pd.DataFrame(FeatECG, columns=['date', 'P_mean', 'P_std', 'Q_mean', 'Q_std', 'S_mean', 'S_std', 'T_mean', 'T_std', 'QT_mean', 'QT_std', 'ST_mean', 'ST_std'])
ECGFeat = pd.DataFrame([])
ECG_HRV = pd.DataFrame(HRVECG)
ECG_QRS = pd.DataFrame(ECGQRS)


# UNCOMMENT LATER
if os.path.isfile(fr'C:\Users\nrust\Downloads\ECG_single_%s_partial2.csv' % participant):
    ECGFeat = pd.read_csv(fr'C:\Users\nrust\Downloads\ECG_single_%s_partial2.csv' % participant, sep=',', parse_dates=[0],
                           engine='python')

#UNCOMENT LATER

#ECGFeat = pd.DataFrame()

# Loop through each file in the folder
for i, file in enumerate(all_files[:]):
    # Read the file
    # file = file.replace('.csv.csv', '.csv')
    data = pd.read_csv('C:/Users/nrust/Downloads/D0_test/' + participant + '/' + file, parse_dates=[0])
    date = data['Time'][15001]
    print(file)
    # Add a Label column (e.g Label 1 for epoch 1)
    data['Label'] = np.full(len(data), str(i + 1))
    # Set index of data to time in seconds
    index = data.index / sampling_rate
    data = data.set_index(pd.Series(index))

    # Append the file into the dictionary
    epochs[str(i + 1)] = data



    x = data['EcgWaveform'].to_numpy()
    #x = x[:-2]

    ecg_signal = pd.DataFrame(data=x, columns=['ECG'])
    ecg_signal = nk.ecg_clean(ecg_signal, sampling_rate=250, method="neurokit") #
    ecg_signal2 = nk.ecg_clean(ecg_signal, sampling_rate=250, method="elgendi2010")
    ecg_signal = pd.DataFrame(ecg_signal, columns=['ECG'])
    ecg_signal['ECG'] = ecg_signal['ECG'].astype('float')
    cleaned = nk.ecg_clean(ecg_signal, sampling_rate=250)
    cleaned3 = nk.ecg_clean(ecg_signal2, sampling_rate=250)

    quality = nk.ecg_quality(cleaned, sampling_rate=250)
    #quality = nk.ecg_quality(cleaned, sampling_rate=250, method="zhao2018", approach="fuzzy")


    cleaned2 = pd.DataFrame(cleaned, columns=['ECG'])

    for k in range(29998):
        if quality[k] < 0.95:
            cleaned[k] = 0

    # plt.show()
    #        return

    # processed_data, info = nk.bio_process(cleaned2, sampling_rate=250)
    # results = nk.bio_analyze(processed_data, sampling_rate=250)
    # results.to_csv(fr'C:\Users\nrust\Downloads\rslt_%s_%s.csv' % (participant, file), index=False, na_rep='')

    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=250)
    _, waves_peak = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=250, method='peaks')

    _, waves_peak2 = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=250, method='dwt')


    rpeaks2 = rpeaks
    rpeaks2['onset'] = rpeaks['ECG_R_Peaks']
    rpeaks2['label'] = rpeaks['ECG_R_Peaks']

    epochs = nk.ecg_segment(cleaned, rpeaks=rpeaks2, sampling_rate=250)  # Define a function to create epochs

    wavesdf = pd.DataFrame.from_dict(waves_peak2)
    #wavesdf.to_csv(fr'C:/Users/nrust/Downloads/raw_dict_%s_%s.csv' % (participant, file), index=False, na_rep='')

    waves_amp_R = cleaned[list(np.array(rpeaks['ECG_R_Peaks']))]

    pk_index = np.array(waves_peak2['ECG_P_Peaks'])
    pk_index[np.isnan(pk_index)] = 1
    waves_amp_P = cleaned[pk_index.astype(int)]

    pk_index = np.array(waves_peak2['ECG_T_Peaks'])
    pk_index[np.isnan(pk_index)] = 1
    waves_amp_T = cleaned[pk_index.astype(int)]

    waves_qt = wavesdf.subtract(list(np.array(waves_peak2['ECG_R_Onsets'])), axis=0)
    waves_st = wavesdf.subtract(list(np.array(waves_peak2['ECG_T_Onsets'])), axis=0)
    wavesdf = wavesdf.subtract(list(np.array(waves_peak2['ECG_P_Onsets'])), axis=0)
    wavesdf = wavesdf.drop(columns=['ECG_P_Onsets'])
    wavesdf['QT'] = waves_qt['ECG_T_Offsets']
    wavesdf['ST'] = waves_st['ECG_R_Offsets']
    wavesdf['R_amp'] = pd.DataFrame(waves_amp_R)
    wavesdf['P_amp'] = pd.DataFrame(waves_amp_P)
    wavesdf['T_amp'] = pd.DataFrame(waves_amp_T)


    #R_off = pd.DataFrame(waves_peak2['ECG_R_Offsets']).fillna(method='bfill')
    #T_on = pd.DataFrame(waves_peak2['ECG_T_Onsets']).fillna(method='bfill')
    #R = (R_off[0].values).astype(int)
    #T = (T_on[0].values).astype(int)
    #print(np.array(T_on['ECG_T_Onsets']))
    #wavesdf['ST_slope'] = cleaned[np.array(R)] - cleaned[np.array(T)]
    #wavesdf['ST_slope'] = wavesdf['ST_slope']/wavesdf['ST']
    #print(np.array(waves_peak2['ECG_T_Onsets']), np.array(waves_peak2['ECG_R_Offsets']))

    #wavesdf['QT'] = list(np.array(waves_peak2['ECG_T_Offsets']) - np.array(waves_peak2['ECG_R_Onsets']))
    #wavesdf['ST'] = list(np.array(waves_peak2['ECG_T_Onsets']) - np.array(waves_peak2['ECG_R_Offsets']))
    #wavesdf['ST_slope'] = cleaned2.iloc[waves_peak2['ECG_T_Onsets']] - cleaned2.iloc[waves_peak2['ECG_R_Offsets']]
    #wavesdf = wavesdf.drop(columns=['ECG_P_Onsets'])

    tpot_tgt = pd.read_csv('C:/Users/nrust/Downloads/Target_%s.csv' % participant, sep=',', parse_dates=[0],
                           engine='python')
    tpot_tgt = tpot_tgt.rename(columns={"date_time": "date", "comments": "target"})
    target = tpot_tgt[abs(tpot_tgt['date'] - date) < datetime.timedelta(seconds=225)]
    print(i)
    #print(target)
    if target.empty:
        continue
    if len(target) == 1:
        wavesdf['target'] = int(target['target'])
    else:
        wavesdf['target'] = int(target['target'].iloc[1])

    #wavesdf.to_csv(fr'C:/Users/nrust/Downloads/single_feat_%s_%s.csv' % (participant, file), index=False, na_rep='')

    # Find peaks
    peaks, info = nk.ecg_peaks(cleaned, sampling_rate=250)

    # Compute HRV indices
    ECG_HRV1 = nk.hrv(peaks, sampling_rate=250)
    ECG_HRV1 = ECG_HRV1.dropna(axis=1)
    ECG_HRV1 = pd.concat([ECG_HRV1]*len(wavesdf), ignore_index=True)

    wavesdf['QTc'] = wavesdf['QT']/(125*np.sqrt(ECG_HRV1['HRV_MeanNN'][1]))
    all_feat = pd.concat([wavesdf, ECG_HRV1], axis=1)
    all_feat = all_feat.iloc[:, 0:14]
    all_feat = all_feat[[c for c in all_feat if c not in ['target']] + ['target']]

    #with pd.option_context('display.max_columns', 40):
    #    print(all_feat.describe(include='all'))

    #with pd.option_context('display.max_columns', 40):
    #    print(all_feat.describe(include='all'))
    #breakpoint()

    ECGFeat = ECGFeat.append(all_feat)
    ECGFeat.to_csv(fr'C:\Users\nrust\Downloads\ECG_single_%s_partial2.csv' % participant, index=False)

ECGFeat.replace('', np.nan, inplace=True)
ECGFeat.dropna(inplace=True)
ECGFeat.to_csv(fr'C:\Users\nrust\Downloads\ECG_single_%s_all3.csv' % participant, index=False)

# peaks, info = nk.ecg_peaks(epochs["EcgWaveform"], sampling_rate=250)

# Extract clean EDA and SCR features
# hrv_time = nk.hrv_time(peaks, sampling_rate=250, show=True)
# hrv_time
