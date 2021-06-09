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


def plot_heartbeats(cleaned_ecg, peaks, sampling_rate=250):
    heartbeats = nk.epochs_create(cleaned_ecg, events=peaks, epochs_start=-0.3, epochs_end=0.4,
                                  sampling_rate=sampling_rate)
    heartbeats = nk.epochs_to_df(heartbeats)
    return heartbeats

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
print(all_files)
ECG_HRV = pd.DataFrame(HRVECG)
ECG_QRS = pd.DataFrame(ECGQRS)
# Loop through each file in the subject folder
for i, file in enumerate(all_files[0:1]):
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

    cleaned = cleaned[1500:]

    processed_data, info = nk.bio_process(data['EcgWaveform'], sampling_rate=250)

    results = nk.bio_analyze(processed_data, sampling_rate=250)
    print(results)
    #results.to_csv(fr'C:\Users\nrust\Downloads\rslt_%s_%s.csv' % (participant, file), index=False, na_rep='')




    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=250)
    _, waves_peak = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=250, show=True, method='peaks', show_type='all')


    _, waves_peak2 = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=250, show=True, method='dwt', show_type='all')


    # Plotting all the heart beats

    rpeaks2 = rpeaks
    rpeaks2['onset'] = rpeaks['ECG_R_Peaks']
    rpeaks2['label'] = rpeaks['ECG_R_Peaks']

    epochs = nk.ecg_segment(cleaned, rpeaks=rpeaks2, sampling_rate=250) # Define a function to create epochs
    #print(epochs)
    #epochs2 = pd.DataFrame.from_dict(epochs, orient='Index')
    #print(epochs2.head())
    heartbeats = plot_heartbeats(cleaned, peaks=rpeaks2, sampling_rate=250)
    heartbeats_pivoted = heartbeats.pivot(index='Time', columns='Label', values='Signal')
    heartbeats_pivoted = heartbeats_pivoted.transpose()
    heartbeats_pivoted['target'] = 1
    print(heartbeats_pivoted)
    heartbeats_pivoted.to_csv(fr'C:/Users/nrust/Downloads/pivot_%s_pos.csv' % file)
    break
    stack = heartbeats_pivoted.stack()
    stack = stack.transpose()
    dfstack = pd.DataFrame(stack)
    if i < 3:
        k = 1
    else:
        k = 2
    dfstack['y'] = k
    #print(dfstack.head())
    #dfstack.to_csv(fr'C:/Users/nrust/Downloads/test%s_%s_t1.csv' % (participant, i))




    wavesdf = pd.DataFrame.from_dict(waves_peak2)
    
    

    #P_peak = list(np.array(waves_peak2['ECG_P_Peaks']))
    #T_peak = list(np.array(waves_peak2['ECG_T_Peaks']))
    wavesdf = wavesdf.subtract(list(np.array(waves_peak2['ECG_P_Onsets'])), axis=0)
    wavesdf['QT'] = list(np.array(waves_peak2['ECG_T_Offsets']) - np.array(waves_peak2['ECG_R_Onsets']))
    wavesdf = wavesdf.drop(columns=['ECG_P_Onsets'])
    if i < 4:
        k = 1
    else:
        k = 0

    wavesdf['target'] = k
    #print(wavesdf.describe())
    wavesdf.to_csv(fr'C:/Users/nrust/Downloads/dict_%s_%s.csv' % (participant, file), index=False, na_rep='')

    #file_rename = file.replace('flt_ECG', 'ECG')
    #plt.savefig('img/%s/%s_.png' % (participant, file_rename))
    

    # Find peaks
    peaks, info = nk.ecg_peaks(cleaned, sampling_rate=250)

    # START of Removing outliers
    #threshold_value = 20000
    #keys = [k for k, v in waves_peak.items() if v['ECG_P_Peaks'] > threshold_value]
    #for x in keys:
    #    del waves_peak[x]

    for (key, value) in waves_peak.items():
        if any(value) > 20000 or any(value) < -20000:
            del waves_peak[key]
    # END of Removing outliers



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

    #result.to_csv(fr'C:\Users\nrust\Downloads\D0_test\008_feat\ecg_feat_all.csv', index=False)


# result.to_csv(fr'C:\Users\nrust\Downloads\D0_test\007_feat\ECG_all.csv', index=False)

# peaks, info = nk.ecg_peaks(epochs["EcgWaveform"], sampling_rate=250)

# Extract clean EDA and SCR features
# hrv_time = nk.hrv_time(peaks, sampling_rate=250, show=True)
# hrv_time
