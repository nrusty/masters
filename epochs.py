import pandas as pd
import numpy as np
import os
import neurokit2 as nk
import matplotlib.pyplot as plt
#%matplotlib inline

import padasip as pa

plt.rcParams['figure.figsize'] = [15, 9]  # Bigger images

participant = '007'

sampling_rate = 250

# List all data files in Subject_001 folder
all_files = os.listdir('C:/Users/nrust/Downloads/D0_test/' + participant)

# Create an empty directory to store your files (events)
epochs = {}
data2 = []
FeatECG = []
HRVECG = []
ECGQRS = []
result = []

ECGFeat = pd.DataFrame(FeatECG, columns=['name', 'P_mean', 'P_std', 'Q_mean', 'Q_std', 'S_mean', 'S_std', 'T_mean',
                                         'T_std', 'QT_mean', 'QT_std', 'ST_mean', 'ST_std', 'QT_mean2', 'QT_std2',
                                         'ST_mean2', 'ST_std2'])
print(ECGFeat.head())
ECG_HRV = pd.DataFrame(HRVECG)
ECG_QRS = pd.DataFrame(ECGQRS)
# Loop through each file in the subject folder
for i, file in enumerate(all_files[:1]): #all_files[290:295]
    # Read the file
    #file = file.replace('.csv.csv', '.csv')
    print(file)
    print(file, i)
    #data = pd.read_csv('C:/Users/nrust/Downloads/D0_test/' + participant + '/' + file)

    data = pd.read_csv('C:/Users/nrust/Downloads/007_flt_ECG_2014-10-01-11-49.csv.csv')
    # Add a Label column (e.g Label 1 for epoch 1)
    data['Label'] = np.full(len(data), str(i+1))
    # Set index of data to time in seconds
    index = data.index/sampling_rate
    data = data.set_index(pd.Series(index))
    # Append the file into the dictionary
    epochs[str(i + 1)] = data

    x = data['EcgWaveform'].to_numpy()
    ecg_signal = pd.DataFrame(data=x, columns=['ECG'])
    ecg_signal = nk.ecg_clean(ecg_signal, sampling_rate=250, method="neurokit")
    ecg_signal = pd.DataFrame(ecg_signal, columns=['ECG'])
    ecg_signal['ECG'] = -ecg_signal['ECG'].astype('float')
    plt.plot(ecg_signal['ECG'][1000:])

    cleaned = nk.ecg_clean(ecg_signal, sampling_rate=250)

    cleaned = cleaned[1030:]
    plt.plot(cleaned, 'r')
    plt.show()
    #print(np.mean(cleaned))
    #print(np.max(cleaned))
    for i in range(len(cleaned)):
        if cleaned[i]**2 > 150**2:
            #print(i, 'is greater than 150')
            cleaned[i-250:i+250] = 0
    #print(np.mean(cleaned))
    #plt.plot(cleaned)
    #plt.show()

    # Plotting all the heart beats
    #single_beat = nk.ecg_segment(cleaned[9000:15000], rpeaks=None, sampling_rate=250, show=True)
    # print(single_beat.values())



    def plot_results():
        # show results
        plt.figure(figsize=(15, 9))
        #plt.subplot(211);
        plt.title("Adaptation")  # ;plt.xlabel("samples - k")
        plt.ylim(-200, 200)
        plt.plot(cleaned[20000:30000], "b", label="d - input");
        plt.legend()
        cleaned2 = pd.DataFrame(cleaned)
        #cleaned2.to_csv(fr'C:\Users\nrust\Downloads\D0_test\clnECG_001_10-04_19-29.csv', index=False)
        plt.show()
        return


    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=250)
    _, waves_peak = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=250, show=True)
    _, waves_peak2 = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=250, method='dwt', show_type='all')
    file_rename = file.replace('flt_ECG', 'ECG')
    plt.savefig('img/%s/%s_conc_up.png' % (participant, file_rename))

    plt.rcParams['figure.figsize'] = [10, 6]  # resize


    # Define a function to create epochs
    def plot_heartbeats(cleaned, peaks, sampling_rate=250):
        heartbeats = nk.epochs_create(cleaned, events=peaks, epochs_start=-0.3, epochs_end=0.4,
                                      sampling_rate=sampling_rate)
        heartbeats = nk.epochs_to_df(heartbeats)
        return heartbeats


    #heartbeats = plot_heartbeats(cleaned, peaks=rpeaks, sampling_rate=250)
    #plt.plot(heartbeats)
    #plt.show()

    # Find peaks
    peaks, info = nk.ecg_peaks(cleaned, sampling_rate=250)

    # Compute HRV indices
    ECG_HRV1 = nk.hrv(peaks, sampling_rate=250)

    #print(waves_peak.keys())
    ECG_QT = list(np.array(waves_peak['ECG_T_Peaks']) - np.array(waves_peak['ECG_Q_Peaks']))
    ECG_ST = list(np.array(waves_peak['ECG_T_Peaks']) - np.array(waves_peak['ECG_S_Peaks']))

    #print(waves_peak2.keys())
    ECG_QT2 = list(np.array(waves_peak2['ECG_T_Offsets']) - np.array(waves_peak2['ECG_R_Onsets']))
    ECG_ST2 = list(np.array(waves_peak2['ECG_T_Offsets']) - np.array(waves_peak2['ECG_R_Offsets']))

    #QTdf = pd.DataFrame(ECG_QT)

    #print(ECG_QT)
    #print(waves_peak['ECG_Q_Peaks'][:5])
    #indexq = waves_peak['ECG_Q_Peaks'][:5]
    #cleanedList = [x for x in indexq if str(x) != 'nan']
    #Q_pk = cleaned[cleanedList]

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
    ST_mean2 = np.nanmean(ECG_ST2)
    ST_std2 = np.nanstd(ECG_ST2)
    QT_mean2 = np.nanmean(ECG_QT2)
    QT_std2 = np.nanstd(ECG_QT2)

    data = {'name': [file],
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
            'QT_mean2': [QT_mean2],
            'QT_std2': [QT_std2],
            'ST_mean2': [ST_mean2],
            'ST_std2': [ST_std2]
            }

    values = [[file, P_mean, P_std, Q_mean, Q_std, S_mean, S_std, T_mean, T_std, QT_mean, QT_std, ST_mean, ST_std,
               QT_mean2, QT_std2, ST_mean2, ST_std2]]

    df2 = pd.DataFrame(values, columns=['name','P_mean', 'P_std', 'Q_mean', 'Q_std', 'S_mean', 'S_std', 'T_mean',
                                          'T_std', 'QT_mean', 'QT_std', 'ST_mean', 'ST_std', 'QT_mean2', 'QT_std2',
                                        'ST_mean2', 'ST_std2'])


    ECGFeat = ECGFeat.append(df2)

    #ECGFeat = ECGFeat.append(data, ignore_index=True)
    ECG_HRV = ECG_HRV.append(ECG_HRV1)

    #uncomment later
    #ECG_HRV.to_csv(fr'C:\Users\nrust\Downloads\D0_test\newest_HRV_ECG{i}.csv', index=False)

    #hist = QTdf.hist(bins=100)
    #hdf = pd.DataFrame(data=hist, columns=['ECG'])
    #hdf.to_csv(fr'C:\Users\nrust\Downloads\D0_test\hist_s{i}.csv', index=False)

    #freq, bins = np.histogram(ECG_QT, 50)
    #hist, bin_edges = np.histogram(ECG_QT, bins=np.linspace(10, 90, num=10)) # originally 0, 125, 100
    #data2.append(hist)
    #data2.append(np.histogram(ECG_QT, bins=np.linspace(0, 500, num=100)))
    #data2.append(zip(*np.histogram(ECG_QT, bins=np.linspace(0, 500, num=100))))



    result = pd.concat([ECGFeat, ECG_HRV], axis=1)
    if (i%10 == 0 or i==385):
        result.to_csv(fr'C:\Users\nrust\Downloads\D0_test\003_feat\ecg_feat_s{i}.csv', index=False)

#uncomment later
result.to_csv(fr'C:\Users\nrust\Downloads\D0_test\003_feat\update_feat_HRVECG_all.csv', index=False)



# Extract clean EDA and SCR features
#hrv_time = nk.hrv_time(peaks, sampling_rate=250, show=True)
#hrv_time