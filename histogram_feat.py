import pandas as pd
import numpy as np
import os
import neurokit2 as nk
import matplotlib.pyplot as plt
#%matplotlib inline

import padasip as pa
import datetime

plt.rcParams['figure.figsize'] = [15, 9]  # Bigger images

participant = '002'
sampling_rate = 250
all_hist = []


# List all data files in participant_00X folder
all_files = os.listdir('C:/Users/nrust/Downloads/D0_test/' + participant)

tpot_tgt = pd.read_csv('C:/Users/nrust/Downloads/Target_%s_new.csv' % participant, sep=',', parse_dates=[0])
tpot_tgt = tpot_tgt.rename(columns={"date_time": "date", "comments": "target"})
tpot_tgt = tpot_tgt.drop(columns=['type'])

# Loop through each file in the subject folder
for i, file in enumerate(all_files[:]):
    data = pd.read_csv('C:/Users/nrust/Downloads/D0_test/' + participant + '/' + file, parse_dates=[0])
    #file = file.replace('.csv.csv', '.csv')
    #file = file.replace('flt_ECG', 'ECG')
    #data_d1 = pd.read_csv('C:/Users/nrust/Downloads/D1_test/' + participant + '/' + file)
    #plt.plot(data_d1['EcgWaveform'])
    #plt.plot(data['EcgWaveform'])
    #plt.show()
    # center time of the reading
    date = data['Time'][15001]
    print(date)
    # Add a Label column (e.g Label 1 for epoch 1)
    data['Label'] = np.full(len(data), str(i+1))
    # Set index of data to time in seconds
    #index = data.index/sampling_rate
    #data = data.set_index(pd.Series(index))
    # Append the file into the dictionary
    #epochs[str(i + 1)] = data

    
    x = data['EcgWaveform'].to_numpy()
    raw_ecg = pd.DataFrame(data=x, columns=['ECG'])
    cleaned = nk.ecg_clean(raw_ecg, sampling_rate=250, method="neurokit")
    df_clean = pd.DataFrame(cleaned, columns=['ECG'])
    df_clean['ECG'] = df_clean['ECG'].astype('float')

    #cleaned = cleaned[500:]

    processed_data, info = nk.bio_process(data['EcgWaveform'], sampling_rate=250)
    results = nk.bio_analyze(processed_data, sampling_rate=250)

    quality = nk.ecg_quality(cleaned, sampling_rate=250)
    quality = np.heaviside(quality-.90, 1)
    cleaned = cleaned*quality

    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=250)
    #plt.plot(raw_ecg[1500:]-2000)
    #plt.plot(np.concatenate([np.zeros(1499), cleaned]))
    #plt.show()
    #breakpoint()

    filtered_rpeaks = list(
        filter(lambda r: (r > 250 or r < 28250), rpeaks["ECG_R_Peaks"]))  # dwt fails when R is close to 0

    _, waves_peak = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=250, show=False, method='peaks', show_type='all')
    _, waves_peak2 = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=250, show=False, method='dwt',
                                      show_type='all')


    wvpk = pd.DataFrame.from_dict(waves_peak2)
    #print(wvpk.shape)
    #print(wvpk.isnull().sum(axis=0))

    QT = wvpk['ECG_T_Offsets'].sub(wvpk['ECG_R_Onsets'], axis=0)
    ST = wvpk['ECG_T_Onsets'].sub(wvpk['ECG_R_Offsets'], axis=0)
    amp_P = cleaned[wvpk['ECG_P_Peaks'].replace(np.nan, 0).astype(int)]
    amp_R = cleaned[rpeaks['ECG_R_Peaks'].astype(int)]
    amp_T = cleaned[wvpk['ECG_T_Peaks'].replace(np.nan, 0).astype(int)]
    #print(np.mean(amp_P), np.median(amp_P), np.std(amp_P))
    #print(np.mean(amp_R), np.median(amp_R), np.std(amp_R))
    #print(np.mean(amp_T), np.median(amp_T), np.std(amp_T))
    #print(len(QT), len(amp_T), len(amp_R))
    #plt.plot(cleaned)
    #plt.show()


    wvpk['ECG_P_Onsets'] = wvpk['ECG_P_Onsets'].interpolate()
    wvpk = wvpk.sub(wvpk['ECG_P_Onsets'], axis=0)
    wvpk = wvpk.drop(['ECG_P_Onsets'], axis=1)
    wvpk['QT'] = QT
    wvpk['ST'] = ST
    wvpk['amp_P'] = amp_P
    wvpk['amp_R'] = amp_R
    wvpk['amp_T'] = amp_T
    wvpk.insert(0, 'Time', date)

    #print(tpot_tgt['date'])
    a = tpot_tgt[abs(tpot_tgt['date'] - date) < datetime.timedelta(seconds=149)]
    a = a.iloc[0]


    wvpk['glucose'] = a['glucose']
    wvpk['target'] = a['target']


    #boxplot = wvpk.boxplot()
    #plt.show()
    date_time = file.replace('flt_ECG_val', 'feat')
    date_time = date_time.replace('.csv.csv', '')
    wvpk.to_csv(fr'C:/Users/nrust/Downloads/D0_test/%s_feat/%s_%s.csv' % (participant, participant, date_time), index=False)


"""
# Create histogram
QT = data['QT'].replace([np.inf, -np.inf, np.nan], np.nan).dropna(axis=0)  #
hist = QT.hist(bins=100)
# hdf = pd.DataFrame(data=hist, columns=['ECG'])

# hdf.to_csv(fr'C:/Users/nrust/Downloads/D0_test/hist_s{i}.csv', index=False)



freq, bins = np.histogram(QT, 50)
hist, bin_edges = np.histogram(QT, bins=np.linspace(10, 90, num=10)) # originally 0, 125, 100
plt.show()
QT.plot.hist(grid=True, bins=20, rwidth=0.9,
                   color='#607c8e')
plt.title('QT interval Histogram')
plt.xlabel('Counts')
plt.ylabel('QT length')
plt.grid(axis='y', alpha=0.75)
plt.show()


all_hist.append(hist)
all_hist.append(np.histogram(data['QT'], bins=np.linspace(0, 500, num=100)))
all_hist.append(zip(*np.histogram(data['QT'], bins=np.linspace(0, 500, num=100))))
"""





