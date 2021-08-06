import pandas as pd
import numpy as np
import os
import neurokit2 as nk
import matplotlib.pyplot as plt
# %matplotlib inline
import padasip as pa
import datetime

plt.rcParams['figure.figsize'] = [15, 9]  # Bigger images

participant = '005'

sampling_rate = 250


def plot_heartbeats(cleaned_ecg, peaks, sampling_rate=250):
    heartbeats = nk.epochs_create(cleaned_ecg, events=peaks, epochs_start=-0.3, epochs_end=0.4,
                                  sampling_rate=sampling_rate)
    heartbeats = nk.epochs_to_df(heartbeats)
    return heartbeats


# List all data files in Subject_001 folder
all_files = os.listdir('C:/Users/nrust/Downloads/D0_test/' + participant)
#all_files = os.listdir('C:/Users/nrust/Downloads/D1_test/' + participant)


# Create an empty directory to store your files (events)
epochs = {}
data2 = []
FeatECG = []
ECGQRS = []
HRVECG = []
result = []

heartbeats_all = pd.DataFrame(result)
#heartbeats_all = pd.read_csv('C:/Users/nrust/Downloads/pivot_%s_partial_0.csv' % participant)
reset = 99
# Loop through each file in the subject folder
#already did from 303 002
for i, file in enumerate(all_files[510:]):
    # Read the file
    data = pd.read_csv('C:/Users/nrust/Downloads/D0_test/' + participant + '/' + file, parse_dates=[0])
    tpot_tgt = pd.read_csv('C:/Users/nrust/Downloads/Target_%s_new.csv' % participant, sep=',', parse_dates=[0], engine='python')
    tpot_tgt = tpot_tgt.rename(columns={"date_time": "date", "comments": "target"})
    file = file.replace('.csv.csv', '')
    file = file.replace('flt_ECG_', '')
    tpot_tgt2 = tpot_tgt['date'].dt.strftime('%Y-%m-%d-%H-%M')
    date = data['Time'][15001]
    date2 = tpot_tgt2.iloc[30+i]

    print(i, date)


    target = tpot_tgt[abs(tpot_tgt['date'] - date) < datetime.timedelta(minutes=4)]
    print(target)
    if len(target) == 1:
        tgt = int(target['target'])
    else:
        tgt = int(target['target'].iloc[1])
    #print(tgt)
    # print(date)
    # Add a Label column (e.g Label 1 for epoch 1)
    data['Label'] = np.full(len(data), str(i + 1))
    # Set index of data to time in seconds
    index = data.index / sampling_rate
    data = data.set_index(pd.Series(index))
    # Append the file into the dictionary
    epochs[str(i + 1)] = data

    x = data['EcgWaveform'].to_numpy()
    ecg_signal_before = pd.DataFrame(data=x-2000, columns=['ECG'])

    ecg_signal = pd.DataFrame(data=x, columns=['ECG'])
    ecg_signal = nk.ecg_clean(ecg_signal, sampling_rate=250, method="neurokit")
    ecg_signal = pd.DataFrame(ecg_signal, columns=['ECG'])
    ecg_signal['ECG'] = ecg_signal['ECG'].astype('float')
    cleaned = nk.ecg_clean(ecg_signal_before, sampling_rate=250)

    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=250)
    HRV = nk.hrv(rpeaks, sampling_rate=250, show=True)

    """
    with pd.option_context('display.max_columns', 50):
        print(HRV.head())
    plt.show()

    breakpoint()


    ecg = ecg_signal_before['ECG']

    signals = pd.DataFrame({
                            "ECG_NeuroKit": nk.ecg_clean(ecg, sampling_rate=250, method="neurokit"),
                            "ECG_BioSPPy": nk.ecg_clean(ecg, sampling_rate=250, method="biosppy"),
                            "ECG_PanTompkins": nk.ecg_clean(ecg, sampling_rate=250, method="pantompkins1985"),
                            "ECG_Hamilton": nk.ecg_clean(ecg, sampling_rate=250, method="hamilton2002"),
                            "ECG_Elgendi": nk.ecg_clean(ecg, sampling_rate=250, method="elgendi2010"),
                            "ECG_EngZeeMod": nk.ecg_clean(ecg, sampling_rate=250, method="engzeemod2012"),
                            "ECG_Raw": ecg_signal_before['ECG']})
    signals.plot()  # doctest: +ELLIPSIS
    plt.show()



    plt.plot(ecg_signal_before['ECG'][1500:], label = 'input')
    #plt.plot(cleaned[1500:])
    crop = np.concatenate((np.zeros(1499), -cleaned[1500:]), axis=0)
    plt.xlabel('sample number')
    plt.ylabel('amplitude (mV)')
    plt.plot(crop, label = 'cleaned')
    plt.legend()
    plt.show()
    


    cleaned2 = pd.DataFrame(cleaned)

    cleaned = cleaned[1500:]
    print(signals.columns)
    for column in signals:

        processed_data, info = nk.bio_process(signals[column], sampling_rate=250) #data['EcgWaveform'][100:],

        results = nk.bio_analyze(processed_data, sampling_rate=250)

    print(np.nanmean(processed_data['ECG_Quality']), np.nanstd(processed_data['ECG_Quality']))

    #plt.show()
    breakpoint()
    """

    # results.to_csv(fr'C:\Users\nrust\Downloads\rslt_%s_%s.csv' % (participant, file), index=False, na_rep='')

    _, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=250)
    _, waves_peak = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=250, method='peaks')

    _, waves_peak2 = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=250, method='dwt')

    # Plotting all the heart beats

    rpeaks2 = rpeaks
    rpeaks2['onset'] = rpeaks['ECG_R_Peaks']
    rpeaks2['label'] = rpeaks['ECG_R_Peaks']

    epochs = nk.ecg_segment(cleaned, rpeaks=rpeaks2, sampling_rate=250)  # Define a function to create epochs

    heartbeats = plot_heartbeats(cleaned, peaks=rpeaks2, sampling_rate=250)
    heartbeats_pivoted = heartbeats.pivot(index='Time', columns='Label', values='Signal')
    heartbeats_pivoted = heartbeats_pivoted.transpose()
    heartbeats_pivoted['target'] = tgt
    #print(heartbeats_pivoted)
    heartbeats_pivoted = heartbeats_pivoted.reset_index(drop=True)
    # Drop first column
    heartbeats_pivoted.drop(columns=heartbeats_pivoted.columns[0], axis=1, inplace=True)
    #print('All\n', heartbeats_all.head())
    #print('Pivoted\n',heartbeats_pivoted.head())
    if i == reset:
        heartbeats_all = heartbeats_pivoted
    else:
        heartbeats_all = pd.concat([heartbeats_all, heartbeats_pivoted], axis=0)

    #heartbeats_all = heartbeats_all.rename(columns={"target": "0.5"})
    heartbeats_all.to_csv(fr'C:/Users/nrust/Downloads/temp/%s_%s_v4.csv' % (date2, participant), index=False)
#heartbeats_pos = heartbeats_all[heartbeats_all['0.5'] == 1]
#heartbeats_neg = heartbeats_all[heartbeats_all['0.5'] == 0]
#heartbeats_pos.to_csv(fr'C:/Users/nrust/Downloads/pivot_%s_pos.csv' % participant, index=None)
#heartbeats_neg.to_csv(fr'C:/Users/nrust/Downloads/pivot_%s_neg.csv' % participant, index=None)
    #stack = heartbeats_pivoted.stack()
    #stack = stack.transpose()
    #dfstack = pd.DataFrame(stack)
    #dfstack['y'] = tgt
    #print(dfstack.head())
    # dfstack.to_csv(fr'C:/Users/nrust/Downloads/test%s_%s_t1.csv' % (participant, i))