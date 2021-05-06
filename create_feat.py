import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import neurokit2 as nk
import padasip as pa
import openpyxl
import xlrd


#ecg = pd.read_csv('C:/Users/nrust/Downloads/D1_test/001/ECG_2014-10-03-10-29.csv', index_col=None)
ecg = pd.read_csv('C:/Users/nrust/Downloads/D0_test/001/fltECG_2014-10-01-19-39.csv.csv', index_col=None)
print(ecg.describe())

ecg['EcgWaveform'] = ecg['EcgWaveform'].astype('float')

x = ecg['EcgWaveform'].to_numpy()


ecg_signal = pd.DataFrame(data=x[1000:], columns=['ECG'])
ecg_signal = nk.ecg_clean(ecg_signal, sampling_rate=250, method="neurokit")
#plt.plot(ecg_signal)
#plt.plot(ecg_signal[1000:2000])
ecg_signal = pd.DataFrame(ecg_signal, columns=['ECG'])
ecg_signal['ECG'] = ecg_signal['ECG'].astype('float')

#ecg_signal['ECG'] = ecg_signal['ECG'].apply(lambda x: '0' if abs(x) > 1 else x)
#ecg_signal['ECG'] = ecg_signal['ECG'].astype(float)
#print(ecg_signal.describe())

cleaned = nk.ecg_clean(ecg_signal, sampling_rate=250) # ecg_signal[7000:8000],
#plt.plot(cleaned)

# Extract R-peaks locations
_, rpeaks = nk.ecg_peaks(cleaned, sampling_rate=250)

# Visualize R-peaks in ECG signal
plot = nk.events_plot(rpeaks['ECG_R_Peaks'][:2], ecg_signal[:500]);plt.xlabel("samples - k");plt.ylabel("amplitude")


# Zooming into the first 5 R-peaks
#plot = nk.events_plot(rpeaks['ECG_R_Peaks'][:2], ecg_signal[:400])

_, waves_peak = nk.ecg_delineate(cleaned, rpeaks, sampling_rate=250)

print(waves_peak['ECG_T_Peaks'])
print(np.array(waves_peak['ECG_T_Peaks']) - np.array(waves_peak['ECG_Q_Peaks']))
print(np.nanmean((np.array(waves_peak['ECG_T_Peaks']) - np.array(waves_peak['ECG_Q_Peaks'])))*4)

ECG_QT = list(np.array(waves_peak['ECG_T_Peaks'])*4 - np.array(waves_peak['ECG_Q_Peaks'])*4)
QTdf = pd.DataFrame(ECG_QT)

bins = np.linspace(100, 500, 100)

hist = QTdf.hist(bins=bins);plt.xlabel("ms");plt.ylabel("samples"); plt.title("normal")
'''
# Visualize the T-peaks, P-peaks, Q-peaks and S-peaks
plot = nk.events_plot([waves_peak['ECG_T_Peaks'],
                       waves_peak['ECG_P_Peaks'],
                       waves_peak['ECG_Q_Peaks'],
                       waves_peak['ECG_S_Peaks']], cleaned)

# Zooming into the first 3 R-peaks, with focus on T_peaks, P-peaks, Q-peaks and S-peaks
plot = nk.events_plot([waves_peak['ECG_T_Peaks'][:2],
                       waves_peak['ECG_P_Peaks'][:2],
                       waves_peak['ECG_Q_Peaks'][:2],
                       waves_peak['ECG_S_Peaks'][:2]], cleaned[:500]);plt.xlabel("samples - k");plt.ylabel("amplitude")

'''
# Delineate
#plt.figure(figsize=(15,9))
signal, waves = nk.ecg_delineate(cleaned[:2500], rpeaks, sampling_rate=250, method="dwt", show=True, show_type='all');plt.xlabel("distance from R peak - s");plt.ylabel("amplitude")



plt.show()
