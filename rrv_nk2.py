# Load NeuroKit and other useful packages
import neurokit2 as nk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
#%matplotlib inline

plt.rcParams['figure.figsize'] = 15, 5  # Bigger images

# Get data
data = pd.read_csv('C:/Users/nrust/Downloads/D1_test/001_other/Breath/Breathing_2014-10-02-21-09.csv')
rsp2 = data["BreathingWaveform"].iloc[0:4500]

normalized_df = (rsp2-rsp2.mean())/rsp2.std()
rsp=(rsp2-rsp2.min())/(rsp2.max()-rsp2.min())

nk.signal_plot(rsp, sampling_rate=25) # Visualize

# Clean signal
cleaned = nk.rsp_clean(rsp, sampling_rate=25, method="biosppy")

# Extract peaks
df, peaks_dict = nk.rsp_peaks(cleaned)
info = nk.rsp_fixpeaks(peaks_dict)
formatted = nk.signal_formatpeaks(info, desired_length=len(cleaned), peak_indices=info["RSP_Peaks"])

nk.signal_plot(pd.DataFrame({"RSP_Raw": rsp, "RSP_Clean": cleaned}), sampling_rate=25, subplots=True)

candidate_peaks = nk.events_plot(peaks_dict['RSP_Peaks'], cleaned)

fixed_peaks = nk.events_plot(info['RSP_Peaks'], cleaned)

# Extract rate
rsp_rate = nk.rsp_rate(cleaned, sampling_rate=25) # Note: You can also replace info with peaks dictionary
#rsp_rate = nk.rsp_rate(formatted, desired_length=len(cleaned), sampling_rate=25) # Note: You can also replace info with peaks dictionary



# Visualize
nk.signal_plot(rsp_rate, sampling_rate=25)
plt.ylabel('BPM')

rrv = nk.rsp_rrv(rsp_rate, info, sampling_rate=25, show=True)
print(rrv.iloc[0,:])

plt.show()