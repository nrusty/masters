import pandas as pd
import numpy as np
import os
import neurokit2 as nk
import matplotlib.pyplot as plt
#%matplotlib inline

#plt.rcParams.update({'figure.max_open_warning': 0})

participant = '001'

sampling_rate=100

# List all data files in Subject_001 folder
all_files = os.listdir('C:/Users/nrust/Downloads/D1_test/' + participant + '/Breathing')
print(all_files)
# Create an empty directory to store your files (events)
epochs = {}
data2 = []
# Loop through each file in the subject folder
for i, file in enumerate(all_files):
    # Read the file
    data = pd.read_csv('C:/Users/nrust/Downloads/D1_test/' + participant + '/Breathing/' + file)
    # Add a Label column (e.g Label 1 for epoch 1)
    data['Label'] = np.full(len(data), str(i+1))
    # Set index of data to time in seconds
    index = data.index/sampling_rate
    data = data.set_index(pd.Series(index))
    # Append the file into the dictionary
    epochs[str(i + 1)] = data

    #rsp2 = data["BreathingWaveform"].iloc[0:4500]
    rsp2 = data["BreathingWaveform"]

    # min max normalization
    rsp = (rsp2 - rsp2.min()) / (rsp2.max() - rsp2.min())

    # Clean signal
    cleaned = nk.rsp_clean(rsp, sampling_rate=25)

    # Extract peaks
    df, peaks_dict = nk.rsp_peaks(cleaned)
    info = nk.rsp_fixpeaks(peaks_dict)
    #formatted = nk.signal_formatpeaks(info, desired_length=len(cleaned), peak_indices=info["RSP_Peaks"])

    # Extract rate
    rsp_rate = nk.rsp_rate(cleaned, sampling_rate=25)

    rrv = nk.rsp_rrv(rsp_rate, info, sampling_rate=25, show=False)

    data2.append(rrv.iloc[0,:])

    hdf = pd.DataFrame(data=data2)
    hdf.to_csv(fr'C:\Users\nrust\Downloads\D0_test\hist_brt_s{i}.csv', index=False)
print(epochs)
print(epochs.describe())
