import numpy as np
import pandas as pd
import neurokit2 as nk


# Generate 15 seconds of ECG signal (recorded at 250 samples / second)
ecg = nk.ecg_simulate(duration=15, sampling_rate=250, heart_rate=70)

# Process it
signals, info = nk.ecg_process(ecg, sampling_rate=250)

# Visualise the processing
nk.ecg_plot(signals, sampling_rate=250)
a = nk.ecg_plot(signals, sampling_rate=250)
a.savefig()

# Part 2
ecg_signal = nk.data(dataset="ecg_3000hz")['ECG']

# Extract R-peaks locations
_, rpeaks = nk.ecg_peaks(ecg_signal, sampling_rate=3000)

# Delineate
signal, waves = nk.ecg_delineate(ecg_signal, rpeaks, sampling_rate=3000, method="dwt", show=True, show_type='all')