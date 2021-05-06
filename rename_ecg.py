import pandas as pd
import numpy as np
import os
import neurokit2 as nk
import matplotlib.pyplot as plt
#%matplotlib inline
import padasip as pa

plt.rcParams['figure.figsize'] = [15, 9]  # Bigger images

participant = '001'

sampling_rate=250

# List all data files in Subject_001 folder
all_files = os.listdir('C:/Users/nrust/Downloads/D1_test/' + participant)

# Create an empty directory to store your files (events)
epochs = {}
data2 = []
FeatECG = []



for i, file in enumerate(all_files):
    data = pd.read_csv('C:/Users/nrust/Downloads/D1_test/' + participant + '/' + file)

    print(file)
    file2 = file.replace('ECG', 'Accel')
    print(file2)

    data2 = pd.read_csv('C:/Users/nrust/Downloads/D1_test/001_Acc' + '/' + file2)

    time = []
    data['Time'] = pd.to_datetime(data['Time'], ).astype('datetime64[ms]')  # + pd.Timedelta('3ms')
    time0 = data['Time'].dt.strftime('%Y-%d-%m-%H-%M')
    time = data['Time'].dt.strftime('%Y-%d-%m-%H-%M')
    data['Time'] = time0


    data.to_csv(fr'C:\Users\nrust\Downloads\D1_test\001_Acc\Acc2_%s.csv' % time.loc[6001], sep=",", index=False)