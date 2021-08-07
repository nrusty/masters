import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import openpyxl
import os
import padasip as pa
import datetime

participant = '001'

sampling_rate = 250

# List all data files in Subject_001 folder
all_files = os.listdir(fr'C:/Users/nrust/Downloads/D1NAMO/diabetes_subset/' + participant + '/sensor_data')

# Create an empty directory to store your files (events)
epochs = {}
data2 = []


tgt = pd.read_csv(fr'C:\Users\nrust\Downloads\Target_%s_new.csv' % participant, sep=",", parse_dates=[0],
                  dayfirst=True, engine='python')
print(tgt['date_time'].head())


for i, file in enumerate(all_files[1:]):
    x0 = pd.read_csv(fr'C:/Users/nrust/Downloads/D1NAMO/diabetes_subset/' + participant + '/sensor_data/' + file + '/' +
                     file + '_Summary.csv', parse_dates=[0], dayfirst=True, engine='python')
    print(file)
    print(x0['Time'].head())
    if i > 0:
        all_data = pd.concat([x0['SystemConfidence'], all_data], axis=1)
    else:
        all_data = x0['SystemConfidence']
        total_q = pd.DataFrame(columns=x0.columns)
    #plt.hist(all_data, bins=100)
    #plt.show()

    for k in range(tgt.shape[0]):
        print(tgt['date_time'][k])
        a = x0[abs(x0['Time'] - tgt['date_time'][k]) < datetime.timedelta(seconds=59)] #
        #print(i, k)
        if not a.empty:
            total_q = pd.concat([total_q, x0.iloc[a.index.values]])
            print(total_q.describe())
    clean_hist = total_q['SystemConfidence'].replace([np.inf, -np.inf], 0).dropna(axis=0)
    plt.hist(clean_hist, bins=100)
plt.show()

print(total_q)
total_q.to_csv(fr'C:\Users\nrust\Downloads\qlty_sec_%s.csv' % participant, sep=",", index=False)
