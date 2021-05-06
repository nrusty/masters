import numpy as np
import pandas as pd
import os
import matplotlib.pylab as plt
import padasip as pa
import openpyxl
import datetime

# Acc for i in range(3260058, 4370059, 30000):  2014_10_01-10_09_39\2014_10_01-10_09_39
# for i in range(7558, 3817559, 30000): 2014_10_02-10_56_44\2014_10_02-10_56_44
# for i in range(9558, 4509559, 30000):   2014_10_03-06_36_24\2014_10_03-06_36_24
# for i in range(18258, 1818259, 30000):   2014_10_04-06_34_57\2014_10_04-06_34_57

subject = '002'

glu = pd.read_csv(r'C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\%s\glucose.csv' % subject,
                  parse_dates=[['date', 'time']], dayfirst=True, engine='python')

# List all data files in Subject folder
all_files = os.listdir(r'C:/Users/nrust/Downloads/D1NAMO/diabetes_subset/' + subject + '/sensor_data/')
#all_files = all_files[1:]

j = 8

print(r'C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\%s\sensor_data\%s\%s_Breathing.csv' % (subject, all_files[j],
                                                                                               all_files[j]))

x0 = pd.read_csv(r'C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\%s\sensor_data\%s\%s_Breathing.csv' % (subject,
                                                                                                    all_files[j],
                                                                                                    all_files[j]),
                 parse_dates=[0], dayfirst=True, engine='python')

x0.columns = ['Time', 'BreathingWaveform']
idx_list = []
diff = datetime.timedelta(milliseconds=89)

for i in range(1, glu.shape[0]):
    new_idx = x0[abs(x0['Time'] - glu['date_time'].iloc[i]) <= diff].index.values
    if new_idx.all() is not None and len(new_idx) > 0:
        idx_list = idx_list + [new_idx[0]-1500]
print(idx_list)

for i in idx_list:
    x0 = pd.read_csv(r'C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\%s\sensor_data\%s\%s_Breathing.csv' % (subject,
                                                                                                          all_files[j],
                                                                                                          all_files[j]),
                     parse_dates=[0], dayfirst=True, skiprows=i, nrows=3000, engine='python')
    x0.columns = ['Time', 'BreathingWaveform']
    time = []
    x0['Time'] = pd.to_datetime(x0['Time'], ).astype('datetime64[ms]') #+ pd.Timedelta('3ms')
    time = x0['Time'].dt.strftime('%Y-%m-%d-%H-%M')

    x0.to_csv(fr'C:\Users\nrust\Downloads\D1_test\%s_Breathing\Breathing_%s.csv' % (subject, time.loc[1501]), sep=",",
              index=False)

