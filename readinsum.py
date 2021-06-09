import numpy as np
import pandas as pd
import os
import matplotlib.pylab as plt
import padasip as pa
import openpyxl
import datetime

subject = '007'

glu = pd.read_csv(r'C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\%s\glucose.csv' % subject,
                  parse_dates=[['date', 'time']], dayfirst=True, engine='python')

# List all data files in Subject folder
all_files = os.listdir(r'C:/Users/nrust/Downloads/D1NAMO/diabetes_subset/' + subject + '/sensor_data/')

j = 2

print(
    r'C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\%s\sensor_data\%s\%s.csv' % (subject, all_files[j], all_files[j]))

a0 = pd.read_csv(r'C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\%s\sensor_data\%s\%s_Summary.csv' % (subject,
                                                                                                        all_files[j],
                                                                                                        all_files[j]),
                 parse_dates=[0], dayfirst=True, engine='python')

idx_list = []
diff = datetime.timedelta(milliseconds=500)

for i in range(1, glu.shape[0]):
    new_idx = a0[abs(a0['Time'] - glu['date_time'].iloc[i]) <= diff].index.values
    if new_idx.all() is not None and len(new_idx) > 0:
        idx_list = idx_list + [new_idx[0]]
print(idx_list)
print(len(idx_list))


#for i in idx_list:
a0 = a0.iloc[idx_list]
print(a0.describe())

# a0 = a0.drop(columns=['HRV_ULF', 'HRV_VLF'])
a0 = a0.rename(columns={'Time': 'date'})
a0['date'] = a0['date'].dt.strftime('%Y-%m-%d-%H-%M')
# Drop rows with any empty cells
# tpot_data.dropna(axis=0, how='all', thresh=None, subset=None, inplace=True)
print(a0['date'])
tpot_tgt = pd.read_csv('C:/Users/nrust/Downloads/Target_%s.csv' %subject, sep=',', parse_dates=[0],  dayfirst=True, engine='python')

tpot_tgt = tpot_tgt.rename(columns={"date_time": "date", "comments": "target"})
print(tpot_tgt['date'])
tpot_tgt['date'] = tpot_tgt['date'].dt.strftime('%Y-%m-%d-%H-%M')
tpot_tgt = tpot_tgt.drop(columns=['glucose', 'type'])
print(tpot_tgt['date'])

tpot = pd.merge(a0, tpot_tgt, how="inner", on=["date"])
print(tpot.head(5))
print(tpot.describe())

tpot.to_csv(fr'C:\Users\nrust\Downloads\summary_%s_true%s.csv' % (subject, j), sep=",", index=False)
