import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import openpyxl
import os
import padasip as pa

t = np.arange(0, 29998, 1)
y = np.sin(4 * np.pi * t)
glucose = pd.read_csv('C:/Users/nrust/Downloads/Target_009.csv', sep=',', parse_dates=[0])
x0 = pd.read_csv('C:/Users/nrust/Downloads/D1_test/006/ECG_2014-10-01-22-05.csv', sep=',', parse_dates=[0], nrows=29998)
v0 = pd.read_csv('C:/Users/nrust/Downloads/D1_test/006_Acc/Accel_2014-10-01-22-05.csv', sep=',', parse_dates=[0], nrows=29998)
tpot_raw = pd.read_csv('C:/Users/nrust/Downloads/D1_test/007/ECG_2014-10-01-11-59.csv', sep=',', parse_dates=[0], nrows=29998)
tpot_data = pd.read_csv('C:/Users/nrust/Downloads/D0_test/007/flt_ECG_2014-10-01-11-59.csv.csv', sep=',', parse_dates=[0], nrows=29998) #, dtype=np.float64
#tpot_data = pd.read_csv('C:/Users/nrust/Downloads/D1_test/007/ECG_2014-10-01-11-49.csv', sep=',', parse_dates=[0], nrows=29998)
x0.set_index('Time', inplace=True)
# resample accelerometer
v0.set_index('Time', inplace=True)
v0 = v0.resample('4ms').bfill()


glucose.set_index('date_time', inplace=True)
glucose['glucose'] = glucose['glucose'].astype('float')
#plt.plot(glucose['glucose'], 'o')
a = glucose
a['glucose'] = 4

"""
plt.plot(a['glucose'], 'r')
plt.xlabel('Date (MM-DD HH)')
plt.ylabel('Glucose (mmol/L)')
plt.title('Glucose Level - 005')
plt.grid(True)

plt.show()
"""


x0['EcgWaveform'] = x0['EcgWaveform'].astype('int')

# Substitutes values higher than 2500 with 2000
x0["EcgWaveform"] = np.where(x0["EcgWaveform"] > 2500, 2000, x0["EcgWaveform"])

v0['Vertical'] = np.where(v0['Vertical'] > 2500, 2000, v0['Vertical'])
v0['Lateral'] = np.where(v0['Lateral'] > 2500, 2000, v0['Lateral'])
v0['Sagittal'] = np.where(v0['Sagittal'] > 2500, 2000, v0['Sagittal'])



# Imperative syntax
plt.figure(1)
plt.clf()
plt.plot(tpot_data['EcgWaveform'][50:29998], 'r')
plt.plot(tpot_raw['EcgWaveform'][50:29998])
#plt.plot(v0['Vertical'][:29998]*10)
#plt.plot(x0['EcgWaveform'][:29998])
plt.xlabel('Time (s)')
plt.ylabel('Amplitude (V)')
plt.grid(True)
"""
# Object oriented syntax
fig = plt.figure(2)
fig.clf()
ax = fig.add_subplot(1,1,1)
ax.plot(v0['Vertical'])
ax.set_xlabel('Time (s)')
ax.set_ylabel('Amplitude (V)')
ax.set_title('Sine Wave')
ax.grid(True)
"""
plt.show()