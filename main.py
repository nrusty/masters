import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import padasip as pa
import openpyxl



#x = genfromtxt('C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\001\sensor_data\2014_10_01-10_09_39\2014_10_01-10_09_39_ECG.csv', delimiter=',')
#v = genfromtxt('C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\001\sensor_data\2014_10_01-10_09_39\2014_10_01-10_09_39_Accel.csv', delimiter=',')
x0 = pd.read_csv(r'C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\001\sensor_data\2014_10_01-10_09_39\2014_10_01'
                 r'-10_09_39_ECG.csv', parse_dates=[0], skiprows=8125149, nrows=29999, engine='python')
v0 = pd.read_csv(r'C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\001\sensor_data\2014_10_01-10_09_39\2014_10_01'
                 r'-10_09_39_Accel.csv', skiprows=3254059, nrows=12000, parse_dates=[0], engine='python') # header=0,header=0, index_col=0, squeeze=True,
#v0.to_excel(r'C:\Users\nrust\Downloads\ACC_ms.xlsx', index=False)

#x1 = pd.read_csv(r'C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\001\sensor_data\2014_10_01-10_09_39\2014_10_01'
# r'-10_09_39_ECG.csv', nrows=100000)
#x1.to_excel(r'C:\Users\nrust\Downloads\ECG_ms.xlsx', index=False)

x0.columns = ['Time', 'EcgWaveform']
v0.columns = ['Time', 'Vertical', 'Lateral', 'Sagittal']


x0['Time'] = pd.to_datetime(x0['Time']).astype('datetime64[ms]')
x0['EcgWaveform'] = x0['EcgWaveform'].astype('int')


v0['Time'] = pd.to_datetime(v0['Time']).astype('datetime64[ms]')
v0['Vertical'] = v0['Vertical'].apply(lambda x: '0' if x == '' else x)
v0['Lateral'] = v0['Lateral'].apply(lambda x: '0' if x == '' else x)
v0['Sagittal'] = v0['Sagittal'].apply(lambda x: '0' if x == '' else x)
v0['Vertical'] = v0['Vertical'].astype('int')
v0['Lateral'] = v0['Lateral'].astype('int')
v0['Sagittal'] = v0['Sagittal'].astype('int')

x0.set_index('Time', inplace=True)
v0.set_index('Time', inplace=True)
print(v0.head())
v0 = v0.resample('4ms', offset=1000000).bfill()

x0 = x0.iloc[10000:11000, :]
v0 = v0.iloc[10000:11000, :]

#print(x0.describe())
#print(v0.describe())

print(x0.head())
print(v0.head())


x = x0['EcgWaveform'].to_numpy()

v = v0['Vertical'].to_numpy()

l = v0['Lateral'].to_numpy()

s = v0['Sagittal'].to_numpy()


a = v+l+s
Ma = max(a)
a = a / Ma

Mx = max(x)
Mv = max(v)
Ml = max(l)
Ms = max(s)

x = x / Mx
v = v / Mv
l = l / Ml
s = s / Ms



#x2 = np.roll(x, 2)

d = x
x = np.stack((x, x), axis=-1)
#x2 = np.stack((d, x2), axis=-1)


# plot data
"""
plt.figure(figsize=(15,9))
plt.subplot(411);plt.title("ECG");plt.xlabel("samples - k")
plt.plot(x, "r", linewidth=4, label="ecg")
plt.subplot(412);plt.title("Vertical");plt.xlabel("samples - k")
plt.plot(v, "b", label="Vertical")
plt.subplot(413);plt.title("Lateral");plt.xlabel("samples - k")
plt.plot(l, "b")
plt.subplot(414);plt.title("Sagittal");plt.xlabel("samples - k")
plt.plot(s, "b")
plt.tight_layout()
plt.show()
"""



# identification

# FilterNSSLMS, FilterLMS
f = pa.filters.FilterLMS(n=2, mu=0.1, w="random")
f2 = pa.filters.FilterLMS(n=2, mu=0.05, w="random")

y, e, w = f2.run(a, x)
yv, ev, wv = f.run(v, x)
yl, el, wl = f.run(l, x)
ys, es, ws = f.run(s, x)
yv2 = np.stack((yv, yv), axis=-1)
yvl, evl, wvl = f.run(l, yv2)
yvl2 = np.stack((yvl, yvl), axis=-1)
yvls, evls, wvls = f.run(s, yvl2)

# show results
plt.figure(figsize=(15,9))
plt.subplot(211);plt.title("Adaptation") #;plt.xlabel("samples - k")
#plt.ylim(0, 2)
plt.plot(d[50:],"b", label="d - input")
plt.plot(y[50:],"k", label="y - sum")
plt.plot(yvls[50:],"g", label="y - concatenate")

plt.subplot(212);plt.title("Filter error") # ;plt.xlabel("samples - k")
plt.plot(10*np.log10(e**2),"r", label="e - error [dB]");plt.legend()
plt.plot(10*np.log10(evls**2),"g", label="e2 - error [dB]");plt.legend()
plt.tight_layout()
plt.show()

"""

#plt.plot(yvls,"k", label="y - v&l&s")
#plt.plot(y,"c", label="y - output v+l+s");plt.legend()


"""

"""
plt.subplot(421);plt.title("Sum")
plt.plot(a,"k", label="y - v&l&s")

plt.subplot(423);plt.title("Vertical")
plt.plot(v,"r", label="y - vertical")

plt.subplot(425);plt.title("Lateral")
plt.plot(l,"y", label="y - lateral")

plt.subplot(427);plt.title("Sagittal")
plt.plot(s,"k", label="y - sagittal")


plt.subplot(422);plt.title("Sum Out")
plt.ylim(0.95, 1.1)
plt.plot(y,"k", label="y - v&l&s")

plt.subplot(424);plt.title("Vertical Out")
plt.plot(yv,"r", label="y - vertical")

plt.subplot(426);plt.title("Lateral Out")
plt.plot(yl,"y", label="y - lateral")

plt.subplot(428);plt.title("Sagittal Out")
plt.plot(ys,"k", label="y - sagittal")

plt.tight_layout()
plt.show()

"""

