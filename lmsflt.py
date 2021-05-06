import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import padasip as pa
import openpyxl


#x = genfromtxt('C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\001\sensor_data\2014_10_01-10_09_39\2014_10_01-10_09_39_ECG.csv', delimiter=',')
#v = genfromtxt('C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\001\sensor_data\2014_10_01-10_09_39\2014_10_01-10_09_39_Accel.csv', delimiter=',')
x0 = pd.read_csv(r'C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\001\sensor_data\2014_10_01-10_09_39\2014_10_01'
                 r'-10_09_39_ECG.csv', parse_dates=[0], skiprows=8135147, nrows=30000, engine='python')
v0 = pd.read_csv(r'C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\001\sensor_data\2014_10_01-10_09_39\2014_10_01'
                 r'-10_09_39_Accel.csv', skiprows=3254060, nrows=12000, header=0, parse_dates=[0], index_col=0, squeeze=True, engine='python')
#x1 = pd.read_csv(r'C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\001\sensor_data\2014_10_01-10_09_39\2014_10_01'
# r'-10_09_39_ECG.csv', nrows=100000)
#x1.to_excel(r'C:\Users\nrust\Downloads\ECG_ms.xlsx', index=False)

#v0 = v0.resample('5ms', base=417).pad()
#v0.to_excel(r'C:\Users\nrust\Downloads\ACC_ms.xlsx', index=False)

x0.describe()
x0['EcgWaveform'] = x0['EcgWaveform'].astype('int')


x0["EcgWaveform"] = np.where(x0["EcgWaveform"] > 2500, 2000, x0["EcgWaveform"])
v0['Vertical'] = np.where(v0['Vertical'] > 2500, 2000, v0['Vertical'])
v0['Lateral'] = np.where(v0['Lateral'] > 2500, 2000, v0['Lateral'])
v0['Sagittal'] = np.where(v0['Sagittal'] > 2500, 2000, v0['Sagittal'])


x = x0['EcgWaveform'].to_numpy()
x = x[24000:28000]
v = v0['Vertical'].to_numpy()
v = v[24000:28000]
l = v0['Lateral'].to_numpy()
l = l[24000:28000]
s = v0['Sagittal'].to_numpy()
s = s[24000:28000]


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

"""
# plot data

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

# creation of data
#N = 500
# x = np.random.normal(0, 1, (N, 4)) # input matrix
# v = np.random.normal(0, 0.1, N) # noise
# d = 2*x[:,0] + 0.1*x[:,1] - 4*x[:,2] + 0.5*x[:,3] + v # target
#d = x - v


"""
# identification

# FilterNSSLMS, FilterLMS
f = pa.filters.FilterLMS(n=2, mu=0.08, w="random")
f2 = pa.filters.FilterLMS(n=2, mu=0.8, w="random")

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
plt.ylim(0, 2)
plt.plot(d,"b", label="d - input")
plt.plot(y[10:],"k", label="y - sum")
plt.plot(yvls[10:],"g", label="y - concatenate")

plt.subplot(212);plt.title("Filter error") # ;plt.xlabel("samples - k")
plt.plot(10*np.log10(e**2),"r", label="e - error [dB]");plt.legend()
plt.plot(10*np.log10(evls**2),"g", label="e2 - error [dB]");plt.legend()
plt.tight_layout()
plt.show()



#plt.plot(yvls,"k", label="y - v&l&s")
#plt.plot(y,"c", label="y - output v+l+s");plt.legend()


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

