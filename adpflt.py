import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import padasip as pa

#x = genfromtxt('C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\001\sensor_data\2014_10_01-10_09_39\2014_10_01-10_09_39_ECG.csv', delimiter=',')
#v = genfromtxt('C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\001\sensor_data\2014_10_01-10_09_39\2014_10_01-10_09_39_Accel.csv', delimiter=',')
x0 = pd.read_csv(r'C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\001\sensor_data\2014_10_01-10_09_39\2014_10_01-10_09_39_ECG.csv', parse_dates=[0], nrows=20000)
v0 = pd.read_csv(r'C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\001\sensor_data\2014_10_01-10_09_39\2014_10_01-10_09_39_Accel.csv', parse_dates=[0], nrows=20000)

x = x0['EcgWaveform'].to_numpy()
x = x[19000:20000]
print (x)
v = v0['Vertical'].to_numpy()
v = v[19000:20000]
print (v)

a = np.random.normal(0, 1, (1000, 4))
print (a)

# creation of data
N = 500
# x = np.random.normal(0, 1, (N, 4)) # input matrix
# v = np.random.normal(0, 0.1, N) # noise
# d = 2*x[:,0] + 0.1*x[:,1] - 4*x[:,2] + 0.5*x[:,3] + v # target
d = x - v

# identification
f = pa.filters.FilterLMS(n=4, mu=0.1, w="random")
y, e, w = f.run(d, x)

# show results
plt.figure(figsize=(15,9))
plt.subplot(211);plt.title("Adaptation");plt.xlabel("samples - k")
plt.plot(d,"b", label="d - target")
plt.plot(y,"g", label="y - output");plt.legend()
plt.subplot(212);plt.title("Filter error");plt.xlabel("samples - k")
plt.plot(10*np.log10(e**2),"r", label="e - error [dB]");plt.legend()
plt.tight_layout()
plt.show()