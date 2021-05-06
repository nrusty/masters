import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import openpyxl
import os
import padasip as pa

participant = '007'

sampling_rate = 250

# List all data files in Subject_001 folder
all_files = os.listdir('C:/Users/nrust/Downloads/D1_test/' + participant)

# Create an empty directory to store your files (events)
epochs = {}
data2 = []
FeatECG = []

#322 3 issue
#3 issue on 007 Accel_2014-10-01-16-20...404 459 490 506 524 553
# 009 64 117
for i, file in enumerate(all_files[:1]):
    x0 = pd.read_csv('C:/Users/nrust/Downloads/D1_test/' + participant + '/' + file, parse_dates=[0],
                     nrows=29998, engine='python')
    print(file)
    print(i)
    file2 = file.replace('ECG', 'Accel')
    #print(file2)
    # add check if file exists
    v0 = pd.read_csv('C:/Users/nrust/Downloads/D1_test/%s_Acc' % participant + '/' + file2, parse_dates=[0],
                     nrows=29998, engine='python')
    # resample accelerometer
    v0.set_index('Time', inplace=True)
    v0 = v0.resample('4ms').bfill()
    #v0 = 0.5 * (v0.resample('4ms').bfill() + v0.resample('4ms').ffill())


    x0['EcgWaveform'] = x0['EcgWaveform'].astype('int')

    # Substitutes values higher than 2500 with 2000
    x0["EcgWaveform"] = np.where(x0["EcgWaveform"] > 2500, 2000, x0["EcgWaveform"])

    v0['Vertical'] = np.where(v0['Vertical'] > 2500, 2000, v0['Vertical'])
    v0['Lateral'] = np.where(v0['Lateral'] > 2500, 2000, v0['Lateral'])
    v0['Sagittal'] = np.where(v0['Sagittal'] > 2500, 2000, v0['Sagittal'])

    x = x0['EcgWaveform'].to_numpy()
    v = v0['Vertical'].to_numpy()
    l = v0['Lateral'].to_numpy()
    s = v0['Sagittal'].to_numpy()

    a = v + l + s
    # testing with squared distance
    #a = np.sqrt(v**2 + l**2 + s**2)
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

    #x2 = np.roll(x, -12)
    x2 = x
    d = x
    x = np.stack((x, x), axis=-1)
    x2 = np.stack((x2, x2), axis=-1)

    # identification
    a = a[:29998]
    v = v[:29998]
    l = l[:29998]
    s = s[:29998]
    print(len(a))
    print(len(x2))

    # FilterNSSLMS, FilterLMS
    f2 = pa.filters.FilterLMS(n=2, mu=0.08, w="random")  # filter for sum
    f = pa.filters.FilterLMS(n=2, mu=0.05, w="random") # concatenate filter


    y, e, w = f2.run(a, x2)
    yv, ev, wv = f.run(v, x2)
    yl, el, wl = f.run(l, x2)
    ys, es, ws = f.run(s, x2)
    yv2 = np.stack((yv, yv), axis=-1)
    yvl, evl, wvl = f.run(l, yv2)
    yvl2 = np.stack((yvl, yvl), axis=-1)
    yvls, evls, wvls = f.run(s, yvl2)

    f3 = pa.filters.FilterNLMS(n=3, mu=0.1, w="random")
    b = x = np.stack((v, l, s), axis=-1)
    x3 = np.stack((x, x, x), axis=-1)
    #y3, e3, w3 = f3.run(b, x3)

    filtECG = x0
    filtECG['EcgWaveform'] = y * 2000
    #filtECG['EcgWaveform'] = y * 2000
    # filtECG2['EcgWaveform'] = yvls*2000
    #filtECG.to_csv(fr'C:\Users\nrust\Downloads\%s_flt2_%s.csv' % (participant, file), sep=",", index=False)
    #filtECG.to_csv(fr'C:\Users\nrust\Downloads\D0_test\%s\flt_%s.csv' % (participant, file), sep=",", index=False)
    #show_results()


# show results
#def show_results(self):
plt.subplot(311)
plt.xlabel('samples')
plt.ylabel('Acc. amplitude - Sagittal Axis')
plt.plot(v0['Sagittal'][2000:8000])

plt.subplot(312)
plt.xlabel('samples')
plt.ylabel('ECG amplitude')
plt.plot(x0["EcgWaveform"][2000:8000])
plt.subplot(312)
plt.plot(filtECG['EcgWaveform'][2000:8000])
plt.subplot(313)
plt.plot(10 * np.log10(e ** 2), "r", label="e - error [dB]");
plt.legend()

yvls = 1 -yvls + 1.2
#y = 1 - y + 1

plt.figure(figsize=(15, 9))
plt.subplot(211);
plt.title("Adaptation")  # ;plt.xlabel("samples - k")
plt.ylim(0, 2)
plt.plot(d[10:]-0.15, "b", label="d - input");
plt.legend()
plt.plot(y[10:], "k", label="y - sum");
plt.legend()
plt.plot(yvls[10:], "g", label="y - concatenate");
plt.legend()

plt.subplot(212);
plt.title("Filter error")  # ;plt.xlabel("samples - k")
plt.plot(10 * np.log10(e ** 2), "r", label="e - error [dB]");
plt.legend()
plt.plot(10 * np.log10(evls ** 2), "g", label="e2 - error [dB]");
plt.legend()
plt.tight_layout()
plt.show()
#    return self
