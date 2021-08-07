import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import openpyxl
import os
import padasip as pa
import datetime
from sklearn import preprocessing

participant = '007'

sampling_rate = 250

summary = pd.read_csv(fr'C:\Users\nrust\Downloads\qlty_sec_%s.csv' % participant, sep=",", parse_dates=[0],
                  dayfirst=True, engine='python')
print(summary.columns)

analysis = summary.filter(['HR', 'BR', 'Posture', 'Activity','ECGAmplitude', 'ECGNoise', 'HRConfidence', 'SystemConfidence'])

clean_hist = analysis.replace([np.inf, -np.inf], 0).dropna(axis=0)
#plt.hist(clean_hist, bins=100)
#plt.show()

x = analysis.values #returns a numpy array
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(x)
df = pd.DataFrame(x_scaled, columns=analysis.columns.values)
df = summary.filter(['ECGAmplitude', 'ECGNoise'])
plt.boxplot(df, labels=df.columns.values, showmeans=True)
#axes = plt.gca()
#axes.set_xlim([xmin,xmax])
#axes.set_ylim([-200,200])
#plt.xticks(rotation=90)
plt.show()


