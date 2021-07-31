import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import openpyxl
import os
import padasip as pa
import datetime

participant = '008'

sampling_rate = 250

# List all data files in Subject_001 folder
all_files = os.listdir('C:/Users/nrust/Downloads/D0_test/' + participant)

# Create an empty directory to store your files (events)
epochs = {}
data2 = []


glu = pd.read_csv(r'C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\%s\glucose.csv' % participant,
                      parse_dates=[['date', 'time']], dayfirst=True, engine='python')
target = pd.DataFrame(columns=glu.columns)
print(glu['date_time'][:])

#print(all_files)

for i, file in enumerate(all_files[:]):
    x0 = pd.read_csv('C:/Users/nrust/Downloads/D0_test/' + participant + '/' + file, parse_dates=[0], dayfirst=True,
                     nrows=29998, engine='python')
    print(file)
    print(x0['Time'][15001])
    a = glu[abs(glu['date_time'] - x0['Time'][15001]) < datetime.timedelta(milliseconds=7)] #
    print(a.index.values)

    if (not a.empty):
        if a['glucose'][a.index.values].item() > 4:
            a['comments'] = 0
        else:
            a['comments'] = 1
        target = target.append(a)
        #print('target: ', target)
    #print(i)
print(target)
target.to_csv(fr'C:\Users\nrust\Downloads\Target_%s.csv' % participant, sep=",",
              index=False)
