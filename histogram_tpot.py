import pandas as pd
import numpy as np
import os
import neurokit2 as nk
import matplotlib.pyplot as plt
# %matplotlib inline

import padasip as pa
import datetime

plt.rcParams['figure.figsize'] = [15, 9]  # Bigger images

participant = '001'
sampling_rate = 250
all_hist, all_beats = [], []

# List all data files in participant_00X folder
all_files = os.listdir('C:/Users/nrust/Downloads/D0_test/' + participant + '_feat')

# Loop through each file in the subject folder
for i, file in enumerate(all_files[:]):
    data = pd.read_csv('C:/Users/nrust/Downloads/D0_test/' + participant + '_feat/' + file, parse_dates=[0])
    print(file)

    # Create new features
    beats_mean = np.nanmean(data.iloc[:, 2:data.shape[1] - 2], axis=0)
    beats_median = np.nanmedian(data.iloc[:, 2:data.shape[1] - 2], axis=0)
    beats_std = np.nanstd(data.iloc[:, 2:data.shape[1] - 2], axis=0)
    beats = np.concatenate([[data['Time'][1]], beats_mean, beats_median, beats_std, [data['target'][1]]], axis=0)  #


    date_time = file.replace('feat', '')
    date_time = date_time.replace('.csv', '')

    all_beats.append(beats)

    original_col = list(data.iloc[:, 2:data.shape[1] - 2].columns.values)
    col_mean = [orig + '_mean' for orig in original_col]
    col_median = [orig + '_median' for orig in original_col]
    col_std = [orig + '_std' for orig in original_col]
    new_col = np.concatenate([['Time'], col_mean, col_median, col_std, ['target']], axis=0)

    df_beats = pd.DataFrame(all_beats, columns=new_col)
    df_beats.to_csv(fr'C:/Users/nrust/Downloads/D0_test/feat_beat/beats_%s.csv' % (participant), index=False)


def histogram_d1namo():
    # Create histogram
    QT = data['QT'].replace([np.inf, -np.inf, np.nan], np.nan).dropna(axis=0)
    print(file)
    if (not QT.empty):
        QT = QT * 4  # QT in milliseconds
        QT = np.clip(QT, 200, 500)
        # hist = QT.hist(bins=np.linspace(300, 500, num=50))
        # center time of the reading
        date = data['Time'][1]
        # hist.insert(0, 'Time', date)
        # hist.insert(0, 'target', data['target'][1])

        # freq, bins = np.histogram(QT, 50)
        # hist, bin_edges = np.histogram(QT, bins=np.linspace(10, 200, num=10))  # originally 0, 125, 100
        # plt.show()

        date_time = file.replace('feat', '')
        date_time = date_time.replace('.csv', '')

        counts, bins = np.histogram(QT, bins=np.linspace(200, 500, num=50))
        counts = np.insert(counts, len(counts), data['target'][1], axis=0)
        # all_hist.append(hist)
        all_hist.append(counts)
        # all_hist.append(zip(*np.histogram(QT, bins=np.linspace(300, 500, num=50))))

        # print(all_hist)
        df_hist = pd.DataFrame(all_hist)
        df_hist.to_csv(fr'C:/Users/nrust/Downloads/D0_test/hist_%s.csv' % (participant), index=False)
        return


all_hist.plot.hist(grid=True, bins=100, rwidth=0.9, color='#607c8e')
plt.title('QT interval Histogram')
plt.xlabel('Counts')
plt.ylabel('QT length')
plt.grid(axis='y', alpha=0.75)
plt.show()

"""
# Create histogram
QT = data['QT'].replace([np.inf, -np.inf, np.nan], np.nan).dropna(axis=0)  #
hist = QT.hist(bins=100)
# hdf = pd.DataFrame(data=hist, columns=['ECG'])

# hdf.to_csv(fr'C:/Users/nrust/Downloads/D0_test/hist_s{i}.csv', index=False)
"""
