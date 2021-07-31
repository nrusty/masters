import pandas as pd
import glob
import numpy as np
import os
import openpyxl
import xlrd

file1 = pd.read_csv('C:/Users/nrust/Downloads/temp/2014-10-02-18-59_001_v4.csv', sep=',', dtype=np.float64)
file2 = pd.read_csv('C:/Users/nrust/Downloads/temp/2014-10-02-18-24_001_v4.csv', sep=',', dtype=np.float64)


2014-10-03-10-49_004_v4


d = {'target': [0,0,0,0,0,0,1,1,1,1,1,1]}
target_id = pd.DataFrame(data=d)
print(target_id.head())
print(target_id.iloc[5]['target'])

path = r'C:/Users/nrust/Downloads/D1_test/001/Breath' # use your path
all_files = glob.glob(path + "/*.csv")

li = []
i = -1
for filename in all_files:
    i = i + 1
    df = pd.read_csv(filename, index_col=None, header=0)
    df['id'] = target_id.iloc[i]['target']
    li.append(df)

frame = pd.concat(li, axis=0, ignore_index=True)



#frame['id'] = target_id['target']

#print(frame.head())
print(frame.describe())
#frame2 = []
#print(frame.iloc[:][100:500])
#frame2 = frame.iloc[:][1000000:2450000]

#print(frame2.describe())


y1 = np.arange(len(frame)/2)
y1.fill(103)
y2 = np.arange(len(frame)/2)
y2.fill(75)
a = np.concatenate((y1, y2), axis=None)
y = pd.Series([0,0,0,0,0,0,1,1,1,1,1,1], index=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

#print(y.head())

if __name__ == '__main__':

    extracted_features = extract_features(frame, column_id="id", column_sort="Time")

    print(extracted_features.head())
    print(extracted_features.tail())
    print(extracted_features.describe())


    impute(extracted_features)
    features_filtered = select_features(extracted_features, y)
    print(features_filtered.head())

    #features_filtered_direct = extract_relevant_features(timeseries, y,
                                                         #column_id='id', column_sort='Time')

    #print(features_filtered_direct.head())