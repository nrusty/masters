import numpy as np
import pandas as pd
import matplotlib.pylab as plt
import padasip as pa
import openpyxl


df = pd.read_csv(r'C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\001\glucose.csv') #, parse_dates=[0]

df['date'] = pd.to_datetime(df['date']).astype('datetime64[ms]')

df['date_eu'] = df['date'].dt.strftime('%d/%m/%Y')
df["match"] = df["date_eu"] + " " + df["time"]
df['match'] = pd.to_datetime(df['match']).astype('datetime64[ms]')

print(df.head())

x0 = pd.read_csv(r'C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\001\sensor_data\2014_10_01-10_09_39\2014_10_01'
                 r'-10_09_39_ECG.csv', parse_dates=[0], skiprows=813514, nrows=30000, engine='python') #8135147, nrows=100,

x0.columns = ['Time', 'EcgWaveform']
x0['Time'] = pd.to_datetime(x0['Time'] - pd.Timedelta('1ms'), ).astype('datetime64[ms]')

x0['index_col'] = x0.index

match = pd.merge(left=x0, left_on='Time',
         right=df, right_on='match')

print(match.head())

match.to_excel(r'C:\Users\nrust\Downloads\ECG_glucose_match.xlsx', index=False)