import pandas as pd
import numpy as np
import datetime


tpot_data = pd.read_csv('C:/Users/nrust/Downloads/ecg_feat_007.csv', sep=',', parse_dates=[0], engine='python')
tpot_data['date'] = tpot_data['date'] #+ datetime.timedelta(seconds=60)
tpot_data = tpot_data.drop(columns=['HRV_ULF', 'HRV_VLF'])
tpot_data['date'] = tpot_data['date'].dt.strftime('%Y-%m-%d-%H-%M')
# Drop rows with any empty cells
#tpot_data.dropna(axis=0, how='all', thresh=None, subset=None, inplace=True)
#print(tpot_data['date'])
tpot_tgt = pd.read_csv('C:/Users/nrust/Downloads/Target_007.csv', sep=',', parse_dates=[0], engine='python')
tpot_tgt = tpot_tgt.rename(columns={"date_time": "date", "comments": "target"})
tpot_tgt['date'] = tpot_tgt['date'].dt.strftime('%Y-%m-%d-%H-%M')
tpot_tgt = tpot_tgt.drop(columns=['glucose', 'type'])
#print(tpot_tgt['date'])

tpot = pd.merge(tpot_data, tpot_tgt, how="inner", on=["date"])
#print(tpot.head(5))
#print(tpot.describe())

#tpot.to_csv(fr'C:\Users\nrust\Downloads\dataset_008.csv', sep=",", index=False)


tpot_data = pd.read_csv(fr'C:\Users\nrust\Downloads\categories.csv', sep=',', engine='python')
tpot_data['percent'] = tpot_data['positive']/(tpot_data['negative']+tpot_data['positive'])
print(tpot_data.describe())
print(tpot_data['percent'].std())
print(tpot_data['positive'].sum()/(tpot_data['negative'].sum()+tpot_data['positive'].sum()))
print(tpot_data['positive'].sum())