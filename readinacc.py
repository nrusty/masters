import numpy as np
import pandas as pd
import os
import matplotlib.pylab as plt
import padasip as pa
import openpyxl
import datetime

subject = '009'

glu = pd.read_csv(r'C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\%s\glucose.csv' % subject,
                  parse_dates=[['date', 'time']], dayfirst=True, engine='python')

# List all data files in Subject folder
all_files = os.listdir(r'C:/Users/nrust/Downloads/D1NAMO/diabetes_subset/' + subject + '/sensor_data/')
#all_files = all_files[1:]

j = 6

print(r'C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\%s\sensor_data\%s\%s_Accel.csv' % (subject, all_files[j], all_files[j]))

v0 = pd.read_csv(r'C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\%s\sensor_data\%s\%s_Accel.csv' % (subject,
                                                                                                    all_files[j],
                                                                                                    all_files[j]),
                 parse_dates=[0], dayfirst=True, engine='python')

v0.columns = ['Time', 'Vertical', 'Lateral', 'Sagittal']
idx_list = []
diff = datetime.timedelta(milliseconds=19)



for i in range(1, glu.shape[0]):
    new_idx = v0[abs(v0['Time'] - glu['date_time'].iloc[i]) <= diff].index.values
    if new_idx.all() is not None and len(new_idx) > 0:
        idx_list = idx_list + [new_idx[0]-6000]
print(idx_list)
"""

idx_list = [9656, 39656, 69656, 99656, 129656, 159656, 189656, 219656, 249656, 279656, 309656, 339656, 369656, 399656, 429656, 459656, 489656, 519656, 549656, 579656, 609656, 639656, 669656, 699656, 729656, 759656, 789656, 819656, 849656, 879656, 909656, 939656, 969656, 999656, 1029656, 1059656, 1089656, 1119656, 1149656, 1179656, 1209656, 1239656, 1269656, 1275556, 1299656, 1329656, 1359656, 1389656, 1419656, 1449656, 1479656, 1509656, 1539656, 1569656, 1599656, 1629656, 1659656, 1689656, 1719656, 1749656, 1779656, 1809656, 1839656, 1869656, 1899656, 1929656, 1959656, 1989656, 2019656, 2049656, 2079656, 2109656, 2139556, 2139656, 2169656, 2199656, 2229656, 2259656, 2289656, 2319656, 2349656, 2379656, 2409656, 2439656, 2469656, 2499656, 2529656, 2559656, 2589656, 2619656, 2649656, 2679656, 2709656, 2739656, 2769656, 2799656, 2829656, 2859656, 2889656, 2919656, 2949656, 2979656, 3009656, 3039656, 3069656, 3099656, 3129656, 3159656, 3189656, 3219656, 3249656, 3279656, 3309656, 3339656, 3369656, 3399656, 3429656, 3459656, 3489656, 3519656, 3549656, 3579656, 3609656, 3639656, 3669656, 3699656, 3729656, 3759656, 3789656, 3819656, 3849656, 3879656, 3909656, 3939656, 3969656, 3999656, 4029656, 4059656, 4089656, 4119656, 4149656, 4179656, 4209656, 4239656, 4269656, 4299656, 4329656, 4359656, 4389656, 4419656, 4449656, 4479656, 4509656, 4539656, 4569656]

"""

for i in idx_list:
    v0 = pd.read_csv(r'C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\%s\sensor_data\%s\%s_Accel.csv' % (subject,
                                                                                                          all_files[j],
                                                                                                          all_files[j]),
                     parse_dates=[0], dayfirst=True, skiprows=i, nrows=12000, engine='python')
    v0.columns = ['Time', 'Vertical', 'Lateral', 'Sagittal']

    #v0['Time'] = pd.to_datetime(v0['Time'] + pd.Timedelta('3ms'), ).astype('datetime64[ms]')
    v0['Time'] = pd.to_datetime(v0['Time']).astype('datetime64[ms]')

    time = []
    time = v0['Time'].dt.strftime('%Y-%m-%d-%H-%M')

    # v0.set_index('Time', inplace=True)
    # v0 = v0.resample('4ms').bfill()  # , offset=1000000

    v0['Vertical'] = v0['Vertical'].apply(lambda x: '0' if x == '' else x)
    v0['Lateral'] = v0['Lateral'].apply(lambda x: '0' if x == '' else x)
    v0['Sagittal'] = v0['Sagittal'].apply(lambda x: '0' if x == '' else x)
    v0['Vertical'] = v0['Vertical'].astype('int')
    v0['Lateral'] = v0['Lateral'].astype('int')
    v0['Sagittal'] = v0['Sagittal'].astype('int')

    v0.to_csv(fr'C:\Users\nrust\Downloads\D1_test\%s_Acc\Accel_%s.csv' % (subject, time.loc[6001]), index=False)
