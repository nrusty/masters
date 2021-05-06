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

j = 5

print(r'C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\%s\sensor_data\%s\%s.csv' % (subject, all_files[j], all_files[j]))


a0 = pd.read_csv(r'C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\%s\sensor_data\%s\%s_ECG.csv' % (subject,
                                                                                                    all_files[j],
                                                                                                    all_files[j]),
                 parse_dates=[0], dayfirst=True, usecols=['Time'], engine='python')
#skiprows=range(2, 2794141)
#curr_date_time = datetime.datetime.strptime(all_files[j], '%Y_%m_%d-%H_%M_%S')

#day_idx = (glu[glu['date_time'] >= a0['Time'].iloc[-1]]).index.values

idx_list = []
diff = datetime.timedelta(milliseconds=7)


for i in range(1, glu.shape[0]):
    new_idx = a0[abs(a0['Time'] - glu['date_time'].iloc[i]) <= diff].index.values
    if new_idx.all() is not None and len(new_idx) > 0:
        idx_list = idx_list + [new_idx[0]-15000]
print(idx_list)

"""
idx_list = [8225144, 8285144, 8300144, 8375144, 8450144, 8525144, 8600144, 8675144, 8750144, 8825144, 8900144, 8975144, 9050144, 9125144, 9200144, 9275144, 9350144, 9425144, 9500144, 9575144, 9650144, 9725144, 9800144, 9875144, 9950144, 10025144, 10100144, 10175144, 10250144, 10325144, 10400144, 10475144, 10550144, 10625144, 10700144, 10775144, 10850144, 10925144]

[18894, 93894, 168894, 243894, 318894, 393894, 468894, 543894, 618894, 693894, 768894, 843894, 918894, 993894, 1068894, 1143894, 1218894, 1293894, 1368894, 1443894, 1518894, 1593894, 1668894, 1743894, 1818894, 1893894, 1968894, 2043894, 2118894, 2193894, 2268894, 2343894, 2418894, 2493894, 2568894, 2643894, 2718894, 2793894, 2868894, 2943894, 3018894, 3093894, 3168894, 3243894, 3318894, 3393894, 3468894, 3543894, 3618894, 3693894, 3768894, 3843894, 3918894, 3993894, 4068894, 4143894, 4218894, 4293894, 4368894, 4443894, 4518894, 4593894, 4668894, 4743894, 4818894, 4893894, 4968894, 5043894, 5118894, 5193894, 5268894, 5283894, 5343894, 5418894, 5493894, 5568894, 5643894, 5718894, 5793894, 5868894, 5943894, 6018894, 6093894, 6168894, 6243894, 6318894, 6393894, 6468894, 6543894, 6618894, 6693894, 6768894, 6843894, 6918894, 6993894, 7068894, 7143894, 7218894, 7293894, 7368894, 7443894, 7518894, 7593894, 7653894, 7668894, 7743894, 7818894, 7893894, 7968894, 8043894, 8118894, 8193894, 8268894, 8343894, 8418894, 8493894, 8568894, 8643894, 8718894, 8793894, 8868894, 8943894, 9018894, 9093894, 9168894, 9243894, 9318894, 9393894, 9468894, 9543894, 9618894]

[24143, 99143, 174143, 249143, 324143, 399143, 474143, 549143, 624143, 699143, 774143, 849143, 924143, 999143, 1074143, 1149143, 1224143, 1299143, 1374143, 1449143, 1524143, 1599143, 1674143, 1749143, 1824143, 1899143, 1974143, 2049143, 2124143, 2199143, 2274143, 2349143, 2424143, 2499143, 2574143, 2649143, 2724143, 2799143, 2874143, 2949143, 3024143, 3099143, 3174143, 3188893, 3249143, 3324143, 3399143, 3474143, 3549143, 3624143, 3699143, 3774143, 3849143, 3924143, 3999143, 4074143, 4149143, 4224143, 4299143, 4374143, 4449143, 4524143, 4599143, 4674143, 4749143, 4824143, 4899143, 4974143, 5049143, 5124143, 5199143, 5274143, 5348893, 5349143, 5424143, 5499143, 5574143, 5649143, 5724143, 5799143, 5874143, 5949143, 6024143, 6099143, 6174143, 6249143, 6324143, 6399143, 6474143, 6549143, 6624143, 6699143, 6774143, 6849143, 6924143, 6999143, 7074143, 7149143, 7224143, 7299143, 7374143, 7449143, 7524143, 7599143, 7674143, 7749143, 7824143, 7899143, 7974143, 8049143, 8124143, 8199143, 8274143, 8349143, 8424143, 8499143, 8574143, 8649143, 8724143, 8799143, 8874143, 8949143, 9024143, 9099143, 9174143, 9249143, 9324143, 9399143, 9474143, 9549143, 9624143, 9699143, 9774143, 9849143, 9924143, 9999143, 10074143, 10149143, 10224143, 10299143, 10374143, 10449143, 10524143, 10599143, 10674143, 10749143, 10824143, 10899143, 10974143, 11049143, 11124143, 11199143, 11274143, 11349143, 11424143]

[45893, 120893, 195893, 270893, 345893, 420893, 495893, 570893, 645893, 720893, 795893, 870893, 945893, 1020893, 1095893, 1170893, 1245893, 1320893, 1395893, 1470893, 1545893, 1620893, 1695893, 1770893, 1845893, 1920893, 1995893, 2070893, 2145893, 2220893, 2295893, 2370893, 2385643, 2445893, 2520893, 2595893, 2670893, 2745893, 2820893, 2895893, 2970893, 3045893, 3120893, 3195893, 3270893, 3345893, 3420893, 3495893, 3570893, 3645893, 3720893, 3795893, 3870893, 3945893, 4020893, 4095893, 4170893, 4245893, 4320893, 4395893, 4470893, 4545893, 4620893, 4695893, 4770893, 4845893, 4920893, 4965643, 4995893, 5070893, 5145893, 5220893, 5295893, 5370893, 5445893, 5520893, 5595893, 5670893, 5745893, 5820893, 5895893, 5970893, 6045893, 6120893, 6195893, 6270893, 6345893, 6420893]


idx_list = [4170893, 4245893, 4320893, 4395893, 4470893, 4545893, 4620893, 4695893, 4770893, 4845893, 4920893, 4965643, 4995893, 5070893, 5145893, 5220893, 5295893, 5370893, 5445893, 5520893, 5595893, 5670893, 5745893, 5820893, 5895893, 5970893, 6045893, 6120893, 6195893, 6270893, 6345893, 6420893]
"""
#idx_list = [x + 2794141 for x in idx_list]
#print(idx_list)

for i in idx_list:
    x0 = pd.read_csv(r'C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\%s\sensor_data\%s\%s_ECG.csv' % (subject,
                                                                                                        all_files[j],
                                                                                                        all_files[j]),
                     parse_dates=[0], dayfirst=True, skiprows=i, nrows=30000, engine='python')
    x0.columns = ['Time', 'EcgWaveform']
    time = []
    x0['Time'] = pd.to_datetime(x0['Time'], ).astype('datetime64[ms]')  # + pd.Timedelta('3ms')
    time = x0['Time'].dt.strftime('%Y-%m-%d-%H-%M')

    x0.to_csv(fr'C:\Users\nrust\Downloads\D1_test\%s\ECG_%s.csv' % (subject, time.loc[15001]), sep=",", index=False)

