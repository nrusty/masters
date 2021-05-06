import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

#gapminder[gapminder.columns[0:2]].head()

df = pd.read_csv(r'C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\001\sensor_data\2014_10_01-10_09_39\2014_10_01-10_09_39_Breathing.csv', parse_dates=[0], nrows=20000)
print (df)
df2 = df.iloc[1:20000, [1]]

pa = pd.read_csv(r'C:\Users\nrust\Downloads\D1NAMO\diabetes_subset\001\sensor_data\2014_10_01-10_09_39\2014_10_01-10_09_39_Summary.csv', parse_dates=[0], nrows=20000)
print (pa)
pa2 = pa.iloc[1:2000, [1,5,6]]

z_scores2 = stats.zscore(pa2)
abs_z_scores2 = np.abs(z_scores2)
filtered_entries2 = (abs_z_scores2 < 3).all(axis=1)
new_pa= pa2[filtered_entries2]

z_scores = stats.zscore(df2)
abs_z_scores = np.abs(z_scores)
filtered_entries = (abs_z_scores < 3).all(axis=1)
new_df = df2[filtered_entries]

print(new_df)

fig, axs = plt.subplots(2, 1)
axs = pa2 = pa.iloc[1:2000, [0]]



pa2.plot().get_figure()
new_pa.plot().get_figure()


#plt.plot(pa2)
#plt.plot(new_pa, '-')

plt.show()
# This is a sample Python script.

# Press Shift+F10 to execute it or replace it with your code.
# Press Double Shift to search everywhere for classes, files, tool windows, actions, and settings.


#def print_hi(name):
    # Use a breakpoint in the code line below to debug your script.
    #print(f'Hi, {name}')  # Press Ctrl+F8 to toggle the breakpoint.


# Press the green button in the gutter to run the script.
#if __name__ == '__main__':
    #print_hi('PyCharm')

# See PyCharm help at https://www.jetbrains.com/help/pycharm/
