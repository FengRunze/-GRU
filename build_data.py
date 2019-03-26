
#读取数据

import numpy as np
import pandas as pd
import os


# directory = 'dataset'
# filenames = os.listdir(directory)
#
# all_data = pd.DataFrame()
#
# for i in range(len(filenames)):
#         filename = filenames[i]
#         filenames2 = os.listdir(directory + "/" + filename)
#         for j in range(len(filenames2)):
#             filename2 = filenames2[j]
#             filenames3 = os.listdir(directory + "/" + filename + "/" + filename2)
#             for k in range(len(filenames3)):
#                 filename3 = filenames3[k]
#                 data = pd.read_csv("originaldataset" + "/" + filename + "/"+ filename2 + "/" + filename3)
#                 all_data = pd.concat([all_data,data])
# all_data.to_csv("finaldataset/all_dataset.csv", index=False, header=False)
# print(all_data.shape)

directory = 'dataset'
filenames = os.listdir(directory)
all_data1 = pd.DataFrame()
all_data2 = pd.DataFrame()
all_data3 = pd.DataFrame()
all_data4 = pd.DataFrame()

for i in range(256):
    filename = filenames[i]
    data = pd.read_csv("dataset" + "/" + filename)
    all_data1 = pd.concat([all_data1, data])
all_data1.to_csv("finaldataset/all_dataset1.csv", index=False, header=False)
print(all_data1.shape)
del all_data1

for i in range(256):
    filename = filenames[256 + i]
    data = pd.read_csv("dataset" + "/" + filename)
    all_data2 = pd.concat([all_data2, data])
all_data2.to_csv("finaldataset/all_dataset2.csv", index=False, header=False)
print(all_data2.shape)
del all_data2

for i in range(256):
    filename = filenames[512 + i]
    data = pd.read_csv("dataset" + "/" + filename)
    all_data3 = pd.concat([all_data3, data])
all_data3.to_csv("finaldataset/all_dataset3.csv", index=False, header=False)
print(all_data3.shape)
del all_data3

for i in range(256):
    filename = filenames[768 + i]
    data = pd.read_csv("dataset" + "/" + filename)
    all_data4 = pd.concat([all_data4, data])
all_data4.to_csv("finaldataset/all_dataset4.csv", index=False, header=False)
print(all_data4.shape)
del all_data4

