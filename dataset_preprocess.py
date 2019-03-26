import numpy as np
import pandas
from sklearn import preprocessing, decomposition


print('Reading csv files...')
dataframe1 = pandas.read_csv("finaldataset/all_dataset1.csv", header=None,low_memory=False)
dataframe2 = pandas.read_csv("finaldataset/all_dataset2.csv", header=None,low_memory=False)
dataframe3 = pandas.read_csv("finaldataset/all_dataset3.csv", header=None,low_memory=False)
#dataframe4 = pandas.read_csv("finaldataset/all_dataset4.csv", header=None,low_memory=False)

# dataframe1 = pandas.read_csv("original_dataset/UNSW_NB15_training-set.csv", header=None)
# dataframe2 = pandas.read_csv("original_dataset/UNSW_NB15_testing-set.csv", header=None)

print('Concatenating...')
total_dataframe = pandas.concat([dataframe1, dataframe2,dataframe3,dataframe4])
print(total_dataframe.shape)
del dataframe1
del dataframe2

"""
===========   ===============   ================
   total_dataframe   (2540047, 49)
===========   ===============   ================
"""

"""
替换标签列里面的空格单元格为normal
"""

# for line in range(total_dataframe.shape[0]):
#     if total_dataframe[line][47] is np.nan:
#         total_dataframe[line][47].replace(np.nan,"normal")
#
# print(total_dataframe[:, 47])


"""SPLIT DATA AND LABEL"""
total_X = total_dataframe.iloc[1:, :83]
total_Y_CONCAT = total_dataframe.iloc[1:, 84:]
total_Y = total_dataframe.iloc[1:, 84]
"""
==============    ============    ======================================
total_X           (805050, 47)     <class 'pandas.core.frame.DataFrame'>
total_Y_CONCAT    (805050, 1)      <class 'pandas.core.frame.DataFrame'>
total_Y           (805050)         <class 'pandas.core.series.Series'>
==============    ============    ======================================
"""
"""DATA PREPROCESS"""
for column_index in range(83):
    if type(total_X.iloc[0, column_index]) == str:
        label_encoder = preprocessing.LabelEncoder()
        total_X.iloc[:, column_index] = label_encoder.fit_transform(total_X.iloc[:, column_index].astype(str))

standard_scaler = preprocessing.StandardScaler()
total_X = standard_scaler.fit_transform(total_X)

pca = decomposition.PCA(n_components=30)
total_X = pca.fit_transform(total_X)




