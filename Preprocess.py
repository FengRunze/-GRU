
#数据预处理

#encoding:utf-8
import pandas as pd
from sklearn import preprocessing, decomposition


"""READ FILE"""
X_Y_1 = pd.read_csv(filepath_or_buffer="finaldataset/all_dataset1.csv", header=None,low_memory=False)
X_Y_2 = pd.read_csv(filepath_or_buffer="finaldataset/all_dataset2.csv", header=None,low_memory=False)
X_Y_3 = pd.read_csv(filepath_or_buffer="finaldataset/all_dataset3.csv", header=None,low_memory=False)
#X_Y_4 = pd.read_csv(filepath_or_buffer="finaldataset/all_dataset1.csv", header=None,low_memory=False)

# (700001, 49)
# (700001, 49)
# (700001, 49)
# (440044, 49)

"""CONCATENATE TRAIN_X_Y AND TEST_X_Y"""
total_X_Y = pd.concat([X_Y_1,X_Y_2,X_Y_3], axis=0, ignore_index=True)


# (2540047, 49)

"""SPLIT DATA AND LABEL"""


total_X = total_X_Y.iloc[:, :84]

total_Y = total_X_Y.iloc[:, 84:]


# for i in range(total_Y.shape[0]):
#     if type(total_Y.iloc[i, 0]) == int:
#         print("第"+ i +"行")
#     # print(total_Y.iloc[i, 0])
#     str(total_Y.iloc[i, 0])
#     total_Y.iloc[i, 0].strip()
#     # total_Y.iloc[i, 0].strip('"')

total_Y_CONCAT = total_Y

print(total_Y)
# for i in range(total_Y.shape[0]):
#     if type(total_Y.iloc[i, 0]) == int:
#         print("第"+ i +"行")
#     print(total_Y.iloc[i, 0])
#     str(total_Y.iloc[i, 0])
#     total_Y.iloc[i, 0].strip()
#     total_Y.iloc[i, 0].strip('"')

# (2540047, 47)
# (2540047, 2)


"""DATA PREPROCESS"""
for column_index in range(83):
    if type(total_X.iloc[0, column_index]) == str:
        #标签编码
        label_encoder = preprocessing.LabelEncoder()
        total_X.iloc[:, column_index] = label_encoder.fit_transform(total_X.iloc[:, column_index])

#去均值和方差归一化
standard_scaler = preprocessing.StandardScaler()
total_X = standard_scaler.fit_transform(total_X)

#数据降维
pca = decomposition.PCA(n_components=24)
total_X = pca.fit_transform(total_X)

"""2 CLASSIFICATION"""
total_Y_2 = total_Y.values
total_Y_2 = preprocessing.label_binarize(total_Y_2, classes=["BENIGN"])
one_hot_encoder1 = preprocessing.OneHotEncoder()
total_Y_2 = one_hot_encoder1.fit_transform(total_Y_2)
total_Y_2 = total_Y_2.toarray()

"""18 CLASSIFICATION"""

total_Y_18 = total_Y.values
label_encoder = preprocessing.LabelEncoder()
total_Y_18 = label_encoder.fit_transform(total_Y_18)
total_Y_18 = total_Y_18.reshape(-1, 1)
one_hot_encoder2 = preprocessing.OneHotEncoder()
total_Y_18 = one_hot_encoder2.fit_transform(total_Y_18)
total_Y_18 = total_Y_18.toarray()




"""TYPE CAST"""
X = pd.DataFrame(total_X)
Y_2 = pd.DataFrame(total_Y_2)
Y_18 = pd.DataFrame(total_Y_18)


print(Y_18)
"""CONCATENATE X AND Y_2 OR Y_10"""
X_Y_2 = pd.concat([X, Y_2, total_Y_CONCAT], axis=1, ignore_index=True)
X_Y_18 = pd.concat([X, Y_18, total_Y_CONCAT], axis=1, ignore_index=True)
"""

"""
#
Y_to_XY2 = {key: value.reset_index(drop=True) for key, value in X_Y_2.groupby(26)}
for key in Y_to_XY2.keys():
    del Y_to_XY2[key][26]
    key1 = key.strip()
    Y_to_XY2[key].to_csv("dataset2/%s.csv" % key1, index=False, header=False)


Y_to_XY18 = {key: value.reset_index(drop=True) for key, value in X_Y_18.groupby(42)}
for key in Y_to_XY18.keys():
     del Y_to_XY18[key][42]
     key2 = key.strip()
     Y_to_XY18[key].to_csv("dataset18/%s.csv" % key2, index=False, header=False)
