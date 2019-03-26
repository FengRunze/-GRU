
#训练集、验证集、测试集划分并写入到文件

#encoding:utf-8
import pandas as pd
from sklearn import model_selection
import random

nor_abnor_name = [
'ADWARE_DOWGIN.',
'ADWARE_EWIND.',
'ADWARE_FEIWO.',
'ADWARE_GOOLIGAN.',
'ADWARE_KEMOGE.',
'ADWARE_KOODOUS.',
'ADWARE_MOBIDASH.',
'ADWARE_SELFMITE.',
'ADWARE_SHUANET.',
'ADWARE_YOUMI.',
'BENIGN.',
# 'RANSOMWARE_CHARGER.',
# 'RANSOMWARE_JISUT.',
# 'RANSOMWARE_KOLER.',
# 'RANSOMWARE_LOCKERPIN.',
# 'RANSOMWARE_PLETOR.',
# 'RANSOMWARE_PORNDROID.',
# 'RANSOMWARE_RANSOMBO.',
# 'RANSOMWARE_SIMPLOCKER.',
# 'RANSOMWARE_SVPENG.',
# 'RANSOMWARE_WANNALOCKER.',
'SCAREWARE_ANDROIDDEFENDER.',
'SCAREWARE_ANDROIDSPY.',
'SCAREWARE_AVFORANDROID.',
'SCAREWARE_AVPASS.',
'SCAREWARE_FAKEAPP.',
'SCAREWARE_FAKEAPPAL.',
'SCAREWARE_FAKEAV.']
# 'SCAREWARE_FAKEJOBOFFER.',
# 'SCAREWARE_FAKETAOBAO.',
# 'SCAREWARE_PENETHO.',
# 'SCAREWARE_VIRUSSHIELD.',
# 'SMSMALWARE_BEANBOT.',
# 'SMSMALWARE_BIIGE.',
# 'SMSMALWARE_FAKEINST.',
# 'SMSMALWARE_FAKEMART.',
# 'SMSMALWARE_FAKENOTIFY.',
# 'SMSMALWARE_JIFAKE.',
# 'SMSMALWARE_MAZARBOT.',
# 'SMSMALWARE_NANDROBOX.',
# 'SMSMALWARE_PLANKTON.',
# 'SMSMALWARE_SMSSNIFFER.',
# 'SMSMALWARE_ZSONE.']

abnor_name = ['ADWARE_DOWGIN.',
'ADWARE_EWIND.',
'ADWARE_FEIWO.',
'ADWARE_GOOLIGAN.',
'ADWARE_KEMOGE.',
'ADWARE_KOODOUS.',
'ADWARE_MOBIDASH.',
'ADWARE_SELFMITE.',
'ADWARE_SHUANET.',
'ADWARE_YOUMI.',
# 'RANSOMWARE_CHARGER.',
# 'RANSOMWARE_JISUT.',
# 'RANSOMWARE_KOLER.',
# 'RANSOMWARE_LOCKERPIN.',
# 'RANSOMWARE_PLETOR.',
# 'RANSOMWARE_PORNDROID.',
# 'RANSOMWARE_RANSOMBO.',
# 'RANSOMWARE_SIMPLOCKER.',
# 'RANSOMWARE_SVPENG.',
# 'RANSOMWARE_WANNALOCKER.',
'SCAREWARE_ANDROIDDEFENDER.',
'SCAREWARE_ANDROIDSPY.',
'SCAREWARE_AVFORANDROID.',
'SCAREWARE_AVPASS.',
'SCAREWARE_FAKEAPP.',
'SCAREWARE_FAKEAPPAL.',
'SCAREWARE_FAKEAV.']
# 'SCAREWARE_FAKEJOBOFFER.',
# 'SCAREWARE_FAKETAOBAO.',
# 'SCAREWARE_PENETHO.',
# 'SCAREWARE_VIRUSSHIELD.',
# 'SMSMALWARE_BEANBOT.',
# 'SMSMALWARE_BIIGE.',
# 'SMSMALWARE_FAKEINST.',
# 'SMSMALWARE_FAKEMART.',
# 'SMSMALWARE_FAKENOTIFY.',
# 'SMSMALWARE_JIFAKE.',
# 'SMSMALWARE_MAZARBOT.',
# 'SMSMALWARE_NANDROBOX.',
# 'SMSMALWARE_PLANKTON.',
# 'SMSMALWARE_SMSSNIFFER.',
# 'SMSMALWARE_ZSONE.']

"""
'srcip.','dstip.','proto.','state.','dur.','sbytes.','dbytes.','sttl.',
'dttl.','sloss.','dloss.','service.','Sload.','Dload.','Spkts.','Dpkts.','swin.','dwin.','stcpb.','dtcpb.',
'smeansz.','dmeansz.','trans_depth.','res_bdy_len.','Sjit.','Djit.','Stime.','Ltime.','Sintpkt.','Dintpkt.',
'tcprtt.','synack.','ackdat.','is_sm_ips_ports.','ct_state_ttl.','ct_flw_http_mthd.','is_ftp_login.',
'ct_srv_src.','ct_srv_dst.','ct_dst_ltm.','ct_src_ ltm.','ct_src_dport_ltm.','ct_dst_sport_ltm.',
'ct_dst_src_ltm.'
"""



nor_name = ['BENIGN.']

name_10 = random.sample(abnor_name, 10)
name_7 = [name for name in abnor_name if name not in name_10]
print(name_10)
print(name_7)

def split(X, Y):
    train_validation_X, test_X, train_validation_Y, test_Y = model_selection.train_test_split(X, Y, test_size=0.20,
                                                                                              random_state=32)

    train_X, validation_X, train_Y, validation_Y = model_selection.train_test_split(train_validation_X,
                                                                                    train_validation_Y, test_size=0.20,
                                                                                    random_state=32)

    return train_X, train_Y, validation_X, validation_Y, test_X, test_Y


def random_X_Y(X, Y):
    for i in range(11):
        X, _, Y, _ = model_selection.train_test_split(X, Y, test_size=0.0000001, random_state=32)

    return X, Y


def write_X_Y_task_one_2(train_X, train_Y, validation_X, validation_Y, test_X, test_Y):
    train_X.to_csv("task_one_2/train_data.csv", index=False, header=False)
    train_Y.to_csv("task_one_2/train_label.csv", index=False, header=False)

    validation_X.to_csv("task_one_2/validation_data.csv", index=False, header=False)
    validation_Y.to_csv("task_one_2/validation_label.csv", index=False, header=False)

    test_X.to_csv("task_one_2/test_data.csv", index=False, header=False)
    test_Y.to_csv("task_one_2/test_label.csv", index=False, header=False)


def write_X_Y_task_one_18(train_X, train_Y, validation_X, validation_Y, test_X, test_Y):
    train_X.to_csv("task_one_18/train_data.csv", index=False, header=False)
    train_Y.to_csv("task_one_18/train_label.csv", index=False, header=False)

    validation_X.to_csv("task_one_18/validation_data.csv", index=False, header=False)
    validation_Y.to_csv("task_one_18/validation_label.csv", index=False, header=False)

    test_X.to_csv("task_one_18/test_data.csv", index=False, header=False)
    test_Y.to_csv("task_one_18/test_label.csv", index=False, header=False)

def write_X_Y_task_two_2(train_X, train_Y, validation_X, validation_Y, test_X, test_Y):
    train_X.to_csv("task_two_2/train_data.csv", index=False, header=False)
    train_Y.to_csv("task_two_2/train_label.csv", index=False, header=False)

    validation_X.to_csv("task_two_2/validation_data.csv", index=False, header=False)
    validation_Y.to_csv("task_two_2/validation_label.csv", index=False, header=False)

    test_X.to_csv("task_two_2/test_data.csv", index=False, header=False)
    test_Y.to_csv("task_two_2/test_label.csv", index=False, header=False)

def print_X_Y(train_X, train_Y, validation_X, validation_Y, test_X, test_Y):
    print("TrainDataShape:", train_X.shape)
    print("TrainLabelShape:", train_Y.shape)

    print("ValidationDataShape:", validation_X.shape)
    print("ValidationLabelShape:", validation_Y.shape)

    print("TestDataShape:", test_X.shape)
    print("TestLabelShape:", test_Y.shape)

    print("TotalDataShape[0]:", train_X.shape[0] + validation_X.shape[0] + test_X.shape[0])

def split_dateset2_for_task_one():
    """训练数据：100%(9种攻击+1种正常)；
       验证数据：100%(9种攻击+1种正常)；
       测试数据：100%(9种攻击+1种正常)；
    """
    train_X = pd.DataFrame()
    validation_X = pd.DataFrame()
    test_X = pd.DataFrame()

    train_Y = pd.DataFrame()
    validation_Y = pd.DataFrame()
    test_Y = pd.DataFrame()

    for name in abnor_name:
        X_Y = pd.read_csv("dataset2/%scsv" % name, header=None)
        X = X_Y.iloc[:, :24]
        Y = X_Y.iloc[:, 24:]

        train_X_val, train_Y_val, validation_X_val, validation_Y_val, test_X_val, test_Y_val = split(X, Y)

        train_X = pd.concat([train_X, train_X_val], ignore_index=True)
        train_Y = pd.concat([train_Y, train_Y_val], ignore_index=True)
        validation_X = pd.concat([validation_X, validation_X_val], ignore_index=True)
        validation_Y = pd.concat([validation_Y, validation_Y_val], ignore_index=True)
        test_X = pd.concat([test_X, test_X_val], ignore_index=True)
        test_Y = pd.concat([test_Y, test_Y_val], ignore_index=True)

    train_X, train_Y = random_X_Y(train_X, train_Y)
    validation_X, validation_Y = random_X_Y(validation_X, validation_Y)
    test_X, test_Y = random_X_Y(test_X, test_Y)

    write_X_Y_task_one_2(train_X, train_Y, validation_X, validation_Y, test_X, test_Y)

    print_X_Y(train_X, train_Y, validation_X, validation_Y, test_X, test_Y)

def split_dateset2_for_task_one_fragtion():
    """训练数据：100%(9种攻击+1种正常)；
       验证数据：100%(9种攻击+1种正常)；
       测试数据：100%(9种攻击+1种正常)；
    """
    train_X = pd.DataFrame()
    validation_X = pd.DataFrame()
    test_X = pd.DataFrame()

    train_Y = pd.DataFrame()
    validation_Y = pd.DataFrame()
    test_Y = pd.DataFrame()

    total_X_Y = pd.DataFrame()
    total_X = pd.DataFrame()
    total_Y = pd.DataFrame()

    for name in abnor_name:
        X_Y = pd.read_csv("dataset2/%scsv" % name, header=None)
        total_X_Y = pd.concat([total_X_Y, X_Y], ignore_index=True)
    total_X_Y.sample(frac=0.5975)
    total_X = total_X_Y.iloc[:, :24]
    total_Y = total_X_Y.iloc[:, 24:]


    train_X_val, train_Y_val, validation_X_val, validation_Y_val, test_X_val, test_Y_val = split(total_X, total_Y)
    train_X = pd.concat([train_X, train_X_val], ignore_index=True)
    train_Y = pd.concat([train_Y, train_Y_val], ignore_index=True)
    validation_X = pd.concat([validation_X, validation_X_val], ignore_index=True)
    validation_Y = pd.concat([validation_Y, validation_Y_val], ignore_index=True)
    test_X = pd.concat([test_X, test_X_val], ignore_index=True)
    test_Y = pd.concat([test_Y, test_Y_val], ignore_index=True)
    print(total_X.shape)

    BENIGN = pd.read_csv("dataset2/BENIGN.csv", header=None)
    BENIGN_X = BENIGN.iloc[0:100000,:24]
    BENIGN_Y = BENIGN.iloc[0:100000,24:]

    print(BENIGN.shape)

    train_X_val, train_Y_val, validation_X_val, validation_Y_val, test_X_val, test_Y_val = split(BENIGN_X, BENIGN_Y)

    train_X = pd.concat([train_X, train_X_val], ignore_index=True)
    train_Y = pd.concat([train_Y, train_Y_val], ignore_index=True)
    validation_X = pd.concat([validation_X, validation_X_val], ignore_index=True)
    validation_Y = pd.concat([validation_Y, validation_Y_val], ignore_index=True)
    test_X = pd.concat([test_X, test_X_val], ignore_index=True)
    test_Y = pd.concat([test_Y, test_Y_val], ignore_index=True)

    train_X, train_Y = random_X_Y(train_X, train_Y)
    validation_X, validation_Y = random_X_Y(validation_X, validation_Y)
    test_X, test_Y = random_X_Y(test_X, test_Y)

    write_X_Y_task_one_2(train_X, train_Y, validation_X, validation_Y, test_X, test_Y)

    print_X_Y(train_X, train_Y, validation_X, validation_Y, test_X, test_Y)

#split_dateset2_for_task_one_fragtion()
def split_dateset18_for_task_one():
    """训练数据：100%(9种攻击+1种正常)；
       验证数据：100%(9种攻击+1种正常)；
       测试数据：100%(9种攻击+1种正常)；
    """
    train_X = pd.DataFrame()
    validation_X = pd.DataFrame()
    test_X = pd.DataFrame()

    train_Y = pd.DataFrame()
    validation_Y = pd.DataFrame()
    test_Y = pd.DataFrame()

    total_X_Y = pd.DataFrame()
    total_X = pd.DataFrame()
    total_Y = pd.DataFrame()

    for name in abnor_name:
        X_Y = pd.read_csv("dataset18/%scsv" % name, header=None)
        total_X_Y = pd.concat([total_X_Y, X_Y], ignore_index=True)
    total_X_Y.sample(frac=0.5975)
    total_X = total_X_Y.iloc[:, :24]
    total_Y = total_X_Y.iloc[:, 24:]

    train_X_val, train_Y_val, validation_X_val, validation_Y_val, test_X_val, test_Y_val = split(total_X, total_Y)
    train_X = pd.concat([train_X, train_X_val], ignore_index=True)
    train_Y = pd.concat([train_Y, train_Y_val], ignore_index=True)
    validation_X = pd.concat([validation_X, validation_X_val], ignore_index=True)
    validation_Y = pd.concat([validation_Y, validation_Y_val], ignore_index=True)
    test_X = pd.concat([test_X, test_X_val], ignore_index=True)
    test_Y = pd.concat([test_Y, test_Y_val], ignore_index=True)
    print(total_X.shape)

    BENIGN = pd.read_csv("dataset18/BENIGN.csv", header=None)
    BENIGN_X = BENIGN.iloc[0:100000, :24]
    BENIGN_Y = BENIGN.iloc[0:100000, 24:]

    print(BENIGN.shape)

    train_X_val, train_Y_val, validation_X_val, validation_Y_val, test_X_val, test_Y_val = split(BENIGN_X, BENIGN_Y)

    train_X = pd.concat([train_X, train_X_val], ignore_index=True)
    train_Y = pd.concat([train_Y, train_Y_val], ignore_index=True)
    validation_X = pd.concat([validation_X, validation_X_val], ignore_index=True)
    validation_Y = pd.concat([validation_Y, validation_Y_val], ignore_index=True)
    test_X = pd.concat([test_X, test_X_val], ignore_index=True)
    test_Y = pd.concat([test_Y, test_Y_val], ignore_index=True)

    train_X, train_Y = random_X_Y(train_X, train_Y)
    validation_X, validation_Y = random_X_Y(validation_X, validation_Y)
    test_X, test_Y = random_X_Y(test_X, test_Y)

    write_X_Y_task_one_18(train_X, train_Y, validation_X, validation_Y, test_X, test_Y)

    print_X_Y(train_X, train_Y, validation_X, validation_Y, test_X, test_Y)


def split_dateset_for_task_two():
    """训练数据：100%(10种攻击+1种正常)；
       验证数据：100%(10种攻击+1种正常)；
       测试数据：100%(18种攻击+1种正常)；
    """
    train_X = pd.DataFrame()
    validation_X = pd.DataFrame()
    test_X = pd.DataFrame()

    train_Y = pd.DataFrame()
    validation_Y = pd.DataFrame()
    test_Y = pd.DataFrame()

    for name in nor_name:
        X_Y = pd.read_csv("dataset2/%scsv" % name, header=None)
        X = X_Y.iloc[:, :24]
        Y = X_Y.iloc[:, 24:]

        train_X_val, train_Y_val, validation_X_val, validation_Y_val, test_X_val, test_Y_val = split(X, Y)

        train_X = pd.concat([train_X, train_X_val], ignore_index=True)
        train_Y = pd.concat([train_Y, train_Y_val], ignore_index=True)
        validation_X = pd.concat([validation_X, validation_X_val], ignore_index=True)
        validation_Y = pd.concat([validation_Y, validation_Y_val], ignore_index=True)
        test_X = pd.concat([test_X, test_X_val], ignore_index=True)
        test_Y = pd.concat([test_Y, test_Y_val], ignore_index=True)

    for name in name_10:
        X_Y = pd.read_csv("dataset2/%scsv" % name, header=None)
        X = X_Y.iloc[:, :24]
        Y = X_Y.iloc[:, 24:]

        train_X_val, train_Y_val, validation_X_val, validation_Y_val, test_X_val, test_Y_val = split(X, Y)

        train_X = pd.concat([train_X, train_X_val], ignore_index=True)
        train_Y = pd.concat([train_Y, train_Y_val], ignore_index=True)
        validation_X = pd.concat([validation_X, validation_X_val], ignore_index=True)
        validation_Y = pd.concat([validation_Y, validation_Y_val], ignore_index=True)
        test_X = pd.concat([test_X, test_X_val], ignore_index=True)
        test_Y = pd.concat([test_Y, test_Y_val], ignore_index=True)

    for name in name_7:
        X_Y = pd.read_csv("dataset2/%scsv" % name, header=None)
        X = X_Y.iloc[:, :24]
        Y = X_Y.iloc[:, 24:]

        test_X_val, _, test_Y_val, _ = model_selection.train_test_split(X, Y, test_size=0.70, random_state=42)

        test_X = pd.concat([test_X, test_X_val], ignore_index=True)
        test_Y = pd.concat([test_Y, test_Y_val], ignore_index=True)

    train_X, train_Y = random_X_Y(train_X, train_Y)
    validation_X, validation_Y = random_X_Y(validation_X, validation_Y)
    test_X, test_Y = random_X_Y(test_X, test_Y)

    write_X_Y_task_two_2(train_X, train_Y, validation_X, validation_Y, test_X, test_Y)

    print_X_Y(train_X, train_Y, validation_X, validation_Y, test_X, test_Y)

def split_dateset_for_task_two_fragtion():
    """训练数据：100%(10种攻击+1种正常)；
       验证数据：100%(10种攻击+1种正常)；
       测试数据：100%(18种攻击+1种正常)；
    """
    train_X = pd.DataFrame()
    validation_X = pd.DataFrame()
    test_X = pd.DataFrame()

    train_Y = pd.DataFrame()
    validation_Y = pd.DataFrame()
    test_Y = pd.DataFrame()

    for name in nor_name:

        X_Y = pd.read_csv("dataset2/%scsv" % name, header=None)
        X_Y = X_Y.iloc[0:160000,:]
        X = X_Y.iloc[:, :24]
        Y = X_Y.iloc[:, 24:]

        train_X_val, train_Y_val, validation_X_val, validation_Y_val, test_X_val, test_Y_val = split(X, Y)

        train_X = pd.concat([train_X, train_X_val], ignore_index=True)
        train_Y = pd.concat([train_Y, train_Y_val], ignore_index=True)
        validation_X = pd.concat([validation_X, validation_X_val], ignore_index=True)
        validation_Y = pd.concat([validation_Y, validation_Y_val], ignore_index=True)
        test_X = pd.concat([test_X, test_X_val], ignore_index=True)
        test_Y = pd.concat([test_Y, test_Y_val], ignore_index=True)

    for name in name_10:
        X_Y = pd.read_csv("dataset2/%scsv" % name, header=None)
        X = X_Y.iloc[:, :24]
        Y = X_Y.iloc[:, 24:]

        train_X_val, train_Y_val, validation_X_val, validation_Y_val, test_X_val, test_Y_val = split(X, Y)

        train_X = pd.concat([train_X, train_X_val], ignore_index=True)
        train_Y = pd.concat([train_Y, train_Y_val], ignore_index=True)
        validation_X = pd.concat([validation_X, validation_X_val], ignore_index=True)
        validation_Y = pd.concat([validation_Y, validation_Y_val], ignore_index=True)
        test_X = pd.concat([test_X, test_X_val], ignore_index=True)
        test_Y = pd.concat([test_Y, test_Y_val], ignore_index=True)

    for name in name_7:
        X_Y = pd.read_csv("dataset2/%scsv" % name, header=None)
        X = X_Y.iloc[:, :24]
        Y = X_Y.iloc[:, 24:]

        test_X_val, _, test_Y_val, _ = model_selection.train_test_split(X, Y, test_size=0.70, random_state=42)

        test_X = pd.concat([test_X, test_X_val], ignore_index=True)
        test_Y = pd.concat([test_Y, test_Y_val], ignore_index=True)

    train_X, train_Y = random_X_Y(train_X, train_Y)
    validation_X, validation_Y = random_X_Y(validation_X, validation_Y)
    test_X, test_Y = random_X_Y(test_X, test_Y)

    write_X_Y_task_two_2(train_X, train_Y, validation_X, validation_Y, test_X, test_Y)

    print_X_Y(train_X, train_Y, validation_X, validation_Y, test_X, test_Y)
split_dateset_for_task_two_fragtion()
# split_dateset2_for_task_one()
# split_dateset18_for_task_one()
# split_dateset_for_task_two()
#
#
#
# FV6
