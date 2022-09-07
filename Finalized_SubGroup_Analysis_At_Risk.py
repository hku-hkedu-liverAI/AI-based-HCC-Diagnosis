import numpy as np
import os
import glob
import pandas as pd
import math
import SimpleITK as sitk
import numpy
from sklearn.metrics import roc_curve, auc, confusion_matrix


data_id_excel_file_fullpath = '/home/ra1/original/Finalized_Mask/All_database_update.xlsx'
xls = pd.ExcelFile(data_id_excel_file_fullpath)
pyn_part_1 = pd.read_excel(xls, 'PYN part 1')
pyn_part_2 = pd.read_excel(xls, 'PYN part 2')
qeh = pd.read_excel(xls, 'QEH')
kwh = pd.read_excel(xls, 'KWH')
qmh = pd.read_excel(xls, 'QMH')
hku = pd.read_excel(xls, 'HKU')
szh = pd.read_excel(xls, 'SZH')
szf = pd.read_excel(xls, 'SZF')

data_id_excel_file_fullpath_update = '/home/ra1/original/Updated_at_risk.xlsx'
xls_update = pd.ExcelFile(data_id_excel_file_fullpath_update)
xls_train_at_risk = pd.read_excel(xls_update, 'Training')
xls_test_at_risk = pd.read_excel(xls_update, 'Testing')

All_At_Risk_Update_Train = []
for i in xls_train_at_risk.index:
    case_term = xls_train_at_risk['ID'][i]
    if 'AI' in case_term:
        case_term_id = 'ID' + case_term[2:]
    elif 'ID_' in case_term:
        case_term_id = case_term
    elif 'HKU' in case_term:
        case_term_id = case_term
    elif 'QMH' in case_term and '_' in case_term:
        case_term_id = case_term
    elif 'QEH' in case_term and '_' in case_term:
        case_term_id = case_term[0:3] + case_term[5:]
    elif 'SZ' in case_term and '_' not in case_term:
        case_term_id = case_term[0:3] + '_' + case_term[3:]
    elif 'KW' in case_term and '_' in case_term:
        case_term_id = case_term
    else:
        print(case_term, 'Check Case ID!')
    All_At_Risk_Update_Train.append(case_term_id)

print(len(All_At_Risk_Update_Train), "This value should be 840!")

All_At_Risk_Update_Test = []
for i in xls_test_at_risk.index:
    case_term = xls_test_at_risk['ID'][i]
    if 'AI' in case_term:
        case_term_id = 'ID' + case_term[2:]
    elif 'ID_' in case_term:
        case_term_id = case_term
    elif 'HKU' in case_term:
        case_term_id = case_term
    elif 'QMH' in case_term and '_' in case_term:
        case_term_id = case_term
    elif 'QEH' in case_term and '_' in case_term:
        case_term_id = case_term[0:3] + case_term[5:]
    elif 'SZ' in case_term and '_' not in case_term:
        case_term_id = case_term[0:3] + '_' + case_term[3:]
    elif 'KW' in case_term and '_' in case_term:
        case_term_id = case_term
    else:
        print(case_term, 'Check Case ID!')
    All_At_Risk_Update_Test.append(case_term_id)

print(len(All_At_Risk_Update_Test), "This value should be 375!")
# In the training set, there are 840 cases, while in the testing set there are 375 cases!
# *********************************************************************************************************************#
# For subgroup analysis
# 1) At-risk      (at-risk = 1 for HCC and Non-HCC)
# 2) Gold-standard
# 3) 2, 3, 4, LR-M
All_At_Risk = []
for i in pyn_part_1.index:
    id = 'ID' + pyn_part_1['Code'][i][2:]
    if int(pyn_part_1['At-risk'][i]) == 1:
        All_At_Risk.append([id, int(pyn_part_1['At-risk'][i])])

for i in pyn_part_2.index:
    id = 'ID' + pyn_part_2['Code'][i][2:]
    if int(pyn_part_2['At-risk'][i]) == 1:
        All_At_Risk.append([id, int(pyn_part_2['At-risk'][i])])

for i in hku.index:
    if int(hku['At-risk'][i]) == 1:
        All_At_Risk.append([hku['Code'][i], int(hku['At-risk'][i])])

for i in qeh.index:
    if int(qeh['At-risk'][i]) == 1:
        All_At_Risk.append([qeh['Code'][i], int(qeh['At-risk'][i])])

for i in kwh.index:
    if int(kwh['At-risk'][i]) == 1:
        All_At_Risk.append([kwh['Code'][i], int(kwh['At-risk'][i])])

for i in qmh.index:
    if int(qmh['At-risk'][i]) == 1:
        All_At_Risk.append([qmh['Code'][i], int(qmh['At-risk'][i])])

for i in szh.index:
    if not(math.isnan(szh['At-risk'][i])) and int(szh['At-risk'][i]) == 1:
        id = szh['Code'][i][0:3] + '_' + szh['Code'][i][3:]
        All_At_Risk.append([id, int(szh['At-risk'][i])])

for i in szf.index:
    if not(math.isnan(szf['At-risk'][i])) and int(szf['At-risk'][i]) == 1:
        id = szf['Code'][i][0:3] + '_' + szf['Code'][i][3:]
        All_At_Risk.append([id, int(szf['At-risk'][i])])

# All_At_Risk stores the information as follows: [ID, at-risk indicator 1]  ---- 1555 cases
# For Gold-Standard Analysis
HCC_Gold_Standard_List = []
NonHCC_Gold_Standard_List = []
for i in pyn_part_1.index:
    id = 'ID' + pyn_part_1['Code'][i][2:]
    if pyn_part_1['Validated Dx1'][i] == 1 and pyn_part_1['Radiological Dx1'][i] == 1:
        HCC_Gold_Standard_List.append([id, 1])
    elif 'exclude' in str(pyn_part_1['Validated Dx1'][i]) or 'Exclude' in str(pyn_part_1['Validated Dx1'][i]):
        continue
    elif type(pyn_part_1['Validated Dx1'][i]) == float:
        continue
    elif type(pyn_part_1['Validated Dx1'][i] == str) or (pyn_part_1['Validated Dx1'][i] != 1):
        NonHCC_Gold_Standard_List.append([id, 0])

for i in pyn_part_2.index:
    id = 'ID' + pyn_part_2['Code'][i][2:]
    if pyn_part_2['Validated Dx1'][i] == 1 and pyn_part_2['Radiological Dx1'][i] == 1:
        HCC_Gold_Standard_List.append([id, 1])
    elif 'exclude' in str(pyn_part_2['Validated Dx1'][i]) or 'Exclude' in str(pyn_part_2['Validated Dx1'][i]):
        continue
    elif type(pyn_part_2['Validated Dx1'][i]) == float:
        continue
    elif type(pyn_part_2['Validated Dx1'][i]) == str or (pyn_part_2['Validated Dx1'][i] != 1):
        NonHCC_Gold_Standard_List.append([id, 0])

for i in hku.index:
    if hku['Validated Dx1'][i] == 1 and hku['Radiological Dx1'][i] == 1:
        HCC_Gold_Standard_List.append([hku['Code'][i], 1])
    elif 'exclude' in str(hku['Validated Dx1'][i]) or 'Exclude' in str(hku['Validated Dx1'][i]):
        continue
    elif type(hku['Validated Dx1'][i]) == float:
        continue
    elif type(hku['Validated Dx1'][i]) == str or (hku['Validated Dx1'][i] != 1):
        NonHCC_Gold_Standard_List.append([hku['Code'][i], 0])

for i in qeh.index:
    if (qeh['Validated Dx1'][i] == 1 and qeh['Radiological Dx1'][i] == 1) or \
            ('1' in str(qeh['Validated Dx1'][i]) and qeh['Radiological Dx1'][i] == 1):
        HCC_Gold_Standard_List.append([qeh['Code'][i], 1])
    elif 'exclude' in str(qeh['Validated Dx1'][i]) or 'Exclude' in str(qeh['Validated Dx1'][i]):
        continue
    elif type(qeh['Validated Dx1'][i]) == float:
        continue
    elif type(qeh['Validated Dx1'][i]) == str or (qeh['Validated Dx1'][i] != 1):
        NonHCC_Gold_Standard_List.append([qeh['Code'][i], 0])

for i in qmh.index:
    if (qmh['Validated Dx1'][i] == 1 and qmh['Radiological Dx1'][i] == 1) or \
            ('1' in str(qmh['Validated Dx1'][i]) and qmh['Radiological Dx1'][i] == 1):
        HCC_Gold_Standard_List.append([qmh['Code'][i], 1])
    elif 'exclude' in str(qmh['Validated Dx1'][i]) or 'Exclude' in str(qmh['Validated Dx1'][i]):
        continue
    elif type(qmh['Validated Dx1'][i]) == float:
        continue
    elif type(qmh['Validated Dx1'][i]) == str or (qmh['Validated Dx1'][i] != 1):
        NonHCC_Gold_Standard_List.append([qmh['Code'][i], 0])
    else:
        continue

for i in kwh.index:
    if (kwh['Validated Dx1'][i] == 1 and kwh['Radiological Dx1'][i] == 1) or \
            ('1' in str(kwh['Validated Dx1'][i]) and kwh['Radiological Dx1'][i] == 1):
        HCC_Gold_Standard_List.append([kwh['Code'][i], 1])
    elif 'exclude' in str(kwh['Validated Dx1'][i]) or 'Exclude' in str(kwh['Validated Dx1'][i]):
        continue
    elif type(kwh['Validated Dx1'][i]) == numpy.float64 and math.isnan(kwh['Validated Dx1'][i]):
        continue
    elif type(kwh['Validated Dx1'][i]) == str or (kwh['Validated Dx1'][i] != 1):
        NonHCC_Gold_Standard_List.append([kwh['Code'][i], 0])
    else:
        print(kwh['Code'][i], kwh['Validated Dx1'][i], kwh['Radiological Dx1'][i])
        continue

for i in szh.index:
    id = szh['Code'][i][0:3] + '_' + szh['Code'][i][3:]
    if szh['CT_Diagnosis_Level_1'][i] == 1:
        HCC_Gold_Standard_List.append([id, 1])
    else:
        continue

for i in szf.index:
    id = szf['Code'][i][0:3] + '_' + szf['Code'][i][3:]
    NonHCC_Gold_Standard_List.append([id, 0])

print("The total numbers of HCC and NonHCC belonging to Gold_Standard", len(HCC_Gold_Standard_List),
      len(NonHCC_Gold_Standard_List))                                 # HCC 611 and Non-HCC 1835 for gold-standard
All_At_Risk_Patient_ID, All_Gold_Standard_Patient_ID = [], []
for case_term in All_At_Risk:
    All_At_Risk_Patient_ID.append(case_term[0])
for case_term in HCC_Gold_Standard_List:
    All_Gold_Standard_Patient_ID.append(case_term[0])
for case_term in NonHCC_Gold_Standard_List:
    All_Gold_Standard_Patient_ID.append(case_term[0])
print('*<>*'*40)
print(len(All_At_Risk_Patient_ID), len(All_At_Risk_Update_Train) + len(All_At_Risk_Update_Test))
print('*<>*'*40)
# ******************************************************************************************************************** #
# Excluding Cases according to Professor Seto.
ID_Exclude_List_by_Seto = ['ID_0862a_P3', 'ID_0862b_P3', 'ID_0584_P3', 'HKU_0340_P3', 'HKU_0503_P3', 'ID_0816_P3',
                           'QMH_0012_P3', 'QMH_0020a_P2']
data_id_excel_file_fullpath = '/home/ra1/original/LiRad lesions combine testing and training.xlsx'
xls = pd.ExcelFile(data_id_excel_file_fullpath)
ALL_Lesions = pd.read_excel(xls, 'All lesions')
All_Lesions_List, Patient_ID_List = [], []
Patient_ID_List_Phase2, Patient_ID_List_Phase3, Patient_ID_List_Phase4 = [], [], []
Patient_ID_List_Clean = []
Exclude_Lesion_Counter, Include_Lesion_Counter = 0, 0
Patient_ID_Lesion_Type_LiRad_Label = []
Patient_ID_Tace_Record_Clean = []
Patient_ID_RFA_Record_Clean = []
Patient_ID_LiRad_Record_Clean = []
for i in ALL_Lesions.index:
    id = ALL_Lesions['Case'][i]
    if not (id in Patient_ID_List):
        Patient_ID_List.append(id)

    if not (id in Patient_ID_List_Phase2) and 'P2' in id:
        Patient_ID_List_Phase2.append(id)
    elif not (id in Patient_ID_List_Phase3) and 'P3' in id:
        Patient_ID_List_Phase3.append(id)
    elif not (id in Patient_ID_List_Phase4) and 'P4' in id:
        Patient_ID_List_Phase4.append(id)

    Lesion_LiRad = ALL_Lesions['LiRad for Cao'][i]
    Lesion_Type = ALL_Lesions['type'][i]

    if (type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M') \
            or (type(Lesion_LiRad) == int and (int(Lesion_Type) != 6)):    #  or int(Lesion_Type) != 14)):
        Include_Lesion_Counter += 1
        if not (id in Patient_ID_List_Clean):                              # and not(id in ID_Exclude_List_by_Seto):
            Patient_ID_List_Clean.append(id)

        if not (id in ID_Exclude_List_by_Seto):
            Patient_ID_Lesion_Type_LiRad_Label.append([id, int(ALL_Lesions['type'][i]), ALL_Lesions['3D'][i], Lesion_LiRad])
    else:
        Exclude_Lesion_Counter += 1

    if (type(Lesion_LiRad) == str and Lesion_LiRad == 'Exclude') and not (id in Patient_ID_LiRad_Record_Clean):
        Patient_ID_LiRad_Record_Clean.append(id)

    if int(Lesion_Type) == 6 and not(id in Patient_ID_Tace_Record_Clean):
        Patient_ID_Tace_Record_Clean.append(id)      # 89 cases

    if int(Lesion_Type) == 14 and not (id in Patient_ID_RFA_Record_Clean):
        Patient_ID_RFA_Record_Clean.append(id)       # 73 cases

# In P2, P3 and P4, there are 16, 2533 and 2 cases. In total, there are 2551 cases
# Before removing, there are 2332 patients with 3765 lesions
# After tace removal, there are 3731 lesions, 2326 patients
# After RFA, there are 3670 lesions, 2288 patients
# After LiRad exclude, there aree 3661 lesion, 2281 patients.
Patient_ID_Lesion_Type_LiRad_Label_Tace_Excluding = []
Patient_ID_List_Clean_Tace_Excluding = []
for case_info in Patient_ID_Lesion_Type_LiRad_Label:
    case_info_id = case_info[0]
    if case_info_id in Patient_ID_Tace_Record_Clean:
        continue
    else:
        Patient_ID_Lesion_Type_LiRad_Label_Tace_Excluding.append(case_info)

for case_id in Patient_ID_List_Clean:
    if case_id in Patient_ID_Tace_Record_Clean:
        continue
    else:
        Patient_ID_List_Clean_Tace_Excluding.append(case_id)

# Second, Remove RFA
Patient_ID_Lesion_Type_LiRad_Label_Tace_RFA_Excluding = []
Patient_ID_List_Clean_Tace_RFA_Excluding = []
for case_info in Patient_ID_Lesion_Type_LiRad_Label_Tace_Excluding:
    case_info_id = case_info[0]
    if case_info_id in Patient_ID_RFA_Record_Clean:
        continue
    else:
        Patient_ID_Lesion_Type_LiRad_Label_Tace_RFA_Excluding.append(case_info)

for case_id in Patient_ID_List_Clean_Tace_Excluding:
    if case_id in Patient_ID_RFA_Record_Clean:
        continue
    else:
        Patient_ID_List_Clean_Tace_RFA_Excluding.append(case_id)

# Third, Remove "Exclude"
Patient_ID_Lesion_Type_LiRad_Label_Tace_RFA_LiRadExclude_Excluding = []
Patient_ID_List_Clean_Tace_RFA_LiRadExclude_Excluding = []
for case_info in Patient_ID_Lesion_Type_LiRad_Label_Tace_RFA_Excluding:
    case_info_id = case_info[0]
    if case_info_id in Patient_ID_LiRad_Record_Clean:
        continue
    else:
        Patient_ID_Lesion_Type_LiRad_Label_Tace_RFA_LiRadExclude_Excluding.append(case_info)
for case_id in Patient_ID_List_Clean_Tace_RFA_Excluding:
    if case_id in Patient_ID_LiRad_Record_Clean:
        continue
    else:
        Patient_ID_List_Clean_Tace_RFA_LiRadExclude_Excluding.append(case_id)

# Patient_ID_Lesion_Type_LiRad_Label_Tace_RFA_LiRadExclude_Excluding with the format (ID, Lesion Type, 3D_Size, LiRads)
# Patient_ID_List_Clean_Tace_RFA_LiRadExclude_Excluding includes the list of patient id

Singe_Lesion_List = []
Multiple_Lesion_List = []
Patient_ID_Lesion_Size_Dictionary = {}
for case_info in Patient_ID_Lesion_Type_LiRad_Label_Tace_RFA_LiRadExclude_Excluding:
    case_info_id = case_info[0]
    case_info_lesion_type = case_info[1]
    case_info_lesion_size = case_info[2]
    case_info_lesion_lirad = case_info[3]
    if case_info_id not in Patient_ID_Lesion_Size_Dictionary.keys():
        value = list([case_info_lesion_type, case_info_lesion_size, case_info_lesion_lirad])
        Patient_ID_Lesion_Size_Dictionary[case_info_id] = value
    else:
        Patient_ID_Lesion_Size_Dictionary[case_info_id].append([case_info_lesion_type, case_info_lesion_size,
                                                                case_info_lesion_lirad])

for key in Patient_ID_Lesion_Size_Dictionary.keys():
    if len(Patient_ID_Lesion_Size_Dictionary[key]) == 3:
        Singe_Lesion_List.append([key, Patient_ID_Lesion_Size_Dictionary[key]])
        # print(key, Patient_ID_Lesion_Size_Dictionary[key], "Line-136")
    elif len(Patient_ID_Lesion_Size_Dictionary[key]) == 4:
        Multiple_Lesion_List.append([key, Patient_ID_Lesion_Size_Dictionary[key][0:3],
                                     Patient_ID_Lesion_Size_Dictionary[key][3]])
    else:
        Multiple_Lesion_List.append([key, Patient_ID_Lesion_Size_Dictionary[key][0:3],
                                     Patient_ID_Lesion_Size_Dictionary[key][3],
                                     Patient_ID_Lesion_Size_Dictionary[key][4]])

# 1391 patients with one lesion, and 890 patients with more than one lesions
# among 890 patients, 255 patients with more than or equal to 2 kinds of lesions; 635 patients with one kind of lesions
Check_Lesion_Possible_Removed = []
for case_info in Multiple_Lesion_List:
    lesion_size, lesion_type = [], []
    lesion_record = []
    for case_lesion in case_info[1:]:       # [Lesion_1_type, Lesion_1_size, Lesion_1_LiRad]
        lesion_size.append(case_lesion[1])
        lesion_type.append(case_lesion[0])
        lesion_record.append([case_lesion])
    lesion_type_unique = np.unique(np.array(lesion_type))
    # print(lesion_record, lesion_size, lesion_type, "Line-161")
    if len(lesion_type_unique) == 1:
        max_value = np.max(lesion_size)
        ratio_list = lesion_size / max_value
        for idx in range(len(ratio_list)):
            if ratio_list[idx] >= 0.1:
                continue
            else:
                # print(case_info[0], ratio_list, lesion_record, lesion_record[idx])
                Check_Lesion_Possible_Removed.append([case_info[0], lesion_record[idx][0]])

# train_test_split = open('/home/ra1/Documents/2D_3D_Classification_Segment_Split_Finalized_31_October_Update.txt', 'r')
train_test_split = open('/home/ra1/Documents/2D_3D_Classification_Segment_Split_Finalized_31_Oct_Check_3_Nov.txt', 'r')
file_lines = train_test_split.readlines()
Train_ID_Patient_Label_GT, Test_ID_Patient_Label_GT = [], []
Train_ID_List, Test_ID_List = [], []
CNT = 0
Patient_ID = []
for line in file_lines:
    CNT += 1
    if CNT == 1:
        continue
    elif 1 < CNT < 1598:
        patient_id_label = line.split('ata/')[1]
        patient_id = patient_id_label.split(' ')[0]
        patient_label = int(patient_id_label.split(' ')[1])
        Train_ID_List.append(patient_id)
        Patient_ID.append(patient_id)
        Train_ID_Patient_Label_GT.append([patient_id, patient_label])
    elif CNT == 1598:
        continue
    elif CNT > 1598:
        patient_id_label = line.split('ata/')[1]
        patient_id = patient_id_label.split(' ')[0]
        patient_label = int(patient_id_label.split(' ')[1])
        Test_ID_List.append(patient_id)
        Test_ID_Patient_Label_GT.append([patient_id, patient_label])
        Patient_ID.append(patient_id)
    else:
        continue

print("The numbers of patients in the training and testing sets are: %d and %d" %
      (len(Train_ID_Patient_Label_GT), len(Test_ID_Patient_Label_GT)))
print('The total number of patients is %d' % (len(Train_ID_Patient_Label_GT) + len(Test_ID_Patient_Label_GT)))

Singe_Lesion_List = []
Multiple_Lesion_List = []
Patient_ID_Lesion_Size_Dictionary = {}
for case_info in Patient_ID_Lesion_Type_LiRad_Label_Tace_RFA_LiRadExclude_Excluding:
    case_info_id = case_info[0]
    case_info_lesion_type = case_info[1]
    case_info_lesion_size = case_info[2]
    case_info_lesion_lirad = case_info[3]
    if case_info_id not in Patient_ID_Lesion_Size_Dictionary.keys():
        value = list([case_info_lesion_type, case_info_lesion_size, case_info_lesion_lirad])
        Patient_ID_Lesion_Size_Dictionary[case_info_id] = value
    else:
        Patient_ID_Lesion_Size_Dictionary[case_info_id].append([case_info_lesion_type, case_info_lesion_size, case_info_lesion_lirad])

for key in Patient_ID_Lesion_Size_Dictionary.keys():
    if len(Patient_ID_Lesion_Size_Dictionary[key]) == 3:
        Singe_Lesion_List.append([key, Patient_ID_Lesion_Size_Dictionary[key]])
        # print(key, Patient_ID_Lesion_Size_Dictionary[key], "Line-136")
    elif len(Patient_ID_Lesion_Size_Dictionary[key]) == 4:
        Multiple_Lesion_List.append([key, Patient_ID_Lesion_Size_Dictionary[key][0:3],
                                     Patient_ID_Lesion_Size_Dictionary[key][3]])
    else:
        Multiple_Lesion_List.append([key, Patient_ID_Lesion_Size_Dictionary[key][0:3],
                                     Patient_ID_Lesion_Size_Dictionary[key][3],
                                     Patient_ID_Lesion_Size_Dictionary[key][4]])

Train_ID_At_Risk, Test_ID_At_Risk = [], []
Train_ID_Label_At_Risk, Test_ID_Label_At_Risk = [], []
Train_ID_Gold_Standard, Test_ID_Gold_Standard = [], []
Train_ID_Label_Gold_Standard, Test_ID_Label_Gold_Standard = [], []

print(len(Train_ID_List), len(Test_ID_List))      # 1596 training vs 685 testing

CNT = 0
for idx in range(len(All_At_Risk_Update_Train)):      # (len(All_At_Risk_Patient_ID)):
    case_id = All_At_Risk_Update_Train[idx]           #   All_At_Risk_Patient_ID[idx]
    for iidx in range(len(Train_ID_List)):
        case_id_select = Train_ID_List[iidx]
        case_id_label_select = Train_ID_Patient_Label_GT[iidx]
        if case_id in case_id_select:
            Train_ID_At_Risk.append(case_id_select)
            Train_ID_Label_At_Risk.append(case_id_label_select)

for idx in range(len(All_At_Risk_Update_Test)):
    case_id = All_At_Risk_Update_Test[idx]
    for iidx in range(len(Test_ID_List)):
        case_id_select = Test_ID_List[iidx]
        case_id_label_select = Test_ID_Patient_Label_GT[iidx]
        if case_id in case_id_select:
            Test_ID_At_Risk.append(case_id_select)
            Test_ID_Label_At_Risk.append(case_id_label_select)
        else:
            continue
print(Test_ID_At_Risk)
print(Test_ID_At_Risk.index('SZH_0409_P3'), "Line-453")
"""
for idx in range(len(All_At_Risk_Patient_ID)):
    case_id = All_At_Risk_Patient_ID[idx]
    for iidx in range(len(Train_ID_List)):
        case_id_select = Train_ID_List[iidx]
        case_id_label_select = Train_ID_Patient_Label_GT[iidx]
        if case_id in case_id_select:
            Train_ID_At_Risk.append(case_id_select)
            Train_ID_Label_At_Risk.append(case_id_label_select)
        else:
            continue

    for iidx in range(len(Test_ID_List)):
        case_id_select = Test_ID_List[iidx]
        case_id_label_select = Test_ID_Patient_Label_GT[iidx]
        if case_id in case_id_select:
            Test_ID_At_Risk.append(case_id_select)
            Test_ID_Label_At_Risk.append(case_id_label_select)
        else:
            continue
"""

print(len(Train_ID_Label_At_Risk), len(Test_ID_Label_At_Risk), len(Train_ID_List) + len(Test_ID_List))
print(III)
for idx in range(len(All_Gold_Standard_Patient_ID)):
    case_id = All_Gold_Standard_Patient_ID[idx]
    for iidx in range(len(Train_ID_List)):
        case_id_select = Train_ID_List[iidx]
        if case_id in case_id_select:
            Train_ID_Gold_Standard.append(case_id_select)
            Train_ID_Label_Gold_Standard.append(Train_ID_Patient_Label_GT[iidx])
        else:
            continue
    for iidx in range(len(Test_ID_List)):
        case_id_select = Test_ID_List[iidx]
        if case_id in case_id_select:
            Test_ID_Gold_Standard.append(case_id_select)
            Test_ID_Label_Gold_Standard.append((Test_ID_Patient_Label_GT[iidx]))
        else:
            continue

# ******************************************************************************************************************** #
# Type 1: (4, 5 -- HCC)  vs (1, 2, 3, LR-M -- NonHCC)
Train_ID_Patient_Label_Radiologist_At_Risk, Test_ID_Patient_Label_Radiologist_At_Risk = [], []
Train_ID_Patient_Label_GT_From_Excel_At_Risk, Test_ID_Patient_Label_GT_From_Excel_At_Risk = [], []
Train_ID_Lesion_Label_Radiologist_At_Risk, Test_ID_Lesion_Label_Radiologist_At_Risk = [], []
Train_ID_Lesion_Label_GT_At_Risk, Test_ID_Lesion_Label_GT_At_Risk = [], []
# Type 2: (3, 4, 5, M --- HCC) vs (1, 2 --- NonHCC)
Train_ID_Patient_Label_Radiologist_type2_At_Risk, Test_ID_Patient_Label_Radiologist_type2_At_Risk = [], []
Train_ID_Patient_Label_GT_From_Excel_type2_At_Risk, Test_ID_Patient_Label_GT_From_Excel_type2_At_Risk = [], []
Train_ID_Lesion_Label_Radiologist_type2_At_Risk, Test_ID_Lesion_Label_Radiologist_type2_At_Risk = [], []
Train_ID_Lesion_Label_GT_type2_At_Risk, Test_ID_Lesion_Label_GT_type2_At_Risk = [], []
# Type 3: (4, 5, M --- HCC) vs (1, 2, 3 --- NonHCC)
Train_ID_Patient_Label_Radiologist_type3_At_Risk, Test_ID_Patient_Label_Radiologist_type3_At_Risk = [], []
Train_ID_Patient_Label_GT_From_Excel_type3_At_Risk, Test_ID_Patient_Label_GT_From_Excel_type3_At_Risk = [], []
Train_ID_Lesion_Label_Radiologist_type3_At_Risk, Test_ID_Lesion_Label_Radiologist_type3_At_Risk = [], []
Train_ID_Lesion_Label_GT_type3_At_Risk, Test_ID_Lesion_Label_GT_type3_At_Risk = [], []
# Type 4: HCC (4, 5) vs Uncertain (3, LR-M) vs NonHCC (1, 2)
Train_ID_Patient_Label_Radiologist_type4_At_Risk, Test_ID_Patient_Label_Radiologist_type4_At_Risk, \
Test_ID_Patient_Label_Radiologist_type4_2_At_Risk, Test_ID_Patient_Label_Radiologist_type4_3_At_Risk = [], [], [], []
Train_ID_Patient_Label_GT_From_Excel_type4_At_Risk, Test_ID_Patient_Label_GT_From_Excel_type4_At_Risk = [], []
Train_ID_Lesion_Label_Radiologist_type4_At_Risk, Test_ID_Lesion_Label_Radiologist_type4_At_Risk, \
Test_ID_Lesion_Label_Radiologist_type4_2_At_Risk, Test_ID_Lesion_Label_Radiologist_type4_3_At_Risk = [], [], [], []
Train_ID_Lesion_Label_GT_type4_At_Risk, Test_ID_Lesion_Label_GT_type4_At_Risk = [], []
# Type 5: HCC 4, 5, LR_M) vs Uncertain (3) vs NonHCC (1, 2)
Train_ID_Patient_Label_Radiologist_type5_At_Risk, Test_ID_Patient_Label_Radiologist_type5_At_Risk, \
Test_ID_Patient_Label_Radiologist_type5_2_At_Risk, Test_ID_Patient_Label_Radiologist_type5_3_At_Risk = [], [], [], []
Train_ID_Patient_Label_GT_From_Excel_type5_At_Risk, Test_ID_Patient_Label_GT_From_Excel_type5_At_Risk = [], []
Train_ID_Lesion_Label_Radiologist_type5_At_Risk, Test_ID_Lesion_Label_Radiologist_type5_At_Risk, \
Test_ID_Lesion_Label_Radiologist_type5_2_At_Risk, Test_ID_Lesion_Label_Radiologist_type5_3_At_Risk = [], [], [], []
Train_ID_Lesion_Label_GT_type5_At_Risk, Test_ID_Lesion_Label_GT_type5_At_Risk = [], []
# Type 6: LiRad 5 --- HCC, LiRad 1, 2 --- Non-HCC, LiRad 3, 4, LR-M --- Uncertain
Train_ID_Patient_Label_Radiologist_type6_At_Risk, Test_ID_Patient_Label_Radiologist_type6_At_Risk, \
Test_ID_Patient_Label_Radiologist_type6_2_At_Risk, Test_ID_Patient_Label_Radiologist_type6_3_At_Risk = [], [], [], []
Train_ID_Patient_Label_GT_From_Excel_type6_At_Risk, Test_ID_Patient_Label_GT_From_Excel_type6_At_Risk = [], []
Train_ID_Lesion_Label_Radiologist_type6_At_Risk, Test_ID_Lesion_Label_Radiologist_type6_At_Risk, \
Test_ID_Lesion_Label_Radiologist_type6_2_At_Risk, Test_ID_Lesion_Label_Radiologist_type6_3_At_Risk = [], [], [], []
Train_ID_Lesion_Label_GT_type6_At_Risk, Test_ID_Lesion_Label_GT_type6_At_Risk = [], []
# Type 7: LiRad 5 --- HCC; LiRad 1, 2, 3 --- Non-HCC; LiRad 4, LR-M --- Uncertain
Train_ID_Patient_Label_Radiologist_type7_At_Risk, Test_ID_Patient_Label_Radiologist_type7_At_Risk, \
Test_ID_Patient_Label_Radiologist_type7_2_At_Risk, Test_ID_Patient_Label_Radiologist_type7_3_At_Risk = [], [], [], []
Train_ID_Patient_Label_GT_From_Excel_type7_At_Risk, Test_ID_Patient_Label_GT_From_Excel_type7_At_Risk = [], []
Train_ID_Lesion_Label_Radiologist_type7_At_Risk, Test_ID_Lesion_Label_Radiologist_type7_At_Risk, \
Test_ID_Lesion_Label_Radiologist_type7_2_At_Risk, Test_ID_Lesion_Label_Radiologist_type7_3_At_Risk, = [], [], [], []
Train_ID_Lesion_Label_GT_type7_At_Risk, Test_ID_Lesion_Label_GT_type7_At_Risk = [], []
# Type 8: LiRad 5 --- HCC  vs  LiRad 1, 2, 3, 4 and LR-M (Non-HCC)
Train_ID_Patient_Label_Radiologist_type8_At_Risk, Test_ID_Patient_Label_Radiologist_type8_At_Risk = [], []
Train_ID_Patient_Label_GT_From_Excel_type8_At_Risk, Test_ID_Patient_Label_GT_From_Excel_type8_At_Risk = [], []
Train_ID_Lesion_Label_Radiologist_type8_At_Risk, Test_ID_Lesion_Label_Radiologist_type8_At_Risk = [], []
Train_ID_Lesion_Label_GT_type8_At_Risk, Test_ID_Lesion_Label_GT_type8_At_Risk = [], []

CNT = 0
for key in Patient_ID_Lesion_Size_Dictionary.keys():
    values = Patient_ID_Lesion_Size_Dictionary[key]
    if key in Train_ID_At_Risk:
        if len(values) == 3:                              # this patient only has one lesion
            Lesion_Type = values[0]
            Lesion_LiRad = values[2]
            # Obtain the ground-truth label from lesion type, which is based on the mask value
            if Lesion_Type == 2 or Lesion_Type == 6 or Lesion_Type == 10 or Lesion_Type == 19 or Lesion_Type == 20 \
                    or Lesion_Type == 21 or Lesion_Type == 22:
                HCC_nonHCC_GT = 1
            else:
                HCC_nonHCC_GT = 0
            # For type 1, LiRads 1, 2, 3 and LR_M are considered as Non-HCC,  vs LiRad 4, 5 are as HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist = 1
            else:
                HCC_nonHCC_Radiologist = 0

            # For type 2, LiRads 3, 4, 5 and LR-M are considered as HCC, vs LiRad 1, 2 are as Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type2 = 0
            else:
                HCC_nonHCC_Radiologist_type2 = 1

            # For type 3, LiRad 4, 5 and LR-M are considered as HCC, vs LiRads 1, 2, and 3 are Non-HCC
            if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 3):
                HCC_nonHCC_Radiologist_type3 = 0
            else:
                HCC_nonHCC_Radiologist_type3 = 1

            # For type 4, LiRad 4, 5 --- HCC; LR-M and LiRad 3 --- Uncertain, LiRad 1, 2 --- Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type4 = 0  # Non-HCC
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type4 = 1  # HCC
            else:
                HCC_nonHCC_Radiologist_type4 = 2  # Uncertain

            # For type 5, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type5 = 0
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type5 = 1
            elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                HCC_nonHCC_Radiologist_type5 = 1
            else:  # LiRads = 3    ------ Uncertain
                HCC_nonHCC_Radiologist_type5 = 2

            # For type 6, LiRad 5 ---- HCC; LiRad 3, 4 and LR-M ---- Uncertain; LiRad 1, 2 ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):  # LiRad 1, 2 --- Non-HCC
                HCC_nonHCC_Radiologist_type6 = 0
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type6 = 1
            else:  # LiRad 3, 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type6 = 2

            # For type 7, LiRad 5 ---- HCC; LiRad 4 and LR-M ---- Uncertain; LiRad 1, 2, 3---- NonHCC
            if type(Lesion_LiRad) == int and 1 <= Lesion_LiRad <= 3:  # LiRad 1, 2, 3 --- Non-HCC
                HCC_nonHCC_Radiologist_type7 = 0
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type7 = 1
            else:  # LiRad 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type7 = 2

            # For type 8, LiRad 5 ---- HCC; LiRad 1, 2 3, 4 and LR-M ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- Non-HCC
                HCC_nonHCC_Radiologist_type8 = 1
            else:  # LiRad 1,2, 3, LR-M --- Non-HCC
                HCC_nonHCC_Radiologist_type8 = 0

            Train_ID_Lesion_Label_GT_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type2_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type3_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type4_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type5_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type6_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type7_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type8_At_Risk.append([key, HCC_nonHCC_GT])

            Train_ID_Lesion_Label_Radiologist_At_Risk.append([key, HCC_nonHCC_Radiologist])
            Train_ID_Lesion_Label_Radiologist_type2_At_Risk.append([key, HCC_nonHCC_Radiologist_type2])
            Train_ID_Lesion_Label_Radiologist_type3_At_Risk.append([key, HCC_nonHCC_Radiologist_type3])
            Train_ID_Lesion_Label_Radiologist_type4_At_Risk.append([key, HCC_nonHCC_Radiologist_type4])
            Train_ID_Lesion_Label_Radiologist_type5_At_Risk.append([key, HCC_nonHCC_Radiologist_type5])
            Train_ID_Lesion_Label_Radiologist_type6_At_Risk.append([key, HCC_nonHCC_Radiologist_type6])
            Train_ID_Lesion_Label_Radiologist_type7_At_Risk.append([key, HCC_nonHCC_Radiologist_type7])
            Train_ID_Lesion_Label_Radiologist_type8_At_Risk.append([key, HCC_nonHCC_Radiologist_type8])

            Train_ID_Patient_Label_GT_From_Excel_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Patient_Label_GT_From_Excel_type2_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Patient_Label_GT_From_Excel_type3_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Patient_Label_GT_From_Excel_type4_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Patient_Label_GT_From_Excel_type5_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Patient_Label_GT_From_Excel_type6_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Patient_Label_GT_From_Excel_type7_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Patient_Label_GT_From_Excel_type8_At_Risk.append([key, HCC_nonHCC_GT])

            Train_ID_Patient_Label_Radiologist_At_Risk.append([key, HCC_nonHCC_Radiologist])
            Train_ID_Patient_Label_Radiologist_type2_At_Risk.append([key, HCC_nonHCC_Radiologist_type2])
            Train_ID_Patient_Label_Radiologist_type3_At_Risk.append([key, HCC_nonHCC_Radiologist_type3])
            Train_ID_Patient_Label_Radiologist_type4_At_Risk.append([key, HCC_nonHCC_Radiologist_type4])
            Train_ID_Patient_Label_Radiologist_type5_At_Risk.append([key, HCC_nonHCC_Radiologist_type5])
            Train_ID_Patient_Label_Radiologist_type6_At_Risk.append([key, HCC_nonHCC_Radiologist_type6])
            Train_ID_Patient_Label_Radiologist_type7_At_Risk.append([key, HCC_nonHCC_Radiologist_type7])
            Train_ID_Patient_Label_Radiologist_type8_At_Risk.append([key, HCC_nonHCC_Radiologist_type8])
        elif len(values) == 4:
            Lesion_First = values[0:3]
            Lesion_Type = Lesion_First[0]
            Lesion_LiRad = Lesion_First[2]
            HCC_nonHCC_GT_Overall, HCC_nonHCC_Radiologist_Overall = [], []
            HCC_nonHCC_Radiologist_type2_Overall = []
            HCC_nonHCC_Radiologist_type3_Overall = []
            HCC_nonHCC_Radiologist_type4_Overall = []
            HCC_nonHCC_Radiologist_type5_Overall = []
            HCC_nonHCC_Radiologist_type6_Overall = []
            HCC_nonHCC_Radiologist_type7_Overall = []
            HCC_nonHCC_Radiologist_type8_Overall = []

            if Lesion_Type == 2 or Lesion_Type == 6 or Lesion_Type == 10 or Lesion_Type == 19 or Lesion_Type == 20 \
                    or Lesion_Type == 21 or Lesion_Type == 22:
                HCC_nonHCC_GT = 1
                HCC_nonHCC_GT_Overall.append(HCC_nonHCC_GT)
            else:
                HCC_nonHCC_GT = 0
                HCC_nonHCC_GT_Overall.append(HCC_nonHCC_GT)

            if type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist = 1
                HCC_nonHCC_Radiologist_Overall.append(HCC_nonHCC_Radiologist)
            elif (type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M') or (Lesion_LiRad > 0 and Lesion_LiRad < 4):
                HCC_nonHCC_Radiologist = 0
                HCC_nonHCC_Radiologist_Overall.append(HCC_nonHCC_Radiologist)
            else:
                HCC_nonHCC_Radiologist = Lesion_LiRad

            # For type 2, LiRads 3, 4, 5 and LR-M are considered as Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type2 = 0
                HCC_nonHCC_Radiologist_type2_Overall.append(HCC_nonHCC_Radiologist_type2)
            else:
                HCC_nonHCC_Radiologist_type2 = 1
                HCC_nonHCC_Radiologist_type2_Overall.append(HCC_nonHCC_Radiologist_type2)

            # For type 3, LiRad 4, 5 and LR-M are considered as HCC, and LiRads 1, 2, and 3 are Non-HCC
            if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 3):
                HCC_nonHCC_Radiologist_type3 = 0
                HCC_nonHCC_Radiologist_type3_Overall.append(HCC_nonHCC_Radiologist_type3)
            else:
                HCC_nonHCC_Radiologist_type3 = 1
                HCC_nonHCC_Radiologist_type3_Overall.append(HCC_nonHCC_Radiologist_type3)

            # For type 4, LiRad 4, 5 --- HCC; LR-M and LiRad 3 --- Uncertain, LiRad 1, 2 --- Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type4 = 0  # Non-HCC
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type4 = 1  # HCC
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)
            else:
                HCC_nonHCC_Radiologist_type4 = 2  # Uncertain
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)

            # For type 5, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type5 = 0
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type5 = 1
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                HCC_nonHCC_Radiologist_type5 = 1
            else:  # LiRads = 3    ------ Uncertain
                HCC_nonHCC_Radiologist_type5 = 2
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)

            # For type 6, LiRad 5 --- HCC, and LR-M, LiRad 3, 4 --- Uncertain, LiRad 1, 2 --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):  # LiRad 1, 2 --- NonHCC
                HCC_nonHCC_Radiologist_type6 = 0
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5  --- HCC
                HCC_nonHCC_Radiologist_type6 = 1
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)
            else:  # LiRad 3, 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type6 = 2
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)

            # For type 7, LiRad 5 --- HCC, LiRad 4, LR-M ---- Uncertain, LiRad 1, 2, 3 --- NonHCC
            if type(Lesion_LiRad) == int and 1 <= Lesion_LiRad <= 3:  # LiRad 1, 2, 3 --- NonHCC
                HCC_nonHCC_Radiologist_type7 = 0
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type7 = 1
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)
            else:  # LiRads 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type7 = 2
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)

            # For type 8, LiRad 5 --- HCC, LiRad 1, 2, 3, 4 and LR-M --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type8 = 1
                HCC_nonHCC_Radiologist_type8_Overall.append(HCC_nonHCC_Radiologist_type8)
            else:  # LiRad 1, 2, 3, 4 and LR-M    --- Non-HCC
                HCC_nonHCC_Radiologist_type8 = 0
                HCC_nonHCC_Radiologist_type8_Overall.append(HCC_nonHCC_Radiologist_type8)

            Train_ID_Lesion_Label_GT_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type2_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type3_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type4_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type5_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type6_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type7_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type8_At_Risk.append([key, HCC_nonHCC_GT])

            Train_ID_Lesion_Label_Radiologist_At_Risk.append([key, HCC_nonHCC_Radiologist])
            Train_ID_Lesion_Label_Radiologist_type2_At_Risk.append([key, HCC_nonHCC_Radiologist_type2])
            Train_ID_Lesion_Label_Radiologist_type3_At_Risk.append([key, HCC_nonHCC_Radiologist_type3])
            Train_ID_Lesion_Label_Radiologist_type4_At_Risk.append([key, HCC_nonHCC_Radiologist_type4])
            Train_ID_Lesion_Label_Radiologist_type5_At_Risk.append([key, HCC_nonHCC_Radiologist_type5])
            Train_ID_Lesion_Label_Radiologist_type6_At_Risk.append([key, HCC_nonHCC_Radiologist_type6])
            Train_ID_Lesion_Label_Radiologist_type7_At_Risk.append([key, HCC_nonHCC_Radiologist_type7])
            Train_ID_Lesion_Label_Radiologist_type8_At_Risk.append([key, HCC_nonHCC_Radiologist_type8])

            Lesion_Second = values[3]
            Lesion_Type = Lesion_Second[0]
            Lesion_LiRad = Lesion_Second[2]
            if Lesion_Type == 2 or Lesion_Type == 6 or Lesion_Type == 10 or Lesion_Type == 19 or Lesion_Type == 20 \
                    or Lesion_Type == 21 or Lesion_Type == 22:
                HCC_nonHCC_GT = 1
                HCC_nonHCC_GT_Overall.append(HCC_nonHCC_GT)
            else:
                HCC_nonHCC_GT = 0
                HCC_nonHCC_GT_Overall.append(HCC_nonHCC_GT)

            if type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist = 1
                HCC_nonHCC_Radiologist_Overall.append(HCC_nonHCC_Radiologist)
            elif (type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M') or (Lesion_LiRad > 0 and Lesion_LiRad < 4):
                HCC_nonHCC_Radiologist = 0
                HCC_nonHCC_Radiologist_Overall.append(HCC_nonHCC_Radiologist)
            else:
                HCC_nonHCC_Radiologist = Lesion_LiRad

            # For type 2, LiRads 3, 4, 5 and LR-M are considered as Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type2 = 0
                HCC_nonHCC_Radiologist_type2_Overall.append(HCC_nonHCC_Radiologist_type2)
            else:
                HCC_nonHCC_Radiologist_type2 = 1
                HCC_nonHCC_Radiologist_type2_Overall.append(HCC_nonHCC_Radiologist_type2)

            # For type 3, LiRad 4, 5 and LR-M are considered as HCC, and LiRads 1, 2, and 3 are Non-HCC
            if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 3):
                HCC_nonHCC_Radiologist_type3 = 0
                HCC_nonHCC_Radiologist_type3_Overall.append(HCC_nonHCC_Radiologist_type3)
            else:
                HCC_nonHCC_Radiologist_type3 = 1
                HCC_nonHCC_Radiologist_type3_Overall.append(HCC_nonHCC_Radiologist_type3)

            # For type 4, LiRad 4, 5 --- HCC; LR-M and LiRad 3 --- Uncertain, LiRad 1, 2 --- Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type4 = 0  # Non-HCC
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type4 = 1  # HCC
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)
            else:
                HCC_nonHCC_Radiologist_type4 = 2  # Uncertain
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)

            # For type 5, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type5 = 0
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type5 = 1
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                HCC_nonHCC_Radiologist_type5 = 1
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            else:  # LiRads = 3    ------ Uncertain
                HCC_nonHCC_Radiologist_type5 = 2
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)

            # For type 6, LiRad 5 --- HCC, and LR-M, LiRad 3, 4 --- Uncertain, LiRad 1, 2 --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):  # LiRad 1, 2 --- NonHCC
                HCC_nonHCC_Radiologist_type6 = 0
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5  --- HCC
                HCC_nonHCC_Radiologist_type6 = 1
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)
            else:  # LiRad 3, 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type6 = 2
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)

            # For type 7, LiRad 5 --- HCC, LiRad 4, LR-M ---- Uncertain, LiRad 1, 2, 3 --- NonHCC
            if type(Lesion_LiRad) == int and 1 <= Lesion_LiRad <= 3:  # LiRad 1, 2, 3 --- NonHCC
                HCC_nonHCC_Radiologist_type7 = 0
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type7 = 1
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)
            else:  # LiRads 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type7 = 2
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)

            # For type 8, LiRad 5 --- HCC, LiRad 1, 2, 3, 4 and LR-M --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type8 = 1
                HCC_nonHCC_Radiologist_type8_Overall.append(HCC_nonHCC_Radiologist_type8)
            else:  # LiRad 1, 2, 3, 4 and LR-M    --- Non-HCC
                HCC_nonHCC_Radiologist_type8 = 0
                HCC_nonHCC_Radiologist_type8_Overall.append(HCC_nonHCC_Radiologist_type8)

            Train_ID_Lesion_Label_GT_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type2_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type3_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type4_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type5_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type6_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type7_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type8_At_Risk.append([key, HCC_nonHCC_GT])

            Train_ID_Lesion_Label_Radiologist_At_Risk.append([key, HCC_nonHCC_Radiologist])
            Train_ID_Lesion_Label_Radiologist_type2_At_Risk.append([key, HCC_nonHCC_Radiologist_type2])
            Train_ID_Lesion_Label_Radiologist_type3_At_Risk.append([key, HCC_nonHCC_Radiologist_type3])
            Train_ID_Lesion_Label_Radiologist_type4_At_Risk.append([key, HCC_nonHCC_Radiologist_type4])
            Train_ID_Lesion_Label_Radiologist_type5_At_Risk.append([key, HCC_nonHCC_Radiologist_type5])
            Train_ID_Lesion_Label_Radiologist_type6_At_Risk.append([key, HCC_nonHCC_Radiologist_type6])
            Train_ID_Lesion_Label_Radiologist_type7_At_Risk.append([key, HCC_nonHCC_Radiologist_type7])
            Train_ID_Lesion_Label_Radiologist_type8_At_Risk.append([key, HCC_nonHCC_Radiologist_type8])

            if np.max(np.array(HCC_nonHCC_GT_Overall)) == 0:  # NonHCC
                Train_ID_Patient_Label_GT_From_Excel_At_Risk.append([key, 0])
                Train_ID_Patient_Label_GT_From_Excel_type2_At_Risk.append([key, 0])
                Train_ID_Patient_Label_GT_From_Excel_type3_At_Risk.append([key, 0])
                Train_ID_Patient_Label_GT_From_Excel_type4_At_Risk.append([key, 0])
                Train_ID_Patient_Label_GT_From_Excel_type5_At_Risk.append([key, 0])
                Train_ID_Patient_Label_GT_From_Excel_type6_At_Risk.append([key, 0])
                Train_ID_Patient_Label_GT_From_Excel_type7_At_Risk.append([key, 0])
                Train_ID_Patient_Label_GT_From_Excel_type8_At_Risk.append([key, 0])
            else:  # HCC
                Train_ID_Patient_Label_GT_From_Excel_At_Risk.append([key, 1])
                Train_ID_Patient_Label_GT_From_Excel_type2_At_Risk.append([key, 1])
                Train_ID_Patient_Label_GT_From_Excel_type3_At_Risk.append([key, 1])
                Train_ID_Patient_Label_GT_From_Excel_type4_At_Risk.append([key, 1])
                Train_ID_Patient_Label_GT_From_Excel_type5_At_Risk.append([key, 1])
                Train_ID_Patient_Label_GT_From_Excel_type6_At_Risk.append([key, 1])
                Train_ID_Patient_Label_GT_From_Excel_type7_At_Risk.append([key, 1])
                Train_ID_Patient_Label_GT_From_Excel_type8_At_Risk.append([key, 1])

            if np.max(np.array(HCC_nonHCC_Radiologist_Overall)) == 0:
                Train_ID_Patient_Label_Radiologist_At_Risk.append([key, 0])
            else:
                Train_ID_Patient_Label_Radiologist_At_Risk.append([key, 1])

            if np.max(np.array(HCC_nonHCC_Radiologist_type2_Overall)) == 0:
                Train_ID_Patient_Label_Radiologist_type2_At_Risk.append([key, 0])
            else:
                Train_ID_Patient_Label_Radiologist_type2_At_Risk.append([key, 1])

            if np.max(np.array(HCC_nonHCC_Radiologist_type3_Overall)) == 0:
                Train_ID_Patient_Label_Radiologist_type3_At_Risk.append([key, 0])
            else:
                Train_ID_Patient_Label_Radiologist_type3_At_Risk.append([key, 1])

            if np.max(np.array(HCC_nonHCC_Radiologist_type4_Overall)) == 0:
                Train_ID_Patient_Label_Radiologist_type4_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type4_Overall)) == 1:
                Train_ID_Patient_Label_Radiologist_type4_At_Risk.append([key, 1])
            else:
                Train_ID_Patient_Label_Radiologist_type4_At_Risk.append([key, 2])
            # For type 5, patient with two lesions
            if np.max(np.array(HCC_nonHCC_Radiologist_type5_Overall)) == 0:
                Train_ID_Patient_Label_Radiologist_type5_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type5_Overall)) == 1:
                Train_ID_Patient_Label_Radiologist_type5_At_Risk.append([key, 1])
            else:
                Train_ID_Patient_Label_Radiologist_type5_At_Risk.append([key, 2])
            # For type 6, patient with two lesions
            if np.max(np.array(HCC_nonHCC_Radiologist_type6_Overall)) == 0:
                Train_ID_Patient_Label_Radiologist_type6_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type6_Overall)) == 1:
                Train_ID_Patient_Label_Radiologist_type6_At_Risk.append([key, 1])
            else:
                Train_ID_Patient_Label_Radiologist_type6_At_Risk.append([key, 2])
            # For type 7, patient with two lesions
            if np.max(np.array(HCC_nonHCC_Radiologist_type7_Overall)) == 0:
                Train_ID_Patient_Label_Radiologist_type7_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type7_Overall)) == 1:
                Train_ID_Patient_Label_Radiologist_type7_At_Risk.append([key, 1])
            else:
                Train_ID_Patient_Label_Radiologist_type7_At_Risk.append([key, 2])
            # For type 8, patient with two lesions
            if np.max(np.array(HCC_nonHCC_Radiologist_type8_Overall)) == 0:
                Train_ID_Patient_Label_Radiologist_type8_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type8_Overall)) == 1:
                Train_ID_Patient_Label_Radiologist_type8_At_Risk.append([key, 1])
            else:
                Train_ID_Patient_Label_Radiologist_type8_At_Risk.append([key, 2])
        elif len(values) == 5:  # For patient with three lesions
            Lesion_First = values[0:3]
            Lesion_Type = Lesion_First[0]
            Lesion_LiRad = Lesion_First[2]
            HCC_nonHCC_GT_Overall, HCC_nonHCC_Radiologist_Overall = [], []
            HCC_nonHCC_Radiologist_type2_Overall, HCC_nonHCC_Radiologist_type3_Overall = [], []
            HCC_nonHCC_Radiologist_type4_Overall, HCC_nonHCC_Radiologist_type5_Overall = [], []
            HCC_nonHCC_Radiologist_type6_Overall, HCC_nonHCC_Radiologist_type7_Overall = [], []
            HCC_nonHCC_Radiologist_type8_Overall = []

            if Lesion_Type == 2 or Lesion_Type == 6 or Lesion_Type == 10 or Lesion_Type == 19 or Lesion_Type == 20 \
                    or Lesion_Type == 21 or Lesion_Type == 22:
                HCC_nonHCC_GT = 1
                HCC_nonHCC_GT_Overall.append(HCC_nonHCC_GT)
            else:
                HCC_nonHCC_GT = 0
                HCC_nonHCC_GT_Overall.append(HCC_nonHCC_GT)

            if type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist = 1
                HCC_nonHCC_Radiologist_Overall.append(HCC_nonHCC_Radiologist)
            elif (type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M') or (Lesion_LiRad > 0 and Lesion_LiRad < 4):
                HCC_nonHCC_Radiologist = 0
                HCC_nonHCC_Radiologist_Overall.append(HCC_nonHCC_Radiologist)
            else:
                HCC_nonHCC_Radiologist = Lesion_LiRad

            # For type 2, LiRads 3, 4, 5 and LR-M are considered as Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type2 = 0
                HCC_nonHCC_Radiologist_type2_Overall.append(HCC_nonHCC_Radiologist_type2)
            else:
                HCC_nonHCC_Radiologist_type2 = 1
                HCC_nonHCC_Radiologist_type2_Overall.append(HCC_nonHCC_Radiologist_type2)

            # For type 3, LiRad 4, 5 and LR-M are considered as HCC, and LiRads 1, 2, and 3 are Non-HCC
            if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 3):
                HCC_nonHCC_Radiologist_type3 = 0
                HCC_nonHCC_Radiologist_type3_Overall.append(HCC_nonHCC_Radiologist_type3)
            else:
                HCC_nonHCC_Radiologist_type3 = 1
                HCC_nonHCC_Radiologist_type3_Overall.append(HCC_nonHCC_Radiologist_type3)

            # For type 4, LiRad 4, 5 --- HCC; LR-M and LiRad 3 --- Uncertain, LiRad 1, 2 --- Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type4 = 0  # Non-HCC
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type4 = 1  # HCC
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)
            else:
                HCC_nonHCC_Radiologist_type4 = 2  # Uncertain
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)

            # For type 5, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type5 = 0
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type5 = 1
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                HCC_nonHCC_Radiologist_type5 = 1
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            else:  # LiRads = 3    ------ Uncertain
                HCC_nonHCC_Radiologist_type5 = 2
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)

            # For type 6, LiRad 5 --- HCC, and LR-M, LiRad 3, 4 --- Uncertain, LiRad 1, 2 --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):  # LiRad 1, 2 --- NonHCC
                HCC_nonHCC_Radiologist_type6 = 0
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5  --- HCC
                HCC_nonHCC_Radiologist_type6 = 1
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)
            else:  # LiRad 3, 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type6 = 2
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)

            # For type 7, LiRad 5 --- HCC, LiRad 4, LR-M ---- Uncertain, LiRad 1, 2, 3 --- NonHCC
            if type(Lesion_LiRad) == int and 1 <= Lesion_LiRad <= 3:  # LiRad 1, 2, 3 --- NonHCC
                HCC_nonHCC_Radiologist_type7 = 0
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type7 = 1
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)
            else:  # LiRads 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type7 = 2
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)

            # For type 8, LiRad 5 --- HCC, LiRad 1, 2, 3, 4 and LR-M --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type8 = 1
                HCC_nonHCC_Radiologist_type8_Overall.append(HCC_nonHCC_Radiologist_type8)
            else:  # LiRad 1, 2, 3, 4 and LR-M    --- Non-HCC
                HCC_nonHCC_Radiologist_type8 = 0
                HCC_nonHCC_Radiologist_type8_Overall.append(HCC_nonHCC_Radiologist_type8)

            Train_ID_Lesion_Label_GT_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type2_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type3_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type4_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type5_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type6_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type7_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type8_At_Risk.append([key, HCC_nonHCC_GT])

            Train_ID_Lesion_Label_Radiologist_At_Risk.append([key, HCC_nonHCC_Radiologist])
            Train_ID_Lesion_Label_Radiologist_type2_At_Risk.append([key, HCC_nonHCC_Radiologist_type2])
            Train_ID_Lesion_Label_Radiologist_type3_At_Risk.append([key, HCC_nonHCC_Radiologist_type3])
            Train_ID_Lesion_Label_Radiologist_type4_At_Risk.append([key, HCC_nonHCC_Radiologist_type4])
            Train_ID_Lesion_Label_Radiologist_type5_At_Risk.append([key, HCC_nonHCC_Radiologist_type5])
            Train_ID_Lesion_Label_Radiologist_type6_At_Risk.append([key, HCC_nonHCC_Radiologist_type6])
            Train_ID_Lesion_Label_Radiologist_type7_At_Risk.append([key, HCC_nonHCC_Radiologist_type7])
            Train_ID_Lesion_Label_Radiologist_type8_At_Risk.append([key, HCC_nonHCC_Radiologist_type8])

            Lesion_Second = values[3]
            Lesion_Type = Lesion_Second[0]
            Lesion_LiRad = Lesion_Second[2]
            if Lesion_Type == 2 or Lesion_Type == 6 or Lesion_Type == 10 or Lesion_Type == 19 or Lesion_Type == 20 \
                    or Lesion_Type == 21 or Lesion_Type == 22:
                HCC_nonHCC_GT = 1
                HCC_nonHCC_GT_Overall.append(HCC_nonHCC_GT)
            else:
                HCC_nonHCC_GT = 0
                HCC_nonHCC_GT_Overall.append(HCC_nonHCC_GT)

            if type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist = 1
                HCC_nonHCC_Radiologist_Overall.append(HCC_nonHCC_Radiologist)
            elif (type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M') or (Lesion_LiRad > 0 and Lesion_LiRad < 4):
                HCC_nonHCC_Radiologist = 0
                HCC_nonHCC_Radiologist_Overall.append(HCC_nonHCC_Radiologist)
            else:
                HCC_nonHCC_Radiologist = Lesion_LiRad

            # For type 2, LiRads 3, 4, 5 and LR-M are considered as Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type2 = 0
                HCC_nonHCC_Radiologist_type2_Overall.append(HCC_nonHCC_Radiologist_type2)
            else:
                HCC_nonHCC_Radiologist_type2 = 1
                HCC_nonHCC_Radiologist_type2_Overall.append(HCC_nonHCC_Radiologist_type2)

            # For type 3, LiRad 4, 5 and LR-M are considered as HCC, and LiRads 1, 2, and 3 are Non-HCC
            if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 3):
                HCC_nonHCC_Radiologist_type3 = 0
                HCC_nonHCC_Radiologist_type3_Overall.append(HCC_nonHCC_Radiologist_type3)
            else:
                HCC_nonHCC_Radiologist_type3 = 1
                HCC_nonHCC_Radiologist_type3_Overall.append(HCC_nonHCC_Radiologist_type3)

            # For type 4, LiRad 4, 5 --- HCC; LR-M and LiRad 3 --- Uncertain, LiRad 1, 2 --- Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type4 = 0  # Non-HCC
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type4 = 1  # HCC
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)
            else:
                HCC_nonHCC_Radiologist_type4 = 2  # Uncertain
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)

            # For type 5, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type5 = 0
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type5 = 1
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                HCC_nonHCC_Radiologist_type5 = 1
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            else:  # LiRads = 3    ------ Uncertain
                HCC_nonHCC_Radiologist_type5 = 2
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)

            # For type 6, LiRad 5 --- HCC, and LR-M, LiRad 3, 4 --- Uncertain, LiRad 1, 2 --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):  # LiRad 1, 2 --- NonHCC
                HCC_nonHCC_Radiologist_type6 = 0
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5  --- HCC
                HCC_nonHCC_Radiologist_type6 = 1
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)
            else:  # LiRad 3, 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type6 = 2
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)

            # For type 7, LiRad 5 --- HCC, LiRad 4, LR-M ---- Uncertain, LiRad 1, 2, 3 --- NonHCC
            if type(Lesion_LiRad) == int and 1 <= Lesion_LiRad <= 3:  # LiRad 1, 2, 3 --- NonHCC
                HCC_nonHCC_Radiologist_type7 = 0
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type7 = 1
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)
            else:  # LiRads 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type7 = 2
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)

            # For type 8, LiRad 5 --- HCC, LiRad 1, 2, 3, 4 and LR-M --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type8 = 1
                HCC_nonHCC_Radiologist_type8_Overall.append(HCC_nonHCC_Radiologist_type8)
            else:  # LiRad 1, 2, 3, 4 and LR-M    --- Non-HCC
                HCC_nonHCC_Radiologist_type8 = 0
                HCC_nonHCC_Radiologist_type8_Overall.append(HCC_nonHCC_Radiologist_type8)

            Train_ID_Lesion_Label_GT_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type2_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type3_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type4_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type5_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type6_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type7_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type8_At_Risk.append([key, HCC_nonHCC_GT])

            Train_ID_Lesion_Label_Radiologist_At_Risk.append([key, HCC_nonHCC_Radiologist])
            Train_ID_Lesion_Label_Radiologist_type2_At_Risk.append([key, HCC_nonHCC_Radiologist_type2])
            Train_ID_Lesion_Label_Radiologist_type3_At_Risk.append([key, HCC_nonHCC_Radiologist_type3])
            Train_ID_Lesion_Label_Radiologist_type4_At_Risk.append([key, HCC_nonHCC_Radiologist_type4])
            Train_ID_Lesion_Label_Radiologist_type5_At_Risk.append([key, HCC_nonHCC_Radiologist_type5])
            Train_ID_Lesion_Label_Radiologist_type6_At_Risk.append([key, HCC_nonHCC_Radiologist_type6])
            Train_ID_Lesion_Label_Radiologist_type7_At_Risk.append([key, HCC_nonHCC_Radiologist_type7])
            Train_ID_Lesion_Label_Radiologist_type8_At_Risk.append([key, HCC_nonHCC_Radiologist_type8])

            Lesion_Third = values[4]
            Lesion_Type = Lesion_Third[0]
            Lesion_LiRad = Lesion_Third[2]
            if Lesion_Type == 2 or Lesion_Type == 6 or Lesion_Type == 10 or Lesion_Type == 19 or Lesion_Type == 20 \
                    or Lesion_Type == 21 or Lesion_Type == 22:
                HCC_nonHCC_GT = 1
                HCC_nonHCC_GT_Overall.append(HCC_nonHCC_GT)
            else:
                HCC_nonHCC_GT = 0
                HCC_nonHCC_GT_Overall.append(HCC_nonHCC_GT)

            if type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist = 1
                HCC_nonHCC_Radiologist_Overall.append(HCC_nonHCC_Radiologist)
            elif (type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M') or (Lesion_LiRad > 0 and Lesion_LiRad < 4):
                HCC_nonHCC_Radiologist = 0
                HCC_nonHCC_Radiologist_Overall.append(HCC_nonHCC_Radiologist)
            else:
                HCC_nonHCC_Radiologist = Lesion_LiRad
            # For type 2, LiRads 3, 4, 5 and LR-M are considered as Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type2 = 0
                HCC_nonHCC_Radiologist_type2_Overall.append(HCC_nonHCC_Radiologist_type2)
            else:
                HCC_nonHCC_Radiologist_type2 = 1
                HCC_nonHCC_Radiologist_type2_Overall.append(HCC_nonHCC_Radiologist_type2)

            # For type 3, LiRad 4, 5 and LR-M are considered as HCC, and LiRads 1, 2, and 3 are Non-HCC
            if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 3):
                HCC_nonHCC_Radiologist_type3 = 0
                HCC_nonHCC_Radiologist_type3_Overall.append(HCC_nonHCC_Radiologist_type3)
            else:
                HCC_nonHCC_Radiologist_type3 = 1
                HCC_nonHCC_Radiologist_type3_Overall.append(HCC_nonHCC_Radiologist_type3)

            # For type 4, LiRad 4, 5 --- HCC; LR-M and LiRad 3 --- Uncertain, LiRad 1, 2 --- Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type4 = 0  # Non-HCC
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type4 = 1  # HCC
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)
            else:
                HCC_nonHCC_Radiologist_type4 = 2  # Uncertain
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)

            # For type 5, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type5 = 0
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type5 = 1
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                HCC_nonHCC_Radiologist_type5 = 1
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            else:  # LiRads = 3    ------ Uncertain
                HCC_nonHCC_Radiologist_type5 = 2
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)

            # For type 6, LiRad 5 --- HCC, and LR-M, LiRad 3, 4 --- Uncertain, LiRad 1, 2 --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):  # LiRad 1, 2 --- NonHCC
                HCC_nonHCC_Radiologist_type6 = 0
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5  --- HCC
                HCC_nonHCC_Radiologist_type6 = 1
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)
            else:  # LiRad 3, 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type6 = 2
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)

            # For type 7, LiRad 5 --- HCC, LiRad 4, LR-M ---- Uncertain, LiRad 1, 2, 3 --- NonHCC
            if type(Lesion_LiRad) == int and 1 <= Lesion_LiRad <= 3:  # LiRad 1, 2, 3 --- NonHCC
                HCC_nonHCC_Radiologist_type7 = 0
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type7 = 1
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)
            else:  # LiRads 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type7 = 2
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)

            # For type 8, LiRad 5 --- HCC, LiRad 1, 2, 3, 4 and LR-M --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type8 = 1
                HCC_nonHCC_Radiologist_type8_Overall.append(HCC_nonHCC_Radiologist_type8)
            else:  # LiRad 1, 2, 3, 4 and LR-M    --- Non-HCC
                HCC_nonHCC_Radiologist_type8 = 0
                HCC_nonHCC_Radiologist_type8_Overall.append(HCC_nonHCC_Radiologist_type8)

            Train_ID_Lesion_Label_GT_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type2_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type3_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type4_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type5_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type6_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type7_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_GT_type8_At_Risk.append([key, HCC_nonHCC_GT])
            Train_ID_Lesion_Label_Radiologist_At_Risk.append([key, HCC_nonHCC_Radiologist])
            Train_ID_Lesion_Label_Radiologist_type2_At_Risk.append([key, HCC_nonHCC_Radiologist_type2])
            Train_ID_Lesion_Label_Radiologist_type3_At_Risk.append([key, HCC_nonHCC_Radiologist_type3])
            Train_ID_Lesion_Label_Radiologist_type4_At_Risk.append([key, HCC_nonHCC_Radiologist_type4])
            Train_ID_Lesion_Label_Radiologist_type5_At_Risk.append([key, HCC_nonHCC_Radiologist_type5])
            Train_ID_Lesion_Label_Radiologist_type6_At_Risk.append([key, HCC_nonHCC_Radiologist_type6])
            Train_ID_Lesion_Label_Radiologist_type7_At_Risk.append([key, HCC_nonHCC_Radiologist_type7])
            Train_ID_Lesion_Label_Radiologist_type8_At_Risk.append([key, HCC_nonHCC_Radiologist_type8])

            if np.max(np.array(HCC_nonHCC_GT_Overall)) == 0:
                Train_ID_Patient_Label_GT_From_Excel_At_Risk.append([key, 0])
                Train_ID_Patient_Label_GT_From_Excel_type2_At_Risk.append([key, 0])
                Train_ID_Patient_Label_GT_From_Excel_type3_At_Risk.append([key, 0])
                Train_ID_Patient_Label_GT_From_Excel_type4_At_Risk.append([key, 0])
                Train_ID_Patient_Label_GT_From_Excel_type5_At_Risk.append([key, 0])
                Train_ID_Patient_Label_GT_From_Excel_type6_At_Risk.append([key, 0])
                Train_ID_Patient_Label_GT_From_Excel_type7_At_Risk.append([key, 0])
                Train_ID_Patient_Label_GT_From_Excel_type8_At_Risk.append([key, 0])
            else:
                Train_ID_Patient_Label_GT_From_Excel_At_Risk.append([key, 1])
                Train_ID_Patient_Label_GT_From_Excel_type2_At_Risk.append([key, 1])
                Train_ID_Patient_Label_GT_From_Excel_type3_At_Risk.append([key, 1])
                Train_ID_Patient_Label_GT_From_Excel_type4_At_Risk.append([key, 1])
                Train_ID_Patient_Label_GT_From_Excel_type5_At_Risk.append([key, 1])
                Train_ID_Patient_Label_GT_From_Excel_type6_At_Risk.append([key, 1])
                Train_ID_Patient_Label_GT_From_Excel_type7_At_Risk.append([key, 1])
                Train_ID_Patient_Label_GT_From_Excel_type8_At_Risk.append([key, 1])

            if np.max(np.array(HCC_nonHCC_Radiologist_Overall)) == 0:
                Train_ID_Patient_Label_Radiologist_At_Risk.append([key, 0])
            else:
                Train_ID_Patient_Label_Radiologist_At_Risk.append([key, 1])

            if np.max(np.array(HCC_nonHCC_Radiologist_type2_Overall)) == 0:
                Train_ID_Patient_Label_Radiologist_type2_At_Risk.append([key, 0])
            else:
                Train_ID_Patient_Label_Radiologist_type2_At_Risk.append([key, 1])

            if np.max(np.array(HCC_nonHCC_Radiologist_type3_Overall)) == 0:
                Train_ID_Patient_Label_Radiologist_type3_At_Risk.append([key, 0])
            else:
                Train_ID_Patient_Label_Radiologist_type3_At_Risk.append([key, 1])

            if np.max(np.array(HCC_nonHCC_Radiologist_type4_Overall)) == 0:
                Train_ID_Patient_Label_Radiologist_type4_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type4_Overall)) == 1:
                Train_ID_Patient_Label_Radiologist_type4_At_Risk.append([key, 1])
            else:
                Train_ID_Patient_Label_Radiologist_type4_At_Risk.append([key, 2])
            # For type 5
            if np.max(np.array(HCC_nonHCC_Radiologist_type5_Overall)) == 0:
                Train_ID_Patient_Label_Radiologist_type5_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type5_Overall)) == 1:
                Train_ID_Patient_Label_Radiologist_type5_At_Risk.append([key, 1])
            else:
                Train_ID_Patient_Label_Radiologist_type5_At_Risk.append([key, 2])
            # For type 7
            if np.max(np.array(HCC_nonHCC_Radiologist_type6_Overall)) == 0:
                Train_ID_Patient_Label_Radiologist_type6_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type6_Overall)) == 1:
                Train_ID_Patient_Label_Radiologist_type6_At_Risk.append([key, 1])
            else:
                Train_ID_Patient_Label_Radiologist_type6_At_Risk.append([key, 2])
            # For type 7
            if np.max(np.array(HCC_nonHCC_Radiologist_type7_Overall)) == 0:
                Train_ID_Patient_Label_Radiologist_type7_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type7_Overall)) == 1:
                Train_ID_Patient_Label_Radiologist_type7_At_Risk.append([key, 1])
            else:
                Train_ID_Patient_Label_Radiologist_type7_At_Risk.append([key, 2])
            # For type 8
            if np.max(np.array(HCC_nonHCC_Radiologist_type8_Overall)) == 0:
                Train_ID_Patient_Label_Radiologist_type8_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type8_Overall)) == 1:
                Train_ID_Patient_Label_Radiologist_type8_At_Risk.append([key, 1])
            else:
                Train_ID_Patient_Label_Radiologist_type8_At_Risk.append([key, 2])
        else:
            print(len(values), "Line-236")
    elif key in Test_ID_At_Risk:
        if len(values) == 3:
            Lesion_Type = values[0]
            Lesion_LiRad = values[2]
            if Lesion_Type == 2 or Lesion_Type == 6 or Lesion_Type == 10 or Lesion_Type == 19 or Lesion_Type == 20 \
                    or Lesion_Type == 21 or Lesion_Type == 22:
                HCC_nonHCC_GT = 1
            else:
                HCC_nonHCC_GT = 0

            if type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist = 1
            elif (type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M') or (Lesion_LiRad > 0 and Lesion_LiRad < 4):
                HCC_nonHCC_Radiologist = 0
            else:
                HCC_nonHCC_Radiologist = Lesion_LiRad

            # For type 2, LiRads 3, 4, 5 and LR-M are considered as Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type2 = 0
            else:
                HCC_nonHCC_Radiologist_type2 = 1

            # For type 3, LiRad 4, 5 and LR-M are considered as HCC, and LiRads 1, 2, and 3 are Non-HCC
            if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 3):
                HCC_nonHCC_Radiologist_type3 = 0
            else:
                HCC_nonHCC_Radiologist_type3 = 1

            # For type 4, LiRad 4, 5 --- HCC; LR-M and LiRad 3 --- Uncertain, LiRad 1, 2 --- Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type4 = 0  # Non-HCC
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type4 = 1  # HCC
            else:
                HCC_nonHCC_Radiologist_type4 = 2  # Uncertain

            # For type 4_2, LiRad 4, 5 --- HCC, LR-M, LiRad 3 --- Uncertain, LiRad 1, 2 --- Non-HCC
            # Uncertain --- HCC
            if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 2):
                HCC_nonHCC_Radiologist_type4_2 = 0  # LiRad 1 and 2
            elif type(Lesion_LiRad) == int and (4 <= Lesion_LiRad <= 5):
                HCC_nonHCC_Radiologist_type4_2 = 1  # LiRad 4, 5
            else:
                HCC_nonHCC_Radiologist_type4_2 = 1  # LiRad 3 and LR-M ---> HCC
            # For type 4_3, LiRad 4, 5 --- HCC, LR-M, LiRad 3 --- Uncertain, LiRad 1, 2 --- Non-HCC
            # Uncertain --- Non-HCC
            if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 2):
                HCC_nonHCC_Radiologist_type4_3 = 0  # LiRad 1 and 2
            elif type(Lesion_LiRad) == int and (4 <= Lesion_LiRad <= 5):
                HCC_nonHCC_Radiologist_type4_3 = 1  # LiRad 4, 5
            else:
                HCC_nonHCC_Radiologist_type4_3 = 0  # LiRad 3 and LR-M ---> Non-HCC

            # For type 5, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type5 = 0
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type5 = 1
            elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                HCC_nonHCC_Radiologist_type5 = 1
            else:  # LiRads = 3    ------ Uncertain
                HCC_nonHCC_Radiologist_type5 = 2

            # For type 5_2, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type5_2 = 0
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type5_2 = 1
            elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                HCC_nonHCC_Radiologist_type5_2 = 1
            else:  # LiRads = 3    --- Uncertain ---> HCC
                HCC_nonHCC_Radiologist_type5_2 = 1

            # For type 5_3, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type5_3 = 0
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type5_3 = 1
            elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                HCC_nonHCC_Radiologist_type5_3 = 1
            else:  # LiRads = 3    ---  Uncertain ---> Non-HCC
                HCC_nonHCC_Radiologist_type5_3 = 0

            # For type 6, LiRad 5 --- HCC, and LR-M, LiRad 3, 4 --- Uncertain, LiRad 1, 2 --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):  # LiRad 1, 2 --- NonHCC
                HCC_nonHCC_Radiologist_type6 = 0
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5  --- HCC
                HCC_nonHCC_Radiologist_type6 = 1
            else:  # LiRad 3, 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type6 = 2

            # For type 6_2, LiRad 5 --- HCC, and LR-M, LiRad 3, 4 --- Uncertain, LiRad 1, 2 --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):  # LiRad 1, 2 --- NonHCC
                HCC_nonHCC_Radiologist_type6_2 = 0
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5  --- HCC
                HCC_nonHCC_Radiologist_type6_2 = 1
            else:  # LiRad 3, 4, LR-M --- Uncertain ---> HCC
                HCC_nonHCC_Radiologist_type6_2 = 1

            # For type 6_3, LiRad 5 --- HCC, and LR-M, LiRad 3, 4 --- Uncertain, LiRad 1, 2 --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):  # LiRad 1, 2 --- NonHCC
                HCC_nonHCC_Radiologist_type6_3 = 0
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5  --- HCC
                HCC_nonHCC_Radiologist_type6_3 = 1
            else:  # LiRad 3, 4, LR-M --- Uncertain ---> Non-HCC
                HCC_nonHCC_Radiologist_type6_3 = 0

            # For type 7, LiRad 5 --- HCC, LiRad 4, LR-M ---- Uncertain, LiRad 1, 2, 3 --- NonHCC
            if type(Lesion_LiRad) == int and 1 <= Lesion_LiRad <= 3:  # LiRad 1, 2, 3 --- NonHCC
                HCC_nonHCC_Radiologist_type7 = 0
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type7 = 1
            else:  # LiRads 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type7 = 2
            # For type 7_2, LiRad 5 --- HCC, LiRad 4, LR-M ---- Uncertain, LiRad 1, 2, 3 --- NonHCC
            if type(Lesion_LiRad) == int and 1 <= Lesion_LiRad <= 3:  # LiRad 1, 2, 3 --- NonHCC
                HCC_nonHCC_Radiologist_type7_2 = 0
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type7_2 = 1
            else:  # LiRads 4, LR-M --- Uncertain ---> HCC
                HCC_nonHCC_Radiologist_type7_2 = 1
            # For type 7_3, LiRad 5 --- HCC, LiRad 4, LR-M ---- Uncertain, LiRad 1, 2, 3 --- NonHCC
            if type(Lesion_LiRad) == int and 1 <= Lesion_LiRad <= 3:  # LiRad 1, 2, 3 --- NonHCC
                HCC_nonHCC_Radiologist_type7_3 = 0
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type7_3 = 1
            else:  # LiRads 4, LR-M --- Uncertain ---> Non-HCC
                HCC_nonHCC_Radiologist_type7_3 = 0

            # For type 8, LiRad 5 --- HCC, LiRad 1, 2, 3, 4 and LR-M --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type8 = 1
            else:  # LiRad 1, 2, 3, 4 and LR-M    --- Non-HCC
                HCC_nonHCC_Radiologist_type8 = 0

            Test_ID_Lesion_Label_GT_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type2_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type3_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type4_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type5_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type6_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type7_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type8_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_Radiologist_At_Risk.append([key, HCC_nonHCC_Radiologist])
            Test_ID_Lesion_Label_Radiologist_type2_At_Risk.append([key, HCC_nonHCC_Radiologist_type2])
            Test_ID_Lesion_Label_Radiologist_type3_At_Risk.append([key, HCC_nonHCC_Radiologist_type3])
            Test_ID_Lesion_Label_Radiologist_type4_At_Risk.append([key, HCC_nonHCC_Radiologist_type4])
            Test_ID_Lesion_Label_Radiologist_type4_2_At_Risk.append([key, HCC_nonHCC_Radiologist_type4_2])
            Test_ID_Lesion_Label_Radiologist_type4_3_At_Risk.append([key, HCC_nonHCC_Radiologist_type4_3])
            Test_ID_Lesion_Label_Radiologist_type5_At_Risk.append([key, HCC_nonHCC_Radiologist_type5])
            Test_ID_Lesion_Label_Radiologist_type5_2_At_Risk.append([key, HCC_nonHCC_Radiologist_type5_2])
            Test_ID_Lesion_Label_Radiologist_type5_3_At_Risk.append([key, HCC_nonHCC_Radiologist_type5_3])
            Test_ID_Lesion_Label_Radiologist_type6_At_Risk.append([key, HCC_nonHCC_Radiologist_type6])
            Test_ID_Lesion_Label_Radiologist_type6_2_At_Risk.append([key, HCC_nonHCC_Radiologist_type6_2])
            Test_ID_Lesion_Label_Radiologist_type6_3_At_Risk.append([key, HCC_nonHCC_Radiologist_type6_3])
            Test_ID_Lesion_Label_Radiologist_type7_At_Risk.append([key, HCC_nonHCC_Radiologist_type7])
            Test_ID_Lesion_Label_Radiologist_type7_2_At_Risk.append([key, HCC_nonHCC_Radiologist_type7_2])
            Test_ID_Lesion_Label_Radiologist_type7_3_At_Risk.append([key, HCC_nonHCC_Radiologist_type7_3])
            Test_ID_Lesion_Label_Radiologist_type8_At_Risk.append([key, HCC_nonHCC_Radiologist_type8])

            Test_ID_Patient_Label_GT_From_Excel_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Patient_Label_GT_From_Excel_type2_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Patient_Label_GT_From_Excel_type3_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Patient_Label_GT_From_Excel_type4_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Patient_Label_GT_From_Excel_type5_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Patient_Label_GT_From_Excel_type6_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Patient_Label_GT_From_Excel_type7_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Patient_Label_GT_From_Excel_type8_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Patient_Label_Radiologist_At_Risk.append([key, HCC_nonHCC_Radiologist])
            Test_ID_Patient_Label_Radiologist_type2_At_Risk.append([key, HCC_nonHCC_Radiologist_type2])
            Test_ID_Patient_Label_Radiologist_type3_At_Risk.append([key, HCC_nonHCC_Radiologist_type3])
            Test_ID_Patient_Label_Radiologist_type4_At_Risk.append([key, HCC_nonHCC_Radiologist_type4])
            Test_ID_Patient_Label_Radiologist_type4_2_At_Risk.append([key, HCC_nonHCC_Radiologist_type4_2])
            Test_ID_Patient_Label_Radiologist_type4_3_At_Risk.append([key, HCC_nonHCC_Radiologist_type4_3])
            Test_ID_Patient_Label_Radiologist_type5_At_Risk.append([key, HCC_nonHCC_Radiologist_type5])
            Test_ID_Patient_Label_Radiologist_type5_2_At_Risk.append([key, HCC_nonHCC_Radiologist_type5_2])
            Test_ID_Patient_Label_Radiologist_type5_3_At_Risk.append([key, HCC_nonHCC_Radiologist_type5_3])
            Test_ID_Patient_Label_Radiologist_type6_At_Risk.append([key, HCC_nonHCC_Radiologist_type6])
            Test_ID_Patient_Label_Radiologist_type6_2_At_Risk.append([key, HCC_nonHCC_Radiologist_type6_2])
            Test_ID_Patient_Label_Radiologist_type6_3_At_Risk.append([key, HCC_nonHCC_Radiologist_type6_3])
            Test_ID_Patient_Label_Radiologist_type7_At_Risk.append([key, HCC_nonHCC_Radiologist_type7])
            Test_ID_Patient_Label_Radiologist_type7_2_At_Risk.append([key, HCC_nonHCC_Radiologist_type7_2])
            Test_ID_Patient_Label_Radiologist_type7_3_At_Risk.append([key, HCC_nonHCC_Radiologist_type7_3])
            Test_ID_Patient_Label_Radiologist_type8_At_Risk.append([key, HCC_nonHCC_Radiologist_type8])
        elif len(values) == 4:
            Lesion_First = values[0:3]
            Lesion_Type = Lesion_First[0]
            Lesion_LiRad = Lesion_First[2]
            HCC_nonHCC_GT_Overall = []
            HCC_nonHCC_Radiologist_Overall, HCC_nonHCC_Radiologist_type2_Overall = [], []
            HCC_nonHCC_Radiologist_type3_Overall, HCC_nonHCC_Radiologist_type4_Overall = [], []
            HCC_nonHCC_Radiologist_type5_Overall, HCC_nonHCC_Radiologist_type6_Overall = [], []
            HCC_nonHCC_Radiologist_type7_Overall, HCC_nonHCC_Radiologist_type8_Overall = [], []

            HCC_nonHCC_Radiologist_type4_2_Overall, HCC_nonHCC_Radiologist_type4_3_Overall = [], []
            HCC_nonHCC_Radiologist_type5_2_Overall, HCC_nonHCC_Radiologist_type5_3_Overall = [], []
            HCC_nonHCC_Radiologist_type6_2_Overall, HCC_nonHCC_Radiologist_type6_3_Overall = [], []
            HCC_nonHCC_Radiologist_type7_2_Overall, HCC_nonHCC_Radiologist_type7_3_Overall = [], []

            if Lesion_Type == 2 or Lesion_Type == 6 or Lesion_Type == 10 or Lesion_Type == 19 or Lesion_Type == 20 \
                    or Lesion_Type == 21 or Lesion_Type == 22:
                HCC_nonHCC_GT = 1
                HCC_nonHCC_GT_Overall.append(HCC_nonHCC_GT)
            else:
                HCC_nonHCC_GT = 0
                HCC_nonHCC_GT_Overall.append(HCC_nonHCC_GT)

            if type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist = 1
                HCC_nonHCC_Radiologist_Overall.append(HCC_nonHCC_Radiologist)
            elif (type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M') or (Lesion_LiRad > 0 and Lesion_LiRad < 4):
                HCC_nonHCC_Radiologist = 0
                HCC_nonHCC_Radiologist_Overall.append(HCC_nonHCC_Radiologist)
            else:
                HCC_nonHCC_Radiologist = Lesion_LiRad

            # For type 2, LiRads 3, 4, 5 and LR-M are considered as Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type2 = 0
                HCC_nonHCC_Radiologist_type2_Overall.append(HCC_nonHCC_Radiologist_type2)
            else:
                HCC_nonHCC_Radiologist_type2 = 1
                HCC_nonHCC_Radiologist_type2_Overall.append(HCC_nonHCC_Radiologist_type2)

            # For type 3, LiRad 4, 5 and LR-M are considered as HCC, and LiRads 1, 2, and 3 are Non-HCC
            if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 3):
                HCC_nonHCC_Radiologist_type3 = 0
                HCC_nonHCC_Radiologist_type3_Overall.append(HCC_nonHCC_Radiologist_type3)
            else:
                HCC_nonHCC_Radiologist_type3 = 1
                HCC_nonHCC_Radiologist_type3_Overall.append(HCC_nonHCC_Radiologist_type3)

            # For type 4, LiRad 4, 5 --- HCC; LR-M and LiRad 3 --- Uncertain, LiRad 1, 2 --- Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type4 = 0  # Non-HCC
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type4 = 1  # HCC
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)
            else:
                HCC_nonHCC_Radiologist_type4 = 2  # Uncertain
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)
            # For type 4_2, LiRad 4, 5 --- HCC; LR-M and LiRad 3 --- Uncertain, LiRad 1, 2 --- Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type4_2 = 0  # Non-HCC
                HCC_nonHCC_Radiologist_type4_2_Overall.append(HCC_nonHCC_Radiologist_type4_2)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type4_2 = 1  # HCC
                HCC_nonHCC_Radiologist_type4_2_Overall.append(HCC_nonHCC_Radiologist_type4_2)
            else:
                HCC_nonHCC_Radiologist_type4_2 = 1  # Uncertain ---> HCC
                HCC_nonHCC_Radiologist_type4_2_Overall.append(HCC_nonHCC_Radiologist_type4_2)
            # For type 4_2, LiRad 4, 5 --- HCC; LR-M and LiRad 3 --- Uncertain, LiRad 1, 2 --- Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type4_3 = 0  # Non-HCC
                HCC_nonHCC_Radiologist_type4_3_Overall.append(HCC_nonHCC_Radiologist_type4_3)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type4_3 = 1  # HCC
                HCC_nonHCC_Radiologist_type4_3_Overall.append(HCC_nonHCC_Radiologist_type4_3)
            else:
                HCC_nonHCC_Radiologist_type4_3 = 0  # Uncertain ---> NonHCC
                HCC_nonHCC_Radiologist_type4_3_Overall.append(HCC_nonHCC_Radiologist_type4_3)

            # For type 5, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type5 = 0
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type5 = 1
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                HCC_nonHCC_Radiologist_type5 = 1
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            else:  # LiRads = 3    ------ Uncertain
                HCC_nonHCC_Radiologist_type5 = 2
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            # For type 5_2, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type5_2 = 0
                HCC_nonHCC_Radiologist_type5_2_Overall.append(HCC_nonHCC_Radiologist_type5_2)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type5_2 = 1
                HCC_nonHCC_Radiologist_type5_2_Overall.append(HCC_nonHCC_Radiologist_type5_2)
            elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                HCC_nonHCC_Radiologist_type5_2 = 1
                HCC_nonHCC_Radiologist_type5_2_Overall.append(HCC_nonHCC_Radiologist_type5_2)
            else:  # LiRads = 3    ------ Uncertain ---> HCC
                HCC_nonHCC_Radiologist_type5_2 = 1
                HCC_nonHCC_Radiologist_type5_2_Overall.append(HCC_nonHCC_Radiologist_type5_2)
            # For type 5_3, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type5_3 = 0
                HCC_nonHCC_Radiologist_type5_3_Overall.append(HCC_nonHCC_Radiologist_type5_3)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type5_3 = 1
                HCC_nonHCC_Radiologist_type5_3_Overall.append(HCC_nonHCC_Radiologist_type5_3)
            elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                HCC_nonHCC_Radiologist_type5_3 = 1
                HCC_nonHCC_Radiologist_type5_3_Overall.append(HCC_nonHCC_Radiologist_type5_3)
            else:  # LiRads = 3    ------ Uncertain ---> Non-HCC
                HCC_nonHCC_Radiologist_type5_3 = 0
                HCC_nonHCC_Radiologist_type5_3_Overall.append(HCC_nonHCC_Radiologist_type5_3)

            # For type 6, LiRad 5 --- HCC, and LR-M, LiRad 3, 4 --- Uncertain, LiRad 1, 2 --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):  # LiRad 1, 2 --- NonHCC
                HCC_nonHCC_Radiologist_type6 = 0
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5  --- HCC
                HCC_nonHCC_Radiologist_type6 = 1
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)
            else:  # LiRad 3, 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type6 = 2
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)
            # For type 6_2, LiRad 5 --- HCC, and LR-M, LiRad 3, 4 --- Uncertain, LiRad 1, 2 --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):  # LiRad 1, 2 --- NonHCC
                HCC_nonHCC_Radiologist_type6_2 = 0
                HCC_nonHCC_Radiologist_type6_2_Overall.append(HCC_nonHCC_Radiologist_type6_2)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5  --- HCC
                HCC_nonHCC_Radiologist_type6_2 = 1
                HCC_nonHCC_Radiologist_type6_2_Overall.append(HCC_nonHCC_Radiologist_type6_2)
            else:  # LiRad 3, 4, LR-M --- Uncertain ---> HCC
                HCC_nonHCC_Radiologist_type6_2 = 1
                HCC_nonHCC_Radiologist_type6_2_Overall.append(HCC_nonHCC_Radiologist_type6_2)
            # For type 6_3, LiRad 5 --- HCC, and LR-M, LiRad 3, 4 --- Uncertain, LiRad 1, 2 --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):  # LiRad 1, 2 --- NonHCC
                HCC_nonHCC_Radiologist_type6_3 = 0
                HCC_nonHCC_Radiologist_type6_3_Overall.append(HCC_nonHCC_Radiologist_type6_3)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5  --- HCC
                HCC_nonHCC_Radiologist_type6_3 = 1
                HCC_nonHCC_Radiologist_type6_3_Overall.append(HCC_nonHCC_Radiologist_type6_3)
            else:  # LiRad 3, 4, LR-M --- Uncertain ---> NonHCC
                HCC_nonHCC_Radiologist_type6_3 = 0
                HCC_nonHCC_Radiologist_type6_3_Overall.append(HCC_nonHCC_Radiologist_type6_3)

            # For type 7, LiRad 5 --- HCC, LiRad 4, LR-M ---- Uncertain, LiRad 1, 2, 3 --- NonHCC
            if type(Lesion_LiRad) == int and 1 <= Lesion_LiRad <= 3:  # LiRad 1, 2, 3 --- NonHCC
                HCC_nonHCC_Radiologist_type7 = 0
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type7 = 1
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)
            else:  # LiRads 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type7 = 2
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)
            # For type 7_2, LiRad 5 --- HCC, LiRad 4, LR-M ---- Uncertain, LiRad 1, 2, 3 --- NonHCC
            if type(Lesion_LiRad) == int and 1 <= Lesion_LiRad <= 3:  # LiRad 1, 2, 3 --- NonHCC
                HCC_nonHCC_Radiologist_type7_2 = 0
                HCC_nonHCC_Radiologist_type7_2_Overall.append(HCC_nonHCC_Radiologist_type7_2)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type7_2 = 1
                HCC_nonHCC_Radiologist_type7_2_Overall.append(HCC_nonHCC_Radiologist_type7_2)
            else:  # LiRads 4, LR-M --- Uncertain ---> HCC
                HCC_nonHCC_Radiologist_type7_2 = 1
                HCC_nonHCC_Radiologist_type7_2_Overall.append(HCC_nonHCC_Radiologist_type7_2)
            # For type 7_3, LiRad 5 --- HCC, LiRad 4, LR-M ---- Uncertain, LiRad 1, 2, 3 --- NonHCC
            if type(Lesion_LiRad) == int and 1 <= Lesion_LiRad <= 3:  # LiRad 1, 2, 3 --- NonHCC
                HCC_nonHCC_Radiologist_type7_3 = 0
                HCC_nonHCC_Radiologist_type7_3_Overall.append(HCC_nonHCC_Radiologist_type7_3)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type7_3 = 1
                HCC_nonHCC_Radiologist_type7_3_Overall.append(HCC_nonHCC_Radiologist_type7_3)
            else:  # LiRads 4, LR-M --- Uncertain  ---> NonHCC
                HCC_nonHCC_Radiologist_type7_3 = 0
                HCC_nonHCC_Radiologist_type7_3_Overall.append(HCC_nonHCC_Radiologist_type7_3)

            # For type 8, LiRad 5 --- HCC, LiRad 1, 2, 3, 4 and LR-M --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type8 = 1
                HCC_nonHCC_Radiologist_type8_Overall.append(HCC_nonHCC_Radiologist_type8)
            else:  # LiRad 1, 2, 3, 4 and LR-M    --- Non-HCC
                HCC_nonHCC_Radiologist_type8 = 0
                HCC_nonHCC_Radiologist_type8_Overall.append(HCC_nonHCC_Radiologist_type8)

            Test_ID_Lesion_Label_GT_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type2_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type3_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type4_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type5_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type6_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type7_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type8_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_Radiologist_At_Risk.append([key, HCC_nonHCC_Radiologist])
            Test_ID_Lesion_Label_Radiologist_type2_At_Risk.append([key, HCC_nonHCC_Radiologist_type2])
            Test_ID_Lesion_Label_Radiologist_type3_At_Risk.append([key, HCC_nonHCC_Radiologist_type3])
            Test_ID_Lesion_Label_Radiologist_type4_At_Risk.append([key, HCC_nonHCC_Radiologist_type4])
            Test_ID_Lesion_Label_Radiologist_type4_2_At_Risk.append([key, HCC_nonHCC_Radiologist_type4_2])
            Test_ID_Lesion_Label_Radiologist_type4_3_At_Risk.append([key, HCC_nonHCC_Radiologist_type4_3])
            Test_ID_Lesion_Label_Radiologist_type5_At_Risk.append([key, HCC_nonHCC_Radiologist_type5])
            Test_ID_Lesion_Label_Radiologist_type5_2_At_Risk.append([key, HCC_nonHCC_Radiologist_type5_2])
            Test_ID_Lesion_Label_Radiologist_type5_3_At_Risk.append([key, HCC_nonHCC_Radiologist_type5_3])
            Test_ID_Lesion_Label_Radiologist_type6_At_Risk.append([key, HCC_nonHCC_Radiologist_type6])
            Test_ID_Lesion_Label_Radiologist_type6_2_At_Risk.append([key, HCC_nonHCC_Radiologist_type6_2])
            Test_ID_Lesion_Label_Radiologist_type6_3_At_Risk.append([key, HCC_nonHCC_Radiologist_type6_3])
            Test_ID_Lesion_Label_Radiologist_type7_At_Risk.append([key, HCC_nonHCC_Radiologist_type7])
            Test_ID_Lesion_Label_Radiologist_type7_2_At_Risk.append([key, HCC_nonHCC_Radiologist_type7_2])
            Test_ID_Lesion_Label_Radiologist_type7_3_At_Risk.append([key, HCC_nonHCC_Radiologist_type7_3])
            Test_ID_Lesion_Label_Radiologist_type8_At_Risk.append([key, HCC_nonHCC_Radiologist_type8])

            Lesion_Second = values[3]
            Lesion_Type = Lesion_Second[0]
            Lesion_LiRad = Lesion_Second[2]
            if Lesion_Type == 2 or Lesion_Type == 6 or Lesion_Type == 10 or Lesion_Type == 19 or Lesion_Type == 20 \
                    or Lesion_Type == 21 or Lesion_Type == 22:
                HCC_nonHCC_GT = 1
                HCC_nonHCC_GT_Overall.append(HCC_nonHCC_GT)
            else:
                HCC_nonHCC_GT = 0
                HCC_nonHCC_GT_Overall.append(HCC_nonHCC_GT)

            if type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist = 1
                HCC_nonHCC_Radiologist_Overall.append(HCC_nonHCC_Radiologist)
            elif (type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M') or (Lesion_LiRad > 0 and Lesion_LiRad < 4):
                HCC_nonHCC_Radiologist = 0
                HCC_nonHCC_Radiologist_Overall.append(HCC_nonHCC_Radiologist)
            else:
                HCC_nonHCC_Radiologist = Lesion_LiRad

            # For type 2, LiRads 3, 4, 5 and LR-M are considered as Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type2 = 0
                HCC_nonHCC_Radiologist_type2_Overall.append(HCC_nonHCC_Radiologist_type2)
            else:
                HCC_nonHCC_Radiologist_type2 = 1
                HCC_nonHCC_Radiologist_type2_Overall.append(HCC_nonHCC_Radiologist_type2)
            # For type 3, LiRad 4, 5 and LR-M are considered as HCC, and LiRads 1, 2, and 3 are Non-HCC
            if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 3):
                HCC_nonHCC_Radiologist_type3 = 0
                HCC_nonHCC_Radiologist_type3_Overall.append(HCC_nonHCC_Radiologist_type3)
            else:
                HCC_nonHCC_Radiologist_type3 = 1
                HCC_nonHCC_Radiologist_type3_Overall.append(HCC_nonHCC_Radiologist_type3)
            # For type 4, LiRad 4, 5 --- HCC; LR-M and LiRad 3 --- Uncertain, LiRad 1, 2 --- Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type4 = 0  # Non-HCC
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type4 = 1  # HCC
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)
            else:
                HCC_nonHCC_Radiologist_type4 = 2  # Uncertain
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)
            # For type 4_2, LiRad 4, 5 --- HCC; LR-M and LiRad 3 --- Uncertain, LiRad 1, 2 --- Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type4_2 = 0  # Non-HCC
                HCC_nonHCC_Radiologist_type4_2_Overall.append(HCC_nonHCC_Radiologist_type4_2)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type4_2 = 1  # HCC
                HCC_nonHCC_Radiologist_type4_2_Overall.append(HCC_nonHCC_Radiologist_type4_2)
            else:
                HCC_nonHCC_Radiologist_type4_2 = 1  # Uncertain ---> HCC
                HCC_nonHCC_Radiologist_type4_2_Overall.append(HCC_nonHCC_Radiologist_type4_2)
            # For type 4_3, LiRad 4, 5 --- HCC; LR-M and LiRad 3 --- Uncertain, LiRad 1, 2 --- Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type4_3 = 0  # Non-HCC
                HCC_nonHCC_Radiologist_type4_3_Overall.append(HCC_nonHCC_Radiologist_type4_3)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type4_3 = 1  # HCC
                HCC_nonHCC_Radiologist_type4_3_Overall.append(HCC_nonHCC_Radiologist_type4_3)
            else:
                HCC_nonHCC_Radiologist_type4_3 = 0  # Uncertain
                HCC_nonHCC_Radiologist_type4_3_Overall.append(HCC_nonHCC_Radiologist_type4_3)

            # For type 5, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type5 = 0
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type5 = 1
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                HCC_nonHCC_Radiologist_type5 = 1
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            else:  # LiRads = 3    ------ Uncertain
                HCC_nonHCC_Radiologist_type5 = 2
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            # For type 5_2, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type5_2 = 0
                HCC_nonHCC_Radiologist_type5_2_Overall.append(HCC_nonHCC_Radiologist_type5_2)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type5_2 = 1
                HCC_nonHCC_Radiologist_type5_2_Overall.append(HCC_nonHCC_Radiologist_type5_2)
            elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                HCC_nonHCC_Radiologist_type5_2 = 1
                HCC_nonHCC_Radiologist_type5_2_Overall.append(HCC_nonHCC_Radiologist_type5_2)
            else:  # LiRads = 3    ------ Uncertain
                HCC_nonHCC_Radiologist_type5_2 = 1
                HCC_nonHCC_Radiologist_type5_2_Overall.append(HCC_nonHCC_Radiologist_type5_2)
            # For type 5_3, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type5_3 = 0
                HCC_nonHCC_Radiologist_type5_3_Overall.append(HCC_nonHCC_Radiologist_type5_3)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type5_3 = 1
                HCC_nonHCC_Radiologist_type5_3_Overall.append(HCC_nonHCC_Radiologist_type5_3)
            elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                HCC_nonHCC_Radiologist_type5_3 = 1
                HCC_nonHCC_Radiologist_type5_3_Overall.append(HCC_nonHCC_Radiologist_type5_3)
            else:  # LiRads = 3    ------ Uncertain
                HCC_nonHCC_Radiologist_type5_3 = 0
                HCC_nonHCC_Radiologist_type5_3_Overall.append(HCC_nonHCC_Radiologist_type5_3)

            # For type 6, LiRad 5 --- HCC, and LR-M, LiRad 3, 4 --- Uncertain, LiRad 1, 2 --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):  # LiRad 1, 2 --- NonHCC
                HCC_nonHCC_Radiologist_type6 = 0
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5  --- HCC
                HCC_nonHCC_Radiologist_type6 = 1
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)
            else:  # LiRad 3, 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type6 = 2
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)
            # For type 6_2, LiRad 5 --- HCC, and LR-M, LiRad 3, 4 --- Uncertain, LiRad 1, 2 --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):  # LiRad 1, 2 --- NonHCC
                HCC_nonHCC_Radiologist_type6_2 = 0
                HCC_nonHCC_Radiologist_type6_2_Overall.append(HCC_nonHCC_Radiologist_type6_2)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5  --- HCC
                HCC_nonHCC_Radiologist_type6_2 = 1
                HCC_nonHCC_Radiologist_type6_2_Overall.append(HCC_nonHCC_Radiologist_type6_2)
            else:  # LiRad 3, 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type6_2 = 1
                HCC_nonHCC_Radiologist_type6_2_Overall.append(HCC_nonHCC_Radiologist_type6_2)
            # For type 6_3, LiRad 5 --- HCC, and LR-M, LiRad 3, 4 --- Uncertain, LiRad 1, 2 --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):  # LiRad 1, 2 --- NonHCC
                HCC_nonHCC_Radiologist_type6_3 = 0
                HCC_nonHCC_Radiologist_type6_3_Overall.append(HCC_nonHCC_Radiologist_type6_3)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5  --- HCC
                HCC_nonHCC_Radiologist_type6_3 = 1
                HCC_nonHCC_Radiologist_type6_3_Overall.append(HCC_nonHCC_Radiologist_type6_3)
            else:  # LiRad 3, 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type6_3 = 0
                HCC_nonHCC_Radiologist_type6_3_Overall.append(HCC_nonHCC_Radiologist_type6_3)

            # For type 7, LiRad 5 --- HCC, LiRad 4, LR-M ---- Uncertain, LiRad 1, 2, 3 --- NonHCC
            if type(Lesion_LiRad) == int and 1 <= Lesion_LiRad <= 3:  # LiRad 1, 2, 3 --- NonHCC
                HCC_nonHCC_Radiologist_type7 = 0
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type7 = 1
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)
            else:  # LiRads 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type7 = 2
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)
            # For type 7_2, LiRad 5 --- HCC, LiRad 4, LR-M ---- Uncertain, LiRad 1, 2, 3 --- NonHCC
            if type(Lesion_LiRad) == int and 1 <= Lesion_LiRad <= 3:  # LiRad 1, 2, 3 --- NonHCC
                HCC_nonHCC_Radiologist_type7_2 = 0
                HCC_nonHCC_Radiologist_type7_2_Overall.append(HCC_nonHCC_Radiologist_type7_2)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type7_2 = 1
                HCC_nonHCC_Radiologist_type7_2_Overall.append(HCC_nonHCC_Radiologist_type7_2)
            else:  # LiRads 4, LR-M --- Uncertain  ---> HCC
                HCC_nonHCC_Radiologist_type7_2 = 1
                HCC_nonHCC_Radiologist_type7_2_Overall.append(HCC_nonHCC_Radiologist_type7_2)
            # For type 7_3, LiRad 5 --- HCC, LiRad 4, LR-M ---- Uncertain, LiRad 1, 2, 3 --- NonHCC
            if type(Lesion_LiRad) == int and 1 <= Lesion_LiRad <= 3:  # LiRad 1, 2, 3 --- NonHCC
                HCC_nonHCC_Radiologist_type7_3 = 0
                HCC_nonHCC_Radiologist_type7_3_Overall.append(HCC_nonHCC_Radiologist_type7_3)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type7_3 = 1
                HCC_nonHCC_Radiologist_type7_3_Overall.append(HCC_nonHCC_Radiologist_type7_3)
            else:  # LiRads 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type7_3 = 0
                HCC_nonHCC_Radiologist_type7_3_Overall.append(HCC_nonHCC_Radiologist_type7_3)

            # For type 8, LiRad 5 --- HCC, LiRad 1, 2, 3, 4 and LR-M --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type8 = 1
                HCC_nonHCC_Radiologist_type8_Overall.append(HCC_nonHCC_Radiologist_type8)
            else:  # LiRad 1, 2, 3, 4 and LR-M    --- Non-HCC
                HCC_nonHCC_Radiologist_type8 = 0
                HCC_nonHCC_Radiologist_type8_Overall.append(HCC_nonHCC_Radiologist_type8)

            Test_ID_Lesion_Label_GT_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type2_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type3_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type4_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type5_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type6_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type7_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type8_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_Radiologist_At_Risk.append([key, HCC_nonHCC_Radiologist])
            Test_ID_Lesion_Label_Radiologist_type2_At_Risk.append([key, HCC_nonHCC_Radiologist_type2])
            Test_ID_Lesion_Label_Radiologist_type3_At_Risk.append([key, HCC_nonHCC_Radiologist_type3])
            Test_ID_Lesion_Label_Radiologist_type4_At_Risk.append([key, HCC_nonHCC_Radiologist_type4])
            Test_ID_Lesion_Label_Radiologist_type5_At_Risk.append([key, HCC_nonHCC_Radiologist_type5])
            Test_ID_Lesion_Label_Radiologist_type6_At_Risk.append([key, HCC_nonHCC_Radiologist_type6])
            Test_ID_Lesion_Label_Radiologist_type7_At_Risk.append([key, HCC_nonHCC_Radiologist_type7])
            Test_ID_Lesion_Label_Radiologist_type8_At_Risk.append([key, HCC_nonHCC_Radiologist_type8])

            Test_ID_Lesion_Label_Radiologist_type4_2_At_Risk.append([key, HCC_nonHCC_Radiologist_type4_2])
            Test_ID_Lesion_Label_Radiologist_type4_3_At_Risk.append([key, HCC_nonHCC_Radiologist_type4_3])
            Test_ID_Lesion_Label_Radiologist_type5_2_At_Risk.append([key, HCC_nonHCC_Radiologist_type5_2])
            Test_ID_Lesion_Label_Radiologist_type5_3_At_Risk.append([key, HCC_nonHCC_Radiologist_type5_3])
            Test_ID_Lesion_Label_Radiologist_type6_2_At_Risk.append([key, HCC_nonHCC_Radiologist_type6_2])
            Test_ID_Lesion_Label_Radiologist_type6_3_At_Risk.append([key, HCC_nonHCC_Radiologist_type6_3])
            Test_ID_Lesion_Label_Radiologist_type7_2_At_Risk.append([key, HCC_nonHCC_Radiologist_type7_2])
            Test_ID_Lesion_Label_Radiologist_type7_3_At_Risk.append([key, HCC_nonHCC_Radiologist_type7_3])

            if np.max(np.array(HCC_nonHCC_GT_Overall)) == 0:
                Test_ID_Patient_Label_GT_From_Excel_At_Risk.append([key, 0])
            else:
                Test_ID_Patient_Label_GT_From_Excel_At_Risk.append([key, 1])

            if np.max(np.array(HCC_nonHCC_Radiologist_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_At_Risk.append([key, 0])
            else:
                Test_ID_Patient_Label_Radiologist_At_Risk.append([key, 1])

            if np.max(np.array(HCC_nonHCC_Radiologist_type2_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type2_At_Risk.append([key, 0])
            else:
                Test_ID_Patient_Label_Radiologist_type2_At_Risk.append([key, 1])

            if np.max(np.array(HCC_nonHCC_Radiologist_type3_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type3_At_Risk.append([key, 0])
            else:
                Test_ID_Patient_Label_Radiologist_type3_At_Risk.append([key, 1])
            # type 4
            if np.max(np.array(HCC_nonHCC_Radiologist_type4_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type4_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type4_Overall)) == 1:
                Test_ID_Patient_Label_Radiologist_type4_At_Risk.append([key, 1])
            else:
                Test_ID_Patient_Label_Radiologist_type4_At_Risk.append([key, 2])
            # type 4_2
            if np.max(np.array(HCC_nonHCC_Radiologist_type4_2_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type4_2_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type4_2_Overall)) == 1:
                Test_ID_Patient_Label_Radiologist_type4_2_At_Risk.append([key, 1])
            else:
                Test_ID_Patient_Label_Radiologist_type4_2_At_Risk.append([key, 2])
            # type 4_3
            if np.max(np.array(HCC_nonHCC_Radiologist_type4_3_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type4_3_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type4_3_Overall)) == 1:
                Test_ID_Patient_Label_Radiologist_type4_3_At_Risk.append([key, 1])
            else:
                Test_ID_Patient_Label_Radiologist_type4_3_At_Risk.append([key, 2])

            # For type 5
            if np.max(np.array(HCC_nonHCC_Radiologist_type5_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type5_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type5_Overall)) == 1:
                Test_ID_Patient_Label_Radiologist_type5_At_Risk.append([key, 1])
            else:
                Test_ID_Patient_Label_Radiologist_type5_At_Risk.append([key, 2])
            # For type 5_2
            if np.max(np.array(HCC_nonHCC_Radiologist_type5_2_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type5_2_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type5_2_Overall)) == 1:
                Test_ID_Patient_Label_Radiologist_type5_2_At_Risk.append([key, 1])
            else:
                Test_ID_Patient_Label_Radiologist_type5_2_At_Risk.append([key, 2])
            # For type 5_3
            if np.max(np.array(HCC_nonHCC_Radiologist_type5_3_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type5_3_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type5_3_Overall)) == 1:
                Test_ID_Patient_Label_Radiologist_type5_3_At_Risk.append([key, 1])
            else:
                Test_ID_Patient_Label_Radiologist_type5_3_At_Risk.append([key, 2])

            # For type 6  -----
            if np.max(np.array(HCC_nonHCC_Radiologist_type6_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type6_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type6_Overall)) == 1:
                Test_ID_Patient_Label_Radiologist_type6_At_Risk.append([key, 1])
            else:
                Test_ID_Patient_Label_Radiologist_type6_At_Risk.append([key, 2])
            # For type 6_2 -----
            if np.max(np.array(HCC_nonHCC_Radiologist_type6_2_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type6_2_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type6_2_Overall)) == 1:
                Test_ID_Patient_Label_Radiologist_type6_2_At_Risk.append([key, 1])
            else:
                Test_ID_Patient_Label_Radiologist_type6_2_At_Risk.append([key, 2])
            # For type 6_3  -----
            if np.max(np.array(HCC_nonHCC_Radiologist_type6_3_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type6_3_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type6_3_Overall)) == 1:
                Test_ID_Patient_Label_Radiologist_type6_3_At_Risk.append([key, 1])
            else:
                Test_ID_Patient_Label_Radiologist_type6_3_At_Risk.append([key, 2])

            # For type 7 ----
            if np.max(np.array(HCC_nonHCC_Radiologist_type7_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type7_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type7_Overall)) == 1:
                Test_ID_Patient_Label_Radiologist_type7_At_Risk.append([key, 1])
            else:
                Test_ID_Patient_Label_Radiologist_type7_At_Risk.append([key, 2])
            # For type 7_2 ----
            if np.max(np.array(HCC_nonHCC_Radiologist_type7_2_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type7_2_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type7_2_Overall)) == 1:
                Test_ID_Patient_Label_Radiologist_type7_2_At_Risk.append([key, 1])
            else:
                Test_ID_Patient_Label_Radiologist_type7_2_At_Risk.append([key, 2])
            # For type 7_3 ----
            if np.max(np.array(HCC_nonHCC_Radiologist_type7_3_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type7_3_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type7_3_Overall)) == 1:
                Test_ID_Patient_Label_Radiologist_type7_3_At_Risk.append([key, 1])
            else:
                Test_ID_Patient_Label_Radiologist_type7_3_At_Risk.append([key, 2])

            # For type 8 -----
            if np.max(np.array(HCC_nonHCC_Radiologist_type8_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type8_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type8_Overall)) == 1:
                Test_ID_Patient_Label_Radiologist_type8_At_Risk.append([key, 1])
            else:
                Test_ID_Patient_Label_Radiologist_type8_At_Risk.append([key, 2])
        elif len(values) == 5:
            Lesion_First = values[0:3]
            Lesion_Type = Lesion_First[0]
            Lesion_LiRad = Lesion_First[2]
            HCC_nonHCC_GT_Overall = []
            HCC_nonHCC_Radiologist_Overall, HCC_nonHCC_Radiologist_type2_Overall = [], []
            HCC_nonHCC_Radiologist_type3_Overall, HCC_nonHCC_Radiologist_type4_Overall = [], []
            HCC_nonHCC_Radiologist_type5_Overall, HCC_nonHCC_Radiologist_type6_Overall = [], []
            HCC_nonHCC_Radiologist_type7_Overall, HCC_nonHCC_Radiologist_type8_Overall = [], []

            HCC_nonHCC_Radiologist_type4_2_Overall, HCC_nonHCC_Radiologist_type4_3_Overall = [], []
            HCC_nonHCC_Radiologist_type5_2_Overall, HCC_nonHCC_Radiologist_type5_3_Overall = [], []
            HCC_nonHCC_Radiologist_type6_2_Overall, HCC_nonHCC_Radiologist_type6_3_Overall = [], []
            HCC_nonHCC_Radiologist_type7_2_Overall, HCC_nonHCC_Radiologist_type7_3_Overall = [], []

            if Lesion_Type == 2 or Lesion_Type == 6 or Lesion_Type == 10 or Lesion_Type == 19 or Lesion_Type == 20 \
                    or Lesion_Type == 21 or Lesion_Type == 22:
                HCC_nonHCC_GT = 1
                HCC_nonHCC_GT_Overall.append(HCC_nonHCC_GT)
            else:
                HCC_nonHCC_GT = 0
                HCC_nonHCC_GT_Overall.append(HCC_nonHCC_GT)

            if type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist = 1
                HCC_nonHCC_Radiologist_Overall.append(HCC_nonHCC_Radiologist)
            elif (type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M') or (Lesion_LiRad > 0 and Lesion_LiRad < 4):
                HCC_nonHCC_Radiologist = 0
                HCC_nonHCC_Radiologist_Overall.append(HCC_nonHCC_Radiologist)
            else:
                HCC_nonHCC_Radiologist = Lesion_LiRad

            # For type 2, LiRads 3, 4, 5 and LR-M are considered as Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type2 = 0
                HCC_nonHCC_Radiologist_type2_Overall.append(HCC_nonHCC_Radiologist_type2)
            else:
                HCC_nonHCC_Radiologist_type2 = 1
                HCC_nonHCC_Radiologist_type2_Overall.append(HCC_nonHCC_Radiologist_type2)
            # For type 3, LiRad 4, 5 and LR-M are considered as HCC, and LiRads 1, 2, and 3 are Non-HCC
            if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 3):
                HCC_nonHCC_Radiologist_type3 = 0
                HCC_nonHCC_Radiologist_type3_Overall.append(HCC_nonHCC_Radiologist_type3)
            else:
                HCC_nonHCC_Radiologist_type3 = 1
                HCC_nonHCC_Radiologist_type3_Overall.append(HCC_nonHCC_Radiologist_type3)
            # For type 4, LiRad 4, 5 --- HCC; LR-M and LiRad 3 --- Uncertain, LiRad 1, 2 --- Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type4 = 0  # Non-HCC
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type4 = 1  # HCC
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)
            else:
                HCC_nonHCC_Radiologist_type4 = 2  # Uncertain
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)
            # For type 4_2, LiRad 4, 5 --- HCC; LR-M and LiRad 3 --- Uncertain, LiRad 1, 2 --- Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type4_2 = 0  # Non-HCC
                HCC_nonHCC_Radiologist_type4_2_Overall.append(HCC_nonHCC_Radiologist_type4_2)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type4_2 = 1  # HCC
                HCC_nonHCC_Radiologist_type4_2_Overall.append(HCC_nonHCC_Radiologist_type4_2)
            else:
                HCC_nonHCC_Radiologist_type4_2 = 1  # Uncertain
                HCC_nonHCC_Radiologist_type4_2_Overall.append(HCC_nonHCC_Radiologist_type4_2)
            # For type 4_3, LiRad 4, 5 --- HCC; LR-M and LiRad 3 --- Uncertain, LiRad 1, 2 --- Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type4_3 = 0  # Non-HCC
                HCC_nonHCC_Radiologist_type4_3_Overall.append(HCC_nonHCC_Radiologist_type4_3)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type4_3 = 1  # HCC
                HCC_nonHCC_Radiologist_type4_3_Overall.append(HCC_nonHCC_Radiologist_type4_3)
            else:
                HCC_nonHCC_Radiologist_type4_3 = 0  # Uncertain
                HCC_nonHCC_Radiologist_type4_3_Overall.append(HCC_nonHCC_Radiologist_type4_3)

            # For type 5, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type5 = 0
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type5 = 1
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                HCC_nonHCC_Radiologist_type5 = 1
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            else:  # LiRads = 3    ------ Uncertain
                HCC_nonHCC_Radiologist_type5 = 2
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            # For type 5_2, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type5_2 = 0
                HCC_nonHCC_Radiologist_type5_2_Overall.append(HCC_nonHCC_Radiologist_type5_2)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type5_2 = 1
                HCC_nonHCC_Radiologist_type5_2_Overall.append(HCC_nonHCC_Radiologist_type5_2)
            elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                HCC_nonHCC_Radiologist_type5_2 = 1
                HCC_nonHCC_Radiologist_type5_2_Overall.append(HCC_nonHCC_Radiologist_type5_2)
            else:  # LiRads = 3    ------ Uncertain
                HCC_nonHCC_Radiologist_type5_2 = 1
                HCC_nonHCC_Radiologist_type5_2_Overall.append(HCC_nonHCC_Radiologist_type5_2)
            # For type 5_3, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type5_3 = 0
                HCC_nonHCC_Radiologist_type5_3_Overall.append(HCC_nonHCC_Radiologist_type5_3)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type5_3 = 1
                HCC_nonHCC_Radiologist_type5_3_Overall.append(HCC_nonHCC_Radiologist_type5_3)
            elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                HCC_nonHCC_Radiologist_type5_3 = 1
                HCC_nonHCC_Radiologist_type5_3_Overall.append(HCC_nonHCC_Radiologist_type5_3)
            else:  # LiRads = 3    ------ Uncertain
                HCC_nonHCC_Radiologist_type5_3 = 0
                HCC_nonHCC_Radiologist_type5_3_Overall.append(HCC_nonHCC_Radiologist_type5_3)

            # For type 6, LiRad 5 --- HCC, and LR-M, LiRad 3, 4 --- Uncertain, LiRad 1, 2 --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):  # LiRad 1, 2 --- NonHCC
                HCC_nonHCC_Radiologist_type6 = 0
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5  --- HCC
                HCC_nonHCC_Radiologist_type6 = 1
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)
            else:  # LiRad 3, 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type6 = 2
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)
            # For type 6_2, LiRad 5 --- HCC, and LR-M, LiRad 3, 4 --- Uncertain, LiRad 1, 2 --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):  # LiRad 1, 2 --- NonHCC
                HCC_nonHCC_Radiologist_type6_2 = 0
                HCC_nonHCC_Radiologist_type6_2_Overall.append(HCC_nonHCC_Radiologist_type6_2)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5  --- HCC
                HCC_nonHCC_Radiologist_type6_2 = 1
                HCC_nonHCC_Radiologist_type6_2_Overall.append(HCC_nonHCC_Radiologist_type6_2)
            else:  # LiRad 3, 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type6_2 = 1
                HCC_nonHCC_Radiologist_type6_2_Overall.append(HCC_nonHCC_Radiologist_type6_2)
            # For type 6_3, LiRad 5 --- HCC, and LR-M, LiRad 3, 4 --- Uncertain, LiRad 1, 2 --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):  # LiRad 1, 2 --- NonHCC
                HCC_nonHCC_Radiologist_type6_3 = 0
                HCC_nonHCC_Radiologist_type6_3_Overall.append(HCC_nonHCC_Radiologist_type6_3)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5  --- HCC
                HCC_nonHCC_Radiologist_type6_3 = 1
                HCC_nonHCC_Radiologist_type6_3_Overall.append(HCC_nonHCC_Radiologist_type6_3)
            else:  # LiRad 3, 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type6_3 = 0
                HCC_nonHCC_Radiologist_type6_3_Overall.append(HCC_nonHCC_Radiologist_type6_3)

            # For type 7, LiRad 5 --- HCC, LiRad 4, LR-M ---- Uncertain, LiRad 1, 2, 3 --- NonHCC
            if type(Lesion_LiRad) == int and 1 <= Lesion_LiRad <= 3:  # LiRad 1, 2, 3 --- NonHCC
                HCC_nonHCC_Radiologist_type7 = 0
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type7 = 1
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)
            else:  # LiRads 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type7 = 2
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)
            # For type 7_2, LiRad 5 --- HCC, LiRad 4, LR-M ---- Uncertain, LiRad 1, 2, 3 --- NonHCC
            if type(Lesion_LiRad) == int and 1 <= Lesion_LiRad <= 3:  # LiRad 1, 2, 3 --- NonHCC
                HCC_nonHCC_Radiologist_type7_2 = 0
                HCC_nonHCC_Radiologist_type7_2_Overall.append(HCC_nonHCC_Radiologist_type7_2)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type7_2 = 1
                HCC_nonHCC_Radiologist_type7_2_Overall.append(HCC_nonHCC_Radiologist_type7_2)
            else:  # LiRads 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type7_2 = 1
                HCC_nonHCC_Radiologist_type7_2_Overall.append(HCC_nonHCC_Radiologist_type7_2)
            # For type 7_3, LiRad 5 --- HCC, LiRad 4, LR-M ---- Uncertain, LiRad 1, 2, 3 --- NonHCC
            if type(Lesion_LiRad) == int and 1 <= Lesion_LiRad <= 3:  # LiRad 1, 2, 3 --- NonHCC
                HCC_nonHCC_Radiologist_type7_3 = 0
                HCC_nonHCC_Radiologist_type7_3_Overall.append(HCC_nonHCC_Radiologist_type7_3)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type7_3 = 1
                HCC_nonHCC_Radiologist_type7_3_Overall.append(HCC_nonHCC_Radiologist_type7_3)
            else:  # LiRads 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type7_3 = 0
                HCC_nonHCC_Radiologist_type7_3_Overall.append(HCC_nonHCC_Radiologist_type7_3)

            # For type 8, LiRad 5 --- HCC, LiRad 1, 2, 3, 4 and LR-M --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type8 = 1
                HCC_nonHCC_Radiologist_type8_Overall.append(HCC_nonHCC_Radiologist_type8)
            else:  # LiRad 1, 2, 3, 4 and LR-M    --- Non-HCC
                HCC_nonHCC_Radiologist_type8 = 0
                HCC_nonHCC_Radiologist_type8_Overall.append(HCC_nonHCC_Radiologist_type8)

            Test_ID_Lesion_Label_GT_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type2_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type3_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type4_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type5_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type6_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type7_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type8_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_Radiologist_At_Risk.append([key, HCC_nonHCC_Radiologist])
            Test_ID_Lesion_Label_Radiologist_type2_At_Risk.append([key, HCC_nonHCC_Radiologist_type2])
            Test_ID_Lesion_Label_Radiologist_type3_At_Risk.append([key, HCC_nonHCC_Radiologist_type3])
            Test_ID_Lesion_Label_Radiologist_type4_At_Risk.append([key, HCC_nonHCC_Radiologist_type4])
            Test_ID_Lesion_Label_Radiologist_type5_At_Risk.append([key, HCC_nonHCC_Radiologist_type5])
            Test_ID_Lesion_Label_Radiologist_type6_At_Risk.append([key, HCC_nonHCC_Radiologist_type6])
            Test_ID_Lesion_Label_Radiologist_type7_At_Risk.append([key, HCC_nonHCC_Radiologist_type7])
            Test_ID_Lesion_Label_Radiologist_type8_At_Risk.append([key, HCC_nonHCC_Radiologist_type8])

            Test_ID_Lesion_Label_Radiologist_type4_2_At_Risk.append([key, HCC_nonHCC_Radiologist_type4_2])
            Test_ID_Lesion_Label_Radiologist_type4_3_At_Risk.append([key, HCC_nonHCC_Radiologist_type4_3])
            Test_ID_Lesion_Label_Radiologist_type5_2_At_Risk.append([key, HCC_nonHCC_Radiologist_type5_2])
            Test_ID_Lesion_Label_Radiologist_type5_3_At_Risk.append([key, HCC_nonHCC_Radiologist_type5_3])
            Test_ID_Lesion_Label_Radiologist_type6_2_At_Risk.append([key, HCC_nonHCC_Radiologist_type6_2])
            Test_ID_Lesion_Label_Radiologist_type6_3_At_Risk.append([key, HCC_nonHCC_Radiologist_type6_3])
            Test_ID_Lesion_Label_Radiologist_type7_2_At_Risk.append([key, HCC_nonHCC_Radiologist_type7_2])
            Test_ID_Lesion_Label_Radiologist_type7_3_At_Risk.append([key, HCC_nonHCC_Radiologist_type7_3])

            Lesion_Second = values[3]
            Lesion_Type = Lesion_Second[0]
            Lesion_LiRad = Lesion_Second[2]
            if Lesion_Type == 2 or Lesion_Type == 6 or Lesion_Type == 10 or Lesion_Type == 19 or Lesion_Type == 20 \
                    or Lesion_Type == 21 or Lesion_Type == 22:
                HCC_nonHCC_GT = 1
                HCC_nonHCC_GT_Overall.append(HCC_nonHCC_GT)
            else:
                HCC_nonHCC_GT = 0
                HCC_nonHCC_GT_Overall.append(HCC_nonHCC_GT)

            if type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist = 1
                HCC_nonHCC_Radiologist_Overall.append(HCC_nonHCC_Radiologist)
            elif (type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M') or (Lesion_LiRad > 0 and Lesion_LiRad < 4):
                HCC_nonHCC_Radiologist = 0
                HCC_nonHCC_Radiologist_Overall.append(HCC_nonHCC_Radiologist)
            else:
                HCC_nonHCC_Radiologist = Lesion_LiRad

            # For type 2, LiRads 3, 4, 5 and LR-M are considered as Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type2 = 0
                HCC_nonHCC_Radiologist_type2_Overall.append(HCC_nonHCC_Radiologist_type2)
            else:
                HCC_nonHCC_Radiologist_type2 = 1
                HCC_nonHCC_Radiologist_type2_Overall.append(HCC_nonHCC_Radiologist_type2)

            # For type 3, LiRad 4, 5 and LR-M are considered as HCC, and LiRads 1, 2, and 3 are Non-HCC
            if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 3):
                HCC_nonHCC_Radiologist_type3 = 0
                HCC_nonHCC_Radiologist_type3_Overall.append(HCC_nonHCC_Radiologist_type3)
            else:
                HCC_nonHCC_Radiologist_type3 = 1
                HCC_nonHCC_Radiologist_type3_Overall.append(HCC_nonHCC_Radiologist_type3)

            # For type 4, LiRad 4, 5 --- HCC; LR-M and LiRad 3 --- Uncertain, LiRad 1, 2 --- Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type4 = 0  # Non-HCC
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type4 = 1  # HCC
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)
            else:
                HCC_nonHCC_Radiologist_type4 = 2  # Uncertain
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)
            # For type 4_2, LiRad 4, 5 --- HCC; LR-M and LiRad 3 --- Uncertain, LiRad 1, 2 --- Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type4_2 = 0  # Non-HCC
                HCC_nonHCC_Radiologist_type4_2_Overall.append(HCC_nonHCC_Radiologist_type4_2)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type4_2 = 1  # HCC
                HCC_nonHCC_Radiologist_type4_2_Overall.append(HCC_nonHCC_Radiologist_type4_2)
            else:
                HCC_nonHCC_Radiologist_type4_2 = 1  # Uncertain
                HCC_nonHCC_Radiologist_type4_2_Overall.append(HCC_nonHCC_Radiologist_type4_2)
            # For type 4_3, LiRad 4, 5 --- HCC; LR-M and LiRad 3 --- Uncertain, LiRad 1, 2 --- Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type4_3 = 0  # Non-HCC
                HCC_nonHCC_Radiologist_type4_3_Overall.append(HCC_nonHCC_Radiologist_type4_3)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type4_3 = 1  # HCC
                HCC_nonHCC_Radiologist_type4_3_Overall.append(HCC_nonHCC_Radiologist_type4_3)
            else:
                HCC_nonHCC_Radiologist_type4_3 = 0  # Uncertain
                HCC_nonHCC_Radiologist_type4_3_Overall.append(HCC_nonHCC_Radiologist_type4_3)

            # For type 5, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type5 = 0
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type5 = 1
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                HCC_nonHCC_Radiologist_type5 = 1
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            else:  # LiRads = 3    ------ Uncertain
                HCC_nonHCC_Radiologist_type5 = 2
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            # For type 5_2, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type5_2 = 0
                HCC_nonHCC_Radiologist_type5_2_Overall.append(HCC_nonHCC_Radiologist_type5_2)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type5_2 = 1
                HCC_nonHCC_Radiologist_type5_2_Overall.append(HCC_nonHCC_Radiologist_type5_2)
            elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                HCC_nonHCC_Radiologist_type5_2 = 1
                HCC_nonHCC_Radiologist_type5_2_Overall.append(HCC_nonHCC_Radiologist_type5_2)
            else:  # LiRads = 3    ------ Uncertain
                HCC_nonHCC_Radiologist_type5_2 = 1
                HCC_nonHCC_Radiologist_type5_2_Overall.append(HCC_nonHCC_Radiologist_type5_2)
            # For type 5_3, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type5_3 = 0
                HCC_nonHCC_Radiologist_type5_3_Overall.append(HCC_nonHCC_Radiologist_type5_3)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type5_3 = 1
                HCC_nonHCC_Radiologist_type5_3_Overall.append(HCC_nonHCC_Radiologist_type5_3)
            elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                HCC_nonHCC_Radiologist_type5_3 = 1
                HCC_nonHCC_Radiologist_type5_3_Overall.append(HCC_nonHCC_Radiologist_type5_3)
            else:  # LiRads = 3    ------ Uncertain
                HCC_nonHCC_Radiologist_type5_3 = 0
                HCC_nonHCC_Radiologist_type5_3_Overall.append(HCC_nonHCC_Radiologist_type5_3)

            # For type 6, LiRad 5 --- HCC, and LR-M, LiRad 3, 4 --- Uncertain, LiRad 1, 2 --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):  # LiRad 1, 2 --- NonHCC
                HCC_nonHCC_Radiologist_type6 = 0
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5  --- HCC
                HCC_nonHCC_Radiologist_type6 = 1
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)
            else:  # LiRad 3, 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type6 = 2
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)
            # For type 6_2, LiRad 5 --- HCC, and LR-M, LiRad 3, 4 --- Uncertain, LiRad 1, 2 --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):  # LiRad 1, 2 --- NonHCC
                HCC_nonHCC_Radiologist_type6_2 = 0
                HCC_nonHCC_Radiologist_type6_2_Overall.append(HCC_nonHCC_Radiologist_type6_2)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5  --- HCC
                HCC_nonHCC_Radiologist_type6_2 = 1
                HCC_nonHCC_Radiologist_type6_2_Overall.append(HCC_nonHCC_Radiologist_type6_2)
            else:  # LiRad 3, 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type6_2 = 1
                HCC_nonHCC_Radiologist_type6_2_Overall.append(HCC_nonHCC_Radiologist_type6_2)
                # For type 6_3, LiRad 5 --- HCC, and LR-M, LiRad 3, 4 --- Uncertain, LiRad 1, 2 --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):  # LiRad 1, 2 --- NonHCC
                HCC_nonHCC_Radiologist_type6_3 = 0
                HCC_nonHCC_Radiologist_type6_3_Overall.append(HCC_nonHCC_Radiologist_type6_3)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5  --- HCC
                HCC_nonHCC_Radiologist_type6_3 = 1
                HCC_nonHCC_Radiologist_type6_3_Overall.append(HCC_nonHCC_Radiologist_type6_3)
            else:  # LiRad 3, 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type6_3 = 0
                HCC_nonHCC_Radiologist_type6_3_Overall.append(HCC_nonHCC_Radiologist_type6_3)

            # For type 7, LiRad 5 --- HCC, LiRad 4, LR-M ---- Uncertain, LiRad 1, 2, 3 --- NonHCC
            if type(Lesion_LiRad) == int and 1 <= Lesion_LiRad <= 3:  # LiRad 1, 2, 3 --- NonHCC
                HCC_nonHCC_Radiologist_type7 = 0
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type7 = 1
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)
            else:  # LiRads 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type7 = 2
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)
            # For type 7_2, LiRad 5 --- HCC, LiRad 4, LR-M ---- Uncertain, LiRad 1, 2, 3 --- NonHCC
            if type(Lesion_LiRad) == int and 1 <= Lesion_LiRad <= 3:  # LiRad 1, 2, 3 --- NonHCC
                HCC_nonHCC_Radiologist_type7_2 = 0
                HCC_nonHCC_Radiologist_type7_2_Overall.append(HCC_nonHCC_Radiologist_type7_2)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type7_2 = 1
                HCC_nonHCC_Radiologist_type7_2_Overall.append(HCC_nonHCC_Radiologist_type7_2)
            else:  # LiRads 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type7_2 = 1
                HCC_nonHCC_Radiologist_type7_2_Overall.append(HCC_nonHCC_Radiologist_type7_2)
            # For type 7_3, LiRad 5 --- HCC, LiRad 4, LR-M ---- Uncertain, LiRad 1, 2, 3 --- NonHCC
            if type(Lesion_LiRad) == int and 1 <= Lesion_LiRad <= 3:  # LiRad 1, 2, 3 --- NonHCC
                HCC_nonHCC_Radiologist_type7_3 = 0
                HCC_nonHCC_Radiologist_type7_3_Overall.append(HCC_nonHCC_Radiologist_type7_3)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type7_3 = 1
                HCC_nonHCC_Radiologist_type7_3_Overall.append(HCC_nonHCC_Radiologist_type7_3)
            else:  # LiRads 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type7_3 = 0
                HCC_nonHCC_Radiologist_type7_3_Overall.append(HCC_nonHCC_Radiologist_type7_3)

            # For type 8, LiRad 5 --- HCC, LiRad 1, 2, 3, 4 and LR-M --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type8 = 1
                HCC_nonHCC_Radiologist_type8_Overall.append(HCC_nonHCC_Radiologist_type8)
            else:  # LiRad 1, 2, 3, 4 and LR-M    --- Non-HCC
                HCC_nonHCC_Radiologist_type8 = 0
                HCC_nonHCC_Radiologist_type8_Overall.append(HCC_nonHCC_Radiologist_type8)

            Test_ID_Lesion_Label_GT_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type2_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type3_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type4_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type5_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type6_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type7_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type8_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_Radiologist_At_Risk.append([key, HCC_nonHCC_Radiologist])
            Test_ID_Lesion_Label_Radiologist_type2_At_Risk.append([key, HCC_nonHCC_Radiologist_type2])
            Test_ID_Lesion_Label_Radiologist_type3_At_Risk.append([key, HCC_nonHCC_Radiologist_type3])
            Test_ID_Lesion_Label_Radiologist_type4_At_Risk.append([key, HCC_nonHCC_Radiologist_type4])
            Test_ID_Lesion_Label_Radiologist_type5_At_Risk.append([key, HCC_nonHCC_Radiologist_type5])
            Test_ID_Lesion_Label_Radiologist_type6_At_Risk.append([key, HCC_nonHCC_Radiologist_type6])
            Test_ID_Lesion_Label_Radiologist_type7_At_Risk.append([key, HCC_nonHCC_Radiologist_type7])
            Test_ID_Lesion_Label_Radiologist_type8_At_Risk.append([key, HCC_nonHCC_Radiologist_type8])

            Test_ID_Lesion_Label_Radiologist_type4_2_At_Risk.append([key, HCC_nonHCC_Radiologist_type4_2])
            Test_ID_Lesion_Label_Radiologist_type4_3_At_Risk.append([key, HCC_nonHCC_Radiologist_type4_3])
            Test_ID_Lesion_Label_Radiologist_type5_2_At_Risk.append([key, HCC_nonHCC_Radiologist_type5_2])
            Test_ID_Lesion_Label_Radiologist_type5_3_At_Risk.append([key, HCC_nonHCC_Radiologist_type5_3])
            Test_ID_Lesion_Label_Radiologist_type6_2_At_Risk.append([key, HCC_nonHCC_Radiologist_type6_2])
            Test_ID_Lesion_Label_Radiologist_type6_3_At_Risk.append([key, HCC_nonHCC_Radiologist_type6_3])
            Test_ID_Lesion_Label_Radiologist_type7_2_At_Risk.append([key, HCC_nonHCC_Radiologist_type7_2])
            Test_ID_Lesion_Label_Radiologist_type7_3_At_Risk.append([key, HCC_nonHCC_Radiologist_type7_3])

            Lesion_Third = values[4]
            Lesion_Type = Lesion_Third[0]
            Lesion_LiRad = Lesion_Third[2]
            if Lesion_Type == 2 or Lesion_Type == 6 or Lesion_Type == 10 or Lesion_Type == 19 or Lesion_Type == 20 \
                    or Lesion_Type == 21 or Lesion_Type == 22:
                HCC_nonHCC_GT = 1
                HCC_nonHCC_GT_Overall.append(HCC_nonHCC_GT)
            else:
                HCC_nonHCC_GT = 0
                HCC_nonHCC_GT_Overall.append(HCC_nonHCC_GT)

            if type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist = 1
                HCC_nonHCC_Radiologist_Overall.append(HCC_nonHCC_Radiologist)
            elif (type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M') or (Lesion_LiRad > 0 and Lesion_LiRad < 4):
                HCC_nonHCC_Radiologist = 0
                HCC_nonHCC_Radiologist_Overall.append(HCC_nonHCC_Radiologist)
            else:
                HCC_nonHCC_Radiologist = Lesion_LiRad

            # For type 2, LiRads 3, 4, 5 and LR-M are considered as Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type2 = 0
                HCC_nonHCC_Radiologist_type2_Overall.append(HCC_nonHCC_Radiologist_type2)
            else:
                HCC_nonHCC_Radiologist_type2 = 1
                HCC_nonHCC_Radiologist_type2_Overall.append(HCC_nonHCC_Radiologist_type2)

            # For type 3, LiRad 4, 5 and LR-M are considered as HCC, and LiRads 1, 2, and 3 are Non-HCC
            if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 3):
                HCC_nonHCC_Radiologist_type3 = 0
                HCC_nonHCC_Radiologist_type3_Overall.append(HCC_nonHCC_Radiologist_type3)
            else:
                HCC_nonHCC_Radiologist_type3 = 1
                HCC_nonHCC_Radiologist_type3_Overall.append(HCC_nonHCC_Radiologist_type3)

            # For type 4, LiRad 4, 5 --- HCC; LR-M and LiRad 3 --- Uncertain, LiRad 1, 2 --- Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type4 = 0  # Non-HCC
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type4 = 1  # HCC
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)
            else:
                HCC_nonHCC_Radiologist_type4 = 2  # Uncertain
                HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)
            # For type 4_2, LiRad 4, 5 --- HCC; LR-M and LiRad 3 --- Uncertain, LiRad 1, 2 --- Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type4_2 = 0  # Non-HCC
                HCC_nonHCC_Radiologist_type4_2_Overall.append(HCC_nonHCC_Radiologist_type4_2)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type4_2 = 1  # HCC
                HCC_nonHCC_Radiologist_type4_2_Overall.append(HCC_nonHCC_Radiologist_type4_2)
            else:
                HCC_nonHCC_Radiologist_type4_2 = 1  # Uncertain
                HCC_nonHCC_Radiologist_type4_2_Overall.append(HCC_nonHCC_Radiologist_type4_2)
            # For type 4_3, LiRad 4, 5 --- HCC; LR-M and LiRad 3 --- Uncertain, LiRad 1, 2 --- Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type4_3 = 0  # Non-HCC
                HCC_nonHCC_Radiologist_type4_3_Overall.append(HCC_nonHCC_Radiologist_type4_3)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type4_3 = 1  # HCC
                HCC_nonHCC_Radiologist_type4_3_Overall.append(HCC_nonHCC_Radiologist_type4_3)
            else:
                HCC_nonHCC_Radiologist_type4_3 = 0  # Uncertain
                HCC_nonHCC_Radiologist_type4_3_Overall.append(HCC_nonHCC_Radiologist_type4_3)

            # For type 5, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type5 = 0
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type5 = 1
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                HCC_nonHCC_Radiologist_type5 = 1
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            else:  # LiRads = 3    ------ Uncertain
                HCC_nonHCC_Radiologist_type5 = 2
                HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)
            # For type 5_2, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type5_2 = 0
                HCC_nonHCC_Radiologist_type5_2_Overall.append(HCC_nonHCC_Radiologist_type5_2)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type5_2 = 1
                HCC_nonHCC_Radiologist_type5_2_Overall.append(HCC_nonHCC_Radiologist_type5_2)
            elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                HCC_nonHCC_Radiologist_type5_2 = 1
                HCC_nonHCC_Radiologist_type5_2_Overall.append(HCC_nonHCC_Radiologist_type5_2)
            else:  # LiRads = 3    ------ Uncertain
                HCC_nonHCC_Radiologist_type5_2 = 1
                HCC_nonHCC_Radiologist_type5_2_Overall.append(HCC_nonHCC_Radiologist_type5_2)
            # For type 5_3, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type5_3 = 0
                HCC_nonHCC_Radiologist_type5_3_Overall.append(HCC_nonHCC_Radiologist_type5_3)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type5_3 = 1
                HCC_nonHCC_Radiologist_type5_3_Overall.append(HCC_nonHCC_Radiologist_type5_3)
            elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                HCC_nonHCC_Radiologist_type5_3 = 1
                HCC_nonHCC_Radiologist_type5_3_Overall.append(HCC_nonHCC_Radiologist_type5_3)
            else:  # LiRads = 3    ------ Uncertain
                HCC_nonHCC_Radiologist_type5_3 = 0
                HCC_nonHCC_Radiologist_type5_3_Overall.append(HCC_nonHCC_Radiologist_type5_3)

            # For type 6, LiRad 5 --- HCC, and LR-M, LiRad 3, 4 --- Uncertain, LiRad 1, 2 --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):  # LiRad 1, 2 --- NonHCC
                HCC_nonHCC_Radiologist_type6 = 0
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5  --- HCC
                HCC_nonHCC_Radiologist_type6 = 1
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)
            else:  # LiRad 3, 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type6 = 2
                HCC_nonHCC_Radiologist_type6_Overall.append(HCC_nonHCC_Radiologist_type6)
            # For type 6_2, LiRad 5 --- HCC, and LR-M, LiRad 3, 4 --- Uncertain, LiRad 1, 2 --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):  # LiRad 1, 2 --- NonHCC
                HCC_nonHCC_Radiologist_type6_2 = 0
                HCC_nonHCC_Radiologist_type6_2_Overall.append(HCC_nonHCC_Radiologist_type6_2)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5  --- HCC
                HCC_nonHCC_Radiologist_type6_2 = 1
                HCC_nonHCC_Radiologist_type6_2_Overall.append(HCC_nonHCC_Radiologist_type6_2)
            else:  # LiRad 3, 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type6_2 = 1
                HCC_nonHCC_Radiologist_type6_2_Overall.append(HCC_nonHCC_Radiologist_type6_2)
                # For type 6_3, LiRad 5 --- HCC, and LR-M, LiRad 3, 4 --- Uncertain, LiRad 1, 2 --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):  # LiRad 1, 2 --- NonHCC
                HCC_nonHCC_Radiologist_type6_3 = 0
                HCC_nonHCC_Radiologist_type6_3_Overall.append(HCC_nonHCC_Radiologist_type6_3)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5  --- HCC
                HCC_nonHCC_Radiologist_type6_3 = 1
                HCC_nonHCC_Radiologist_type6_3_Overall.append(HCC_nonHCC_Radiologist_type6_3)
            else:  # LiRad 3, 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type6_3 = 0
                HCC_nonHCC_Radiologist_type6_3_Overall.append(HCC_nonHCC_Radiologist_type6_3)

            # For type 7, LiRad 5 --- HCC, LiRad 4, LR-M ---- Uncertain, LiRad 1, 2, 3 --- NonHCC
            if type(Lesion_LiRad) == int and 1 <= Lesion_LiRad <= 3:  # LiRad 1, 2, 3 --- NonHCC
                HCC_nonHCC_Radiologist_type7 = 0
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type7 = 1
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)
            else:  # LiRads 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type7 = 2
                HCC_nonHCC_Radiologist_type7_Overall.append(HCC_nonHCC_Radiologist_type7)
            # For type 7_2, LiRad 5 --- HCC, LiRad 4, LR-M ---- Uncertain, LiRad 1, 2, 3 --- NonHCC
            if type(Lesion_LiRad) == int and 1 <= Lesion_LiRad <= 3:  # LiRad 1, 2, 3 --- NonHCC
                HCC_nonHCC_Radiologist_type7_2 = 0
                HCC_nonHCC_Radiologist_type7_2_Overall.append(HCC_nonHCC_Radiologist_type7_2)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type7_2 = 1
                HCC_nonHCC_Radiologist_type7_2_Overall.append(HCC_nonHCC_Radiologist_type7_2)
            else:  # LiRads 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type7_2 = 1
                HCC_nonHCC_Radiologist_type7_2_Overall.append(HCC_nonHCC_Radiologist_type7_2)
            # For type 7_3, LiRad 5 --- HCC, LiRad 4, LR-M ---- Uncertain, LiRad 1, 2, 3 --- NonHCC
            if type(Lesion_LiRad) == int and 1 <= Lesion_LiRad <= 3:  # LiRad 1, 2, 3 --- NonHCC
                HCC_nonHCC_Radiologist_type7_3 = 0
                HCC_nonHCC_Radiologist_type7_3_Overall.append(HCC_nonHCC_Radiologist_type7_3)
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type7_3 = 1
                HCC_nonHCC_Radiologist_type7_3_Overall.append(HCC_nonHCC_Radiologist_type7_3)
            else:  # LiRads 4, LR-M --- Uncertain
                HCC_nonHCC_Radiologist_type7_3 = 0
                HCC_nonHCC_Radiologist_type7_3_Overall.append(HCC_nonHCC_Radiologist_type7_3)

            # For type 8, LiRad 5 --- HCC, LiRad 1, 2, 3, 4 and LR-M --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 5):  # LiRad 5 --- HCC
                HCC_nonHCC_Radiologist_type8 = 1
                HCC_nonHCC_Radiologist_type8_Overall.append(HCC_nonHCC_Radiologist_type8)
            else:  # LiRad 1, 2, 3, 4 and LR-M    --- Non-HCC
                HCC_nonHCC_Radiologist_type8 = 0
                HCC_nonHCC_Radiologist_type8_Overall.append(HCC_nonHCC_Radiologist_type8)

            Test_ID_Lesion_Label_GT_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type2_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type3_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type4_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type5_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type6_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type7_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_GT_type8_At_Risk.append([key, HCC_nonHCC_GT])
            Test_ID_Lesion_Label_Radiologist_At_Risk.append([key, HCC_nonHCC_Radiologist])
            Test_ID_Lesion_Label_Radiologist_type2_At_Risk.append([key, HCC_nonHCC_Radiologist_type2])
            Test_ID_Lesion_Label_Radiologist_type3_At_Risk.append([key, HCC_nonHCC_Radiologist_type3])
            Test_ID_Lesion_Label_Radiologist_type4_At_Risk.append([key, HCC_nonHCC_Radiologist_type4])
            Test_ID_Lesion_Label_Radiologist_type5_At_Risk.append([key, HCC_nonHCC_Radiologist_type5])
            Test_ID_Lesion_Label_Radiologist_type6_At_Risk.append([key, HCC_nonHCC_Radiologist_type6])
            Test_ID_Lesion_Label_Radiologist_type7_At_Risk.append([key, HCC_nonHCC_Radiologist_type7])
            Test_ID_Lesion_Label_Radiologist_type8_At_Risk.append([key, HCC_nonHCC_Radiologist_type8])

            Test_ID_Lesion_Label_Radiologist_type4_2_At_Risk.append([key, HCC_nonHCC_Radiologist_type4_2])
            Test_ID_Lesion_Label_Radiologist_type4_3_At_Risk.append([key, HCC_nonHCC_Radiologist_type4_3])
            Test_ID_Lesion_Label_Radiologist_type5_2_At_Risk.append([key, HCC_nonHCC_Radiologist_type5_2])
            Test_ID_Lesion_Label_Radiologist_type5_3_At_Risk.append([key, HCC_nonHCC_Radiologist_type5_3])
            Test_ID_Lesion_Label_Radiologist_type6_2_At_Risk.append([key, HCC_nonHCC_Radiologist_type6_2])
            Test_ID_Lesion_Label_Radiologist_type6_3_At_Risk.append([key, HCC_nonHCC_Radiologist_type6_3])
            Test_ID_Lesion_Label_Radiologist_type7_2_At_Risk.append([key, HCC_nonHCC_Radiologist_type7_2])
            Test_ID_Lesion_Label_Radiologist_type7_3_At_Risk.append([key, HCC_nonHCC_Radiologist_type7_3])

            if np.max(np.array(HCC_nonHCC_GT_Overall)) == 0:
                Test_ID_Patient_Label_GT_From_Excel_At_Risk.append([key, 0])
            else:
                Test_ID_Patient_Label_GT_From_Excel_At_Risk.append([key, 1])

            if np.max(np.array(HCC_nonHCC_Radiologist_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_At_Risk.append([key, 0])
            else:
                Test_ID_Patient_Label_Radiologist_At_Risk.append([key, 1])

            if np.max(np.array(HCC_nonHCC_Radiologist_type2_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type2_At_Risk.append([key, 0])
            else:
                Test_ID_Patient_Label_Radiologist_type2_At_Risk.append([key, 1])

            if np.max(np.array(HCC_nonHCC_Radiologist_type3_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type3_At_Risk.append([key, 0])
            else:
                Test_ID_Patient_Label_Radiologist_type3_At_Risk.append([key, 1])
            # Type 4
            if np.max(np.array(HCC_nonHCC_Radiologist_type4_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type4_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type4_Overall)) == 1:
                Test_ID_Patient_Label_Radiologist_type4_At_Risk.append([key, 1])
            else:
                Test_ID_Patient_Label_Radiologist_type4_At_Risk.append([key, 2])
            # Type 4_2
            if np.max(np.array(HCC_nonHCC_Radiologist_type4_2_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type4_2_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type4_2_Overall)) == 1:
                Test_ID_Patient_Label_Radiologist_type4_2_At_Risk.append([key, 1])
            else:
                Test_ID_Patient_Label_Radiologist_type4_2_At_Risk.append([key, 2])
            # Type 4_3
            if np.max(np.array(HCC_nonHCC_Radiologist_type4_3_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type4_3_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type4_3_Overall)) == 1:
                Test_ID_Patient_Label_Radiologist_type4_3_At_Risk.append([key, 1])
            else:
                Test_ID_Patient_Label_Radiologist_type4_3_At_Risk.append([key, 2])

            # Type 5
            if np.max(np.array(HCC_nonHCC_Radiologist_type5_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type5_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type5_Overall)) == 1:
                Test_ID_Patient_Label_Radiologist_type5_At_Risk.append([key, 1])
            else:
                Test_ID_Patient_Label_Radiologist_type5_At_Risk.append([key, 2])
            # Type 5_2
            if np.max(np.array(HCC_nonHCC_Radiologist_type5_2_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type5_2_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type5_2_Overall)) == 1:
                Test_ID_Patient_Label_Radiologist_type5_2_At_Risk.append([key, 1])
            else:
                Test_ID_Patient_Label_Radiologist_type5_2_At_Risk.append([key, 2])
            # Type 5_3
            if np.max(np.array(HCC_nonHCC_Radiologist_type5_3_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type5_3_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type5_3_Overall)) == 1:
                Test_ID_Patient_Label_Radiologist_type5_3_At_Risk.append([key, 1])
            else:
                Test_ID_Patient_Label_Radiologist_type5_3_At_Risk.append([key, 2])

            # Type 6
            if np.max(np.array(HCC_nonHCC_Radiologist_type6_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type6_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type6_Overall)) == 1:
                Test_ID_Patient_Label_Radiologist_type6_At_Risk.append([key, 1])
            else:
                Test_ID_Patient_Label_Radiologist_type6_At_Risk.append([key, 2])
            # Type 6_2
            if np.max(np.array(HCC_nonHCC_Radiologist_type6_2_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type6_2_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type6_2_Overall)) == 1:
                Test_ID_Patient_Label_Radiologist_type6_2_At_Risk.append([key, 1])
            else:
                Test_ID_Patient_Label_Radiologist_type6_2_At_Risk.append([key, 2])
            # Type 6_3
            if np.max(np.array(HCC_nonHCC_Radiologist_type6_3_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type6_3_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type6_3_Overall)) == 1:
                Test_ID_Patient_Label_Radiologist_type6_3_At_Risk.append([key, 1])
            else:
                Test_ID_Patient_Label_Radiologist_type6_3_At_Risk.append([key, 2])

            # Type 7
            if np.max(np.array(HCC_nonHCC_Radiologist_type7_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type7_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type7_Overall)) == 1:
                Test_ID_Patient_Label_Radiologist_type7_At_Risk.append([key, 1])
            else:
                Test_ID_Patient_Label_Radiologist_type7_At_Risk.append([key, 2])
            # Type 7_2
            if np.max(np.array(HCC_nonHCC_Radiologist_type7_2_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type7_2_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type7_2_Overall)) == 1:
                Test_ID_Patient_Label_Radiologist_type7_2_At_Risk.append([key, 1])
            else:
                Test_ID_Patient_Label_Radiologist_type7_2_At_Risk.append([key, 2])
            # Type 7_3
            if np.max(np.array(HCC_nonHCC_Radiologist_type7_3_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type7_3_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type7_3_Overall)) == 1:
                Test_ID_Patient_Label_Radiologist_type7_3_At_Risk.append([key, 1])
            else:
                Test_ID_Patient_Label_Radiologist_type7_3_At_Risk.append([key, 2])

            # Type 8
            if np.max(np.array(HCC_nonHCC_Radiologist_type8_Overall)) == 0:
                Test_ID_Patient_Label_Radiologist_type8_At_Risk.append([key, 0])
            elif np.max(np.array(HCC_nonHCC_Radiologist_type8_Overall)) == 1:
                Test_ID_Patient_Label_Radiologist_type8_At_Risk.append([key, 1])
            else:
                Test_ID_Patient_Label_Radiologist_type8_At_Risk.append([key, 2])
    else:
        CNT += 1

# Train_ID_List and Test_ID_List = [1595, 685]  ---> ID_List
print('The number of cases in training and testing sets are: ', len(Train_ID_List), len(Test_ID_List), len(Patient_ID))
print('The number of cases in training and testing sets belonging to sub-group at-risk  are: ',
      len(Train_ID_At_Risk), len(Test_ID_At_Risk))

# HKU_0571_P3 in the training set, but should be removed from the list
# HKU_0030_P3 should be in the list, but during to mismatch of lesion mask and excel file, it seems not to be included
Train_Lesion_Label_GT_At_Risk, Train_Lesion_Label_Radiologist_At_Risk = [], []
Train_Lesion_Label_Radiologist_type2_At_Risk, Train_Lesion_Label_Radiologist_type3_At_Risk, \
Train_Lesion_Label_Radiologist_type4_At_Risk, Train_Lesion_Label_Radiologist_type5_At_Risk, \
Train_Lesion_Label_Radiologist_type6_At_Risk, Train_Lesion_Label_Radiologist_type7_At_Risk, \
Train_Lesion_Label_Radiologist_type8_At_Risk = [], [], [], [], [], [], []

Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_At_Risk = [], []
Test_Lesion_Label_Radiologist_type2_At_Risk, Test_Lesion_Label_Radiologist_type3_At_Risk, \
Test_Lesion_Label_Radiologist_type4_At_Risk, Test_Lesion_Label_Radiologist_type5_At_Risk, \
Test_Lesion_Label_Radiologist_type6_At_Risk, Test_Lesion_Label_Radiologist_type7_At_Risk, \
Test_Lesion_Label_Radiologist_type8_At_Risk = [], [], [], [], [], [], []

Test_Lesion_Label_Radiologist_type4_2_At_Risk, Test_Lesion_Label_Radiologist_type4_3_At_Risk, \
Test_Lesion_Label_Radiologist_type5_2_At_Risk, Test_Lesion_Label_Radiologist_type5_3_At_Risk, \
Test_Lesion_Label_Radiologist_type6_2_At_Risk, Test_Lesion_Label_Radiologist_type6_3_At_Risk, \
Test_Lesion_Label_Radiologist_type7_2_At_Risk, Test_Lesion_Label_Radiologist_type7_3_At_Risk = \
    [], [], [], [], [], [], [], []

for i in range(len(Train_ID_Lesion_Label_GT_At_Risk)):
    Train_Lesion_Label_GT_At_Risk.append(int(Train_ID_Lesion_Label_GT_At_Risk[i][1]))
    Train_Lesion_Label_Radiologist_At_Risk.append(int(Train_ID_Lesion_Label_Radiologist_At_Risk[i][1]))
    Train_Lesion_Label_Radiologist_type2_At_Risk.append(int(Train_ID_Lesion_Label_Radiologist_type2_At_Risk[i][1]))
    Train_Lesion_Label_Radiologist_type3_At_Risk.append(int(Train_ID_Lesion_Label_Radiologist_type3_At_Risk[i][1]))
    Train_Lesion_Label_Radiologist_type4_At_Risk.append(int(Train_ID_Lesion_Label_Radiologist_type4_At_Risk[i][1]))
    Train_Lesion_Label_Radiologist_type5_At_Risk.append(int(Train_ID_Lesion_Label_Radiologist_type5_At_Risk[i][1]))
    Train_Lesion_Label_Radiologist_type6_At_Risk.append(int(Train_ID_Lesion_Label_Radiologist_type6_At_Risk[i][1]))
    Train_Lesion_Label_Radiologist_type7_At_Risk.append(int(Train_ID_Lesion_Label_Radiologist_type7_At_Risk[i][1]))
    Train_Lesion_Label_Radiologist_type8_At_Risk.append(int(Train_ID_Lesion_Label_Radiologist_type8_At_Risk[i][1]))

for i in range(len(Test_ID_Lesion_Label_GT_At_Risk)):
    Test_Lesion_Label_GT_At_Risk.append(int(Test_ID_Lesion_Label_GT_At_Risk[i][1]))
    Test_Lesion_Label_Radiologist_At_Risk.append(int(Test_ID_Lesion_Label_Radiologist_At_Risk[i][1]))
    Test_Lesion_Label_Radiologist_type2_At_Risk.append(int(Test_ID_Lesion_Label_Radiologist_type2_At_Risk[i][1]))
    Test_Lesion_Label_Radiologist_type3_At_Risk.append(int(Test_ID_Lesion_Label_Radiologist_type3_At_Risk[i][1]))
    Test_Lesion_Label_Radiologist_type4_At_Risk.append(int(Test_ID_Lesion_Label_Radiologist_type4_At_Risk[i][1]))
    Test_Lesion_Label_Radiologist_type5_At_Risk.append(int(Test_ID_Lesion_Label_Radiologist_type5_At_Risk[i][1]))
    Test_Lesion_Label_Radiologist_type6_At_Risk.append(int(Test_ID_Lesion_Label_Radiologist_type6_At_Risk[i][1]))
    Test_Lesion_Label_Radiologist_type7_At_Risk.append(int(Test_ID_Lesion_Label_Radiologist_type7_At_Risk[i][1]))
    Test_Lesion_Label_Radiologist_type8_At_Risk.append(int(Test_ID_Lesion_Label_Radiologist_type8_At_Risk[i][1]))

    Test_Lesion_Label_Radiologist_type4_2_At_Risk.append(int(Test_ID_Lesion_Label_Radiologist_type4_2_At_Risk[i][1]))
    Test_Lesion_Label_Radiologist_type4_3_At_Risk.append(int(Test_ID_Lesion_Label_Radiologist_type4_3_At_Risk[i][1]))
    Test_Lesion_Label_Radiologist_type5_2_At_Risk.append(int(Test_ID_Lesion_Label_Radiologist_type5_2_At_Risk[i][1]))
    Test_Lesion_Label_Radiologist_type5_3_At_Risk.append(int(Test_ID_Lesion_Label_Radiologist_type5_3_At_Risk[i][1]))
    Test_Lesion_Label_Radiologist_type6_2_At_Risk.append(int(Test_ID_Lesion_Label_Radiologist_type6_2_At_Risk[i][1]))
    Test_Lesion_Label_Radiologist_type6_3_At_Risk.append(int(Test_ID_Lesion_Label_Radiologist_type6_3_At_Risk[i][1]))
    Test_Lesion_Label_Radiologist_type7_2_At_Risk.append(int(Test_ID_Lesion_Label_Radiologist_type7_2_At_Risk[i][1]))
    Test_Lesion_Label_Radiologist_type7_3_At_Risk.append(int(Test_ID_Lesion_Label_Radiologist_type7_3_At_Risk[i][1]))

print('*=*'*60)
HCC_Patient_At_Risk_Train, NonHCC_Patient_At_Risk_Train = 0, 0
HCC_Patient_At_Risk_Test, NonHCC_Patient_At_Risk_Test = 0, 0
for case in Train_ID_Label_At_Risk:
    if case[1] == 1:
        HCC_Patient_At_Risk_Train += 1
    else:
        NonHCC_Patient_At_Risk_Train += 1
for case in Test_ID_Label_At_Risk:
    if case[1] == 1:
        HCC_Patient_At_Risk_Test += 1
    else:
        NonHCC_Patient_At_Risk_Test += 1
print("The numbers of HCC and Non-HCC cases in the training set are %d %d" % (
    HCC_Patient_At_Risk_Train, NonHCC_Patient_At_Risk_Train),
      "The number of cases in the training set is %d" % (len(Train_ID_Label_At_Risk)))
print("The numbers of HCC and Non-HCC cases in the testing set are %d %d" % (
    HCC_Patient_At_Risk_Test, NonHCC_Patient_At_Risk_Test),
      "The number of cases in the testing set is %d" % (len(Test_ID_Label_At_Risk)))
print(len(Train_ID_Label_At_Risk), len(Test_ID_Label_At_Risk))
print('*=*'*60)
HCC_Lesion_At_Risk_Train, NonHCC_Lesion_At_Risk_Train = 0, 0
HCC_Lesion_At_Risk_Test, NonHCC_Lesion_At_Risk_Test = 0, 0
for case in Train_ID_Lesion_Label_GT_At_Risk:
    if case[1] == 1:
        HCC_Lesion_At_Risk_Train += 1
    else:
        NonHCC_Lesion_At_Risk_Train += 1
for case in Test_ID_Lesion_Label_GT_At_Risk:
    if case[1] == 1:
        HCC_Lesion_At_Risk_Test += 1
    else:
        NonHCC_Lesion_At_Risk_Test += 1

print("The numbers of HCC and Non-HCC lesions in the training set are %d %d" % (
HCC_Lesion_At_Risk_Train, NonHCC_Lesion_At_Risk_Train),
      "The number of lesions in the training set is %d" % (len(Train_ID_Lesion_Label_GT_At_Risk)))
print("The numbers of HCC and Non-HCC lesions in the training set are %d %d" % (
HCC_Lesion_At_Risk_Test, NonHCC_Lesion_At_Risk_Test),
      "The number of lesions in the testing set is %d" % (len(Test_ID_Lesion_Label_GT_At_Risk)))
print(len(Train_ID_Lesion_Label_GT_At_Risk), len(Test_ID_Lesion_Label_GT_At_Risk))

"""
print('the ground-truth label for lesion at the at-risk sub-group analysis: ')
print(Test_Lesion_Label_GT_At_Risk)
print(len(Test_Lesion_Label_GT_At_Risk))
print('*<>'*40)
print(Test_Lesion_Label_Radiologist_type3_At_Risk)
print('<#>'*40)
print(Test_Lesion_Label_Radiologist_type8_At_Risk)
"""

print('The confusion matrix and auc of the testing set at the lesion level ======== type 1 ======== -  line-1881: ')
confusion_test_lesion_level = confusion_matrix(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_At_Risk)
print(confusion_test_lesion_level)
fpr_lesion_type1, tpr_lesion_type1, _ = roc_curve(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_At_Risk)
auc_lesion_type1 = auc(fpr_lesion_type1, tpr_lesion_type1)
print(fpr_lesion_type1, tpr_lesion_type1, auc_lesion_type1)

print('The confusion matrix and auc of the testing set at the lesion level ======== type 2 ======== -  line-1888: ')
confusion_test_lesion_level = confusion_matrix(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type2_At_Risk)
print(confusion_test_lesion_level)
fpr_lesion_type2, tpr_lesion_type2, _ = roc_curve(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type2_At_Risk)
auc_lesion_type2 = auc(fpr_lesion_type2, tpr_lesion_type2)
print(fpr_lesion_type2, tpr_lesion_type2, auc_lesion_type2)

print('The confusion matrix and auc of the testing set at the lesion level ======== type 3 ======== -   line-1895: ')
confusion_test_lesion_level = confusion_matrix(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type3_At_Risk)
print(confusion_test_lesion_level)
fpr_lesion_type3, tpr_lesion_type3, _ = roc_curve(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type3_At_Risk)
auc_lesion_type3 = auc(fpr_lesion_type3, tpr_lesion_type3)
print(fpr_lesion_type3, tpr_lesion_type3, auc_lesion_type3)

print('The confusion matrix and auc of the testing set at the lesion level ======== type 4 ======== -   line-1902: ')
confusion_test_lesion_level = confusion_matrix(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type4_At_Risk)
print(confusion_test_lesion_level)
fpr_group1, tpr_group1, _ = roc_curve(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type4_At_Risk)
auc_group1 = auc(fpr_group1, tpr_group1)
print(auc_group1)
print('The confusion matrix and auc of the testing set at the lesion level ======== type 4 -- 2 ======== - line-2636: ')
confusion_test_lesion_level = confusion_matrix(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type4_2_At_Risk)
print(confusion_test_lesion_level)
fpr_lesion_type_4_2, tpr_lesion_type_4_2, _ = roc_curve(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type4_2_At_Risk)
auc_lesion_type_4_2 = auc(fpr_lesion_type_4_2, tpr_lesion_type_4_2)
print(fpr_lesion_type_4_2, tpr_lesion_type_4_2, auc_lesion_type_4_2)
print('The confusion matrix and auc of the testing set at the lesion level ======== type 4 -- 3 ======== - line-2642: ')
confusion_test_lesion_level = confusion_matrix(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type4_3_At_Risk)
print(confusion_test_lesion_level)
fpr_lesion_type_4_3, tpr_lesion_type_4_3, _ = roc_curve(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type4_3_At_Risk)
auc_lesion_type_4_3 = auc(fpr_lesion_type_4_3, tpr_lesion_type_4_3)
print(fpr_lesion_type_4_3, tpr_lesion_type_4_3, auc_lesion_type_4_3)

print('The confusion matrix and auc of the testing set at the lesion level ========= type 5 ======== -   line-1909: ')
confusion_test_lesion_level = confusion_matrix(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type5_At_Risk)
print(confusion_test_lesion_level)
fpr_group1, tpr_group1, _ = roc_curve(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type5_At_Risk)
auc_group1 = auc(fpr_group1, tpr_group1)
print(auc_group1)
print('The confusion matrix and auc of the testing set at the lesion level ======= type 5 -- 2 ======== - line-2655: ')
confusion_test_lesion_level = confusion_matrix(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type5_2_At_Risk)
print(confusion_test_lesion_level)
fpr_lesion_type_5_2, tpr_lesion_type_5_2, _ = roc_curve(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type5_2_At_Risk)
auc_lesion_type_5_2 = auc(fpr_lesion_type_5_2, tpr_lesion_type_5_2)
print(fpr_lesion_type_5_2, tpr_lesion_type_5_2, auc_lesion_type_5_2)
print('The confusion matrix and auc of the testing set at the lesion level ======= type 5 -- 3 ======== - line-2661: ')
confusion_test_lesion_level = confusion_matrix(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type5_3_At_Risk)
print(confusion_test_lesion_level)
fpr_lesion_type_5_3, tpr_lesion_type_5_3, _ = roc_curve(Test_Lesion_Label_GT_At_Risk,
                                                        Test_Lesion_Label_Radiologist_type5_3_At_Risk)
auc_lesion_type_5_3 = auc(fpr_lesion_type_5_3, tpr_lesion_type_5_3)
print(fpr_lesion_type_5_3, tpr_lesion_type_5_3, auc_lesion_type_5_3)

print('The confusion matrix and auc of the testing set at the lesion level ====== type 6 ======== -   line-1916: ')
confusion_test_lesion_level = confusion_matrix(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type6_At_Risk)
print(confusion_test_lesion_level)
fpr_group1, tpr_group1, _ = roc_curve(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type6_At_Risk)
auc_group1 = auc(fpr_group1, tpr_group1)
print(auc_group1)
print('The confusion matrix and auc of the testing set at the lesion level ====== type 6 -- 2 ======== - line-2674: ')
confusion_test_lesion_level = confusion_matrix(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type6_2_At_Risk)
print(confusion_test_lesion_level)
fpr_lesion_type_6_2, tpr_lesion_type_6_2, _ = roc_curve(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type6_2_At_Risk)
auc_lesion_type_6_2 = auc(fpr_lesion_type_6_2, tpr_lesion_type_6_2)
print(fpr_lesion_type_6_2, tpr_lesion_type_6_2, auc_lesion_type_6_2)
print('The confusion matrix and auc of the testing set at the lesion level ====== type 6 -- 3 ======== - line-2680: ')
confusion_test_lesion_level = confusion_matrix(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type6_3_At_Risk)
print(confusion_test_lesion_level)
fpr_lesion_type_6_3, tpr_lesion_type_6_3, _ = roc_curve(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type6_3_At_Risk)
auc_lesion_type_6_3 = auc(fpr_lesion_type_6_3, tpr_lesion_type_6_3)
print(fpr_lesion_type_6_3, tpr_lesion_type_6_3, auc_lesion_type_6_3)

print('The confusion matrix and auc of the testing set at the lesion level ======== type 7 ======== -   line-1923: ')
confusion_test_lesion_level = confusion_matrix(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type7_At_Risk)
print(confusion_test_lesion_level)
fpr_group1, tpr_group1, _ = roc_curve(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type7_At_Risk)
auc_group1 = auc(fpr_group1, tpr_group1)
print(auc_group1)
print('The confusion matrix and auc of the testing set at the lesion level ======== type 7 -- 2 ======== - line-2693: ')
confusion_test_lesion_level = confusion_matrix(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type7_2_At_Risk)
print(confusion_test_lesion_level)
fpr_lesion_type_7_2, tpr_lesion_type_7_2, _ = roc_curve(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type7_2_At_Risk)
auc_lesion_type_7_2 = auc(fpr_lesion_type_7_2, tpr_lesion_type_7_2)
print(fpr_lesion_type_7_2, tpr_lesion_type_7_2, auc_lesion_type_7_2)
print('The confusion matrix and auc of the testing set at the lesion level ======== type 7 --3 ======== - line-2699: ')
confusion_test_lesion_level = confusion_matrix(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type7_3_At_Risk)
print(confusion_test_lesion_level)
fpr_lesion_type_7_3, tpr_lesion_type_7_3, _ = roc_curve(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type7_3_At_Risk)
auc_lesion_type_7_3 = auc(fpr_lesion_type_7_3, tpr_lesion_type_7_3)
print(fpr_lesion_type_7_3, tpr_lesion_type_7_3, auc_lesion_type_7_3)

print('The confusion matrix and auc of the testing set at the lesion level ========= type 8 ======== -   line-1930: ')
confusion_test_lesion_level = confusion_matrix(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type8_At_Risk)
print(confusion_test_lesion_level)
fpr_lesion_type_8, tpr_lesion_type_8, _ = roc_curve(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_Radiologist_type8_At_Risk)
auc_lesion_type_8 = auc(fpr_lesion_type_8, tpr_lesion_type_8)
print(fpr_lesion_type_8, tpr_lesion_type_8, auc_lesion_type_8)

Train_Patient_Label_GT_At_Risk, Train_Patient_Label_GT_Excel_At_Risk, Train_Patient_Label_Radiologist_At_Risk = [], [], []
Train_Patient_Label_Radiologist_type2_At_Risk, Train_Patient_Label_Radiologist_type3_At_Risk, \
Train_Patient_Label_Radiologist_type4_At_Risk, Train_Patient_Label_Radiologist_type5_At_Risk, \
Train_Patient_Label_Radiologist_type6_At_Risk, Train_Patient_Label_Radiologist_type7_At_Risk, \
Train_Patient_Label_Radiologist_type8_At_Risk = [], [], [], [], [], [], []

Test_Patient_Label_GT_At_Risk, Test_Patient_Label_GT_Excel_At_Risk, Test_Patient_Label_Radiologist_At_Risk = [], [], []
Test_Patient_Label_Radiologist_type2_At_Risk, Test_Patient_Label_Radiologist_type3_At_Risk, \
Test_Patient_Label_Radiologist_type4_At_Risk, Test_Patient_Label_Radiologist_type5_At_Risk, \
Test_Patient_Label_Radiologist_type6_At_Risk, Test_Patient_Label_Radiologist_type7_At_Risk, \
Test_Patient_Label_Radiologist_type8_At_Risk = [], [], [], [], [], [], []
Test_Patient_Label_Radiologist_type4_2_At_Risk, Test_Patient_Label_Radiologist_type4_3_At_Risk, \
Test_Patient_Label_Radiologist_type5_2_At_Risk, Test_Patient_Label_Radiologist_type5_3_At_Risk, \
Test_Patient_Label_Radiologist_type6_2_At_Risk, Test_Patient_Label_Radiologist_type6_3_At_Risk, \
Test_Patient_Label_Radiologist_type7_2_At_Risk, Test_Patient_Label_Radiologist_type7_3_At_Risk \
    = [], [], [], [], [], [], [], []

Train_Patient_ID_At_Risk_for_Check = []
for i in range(len(Train_ID_Patient_Label_GT_From_Excel_At_Risk)):
    Train_Patient_ID_At_Risk_for_Check.append(Train_ID_Patient_Label_GT_From_Excel_At_Risk[i][0])
    Train_Patient_Label_GT_Excel_At_Risk.append(Train_ID_Patient_Label_GT_From_Excel_At_Risk[i][1])
    Train_Patient_Label_Radiologist_At_Risk.append(Train_ID_Patient_Label_Radiologist_At_Risk[i][1])
    Train_Patient_Label_Radiologist_type2_At_Risk.append(Train_ID_Patient_Label_Radiologist_type2_At_Risk[i][1])
    Train_Patient_Label_Radiologist_type3_At_Risk.append(Train_ID_Patient_Label_Radiologist_type3_At_Risk[i][1])
    Train_Patient_Label_Radiologist_type4_At_Risk.append(Train_ID_Patient_Label_Radiologist_type4_At_Risk[i][1])
    Train_Patient_Label_Radiologist_type5_At_Risk.append(Train_ID_Patient_Label_Radiologist_type5_At_Risk[i][1])
    Train_Patient_Label_Radiologist_type6_At_Risk.append(Train_ID_Patient_Label_Radiologist_type6_At_Risk[i][1])
    Train_Patient_Label_Radiologist_type7_At_Risk.append(Train_ID_Patient_Label_Radiologist_type7_At_Risk[i][1])
    Train_Patient_Label_Radiologist_type8_At_Risk.append(Train_ID_Patient_Label_Radiologist_type8_At_Risk[i][1])
    patient_id = Train_ID_Patient_Label_GT_From_Excel_At_Risk[i][0]
    for idx in range(len(Train_ID_Patient_Label_GT)):
        check_id = Train_ID_Patient_Label_GT[idx][0]
        if check_id == patient_id:
            Train_Patient_Label_GT_At_Risk.append(Train_ID_Patient_Label_GT[idx][1])
print(len(Train_Patient_Label_GT_At_Risk), len(Train_Patient_Label_GT_Excel_At_Risk),
      len(Train_Patient_Label_Radiologist_At_Risk), "Line-610")

Test_Patient_ID_Label_Order = []
Test_Patient_ID_At_Risk_for_Check = []
for i in range(len(Test_ID_Patient_Label_GT_From_Excel_At_Risk)):
    Test_Patient_ID_At_Risk_for_Check.append(Test_ID_Patient_Label_GT_From_Excel_At_Risk[i][0])
    Test_Patient_Label_GT_Excel_At_Risk.append(Test_ID_Patient_Label_GT_From_Excel_At_Risk[i][1])
    Test_Patient_Label_Radiologist_At_Risk.append(Test_ID_Patient_Label_Radiologist_At_Risk[i][1])
    Test_Patient_Label_Radiologist_type2_At_Risk.append(Test_ID_Patient_Label_Radiologist_type2_At_Risk[i][1])
    Test_Patient_Label_Radiologist_type3_At_Risk.append(Test_ID_Patient_Label_Radiologist_type3_At_Risk[i][1])
    Test_Patient_Label_Radiologist_type4_At_Risk.append(Test_ID_Patient_Label_Radiologist_type4_At_Risk[i][1])
    Test_Patient_Label_Radiologist_type5_At_Risk.append(Test_ID_Patient_Label_Radiologist_type5_At_Risk[i][1])
    Test_Patient_Label_Radiologist_type6_At_Risk.append(Test_ID_Patient_Label_Radiologist_type6_At_Risk[i][1])
    Test_Patient_Label_Radiologist_type7_At_Risk.append(Test_ID_Patient_Label_Radiologist_type7_At_Risk[i][1])
    Test_Patient_Label_Radiologist_type8_At_Risk.append(Test_ID_Patient_Label_Radiologist_type8_At_Risk[i][1])

    Test_Patient_Label_Radiologist_type4_2_At_Risk.append(Test_ID_Patient_Label_Radiologist_type4_2_At_Risk[i][1])
    Test_Patient_Label_Radiologist_type4_3_At_Risk.append(Test_ID_Patient_Label_Radiologist_type4_3_At_Risk[i][1])
    Test_Patient_Label_Radiologist_type5_2_At_Risk.append(Test_ID_Patient_Label_Radiologist_type5_2_At_Risk[i][1])
    Test_Patient_Label_Radiologist_type5_3_At_Risk.append(Test_ID_Patient_Label_Radiologist_type5_3_At_Risk[i][1])
    Test_Patient_Label_Radiologist_type6_2_At_Risk.append(Test_ID_Patient_Label_Radiologist_type6_2_At_Risk[i][1])
    Test_Patient_Label_Radiologist_type6_3_At_Risk.append(Test_ID_Patient_Label_Radiologist_type6_3_At_Risk[i][1])
    Test_Patient_Label_Radiologist_type7_2_At_Risk.append(Test_ID_Patient_Label_Radiologist_type7_2_At_Risk[i][1])
    Test_Patient_Label_Radiologist_type7_3_At_Risk.append(Test_ID_Patient_Label_Radiologist_type7_3_At_Risk[i][1])

    patient_id = Test_ID_Patient_Label_GT_From_Excel_At_Risk[i][0]
    for idx in range(len(Test_ID_Patient_Label_GT)):
        check_id = Test_ID_Patient_Label_GT[idx][0]
        if check_id == patient_id:
            Test_Patient_Label_GT_At_Risk.append(int(Test_ID_Patient_Label_GT[idx][1]))
            # print(int(Test_ID_Patient_Label_GT[idx][1][0]), Test_ID_Patient_Label_GT_From_Excel[i], Test_ID_Patient_Label_Radiologist[i])
            Test_Patient_ID_Label_Order.append([check_id, int(Test_ID_Patient_Label_GT[idx][1])])

print(len(Train_Patient_ID_At_Risk_for_Check))
print(len(Test_Patient_ID_At_Risk_for_Check))

file_at_risk_train = open('/home/ra1/Desktop/Patient_ID_for_At_Risk_Check_Train.txt', 'w')
file_at_risk_train.write("The patient id list of sub-group at risk for training set is as below:" + '\n')
for i in range(len(Train_Patient_ID_At_Risk_for_Check)):
    file_at_risk_train.write(Train_Patient_ID_At_Risk_for_Check[i] + '\n')
file_at_risk_train.close()

file_at_risk = open('/home/ra1/Desktop/Patient_ID_for_At_Risk_Check.txt', "w")
file_at_risk.write("The patient id list of sub-group at risk is as below:" + '\n')
for i in range(len(Test_Patient_ID_At_Risk_for_Check)):
    file_at_risk.write(Test_Patient_ID_At_Risk_for_Check[i] + '\n')
file_at_risk.close()

print(len(Test_Patient_Label_GT_At_Risk), len(Test_Patient_Label_GT_Excel_At_Risk), len(Test_Patient_Label_Radiologist_At_Risk), "Line-621")
print('*=*'*40)
print(Test_Patient_Label_GT_Excel_At_Risk)
print('*+*'*40)
print(Test_Patient_Label_Radiologist_type3_At_Risk)
print('*#*'*40)
print(Test_Patient_Label_Radiologist_type8_At_Risk)

print('The confusion matrix and auc of testing set at the patient level ======== type 1 ======  - line-2005: ')
confusion_test_patient_level_excel = confusion_matrix(Test_Patient_Label_GT_Excel_At_Risk,
                                                      Test_Patient_Label_Radiologist_At_Risk)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_patient_type1, tpr_patient_type1, _ = roc_curve(Test_Patient_Label_GT_Excel_At_Risk,
                                                    Test_Patient_Label_Radiologist_At_Risk)
auc_patient_type1 = auc(fpr_patient_type1, tpr_patient_type1)
print(fpr_patient_type1, tpr_patient_type1, auc_patient_type1)


print('The confusion matrix and auc of testing set at the patient level ======== type 2 ======  - line-2012: ')
confusion_test_patient_level_excel = confusion_matrix(Test_Patient_Label_GT_Excel_At_Risk,
                                                      Test_Patient_Label_Radiologist_type2_At_Risk)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_patient_type2, tpr_patient_type2, _ = roc_curve(Test_Patient_Label_GT_Excel_At_Risk,
                                                    Test_Patient_Label_Radiologist_type2_At_Risk)
auc_patient_type2 = auc(fpr_patient_type2, tpr_patient_type2)
print(fpr_patient_type2, tpr_patient_type2, auc_patient_type2)

print('The confusion matrix and auc of testing set at the patient level ======== type 3 ======  - line-2019: ')
confusion_test_patient_level_excel = confusion_matrix(Test_Patient_Label_GT_Excel_At_Risk,
                                                      Test_Patient_Label_Radiologist_type3_At_Risk)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_patient_type3, tpr_patient_type3, _ = roc_curve(Test_Patient_Label_GT_Excel_At_Risk,
                                                    Test_Patient_Label_Radiologist_type3_At_Risk)
auc_patient_type3 = auc(fpr_patient_type3, tpr_patient_type3)
print(fpr_patient_type3, tpr_patient_type3, auc_patient_type3)

print('The confusion matrix and auc of testing set at the patient level ======== type 4 ======  - line-2026: ')
confusion_test_patient_level_excel = confusion_matrix(Test_Patient_Label_GT_Excel_At_Risk,
                                                      Test_Patient_Label_Radiologist_type4_At_Risk)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_group1, tpr_group1, _ = roc_curve(Test_Patient_Label_GT_Excel_At_Risk, Test_Patient_Label_Radiologist_type4_At_Risk)
auc_group1 = auc(fpr_group1, tpr_group1)
print(auc_group1)
print('The confusion matrix and auc of testing set at the patient level ======== type 4 -- 2 ====== - line-2822: ')
confusion_test_patient_level_excel = confusion_matrix(Test_Patient_Label_GT_Excel_At_Risk,
                                                      Test_Patient_Label_Radiologist_type4_2_At_Risk)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_patient_type4_2, tpr_patient_type4_2, _ = roc_curve(Test_Patient_Label_GT_Excel_At_Risk,
                                                        Test_Patient_Label_Radiologist_type4_2_At_Risk)
auc_patient_type4_2 = auc(fpr_patient_type4_2, tpr_patient_type4_2)
print(fpr_patient_type4_2, tpr_patient_type4_2, auc_patient_type4_2)

print('The confusion matrix and auc of testing set at the patient level ======== type 4 -- 3 ====== - line-2828: ')
confusion_test_patient_level_excel = confusion_matrix(Test_Patient_Label_GT_Excel_At_Risk,
                                                      Test_Patient_Label_Radiologist_type4_3_At_Risk)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_patient_type4_3, tpr_patient_type4_3, _ = roc_curve(Test_Patient_Label_GT_Excel_At_Risk,
                                                Test_Patient_Label_Radiologist_type4_3_At_Risk)
auc_patient_type4_3 = auc(fpr_patient_type4_3, tpr_patient_type4_3)
print(fpr_patient_type4_3, tpr_patient_type4_3, auc_patient_type4_3)

print('The confusion matrix and auc of testing set at the patient level ======== type 5 ======  - line-2033: ')
confusion_test_patient_level_excel = confusion_matrix(Test_Patient_Label_GT_Excel_At_Risk,
                                                      Test_Patient_Label_Radiologist_type5_At_Risk)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_group1, tpr_group1, _ = roc_curve(Test_Patient_Label_GT_Excel_At_Risk, Test_Patient_Label_Radiologist_type5_At_Risk)
auc_group1 = auc(fpr_group1, tpr_group1)
print(auc_group1)
print('The confusion matrix and auc of testing set at the patient level ======== type 5 -- 2 ======  - line-2841: ')
confusion_test_patient_level_excel = confusion_matrix(Test_Patient_Label_GT_Excel_At_Risk,
                                                      Test_Patient_Label_Radiologist_type5_2_At_Risk)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_patient_type5_2, tpr_patient_type5_2, _ = roc_curve(Test_Patient_Label_GT_Excel_At_Risk,
                                                        Test_Patient_Label_Radiologist_type5_2_At_Risk)
auc_patient_type5_2 = auc(fpr_patient_type5_2, tpr_patient_type5_2)
print(fpr_patient_type5_2, tpr_patient_type5_2, auc_patient_type5_2)

print('The confusion matrix and auc of testing set at the patient level ======== type 5 -- 3 ======  - line-2847: ')
confusion_test_patient_level_excel = confusion_matrix(Test_Patient_Label_GT_Excel_At_Risk,
                                                      Test_Patient_Label_Radiologist_type5_3_At_Risk)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_patient_type5_3, tpr_patient_type5_3, _ = roc_curve(Test_Patient_Label_GT_Excel_At_Risk,
                                                        Test_Patient_Label_Radiologist_type5_3_At_Risk)
auc_patient_type5_3 = auc(fpr_patient_type5_3, tpr_patient_type5_3)
print(fpr_patient_type5_3, tpr_patient_type5_3, auc_patient_type5_3)

print('The confusion matrix and auc of testing set at the patient level ======== type 6 ======  - line-2040: ')
confusion_test_patient_level_excel = confusion_matrix(Test_Patient_Label_GT_Excel_At_Risk,
                                                      Test_Patient_Label_Radiologist_type6_At_Risk)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_group1, tpr_group1, _ = roc_curve(Test_Patient_Label_GT_Excel_At_Risk,
                                      Test_Patient_Label_Radiologist_type6_At_Risk)
auc_group1 = auc(fpr_group1, tpr_group1)
print(auc_group1)
print('The confusion matrix and auc of testing set at the patient level ======== type 6 -- 2 ======  - line-2860: ')
confusion_test_patient_level_excel = confusion_matrix(Test_Patient_Label_GT_Excel_At_Risk,
                                                      Test_Patient_Label_Radiologist_type6_2_At_Risk)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_patient_type6_2, tpr_patient_type6_2, _ = roc_curve(Test_Patient_Label_GT_Excel_At_Risk,
                                      Test_Patient_Label_Radiologist_type6_2_At_Risk)
auc_patient_type6_2 = auc(fpr_patient_type6_2, tpr_patient_type6_2)
print(fpr_patient_type6_2, tpr_patient_type6_2, auc_patient_type6_2)

print('The confusion matrix and auc of testing set at the patient level ======== type 6 -- 3 ======  - line-2866: ')
confusion_test_patient_level_excel = confusion_matrix(Test_Patient_Label_GT_Excel_At_Risk,
                                                      Test_Patient_Label_Radiologist_type6_3_At_Risk)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_patient_type6_3, tpr_patient_type6_3, _ = roc_curve(Test_Patient_Label_GT_Excel_At_Risk,
                                      Test_Patient_Label_Radiologist_type6_3_At_Risk)
auc_patient_type6_3 = auc(fpr_patient_type6_3, tpr_patient_type6_3)
print(fpr_patient_type6_3, tpr_patient_type6_3, auc_patient_type6_3)

print('The confusion matrix and auc of testing set at the patient level ======== type 7 ======  - line-2047: ')
confusion_test_patient_level_excel = confusion_matrix(Test_Patient_Label_GT_Excel_At_Risk,
                                                      Test_Patient_Label_Radiologist_type7_At_Risk)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_group1, tpr_group1, _ = roc_curve(Test_Patient_Label_GT_Excel_At_Risk,
                                      Test_Patient_Label_Radiologist_type7_At_Risk)
auc_group1 = auc(fpr_group1, tpr_group1)
print(auc_group1)
print('The confusion matrix and auc of testing set at the patient level ======== type 7 -- 2======  - line-2879: ')
confusion_test_patient_level_excel = confusion_matrix(Test_Patient_Label_GT_Excel_At_Risk,
                                                      Test_Patient_Label_Radiologist_type7_2_At_Risk)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_patient_type7_2, tpr_patient_type7_2, _ = roc_curve(Test_Patient_Label_GT_Excel_At_Risk,
                                      Test_Patient_Label_Radiologist_type7_2_At_Risk)
auc_patient_type7_2 = auc(fpr_patient_type7_2, tpr_patient_type7_2)
print(fpr_patient_type7_2, tpr_patient_type7_2, auc_patient_type7_2)
print('The confusion matrix and auc of testing set at the patient level ======== type 7 -- 3 ======  - line-2885: ')
confusion_test_patient_level_excel = confusion_matrix(Test_Patient_Label_GT_Excel_At_Risk,
                                                      Test_Patient_Label_Radiologist_type7_3_At_Risk)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_patient_type7_3, tpr_patient_type7_3, _ = roc_curve(Test_Patient_Label_GT_Excel_At_Risk,
                                                        Test_Patient_Label_Radiologist_type7_3_At_Risk)
auc_patient_type7_3 = auc(fpr_patient_type7_3, tpr_patient_type7_3)
print(fpr_patient_type7_3, tpr_patient_type7_3, auc_patient_type7_3)

print('The confusion matrix and auc of testing set at the patient level ======== type 8 ======  - line-2054: ')
confusion_test_patient_level_excel = confusion_matrix(Test_Patient_Label_GT_Excel_At_Risk, Test_Patient_Label_Radiologist_type8_At_Risk)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_patient_type8, tpr_patient_type8, _ = roc_curve(Test_Patient_Label_GT_Excel_At_Risk,
                                                    Test_Patient_Label_Radiologist_type8_At_Risk)
auc_patient_type8 = auc(fpr_patient_type8, tpr_patient_type8)
print(fpr_patient_type8, tpr_patient_type8, auc_patient_type8)



# For our proposed model
print(len(Test_ID_Lesion_Label_GT_At_Risk), len(Test_ID_Patient_Label_GT_From_Excel_At_Risk))
Patient_ID_List_Predict_InCorrect = ['HKU_0427_P3', 'ID_0160_P3', 'ID_0417_P3', 'ID_0516_P3 ', 'ID_0856b_P3 ',
                                     'QMH_0001_P3', 'QMH_0139_P3', 'SZH_0075_P3', 'SZH_0362_P3', 'SZH_0404_P3',
                                     'HKU_0065_P3', 'HKU_0069_P3', 'HKU_0389_P3', 'ID_0707_P3', 'QMH_0143_P3',
                                     'QMH_0074_P3', 'ID_0254_P3', 'SZH_0390_P3', 'ID_0761_P3', 'ID_0444_P3',
                                     'QMH_0126_P3', 'ID_0106_P3']

CNT_Patient_Level_At_Risk, CNT_Lesion_Level_At_Risk = 0, 0
Test_ID_Patient_Label_At_Risk_Model = []
Test_ID_Lesion_Label_At_Risk_Model = []
for i in range(len(Test_ID_Patient_Label_GT_From_Excel_At_Risk)):
    case_id = Test_ID_Patient_Label_GT_From_Excel_At_Risk[i][0]
    if case_id in Patient_ID_List_Predict_InCorrect:
        CNT_Patient_Level_At_Risk += 1
        case_label = Test_ID_Patient_Label_GT_From_Excel_At_Risk[i][1]
        if case_label == 0:
            case_label_pre = 1
        else:
            case_label_pre = 0
        Test_ID_Patient_Label_At_Risk_Model.append([case_id, case_label_pre])
    else:
        Test_ID_Patient_Label_At_Risk_Model.append([case_id, Test_ID_Patient_Label_GT_From_Excel_At_Risk[i][1]])

for i in range(len(Test_ID_Lesion_Label_GT_At_Risk)):
    case_id = Test_ID_Lesion_Label_GT_At_Risk[i][0]
    if case_id in Patient_ID_List_Predict_InCorrect:
        CNT_Lesion_Level_At_Risk += 1
        case_label = Test_ID_Lesion_Label_GT_At_Risk[i][1]
        if case_label == 1:
            case_label_pre = 0
        else:
            case_label_pre = 1
        Test_ID_Lesion_Label_At_Risk_Model.append([case_id, case_label_pre])
    else:
        Test_ID_Lesion_Label_At_Risk_Model.append([case_id, Test_ID_Lesion_Label_GT_At_Risk[i][1]])

print(CNT_Patient_Level_At_Risk, CNT_Lesion_Level_At_Risk)
print(len(Test_ID_Label_At_Risk), len(Test_ID_Patient_Label_At_Risk_Model))
print(len(Test_ID_Lesion_Label_GT_At_Risk), len(Test_ID_Lesion_Label_At_Risk_Model))
Test_Patient_Label_At_Risk_Model, Test_Lesion_Label_At_Risk_Model = [], []

for i in Test_ID_Patient_Label_At_Risk_Model:
    label_pre = i[1]
    Test_Patient_Label_At_Risk_Model.append(label_pre)

for i in Test_ID_Lesion_Label_At_Risk_Model:
    label_pre = i[1]
    Test_Lesion_Label_At_Risk_Model.append(label_pre)

print(len(Test_Patient_Label_GT_Excel_At_Risk))
print(len(Test_Lesion_Label_GT_At_Risk))

print('The confusion matrix and auc of testing set at the patient level ======== Model ======  - line-3198: ')
confusion_test_patient_level_excel = confusion_matrix(Test_Patient_Label_GT_Excel_At_Risk, Test_Patient_Label_At_Risk_Model)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_group1, tpr_group1, _ = roc_curve(Test_Patient_Label_GT_Excel_At_Risk, Test_Patient_Label_At_Risk_Model)
auc_group1 = auc(fpr_group1, tpr_group1)
print(auc_group1)

print('The confusion matrix and auc of testing set at the lesion level ======== Model ======  - line-3205: ')
confusion_test_patient_level_excel = confusion_matrix(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_At_Risk_Model)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_group1, tpr_group1, _ = roc_curve(Test_Lesion_Label_GT_At_Risk, Test_Lesion_Label_At_Risk_Model)
auc_group1 = auc(fpr_group1, tpr_group1)
print(auc_group1)

print('Test_Patient_Label_At-Risk_Model:')
print(Test_Patient_Label_At_Risk_Model)
print('>#*<'*40)
print(Test_Lesion_Label_At_Risk_Model)
print('<##>'*40)
print(len(Test_Patient_Label_At_Risk_Model), len(Test_Lesion_Label_At_Risk_Model))