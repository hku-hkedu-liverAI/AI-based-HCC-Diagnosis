import numpy as np
import pandas as pd
import os
from utils_library_finalized_preprocess import read_dicom, read_mask, obtain_hcc_nonhcc, vol_ct_mask_choose, \
    input_data_reshape, compute_maximum_nonzero_element
import SimpleITK as sitk
from PIL import Image
from sklearn.metrics import roc_curve, auc, confusion_matrix

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
            or (type(Lesion_LiRad) == int and (int(Lesion_Type) != 6)):  # or int(Lesion_Type) != 14)):
        Include_Lesion_Counter += 1
        if not (id in Patient_ID_List_Clean):  # and not(id in ID_Exclude_List_by_Seto):
            Patient_ID_List_Clean.append(id)

        if not (id in ID_Exclude_List_by_Seto):
            Patient_ID_Lesion_Type_LiRad_Label.append(
                [id, int(ALL_Lesions['type'][i]), ALL_Lesions['3D'][i], Lesion_LiRad])
    else:
        Exclude_Lesion_Counter += 1

    if (type(Lesion_LiRad) == str and Lesion_LiRad == 'Exclude') and not (id in Patient_ID_LiRad_Record_Clean):
        Patient_ID_LiRad_Record_Clean.append(id)

    if int(Lesion_Type) == 6 and not (id in Patient_ID_Tace_Record_Clean):
        Patient_ID_Tace_Record_Clean.append(id)  # 89 cases

    if int(Lesion_Type) == 14 and not (id in Patient_ID_RFA_Record_Clean):
        Patient_ID_RFA_Record_Clean.append(id)  # 73 cases

print(len(Patient_ID_LiRad_Record_Clean), len(Patient_ID_Tace_Record_Clean), len(Patient_ID_RFA_Record_Clean),
      "Line-61")
# In P2, P3 and P4, there are 16, 2533 and 2 cases. In total, there are 2551 cases
print(len(Patient_ID_List_Phase2), len(Patient_ID_List_Phase3), len(Patient_ID_List_Phase4))

print(len(Patient_ID_Lesion_Type_LiRad_Label), len(Patient_ID_List_Clean), "Line-71")  # 3765 Lesions, 2332 patients
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
print(len(Patient_ID_Lesion_Type_LiRad_Label_Tace_Excluding), len(Patient_ID_List_Clean_Tace_Excluding), "Line-80")
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
print(len(Patient_ID_Lesion_Type_LiRad_Label_Tace_RFA_Excluding), len(Patient_ID_List_Clean_Tace_RFA_Excluding),
      "Line-97")
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
print(len(Patient_ID_Lesion_Type_LiRad_Label_Tace_RFA_LiRadExclude_Excluding),
      len(Patient_ID_List_Clean_Tace_RFA_LiRadExclude_Excluding), "Line-115")
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
        Patient_ID_Lesion_Size_Dictionary[case_info_id].append(
            [case_info_lesion_type, case_info_lesion_size, case_info_lesion_lirad])

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
print("The numbers of patients having one lesion and more than one lesion are %d and %d, respectively " %
      (len(Singe_Lesion_List), len(Multiple_Lesion_List)))

# 1391 patients with one lesion, and 890 patients with more than one lesions
# among 890 patients, 255 patients with more than or equal to 2 kinds of lesions; 635 patients with one kind of lesions
Check_Lesion_Possible_Removed = []
for case_info in Multiple_Lesion_List:
    lesion_size, lesion_type = [], []
    lesion_record = []
    for case_lesion in case_info[1:]:  # [Lesion_1_type, Lesion_1_size, Lesion_1_LiRad]
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
        patient_label = patient_id_label.split(' ')[1]
        Test_ID_List.append(patient_id)
        Test_ID_Patient_Label_GT.append([patient_id, patient_label])
        Patient_ID.append(patient_id)
    else:
        continue

print(len(Train_ID_List), len(np.unique(np.array(Train_ID_List))), len(Test_ID_List),
      len(np.unique(np.array(Test_ID_List))))
print("The numbers of patients in the training and testing sets are: %d and %d" %
      (len(Train_ID_Patient_Label_GT), len(Test_ID_Patient_Label_GT)))
print('The total number of patients is %d' % (len(Train_ID_Patient_Label_GT) + len(Test_ID_Patient_Label_GT)))

#
Test_ID_Patient_Label_SubGroup4_GT = []
Test_ID_Patient_Label_SubGroup4_Radiologist_type1 = []
Test_ID_Patient_Label_SubGroup4_Radiologist_type2 = []
Test_ID_Patient_Label_SubGroup4_Radiologist_type3 = []
Test_ID_Patient_Label_SubGroup4_Radiologist_type4 = []
Test_ID_Patient_Label_SubGroup4_Radiologist_type4_2 = []
Test_ID_Patient_Label_SubGroup4_Radiologist_type4_3 = []
Test_ID_Patient_Label_SubGroup4_Radiologist_type5 = []
Test_ID_Patient_Label_SubGroup4_Radiologist_type5_2 = []
Test_ID_Patient_Label_SubGroup4_Radiologist_type5_3 = []

Test_ID_Lesion_Label_SubGroup4_GT = []
Test_ID_Lesion_Label_SubGroup4_Radiologist_type1 = []
Test_ID_Lesion_Label_SubGroup4_Radiologist_type2 = []
Test_ID_Lesion_Label_SubGroup4_Radiologist_type3 = []
Test_ID_Lesion_Label_SubGroup4_Radiologist_type4 = []
Test_ID_Lesion_Label_SubGroup4_Radiologist_type4_2 = []
Test_ID_Lesion_Label_SubGroup4_Radiologist_type4_3 = []
Test_ID_Lesion_Label_SubGroup4_Radiologist_type5 = []
Test_ID_Lesion_Label_SubGroup4_Radiologist_type5_2 = []
Test_ID_Lesion_Label_SubGroup4_Radiologist_type5_3 = []

case_id_list_check = \
    ['ID_0036_P3', 'ID_0106_P3', 'ID_0145_P3', 'ID_0270_P3', 'ID_0490b_P3', 'ID_0509_P3', 'ID_0662_P3',
     'ID_0707_P3', 'ID_0716_P3', 'ID_0742_P3', 'KWH_0045_P3', 'QEH020_P3', 'QMH_0074_P3', 'SZF_0007_P3',
     'SZF_0102_P3', 'SZF_0104_P3', 'SZF_0198_P3', 'SZF_0203_P3', 'SZF_0206_P3', 'SZF_0256_P3', 'SZF_0292_P3',
     'SZF_0591_P3', 'SZH_0317_P3']

for key in Patient_ID_Lesion_Size_Dictionary.keys():
    values = Patient_ID_Lesion_Size_Dictionary[key]
    if key in Test_ID_List:
        if len(values) == 3:              # ------- For testing cases with single lesion -------  #
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

            if (type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M') or  \
                    (type(Lesion_LiRad) == int and 2 <= Lesion_LiRad <= 4):
                Test_ID_Patient_Label_SubGroup4_GT.append([key, HCC_nonHCC_GT])
                Test_ID_Patient_Label_SubGroup4_Radiologist_type1.append([key, HCC_nonHCC_Radiologist])
                Test_ID_Patient_Label_SubGroup4_Radiologist_type2.append([key, HCC_nonHCC_Radiologist_type2])
                Test_ID_Patient_Label_SubGroup4_Radiologist_type3.append([key, HCC_nonHCC_Radiologist_type3])
                Test_ID_Patient_Label_SubGroup4_Radiologist_type4.append([key, HCC_nonHCC_Radiologist_type4])
                Test_ID_Patient_Label_SubGroup4_Radiologist_type5.append([key, HCC_nonHCC_Radiologist_type5])

                Test_ID_Lesion_Label_SubGroup4_GT.append([key, HCC_nonHCC_GT])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type1.append([key, HCC_nonHCC_Radiologist])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type2.append([key, HCC_nonHCC_Radiologist_type2])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type3.append([key, HCC_nonHCC_Radiologist_type3])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type4.append([key, HCC_nonHCC_Radiologist_type4])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type5.append([key, HCC_nonHCC_Radiologist_type5])

                # SubGroup4 type 4_2, LiRad 4, 5 --- HCC; LR-M, LiRad 3 --- Uncertain; LiRad 1, 2 --- Non-HCC
                if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 2):
                    HCC_nonHCC_Radiologist_SubGroup4_type4_2 = 0           # LiRad 1 and 2
                elif type(Lesion_LiRad) == int and (4 <= Lesion_LiRad <= 5):
                    HCC_nonHCC_Radiologist_SubGroup4_type4_2 = 1           # LiRad 4 and 5
                else:
                    HCC_nonHCC_Radiologist_SubGroup4_type4_2 = 1           # LiRad 3 and LR-M: Uncertain ---> HCC
                # SubGroup4 type 4_3, LiRad 4, 5 --- HCC; LR-M, LiRad 3 --- Uncertain; LiRad 1, 2 --- Non-HCC
                if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 2):
                    HCC_nonHCC_Radiologist_SubGroup4_type4_3 = 0           # LiRad 1 and 2
                elif type(Lesion_LiRad) == int and (4 <= Lesion_LiRad <= 5):
                    HCC_nonHCC_Radiologist_SubGroup4_type4_3 = 1           # LiRad 4, 5
                else:
                    HCC_nonHCC_Radiologist_SubGroup4_type4_3 = 0           # LiRad 3 and LR-M: Uncertain ---> Non-HCC

                # SubGroup4 type 5_2, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
                if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                    HCC_nonHCC_Radiologist_SubGroup4_type5_2 = 0
                elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                    HCC_nonHCC_Radiologist_SubGroup4_type5_2 = 1
                elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                    HCC_nonHCC_Radiologist_SubGroup4_type5_2 = 1
                else:                                                      # LiRads = 3: Uncertain ---> HCC
                    HCC_nonHCC_Radiologist_SubGroup4_type5_2 = 1
                # SubGroup4 type 5_3, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
                if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                    HCC_nonHCC_Radiologist_SubGroup4_type5_3 = 0
                elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                    HCC_nonHCC_Radiologist_SubGroup4_type5_3 = 1
                elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                    HCC_nonHCC_Radiologist_SubGroup4_type5_3 = 1
                else:                                                     # LiRads = 3:  Uncertain ---> Non-HCC
                    HCC_nonHCC_Radiologist_SubGroup4_type5_3 = 0

                # Patient Level
                Test_ID_Patient_Label_SubGroup4_Radiologist_type4_2.append([key, HCC_nonHCC_Radiologist_SubGroup4_type4_2])
                Test_ID_Patient_Label_SubGroup4_Radiologist_type4_3.append([key, HCC_nonHCC_Radiologist_SubGroup4_type4_3])
                Test_ID_Patient_Label_SubGroup4_Radiologist_type5_2.append([key, HCC_nonHCC_Radiologist_SubGroup4_type5_2])
                Test_ID_Patient_Label_SubGroup4_Radiologist_type5_3.append([key, HCC_nonHCC_Radiologist_SubGroup4_type5_3])
                # Lesion Level
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type4_2.append([key, HCC_nonHCC_Radiologist_SubGroup4_type4_2])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type4_3.append([key, HCC_nonHCC_Radiologist_SubGroup4_type4_3])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type5_2.append([key, HCC_nonHCC_Radiologist_SubGroup4_type5_2])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type5_3.append([key, HCC_nonHCC_Radiologist_SubGroup4_type5_3])

        elif len(values) == 4:     # --------- For Testing Case with two lesions -------- #
            #### **********************  the first lesion in two lesions of the testing case  ******************   ####
            Lesion_First = values[0:3]
            Lesion_Type = Lesion_First[0]
            Lesion_LiRad = Lesion_First[2]
            HCC_nonHCC_GT_Overall, HCC_nonHCC_Radiologist_Overall, HCC_nonHCC_Radiologist_type2_Overall = [], [], []
            HCC_nonHCC_Radiologist_type3_Overall, HCC_nonHCC_Radiologist_type4_Overall = [], []
            HCC_nonHCC_Radiologist_type5_Overall = []

            HCC_nonHCC_Radiologist_type4_2_Overall, HCC_nonHCC_Radiologist_type4_3_Overall = [], []
            HCC_nonHCC_Radiologist_type5_2_Overall, HCC_nonHCC_Radiologist_type5_3_Overall = [], []

            if Lesion_Type == 2 or Lesion_Type == 6 or Lesion_Type == 10 or Lesion_Type == 19 or Lesion_Type == 20 \
                    or Lesion_Type == 21 or Lesion_Type == 22:
                HCC_nonHCC_GT = 1
            else:
                HCC_nonHCC_GT = 0
            HCC_nonHCC_GT_Overall.append(HCC_nonHCC_GT)
            # Type 1:      LiRad 4 and LiRad 5 --- HCC, LiRad 1, 2, 3 and LR-M --- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):  # LiRad 4 and LiRad 5: HCC
                HCC_nonHCC_Radiologist = 1
            else:                                                                       # LiRad 1, 2, 3 and LR-M: NonHCC
                HCC_nonHCC_Radiologist = 0
            HCC_nonHCC_Radiologist_Overall.append(HCC_nonHCC_Radiologist)
            # Type 2, LiRads 3, 4, 5 and LR-M ---> HCC,  LiRad 1, 2 ---> NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2): # LiRad 1 and 2: NonHCC
                HCC_nonHCC_Radiologist_type2 = 0
            else:                                                                      # LiRad 3, 4, 5 and LR-M: HCC
                HCC_nonHCC_Radiologist_type2 = 1
            HCC_nonHCC_Radiologist_type2_Overall.append(HCC_nonHCC_Radiologist_type2)
            # Type 3, LiRad 4, 5 and LR-M are considered as HCC, and LiRads 1, 2, and 3 are Non-HCC
            if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 3):       # LiRad 1, 2, 3    --- NonHCC
                HCC_nonHCC_Radiologist_type3 = 0
            else:                                                            # LiRad 4, 5, LR-M --- HCC
                HCC_nonHCC_Radiologist_type3 = 1
            HCC_nonHCC_Radiologist_type3_Overall.append(HCC_nonHCC_Radiologist_type3)
            # Type 4, LiRad 4, 5 --- HCC; LR-M and LiRad 3 --- Uncertain, LiRad 1, 2 --- Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type4 = 0  # Non-HCC --- LiRad 1, 2
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type4 = 1  # HCC --- LiRad 4, 5
            else:                                 # Uncertain --- LiRad 3 and LR-M
                HCC_nonHCC_Radiologist_type4 = 2
            HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)
            # For type 5, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type5 = 0
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type5 = 1
            elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                HCC_nonHCC_Radiologist_type5 = 1
            else:  # LiRads = 3    ------ Uncertain
                HCC_nonHCC_Radiologist_type5 = 2
            HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)

            # For SubGroup4, type 4 and type 5
            HCC_nonHCC_Radiologist_SubGroup4_type4_2_Overall, HCC_nonHCC_Radiologist_SubGroup4_type4_3_Overall = [], []
            HCC_nonHCC_Radiologist_SubGroup4_type5_2_Overall, HCC_nonHCC_Radiologist_SubGroup4_type5_3_Overall = [], []
            if (type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M') or \
                    (type(Lesion_LiRad) == int and 2 <= Lesion_LiRad <= 4):
                Test_ID_Lesion_Label_SubGroup4_GT.append([key, HCC_nonHCC_GT])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type1.append([key, HCC_nonHCC_Radiologist])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type2.append([key, HCC_nonHCC_Radiologist_type2])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type3.append([key, HCC_nonHCC_Radiologist_type3])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type4.append([key, HCC_nonHCC_Radiologist_type4])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type5.append([key, HCC_nonHCC_Radiologist_type5])

                # SubGroup4 type 4_2, LiRad 4, 5 --- HCC; LR-M, LiRad 3 --- Uncertain; LiRad 1, 2 --- Non-HCC
                if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 2):
                    HCC_nonHCC_Radiologist_SubGroup4_type4_2 = 0           # LiRad 1 and 2
                elif type(Lesion_LiRad) == int and (4 <= Lesion_LiRad <= 5):
                    HCC_nonHCC_Radiologist_SubGroup4_type4_2 = 1           # LiRad 4 and 5
                else:
                    HCC_nonHCC_Radiologist_SubGroup4_type4_2 = 1           # LiRad 3 and LR-M: Uncertain ---> HCC
                HCC_nonHCC_Radiologist_SubGroup4_type4_2_Overall.append(HCC_nonHCC_Radiologist_SubGroup4_type4_2)
                # SubGroup4 type 4_3, LiRad 4, 5 --- HCC; LR-M, LiRad 3 --- Uncertain; LiRad 1, 2 --- Non-HCC
                if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 2):
                    HCC_nonHCC_Radiologist_SubGroup4_type4_3 = 0           # LiRad 1 and 2
                elif type(Lesion_LiRad) == int and (4 <= Lesion_LiRad <= 5):
                    HCC_nonHCC_Radiologist_SubGroup4_type4_3 = 1           # LiRad 4, 5
                else:
                    HCC_nonHCC_Radiologist_SubGroup4_type4_3 = 0           # LiRad 3 and LR-M: Uncertain ---> Non-HCC
                HCC_nonHCC_Radiologist_SubGroup4_type4_3_Overall.append(HCC_nonHCC_Radiologist_SubGroup4_type4_3)
                # Lesion Level
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type4_2.append([key, HCC_nonHCC_Radiologist_SubGroup4_type4_2])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type4_3.append([key, HCC_nonHCC_Radiologist_SubGroup4_type4_3])

                # SubGroup4 type 5_2, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
                if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                    HCC_nonHCC_Radiologist_SubGroup4_type5_2 = 0
                elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                    HCC_nonHCC_Radiologist_SubGroup4_type5_2 = 1
                elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                    HCC_nonHCC_Radiologist_SubGroup4_type5_2 = 1
                else:  # LiRads = 3    --- Uncertain ---> HCC
                    HCC_nonHCC_Radiologist_SubGroup4_type5_2 = 1
                HCC_nonHCC_Radiologist_SubGroup4_type5_2_Overall.append(HCC_nonHCC_Radiologist_SubGroup4_type5_2)
                # SubGroup4 type 5_3, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
                if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                    HCC_nonHCC_Radiologist_SubGroup4_type5_3 = 0
                elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                    HCC_nonHCC_Radiologist_SubGroup4_type5_3 = 1
                elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                    HCC_nonHCC_Radiologist_SubGroup4_type5_3 = 1
                else:  # LiRads = 3    ---  Uncertain ---> Non-HCC
                    HCC_nonHCC_Radiologist_SubGroup4_type5_3 = 0
                HCC_nonHCC_Radiologist_SubGroup4_type5_3_Overall.append(HCC_nonHCC_Radiologist_SubGroup4_type5_3)
                # Lesion Level
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type5_2.append([key, HCC_nonHCC_Radiologist_SubGroup4_type5_2])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type5_3.append([key, HCC_nonHCC_Radiologist_SubGroup4_type5_3])

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

            if (type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M') or \
                    (type(Lesion_LiRad) == int and 2 <= Lesion_LiRad <= 4):
                Test_ID_Lesion_Label_SubGroup4_GT.append([key, HCC_nonHCC_GT])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type1.append([key, HCC_nonHCC_Radiologist])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type2.append([key, HCC_nonHCC_Radiologist_type2])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type3.append([key, HCC_nonHCC_Radiologist_type3])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type4.append([key, HCC_nonHCC_Radiologist_type4])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type5.append([key, HCC_nonHCC_Radiologist_type5])

                # SubGroup4 type 4_2, LiRad 4, 5 --- HCC; LR-M, LiRad 3 --- Uncertain; LiRad 1, 2 --- Non-HCC
                # Uncertain --- HCC
                if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 2):
                    HCC_nonHCC_Radiologist_SubGroup4_type4_2 = 0           # LiRad 1 and 2
                elif type(Lesion_LiRad) == int and (4 <= Lesion_LiRad <= 5):
                    HCC_nonHCC_Radiologist_SubGroup4_type4_2 = 1           # LiRad 4 and 5
                else:
                    HCC_nonHCC_Radiologist_SubGroup4_type4_2 = 1           # LiRad 3 and LR-M ---> HCC
                HCC_nonHCC_Radiologist_SubGroup4_type4_2_Overall.append(HCC_nonHCC_Radiologist_SubGroup4_type4_2)
                # SubGroup4 type 4_3, LiRad 4, 5 --- HCC; LR-M, LiRad 3 --- Uncertain; LiRad 1, 2 --- Non-HCC
                # Uncertain --- Non-HCC
                if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 2):
                    HCC_nonHCC_Radiologist_SubGroup4_type4_3 = 0           # LiRad 1 and 2
                elif type(Lesion_LiRad) == int and (4 <= Lesion_LiRad <= 5):
                    HCC_nonHCC_Radiologist_SubGroup4_type4_3 = 1           # LiRad 4, 5
                else:
                    HCC_nonHCC_Radiologist_SubGroup4_type4_3 = 0           # LiRad 3 and LR-M ---> Non-HCC
                HCC_nonHCC_Radiologist_SubGroup4_type4_3_Overall.append(HCC_nonHCC_Radiologist_SubGroup4_type4_3)
                # Lesion Level
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type4_2.append([key, HCC_nonHCC_Radiologist_SubGroup4_type4_2])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type4_3.append([key, HCC_nonHCC_Radiologist_SubGroup4_type4_3])

                # SubGroup4 type 5_2, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
                if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                    HCC_nonHCC_Radiologist_SubGroup4_type5_2 = 0
                elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                    HCC_nonHCC_Radiologist_SubGroup4_type5_2 = 1
                elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                    HCC_nonHCC_Radiologist_SubGroup4_type5_2 = 1
                else:  # LiRads = 3    --- Uncertain ---> HCC
                    HCC_nonHCC_Radiologist_SubGroup4_type5_2 = 1
                HCC_nonHCC_Radiologist_SubGroup4_type5_2_Overall.append(HCC_nonHCC_Radiologist_SubGroup4_type5_2)
                # SubGroup4 type 5_3, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
                if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                    HCC_nonHCC_Radiologist_SubGroup4_type5_3 = 0
                elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                    HCC_nonHCC_Radiologist_SubGroup4_type5_3 = 1
                elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                    HCC_nonHCC_Radiologist_SubGroup4_type5_3 = 1
                else:  # LiRads = 3    ---  Uncertain ---> Non-HCC
                    HCC_nonHCC_Radiologist_SubGroup4_type5_3 = 0
                HCC_nonHCC_Radiologist_SubGroup4_type5_3_Overall.append(HCC_nonHCC_Radiologist_SubGroup4_type5_3)
                # Lesion Level
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type5_2.append([key, HCC_nonHCC_Radiologist_SubGroup4_type5_2])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type5_3.append([key, HCC_nonHCC_Radiologist_SubGroup4_type5_3])

            if (type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M') or \
                    (type(Lesion_LiRad) == int and 2 <= Lesion_LiRad <= 4):
                if np.max(np.array(HCC_nonHCC_GT_Overall)) == 0:
                    Test_ID_Patient_Label_SubGroup4_GT.append([key, 0])
                else:
                    Test_ID_Patient_Label_SubGroup4_GT.append([key, 1])
                # type 1
                if np.max(np.array(HCC_nonHCC_Radiologist_Overall)) == 0:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type1.append([key, 0])
                else:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type1.append([key, 1])
                # type 2
                if np.max(np.array(HCC_nonHCC_Radiologist_type2_Overall)) == 0:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type2.append([key, 0])
                else:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type2.append([key, 1])
                # type 3
                if np.max(np.array(HCC_nonHCC_Radiologist_type3_Overall)) == 0:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type3.append([key, 0])
                else:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type3.append([key, 1])
                # type 4
                if np.max(np.array(HCC_nonHCC_Radiologist_type4_Overall)) == 0:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type4.append([key, 0])
                elif np.max(np.array(HCC_nonHCC_Radiologist_type4_Overall)) == 1:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type4.append([key, 1])
                else:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type4.append([key, 2])
                # subgroup 4 -----type 5
                if np.max(np.array(HCC_nonHCC_Radiologist_type5_Overall)) == 0:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type5.append([key, 0])
                elif np.max(np.array(HCC_nonHCC_Radiologist_type4_Overall)) == 1:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type5.append([key, 1])
                else:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type5.append([key, 2])

                # subgroup 4 ---- type 4_2
                if np.max(np.array(HCC_nonHCC_Radiologist_SubGroup4_type4_2_Overall)) == 0:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type4_2.append([key, 0])
                else:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type4_2.append([key, 1])
                # subgroup 4 ---- type 4_3
                if np.max(np.array(HCC_nonHCC_Radiologist_SubGroup4_type4_3_Overall)) == 0:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type4_3.append([key, 0])
                else:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type4_3.append([key, 1])
                # subgroup 5 ---- type 4_2
                if np.max(np.array(HCC_nonHCC_Radiologist_SubGroup4_type5_2_Overall)) == 0:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type5_2.append([key, 0])
                else:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type5_2.append([key, 1])
                # subgroup 5 ---- type 4_3
                if np.max(np.array(HCC_nonHCC_Radiologist_SubGroup4_type5_3_Overall)) == 0:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type5_3.append([key, 0])
                else:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type5_3.append([key, 1])
        elif len(values) == 5:
            Lesion_First = values[0:3]
            Lesion_Type = Lesion_First[0]
            Lesion_LiRad = Lesion_First[2]
            HCC_nonHCC_GT_Overall, HCC_nonHCC_Radiologist_Overall, HCC_nonHCC_Radiologist_type2_Overall = [], [], []
            HCC_nonHCC_Radiologist_type3_Overall, HCC_nonHCC_Radiologist_type4_Overall = [], []
            HCC_nonHCC_Radiologist_type5_Overall = []



            HCC_nonHCC_Radiologist_type4_2_Overall, HCC_nonHCC_Radiologist_type4_3_Overall = [], []
            HCC_nonHCC_Radiologist_type5_2_Overall, HCC_nonHCC_Radiologist_type5_3_Overall = [], []

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
            else:
                HCC_nonHCC_Radiologist_type2 = 1
            HCC_nonHCC_Radiologist_type2_Overall.append(HCC_nonHCC_Radiologist_type2)
            # For type 3, LiRad 4, 5 and LR-M are considered as HCC, and LiRads 1, 2, and 3 are Non-HCC
            if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 3):
                HCC_nonHCC_Radiologist_type3 = 0
            else:
                HCC_nonHCC_Radiologist_type3 = 1
            HCC_nonHCC_Radiologist_type3_Overall.append(HCC_nonHCC_Radiologist_type3)
            # For type 4, LiRad 4, 5 --- HCC; LR-M and LiRad 3 --- Uncertain, LiRad 1, 2 --- Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type4 = 0  # Non-HCC
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type4 = 1  # HCC
            else:
                HCC_nonHCC_Radiologist_type4 = 2  # Uncertain
            HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)
            # For type 5, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type5 = 0
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type5 = 1
            elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                HCC_nonHCC_Radiologist_type5 = 1
            else:  # LiRads = 3    ------ Uncertain
                HCC_nonHCC_Radiologist_type5 = 2
            HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)

            HCC_nonHCC_Radiologist_SubGroup4_type4_2_Overall, HCC_nonHCC_Radiologist_SubGroup4_type4_3_Overall = [], []
            HCC_nonHCC_Radiologist_SubGroup4_type5_2_Overall, HCC_nonHCC_Radiologist_SubGroup4_type5_3_Overall = [], []
            if (type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M') or \
                    (type(Lesion_LiRad) == int and 2 <= Lesion_LiRad <= 4):
                Test_ID_Lesion_Label_SubGroup4_GT.append([key, HCC_nonHCC_GT])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type1.append([key, HCC_nonHCC_Radiologist])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type2.append([key, HCC_nonHCC_Radiologist_type2])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type3.append([key, HCC_nonHCC_Radiologist_type3])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type4.append([key, HCC_nonHCC_Radiologist_type4])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type5.append([key, HCC_nonHCC_Radiologist_type5])
                # SubGroup4 type 4_2, LiRad 4, 5 --- HCC; LR-M, LiRad 3 --- Uncertain; LiRad 1, 2 --- Non-HCC
                if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 2):
                    HCC_nonHCC_Radiologist_SubGroup4_type4_2 = 0           # LiRad 1 and 2
                elif type(Lesion_LiRad) == int and (4 <= Lesion_LiRad <= 5):
                    HCC_nonHCC_Radiologist_SubGroup4_type4_2 = 1           # LiRad 4 and 5
                else:
                    HCC_nonHCC_Radiologist_SubGroup4_type4_2 = 1           # LiRad 3 and LR-M: Uncertain ---> HCC
                HCC_nonHCC_Radiologist_SubGroup4_type4_2_Overall.append(HCC_nonHCC_Radiologist_SubGroup4_type4_2)
                # SubGroup4 type 4_3, LiRad 4, 5 --- HCC; LR-M, LiRad 3 --- Uncertain; LiRad 1, 2 --- Non-HCC
                if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 2):
                    HCC_nonHCC_Radiologist_SubGroup4_type4_3 = 0           # LiRad 1 and 2
                elif type(Lesion_LiRad) == int and (4 <= Lesion_LiRad <= 5):
                    HCC_nonHCC_Radiologist_SubGroup4_type4_3 = 1           # LiRad 4, 5
                else:
                    HCC_nonHCC_Radiologist_SubGroup4_type4_3 = 0           # LiRad 3 and LR-M ---> Non-HCC
                HCC_nonHCC_Radiologist_SubGroup4_type4_3_Overall.append(HCC_nonHCC_Radiologist_SubGroup4_type4_3)

                # SubGroup4 type 5_2, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
                if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                    HCC_nonHCC_Radiologist_SubGroup4_type5_2 = 0
                elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                    HCC_nonHCC_Radiologist_SubGroup4_type5_2 = 1
                elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                    HCC_nonHCC_Radiologist_SubGroup4_type5_2 = 1
                else:  # LiRads = 3    --- Uncertain ---> HCC
                    HCC_nonHCC_Radiologist_SubGroup4_type5_2 = 1
                HCC_nonHCC_Radiologist_SubGroup4_type5_2_Overall.append(HCC_nonHCC_Radiologist_SubGroup4_type5_2)
                # SubGroup4 type 5_3, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
                if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                    HCC_nonHCC_Radiologist_SubGroup4_type5_3 = 0
                elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                    HCC_nonHCC_Radiologist_SubGroup4_type5_3 = 1
                elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                    HCC_nonHCC_Radiologist_SubGroup4_type5_3 = 1
                else:  # LiRads = 3    ---  Uncertain ---> Non-HCC
                    HCC_nonHCC_Radiologist_SubGroup4_type5_3 = 0
                HCC_nonHCC_Radiologist_SubGroup4_type5_3_Overall.append(HCC_nonHCC_Radiologist_SubGroup4_type5_3)
                # Lesion Level
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type4_2.append([key, HCC_nonHCC_Radiologist_SubGroup4_type4_2])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type4_3.append([key, HCC_nonHCC_Radiologist_SubGroup4_type4_3])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type5_2.append([key, HCC_nonHCC_Radiologist_SubGroup4_type5_2])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type5_3.append([key, HCC_nonHCC_Radiologist_SubGroup4_type5_3])

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
            else:
                HCC_nonHCC_Radiologist_type2 = 1
            HCC_nonHCC_Radiologist_type2_Overall.append(HCC_nonHCC_Radiologist_type2)
            # For type 3, LiRad 4, 5 and LR-M are considered as HCC, and LiRads 1, 2, and 3 are Non-HCC
            if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 3):
                HCC_nonHCC_Radiologist_type3 = 0
            else:
                HCC_nonHCC_Radiologist_type3 = 1
            HCC_nonHCC_Radiologist_type3_Overall.append(HCC_nonHCC_Radiologist_type3)
            # For type 4, LiRad 4, 5 --- HCC; LR-M and LiRad 3 --- Uncertain, LiRad 1, 2 --- Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type4 = 0  # Non-HCC
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type4 = 1  # HCC
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

            if (type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M') or \
                    (type(Lesion_LiRad) == int and 2 <= Lesion_LiRad <= 4):
                Test_ID_Lesion_Label_SubGroup4_GT.append([key, HCC_nonHCC_GT])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type1.append([key, HCC_nonHCC_Radiologist])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type2.append([key, HCC_nonHCC_Radiologist_type2])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type3.append([key, HCC_nonHCC_Radiologist_type3])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type4.append([key, HCC_nonHCC_Radiologist_type4])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type5.append([key, HCC_nonHCC_Radiologist_type5])
                # SubGroup4 type 4_2, LiRad 4, 5 --- HCC; LR-M, LiRad 3 --- Uncertain; LiRad 1, 2 --- Non-HCC
                if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 2):
                    HCC_nonHCC_Radiologist_SubGroup4_type4_2 = 0           # LiRad 1 and 2
                elif type(Lesion_LiRad) == int and (4 <= Lesion_LiRad <= 5):
                    HCC_nonHCC_Radiologist_SubGroup4_type4_2 = 1           # LiRad 4 and 5
                else:
                    HCC_nonHCC_Radiologist_SubGroup4_type4_2 = 1           # LiRad 3 and LR-M ---> HCC
                HCC_nonHCC_Radiologist_SubGroup4_type4_2_Overall.append(HCC_nonHCC_Radiologist_SubGroup4_type4_2)
                # SubGroup4 type 4_3, LiRad 4, 5 --- HCC; LR-M, LiRad 3 --- Uncertain; LiRad 1, 2 --- Non-HCC
                if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 2):
                    HCC_nonHCC_Radiologist_SubGroup4_type4_3 = 0           # LiRad 1 and 2
                elif type(Lesion_LiRad) == int and (4 <= Lesion_LiRad <= 5):
                    HCC_nonHCC_Radiologist_SubGroup4_type4_3 = 1           # LiRad 4, 5
                else:
                    HCC_nonHCC_Radiologist_SubGroup4_type4_3 = 0           # LiRad 3 and LR-M ---> Non-HCC
                HCC_nonHCC_Radiologist_SubGroup4_type4_3_Overall.append(HCC_nonHCC_Radiologist_SubGroup4_type4_3)

                # SubGroup4 type 5_2, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
                if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                    HCC_nonHCC_Radiologist_SubGroup4_type5_2 = 0
                elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                    HCC_nonHCC_Radiologist_SubGroup4_type5_2 = 1
                elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                    HCC_nonHCC_Radiologist_SubGroup4_type5_2 = 1
                else:  # LiRads = 3    --- Uncertain ---> HCC
                    HCC_nonHCC_Radiologist_SubGroup4_type5_2 = 1
                HCC_nonHCC_Radiologist_SubGroup4_type5_2_Overall.append(HCC_nonHCC_Radiologist_SubGroup4_type5_2)
                # SubGroup4 type 5_3, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
                if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                    HCC_nonHCC_Radiologist_SubGroup4_type5_3 = 0
                elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                    HCC_nonHCC_Radiologist_SubGroup4_type5_3 = 1
                elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                    HCC_nonHCC_Radiologist_SubGroup4_type5_3 = 1
                else:  # LiRads = 3    ---  Uncertain ---> Non-HCC
                    HCC_nonHCC_Radiologist_SubGroup4_type5_3 = 0
                HCC_nonHCC_Radiologist_SubGroup4_type5_3_Overall.append(HCC_nonHCC_Radiologist_SubGroup4_type5_3)
                # Lesion Level
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type4_2.append([key, HCC_nonHCC_Radiologist_SubGroup4_type4_2])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type4_3.append([key, HCC_nonHCC_Radiologist_SubGroup4_type4_3])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type5_2.append([key, HCC_nonHCC_Radiologist_SubGroup4_type5_2])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type5_3.append([key, HCC_nonHCC_Radiologist_SubGroup4_type5_3])

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
            else:
                HCC_nonHCC_Radiologist_type2 = 1
            HCC_nonHCC_Radiologist_type2_Overall.append(HCC_nonHCC_Radiologist_type2)
            # For type 3, LiRad 4, 5 and LR-M are considered as HCC, and LiRads 1, 2, and 3 are Non-HCC
            if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 3):
                HCC_nonHCC_Radiologist_type3 = 0
            else:
                HCC_nonHCC_Radiologist_type3 = 1
            HCC_nonHCC_Radiologist_type3_Overall.append(HCC_nonHCC_Radiologist_type3)
            # For type 4, LiRad 4, 5 --- HCC; LR-M and LiRad 3 --- Uncertain, LiRad 1, 2 --- Non-HCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type4 = 0  # Non-HCC
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type4 = 1  # HCC
            else:
                HCC_nonHCC_Radiologist_type4 = 2  # Uncertain
            HCC_nonHCC_Radiologist_type4_Overall.append(HCC_nonHCC_Radiologist_type4)

            # For type 5, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
            if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                HCC_nonHCC_Radiologist_type5 = 0
            elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                HCC_nonHCC_Radiologist_type5 = 1
            elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                HCC_nonHCC_Radiologist_type5 = 1
            else:  # LiRads = 3    ------ Uncertain
                HCC_nonHCC_Radiologist_type5 = 2
            HCC_nonHCC_Radiologist_type5_Overall.append(HCC_nonHCC_Radiologist_type5)

            if (type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M') or \
                    (type(Lesion_LiRad) == int and 2 <= Lesion_LiRad <= 4):
                Test_ID_Lesion_Label_SubGroup4_GT.append([key, HCC_nonHCC_GT])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type1.append([key, HCC_nonHCC_Radiologist])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type2.append([key, HCC_nonHCC_Radiologist_type2])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type3.append([key, HCC_nonHCC_Radiologist_type3])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type4.append([key, HCC_nonHCC_Radiologist_type4])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type5.append([key, HCC_nonHCC_Radiologist_type5])
                # SubGroup4 type 4_2, LiRad 4, 5 --- HCC; LR-M, LiRad 3 --- Uncertain; LiRad 1, 2 --- Non-HCC
                # Uncertain --- HCC
                if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 2):
                    HCC_nonHCC_Radiologist_SubGroup4_type4_2 = 0           # LiRad 1 and 2
                elif type(Lesion_LiRad) == int and (4 <= Lesion_LiRad <= 5):
                    HCC_nonHCC_Radiologist_SubGroup4_type4_2 = 1           # LiRad 4 and 5
                else:
                    HCC_nonHCC_Radiologist_SubGroup4_type4_2 = 1           # LiRad 3 and LR-M ---> HCC
                HCC_nonHCC_Radiologist_SubGroup4_type4_2_Overall.append(HCC_nonHCC_Radiologist_SubGroup4_type4_2)
                # SubGroup4 type 4_3, LiRad 4, 5 --- HCC; LR-M, LiRad 3 --- Uncertain; LiRad 1, 2 --- Non-HCC
                # Uncertain --- Non-HCC
                if type(Lesion_LiRad) == int and (1 <= Lesion_LiRad <= 2):
                    HCC_nonHCC_Radiologist_SubGroup4_type4_3 = 0           # LiRad 1 and 2
                elif type(Lesion_LiRad) == int and (4 <= Lesion_LiRad <= 5):
                    HCC_nonHCC_Radiologist_SubGroup4_type4_3 = 1           # LiRad 4, 5
                else:
                    HCC_nonHCC_Radiologist_SubGroup4_type4_3 = 0           # LiRad 3 and LR-M ---> Non-HCC
                HCC_nonHCC_Radiologist_SubGroup4_type4_3_Overall.append(HCC_nonHCC_Radiologist_SubGroup4_type4_3)

                # SubGroup4 type 5_2, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
                if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                    HCC_nonHCC_Radiologist_SubGroup4_type5_2 = 0
                elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                    HCC_nonHCC_Radiologist_SubGroup4_type5_2 = 1
                elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                    HCC_nonHCC_Radiologist_SubGroup4_type5_2 = 1
                else:  # LiRads = 3    --- Uncertain ---> HCC
                    HCC_nonHCC_Radiologist_SubGroup4_type5_2 = 1
                HCC_nonHCC_Radiologist_SubGroup4_type5_2_Overall.append(HCC_nonHCC_Radiologist_SubGroup4_type5_2)
                # SubGroup4 type 5_3, LiRad 4, 5 and LR-M ---- HCC, LiRad 3 ---- Uncertain, LiRad 1, 2 ---- NonHCC
                if type(Lesion_LiRad) == int and (Lesion_LiRad == 1 or Lesion_LiRad == 2):
                    HCC_nonHCC_Radiologist_SubGroup4_type5_3 = 0
                elif type(Lesion_LiRad) == int and (Lesion_LiRad == 4 or Lesion_LiRad == 5):
                    HCC_nonHCC_Radiologist_SubGroup4_type5_3 = 1
                elif type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M':
                    HCC_nonHCC_Radiologist_SubGroup4_type5_3 = 1
                else:  # LiRads = 3    ---  Uncertain ---> Non-HCC
                    HCC_nonHCC_Radiologist_SubGroup4_type5_3 = 0
                HCC_nonHCC_Radiologist_SubGroup4_type5_3_Overall.append(HCC_nonHCC_Radiologist_SubGroup4_type5_3)
                # Lesion Level
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type4_2.append([key, HCC_nonHCC_Radiologist_SubGroup4_type4_2])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type4_3.append([key, HCC_nonHCC_Radiologist_SubGroup4_type4_3])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type5_2.append([key, HCC_nonHCC_Radiologist_SubGroup4_type5_2])
                Test_ID_Lesion_Label_SubGroup4_Radiologist_type5_3.append([key, HCC_nonHCC_Radiologist_SubGroup4_type5_3])

            if (type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M') or \
                    (type(Lesion_LiRad) == int and 2 <= Lesion_LiRad <= 4):
                if np.max(np.array(HCC_nonHCC_GT_Overall)) == 0:
                    Test_ID_Patient_Label_SubGroup4_GT.append([key, 0])
                else:
                    Test_ID_Patient_Label_SubGroup4_GT.append([key, 1])
                # SubGroup 4 --- type 1
                if np.max(np.array(HCC_nonHCC_Radiologist_Overall)) == 0:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type1.append([key, 0])
                else:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type1.append([key, 1])
                # Subgroup4 ---- type 2
                if np.max(np.array(HCC_nonHCC_Radiologist_type2_Overall)) == 0:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type2.append([key, 0])
                else:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type2.append([key, 1])
                # SubGroup 4 --- type 3
                if np.max(np.array(HCC_nonHCC_Radiologist_type3_Overall)) == 0:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type3.append([key, 0])
                else:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type3.append([key, 1])
                # SubGroup 4 --- type 4
                if np.max(np.array(HCC_nonHCC_Radiologist_type4_Overall)) == 0:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type4.append([key, 0])
                elif np.max(np.array(HCC_nonHCC_Radiologist_type4_Overall)) == 1:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type4.append([key, 1])
                else:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type4.append([key, 2])
                # SubGroup 4 --- type 5
                if np.max(np.array(HCC_nonHCC_Radiologist_type5_Overall)) == 0:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type5.append([key, 0])
                elif np.max(np.array(HCC_nonHCC_Radiologist_type5_Overall)):
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type5.append([key, 1])
                else:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type5.append([key, 2])

                # SubGroup 4 ---- type 4_2
                if np.max(np.array(HCC_nonHCC_Radiologist_SubGroup4_type4_2_Overall)) == 0:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type4_2.append([key, 0])
                else:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type4_2.append([key, 1])
                # SubGroup 4 ---- type 4_3
                if np.max(np.array(HCC_nonHCC_Radiologist_SubGroup4_type4_3_Overall)) == 0:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type4_3.append([key, 0])
                else:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type4_3.append([key, 1])
                # SubGroup 4 ---- type 5_2
                if np.max(np.array(HCC_nonHCC_Radiologist_SubGroup4_type5_2_Overall)) == 0:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type5_2.append([key, 0])
                else:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type5_2.append([key, 1])
                # SubGroup 4 ---- type 5_3
                if np.max(np.array(HCC_nonHCC_Radiologist_SubGroup4_type5_3_Overall)) == 0:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type5_3.append([key, 0])
                else:
                    Test_ID_Patient_Label_SubGroup4_Radiologist_type5_3.append([key, 1])

keys = Patient_ID_Lesion_Size_Dictionary.keys()
for patient in Patient_ID:
    if patient in list(keys):
        continue
    else:
        print(patient, "Line-230")

# Train_ID_List and Test_ID_List = [1595, 685]  ---> ID_List

# HKU_0571_P3 in the training set, but should be removed from the list
# HKU_0030_P3 should be in the list, but during to mismatch of lesion mask and excel file, it seems not to be included

Test_Lesion_Label_SubGroup4_GT, Test_Lesion_Label_SubGroup4_Radiologist_type1, \
Test_Lesion_Label_SubGroup4_Radiologist_type2, Test_Lesion_Label_SubGroup4_Radiologist_type3, \
Test_Lesion_Label_SubGroup4_Radiologist_type4, Test_Lesion_Label_SubGroup4_Radiologist_type5 = [], [], [], [], [], []
Test_Lesion_Label_SubGroup4_Radiologist_type4_2, Test_Lesion_Label_SubGroup4_Radiologist_type4_3, \
Test_Lesion_Label_SubGroup4_Radiologist_type5_2, Test_Lesion_Label_SubGroup4_Radiologist_type5_3 = [], [], [], []

for i in range(len(Test_ID_Lesion_Label_SubGroup4_GT)):
    Test_Lesion_Label_SubGroup4_GT.append(int(Test_ID_Lesion_Label_SubGroup4_GT[i][1]))
    Test_Lesion_Label_SubGroup4_Radiologist_type1.append(int(Test_ID_Lesion_Label_SubGroup4_Radiologist_type1[i][1]))
    Test_Lesion_Label_SubGroup4_Radiologist_type2.append(int(Test_ID_Lesion_Label_SubGroup4_Radiologist_type2[i][1]))
    Test_Lesion_Label_SubGroup4_Radiologist_type3.append(int(Test_ID_Lesion_Label_SubGroup4_Radiologist_type3[i][1]))
    Test_Lesion_Label_SubGroup4_Radiologist_type4.append(int(Test_ID_Lesion_Label_SubGroup4_Radiologist_type4[i][1]))
    Test_Lesion_Label_SubGroup4_Radiologist_type5.append(int(Test_ID_Lesion_Label_SubGroup4_Radiologist_type5[i][1]))
    Test_Lesion_Label_SubGroup4_Radiologist_type4_2.append(int(Test_ID_Lesion_Label_SubGroup4_Radiologist_type4_2[i][1]))
    Test_Lesion_Label_SubGroup4_Radiologist_type4_3.append(int(Test_ID_Lesion_Label_SubGroup4_Radiologist_type4_3[i][1]))
    Test_Lesion_Label_SubGroup4_Radiologist_type5_2.append(int(Test_ID_Lesion_Label_SubGroup4_Radiologist_type5_2[i][1]))
    Test_Lesion_Label_SubGroup4_Radiologist_type5_3.append(int(Test_ID_Lesion_Label_SubGroup4_Radiologist_type5_3[i][1]))


print(Test_Lesion_Label_SubGroup4_GT)
print('*-'*50)
print(Test_Lesion_Label_SubGroup4_Radiologist_type1)
print('*#'*50)
print(Test_Lesion_Label_SubGroup4_Radiologist_type2)
print('*<'*50)
print(Test_Lesion_Label_SubGroup4_Radiologist_type3)

print('The confusion matrix and auc of the testing set at the lesion level ==== SubGroup4 -type1 ====  line-2836: ')
confusion_test_lesion_level = confusion_matrix(Test_Lesion_Label_SubGroup4_GT,
                                               Test_Lesion_Label_SubGroup4_Radiologist_type1)
print(confusion_test_lesion_level)
fpr_group1, tpr_group1, _ = roc_curve(Test_Lesion_Label_SubGroup4_GT, Test_Lesion_Label_SubGroup4_Radiologist_type1)
auc_group1 = auc(fpr_group1, tpr_group1)
print(fpr_group1, tpr_group1, "Line-Type1")
print(auc_group1)

print('The confusion matrix and auc of the testing set at the lesion level ==== SubGroup4 -type2 ====  line-2843: ')
confusion_test_lesion_level = confusion_matrix(Test_Lesion_Label_SubGroup4_GT,
                                               Test_Lesion_Label_SubGroup4_Radiologist_type2)
print(confusion_test_lesion_level)
fpr_group1, tpr_group1, _ = roc_curve(Test_Lesion_Label_SubGroup4_GT, Test_Lesion_Label_SubGroup4_Radiologist_type2)
auc_group1 = auc(fpr_group1, tpr_group1)
print(fpr_group1, tpr_group1, "Line-Type2")
print(auc_group1)

print('The confusion matrix and auc of the testing set at the lesion level ==== SubGroup4 -type3 ====  line-2849: ')
confusion_test_lesion_level = confusion_matrix(Test_Lesion_Label_SubGroup4_GT,
                                               Test_Lesion_Label_SubGroup4_Radiologist_type3)
print(confusion_test_lesion_level)
fpr_group1, tpr_group1, _ = roc_curve(Test_Lesion_Label_SubGroup4_GT, Test_Lesion_Label_SubGroup4_Radiologist_type3)
auc_group1 = auc(fpr_group1, tpr_group1)
print(fpr_group1, tpr_group1, "Line-Type3")
print(auc_group1)

print('The confusion matrix and auc of the testing set at the lesion level ==== SubGroup4 -type4 ====  line-2856: ')
confusion_test_lesion_level = confusion_matrix(Test_Lesion_Label_SubGroup4_GT,
                                               Test_Lesion_Label_SubGroup4_Radiologist_type4)
print(confusion_test_lesion_level)
fpr_group1, tpr_group1, _ = roc_curve(Test_Lesion_Label_SubGroup4_GT, Test_Lesion_Label_SubGroup4_Radiologist_type4)
auc_group1 = auc(fpr_group1, tpr_group1)
print(auc_group1)

print('The confusion matrix and auc of the testing set at the lesion level ==== SubGroup4 -type5 ====  line-2863: ')
confusion_test_lesion_level = confusion_matrix(Test_Lesion_Label_SubGroup4_GT,
                                               Test_Lesion_Label_SubGroup4_Radiologist_type5)
print(confusion_test_lesion_level)
fpr_group1, tpr_group1, _ = roc_curve(Test_Lesion_Label_SubGroup4_GT, Test_Lesion_Label_SubGroup4_Radiologist_type5)
auc_group1 = auc(fpr_group1, tpr_group1)
print(auc_group1)

print('The confusion matrix and auc of the testing set at the lesion level ==== SubGroup4 -type4-2 ====  line-2856: ')
confusion_test_lesion_level = confusion_matrix(Test_Lesion_Label_SubGroup4_GT,
                                               Test_Lesion_Label_SubGroup4_Radiologist_type4_2)
print(confusion_test_lesion_level)
fpr_group1, tpr_group1, _ = roc_curve(Test_Lesion_Label_SubGroup4_GT, Test_Lesion_Label_SubGroup4_Radiologist_type4_2)
auc_group1 = auc(fpr_group1, tpr_group1)
print(auc_group1)

print('The confusion matrix and auc of the testing set at the lesion level ==== SubGroup4 -type4-3 ====  line-2863: ')
confusion_test_lesion_level = confusion_matrix(Test_Lesion_Label_SubGroup4_GT,
                                               Test_Lesion_Label_SubGroup4_Radiologist_type4_3)
print(confusion_test_lesion_level)
fpr_group1, tpr_group1, _ = roc_curve(Test_Lesion_Label_SubGroup4_GT, Test_Lesion_Label_SubGroup4_Radiologist_type4_3)
auc_group1 = auc(fpr_group1, tpr_group1)
print(auc_group1)

print('The confusion matrix and auc of the testing set at the lesion level ==== SubGroup4 -type5-2 ====  line-2856: ')
confusion_test_lesion_level = confusion_matrix(Test_Lesion_Label_SubGroup4_GT,
                                               Test_Lesion_Label_SubGroup4_Radiologist_type5_2)
print(confusion_test_lesion_level)
fpr_group1, tpr_group1, _ = roc_curve(Test_Lesion_Label_SubGroup4_GT, Test_Lesion_Label_SubGroup4_Radiologist_type5_2)
auc_group1 = auc(fpr_group1, tpr_group1)
print(auc_group1)

print('The confusion matrix and auc of the testing set at the lesion level ==== SubGroup4 -type5-3 ====  line-2863: ')
confusion_test_lesion_level = confusion_matrix(Test_Lesion_Label_SubGroup4_GT,
                                               Test_Lesion_Label_SubGroup4_Radiologist_type5_3)
print(confusion_test_lesion_level)
fpr_group1, tpr_group1, _ = roc_curve(Test_Lesion_Label_SubGroup4_GT, Test_Lesion_Label_SubGroup4_Radiologist_type5_3)
auc_group1 = auc(fpr_group1, tpr_group1)
print(auc_group1)

Test_Patient_Label_SubGroup4_GT, Test_Patient_Label_SubGroup4_Radiologist_type1, \
Test_Patient_Label_SubGroup4_Radiologist_type2, Test_Patient_Label_SubGroup4_Radiologist_type3, \
Test_Patient_Label_SubGroup4_Radiologist_type4, Test_Patient_Label_SubGroup4_Radiologist_type5 = [], [], [], [], [], []

Test_Patient_Label_SubGroup4_Radiologist_type4_2, Test_Patient_Label_SubGroup4_Radiologist_type4_3, \
Test_Patient_Label_SubGroup4_Radiologist_type5_2, Test_Patient_Label_SubGroup4_Radiologist_type5_3 = [], [], [], []
            
for i in range(len(Test_ID_Patient_Label_SubGroup4_GT)):
    Test_Patient_Label_SubGroup4_GT.append(Test_ID_Patient_Label_SubGroup4_GT[i][1])
    Test_Patient_Label_SubGroup4_Radiologist_type1.append(Test_ID_Patient_Label_SubGroup4_Radiologist_type1[i][1])
    Test_Patient_Label_SubGroup4_Radiologist_type2.append(Test_ID_Patient_Label_SubGroup4_Radiologist_type2[i][1])
    Test_Patient_Label_SubGroup4_Radiologist_type3.append(Test_ID_Patient_Label_SubGroup4_Radiologist_type3[i][1])
    Test_Patient_Label_SubGroup4_Radiologist_type4.append(Test_ID_Patient_Label_SubGroup4_Radiologist_type4[i][1])
    Test_Patient_Label_SubGroup4_Radiologist_type5.append(Test_ID_Patient_Label_SubGroup4_Radiologist_type5[i][1])

    Test_Patient_Label_SubGroup4_Radiologist_type4_2.append(Test_ID_Patient_Label_SubGroup4_Radiologist_type4_2[i][1])
    Test_Patient_Label_SubGroup4_Radiologist_type4_3.append(Test_ID_Patient_Label_SubGroup4_Radiologist_type4_3[i][1])
    Test_Patient_Label_SubGroup4_Radiologist_type5_2.append(Test_ID_Patient_Label_SubGroup4_Radiologist_type5_2[i][1])
    Test_Patient_Label_SubGroup4_Radiologist_type5_3.append(Test_ID_Patient_Label_SubGroup4_Radiologist_type5_3[i][1])


print(Test_Patient_Label_SubGroup4_GT)
print('*-'*50)
print(Test_Patient_Label_SubGroup4_Radiologist_type1)
print('*#'*50)
print(Test_Patient_Label_SubGroup4_Radiologist_type2)
print('*<'*50)
print(Test_Patient_Label_SubGroup4_Radiologist_type3)

print('The confusion matrix and auc of testing set at the patient level =======SubGroup4 type 1 ======  - line-3058: ')
confusion_test_patient_level_excel = confusion_matrix(Test_Patient_Label_SubGroup4_GT,
                                                      Test_Patient_Label_SubGroup4_Radiologist_type1)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_group1, tpr_group1, _ = roc_curve(Test_Patient_Label_SubGroup4_GT, Test_Patient_Label_SubGroup4_Radiologist_type1)
auc_group1 = auc(fpr_group1, tpr_group1)
print(auc_group1)

print('The confusion matrix and auc of testing set at the patient level =======SubGroup4 type 2 ======  - line-3066: ')
confusion_test_patient_level_excel = confusion_matrix(Test_Patient_Label_SubGroup4_GT,
                                                      Test_Patient_Label_SubGroup4_Radiologist_type2)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_group1, tpr_group1, _ = roc_curve(Test_Patient_Label_SubGroup4_GT, Test_Patient_Label_SubGroup4_Radiologist_type2)
auc_group1 = auc(fpr_group1, tpr_group1)
print(auc_group1)

print('The confusion matrix and auc of testing set at the patient level =======SubGroup4 type 3 ======  - line-3074: ')
confusion_test_patient_level_excel = confusion_matrix(Test_Patient_Label_SubGroup4_GT,
                                                      Test_Patient_Label_SubGroup4_Radiologist_type3)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_group1, tpr_group1, _ = roc_curve(Test_Patient_Label_SubGroup4_GT, Test_Patient_Label_SubGroup4_Radiologist_type3)
auc_group1 = auc(fpr_group1, tpr_group1)
print(auc_group1)

print('The confusion matrix and auc of testing set at the patient level =======SubGroup4 type 4 ======  - line-3082: ')
confusion_test_patient_level_excel = confusion_matrix(Test_Patient_Label_SubGroup4_GT,
                                                      Test_Patient_Label_SubGroup4_Radiologist_type4)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_group1, tpr_group1, _ = roc_curve(Test_Patient_Label_SubGroup4_GT, Test_Patient_Label_SubGroup4_Radiologist_type4)
auc_group1 = auc(fpr_group1, tpr_group1)
print(auc_group1)

print('The confusion matrix and auc of testing set at the patient level =======SubGroup4 type 5 ======  - line-3090: ')
confusion_test_patient_level_excel = confusion_matrix(Test_Patient_Label_SubGroup4_GT,
                                                      Test_Patient_Label_SubGroup4_Radiologist_type5)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_group1, tpr_group1, _ = roc_curve(Test_Patient_Label_SubGroup4_GT, Test_Patient_Label_SubGroup4_Radiologist_type5)
auc_group1 = auc(fpr_group1, tpr_group1)
print(auc_group1)
#####################
print('The confusion matrix and auc of testing set at the patient level =======SubGroup4 type 4_2 ====== - line-3066: ')
confusion_test_patient_level_excel = confusion_matrix(Test_Patient_Label_SubGroup4_GT,
                                                      Test_Patient_Label_SubGroup4_Radiologist_type4_2)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_group1, tpr_group1, _ = roc_curve(Test_Patient_Label_SubGroup4_GT, Test_Patient_Label_SubGroup4_Radiologist_type4_2)
auc_group1 = auc(fpr_group1, tpr_group1)
print(auc_group1)

print('The confusion matrix and auc of testing set at the patient level =======SubGroup4 type 4_3 ====== - line-3074: ')
confusion_test_patient_level_excel = confusion_matrix(Test_Patient_Label_SubGroup4_GT,
                                                      Test_Patient_Label_SubGroup4_Radiologist_type4_3)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_group1, tpr_group1, _ = roc_curve(Test_Patient_Label_SubGroup4_GT, Test_Patient_Label_SubGroup4_Radiologist_type4_3)
auc_group1 = auc(fpr_group1, tpr_group1)
print(auc_group1)

print('The confusion matrix and auc of testing set at the patient level =======SubGroup4 type 5_2 =====  - line-3082: ')
confusion_test_patient_level_excel = confusion_matrix(Test_Patient_Label_SubGroup4_GT,
                                                      Test_Patient_Label_SubGroup4_Radiologist_type5_2)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_group1, tpr_group1, _ = roc_curve(Test_Patient_Label_SubGroup4_GT, Test_Patient_Label_SubGroup4_Radiologist_type5_2)
auc_group1 = auc(fpr_group1, tpr_group1)
print(auc_group1)

print('The confusion matrix and auc of testing set at the patient level =======SubGroup4 type 5_3 =====  - line-3090: ')
confusion_test_patient_level_excel = confusion_matrix(Test_Patient_Label_SubGroup4_GT,
                                                      Test_Patient_Label_SubGroup4_Radiologist_type5_3)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_group1, tpr_group1, _ = roc_curve(Test_Patient_Label_SubGroup4_GT, Test_Patient_Label_SubGroup4_Radiologist_type5_3)
auc_group1 = auc(fpr_group1, tpr_group1)
print(auc_group1)


"""
print('The confusion matrix and auc of testing set at the patient level - line-647: ')
confusion_test_patient_level = confusion_matrix(Test_Patient_Label_GT, Test_Patient_Label_Radiologist)
print(confusion_test_patient_level, np.sum(confusion_test_patient_level))
fpr_group1, tpr_group1, _ = roc_curve(Test_Patient_Label_GT, Test_Patient_Label_Radiologist)
auc_group1 = auc(fpr_group1, tpr_group1)
print(auc_group1)
"""
"""
print("Patient ID Test of Ground-Truth and Prediction from Radiologist at the patient level:")
for i in range(len(Test_ID_Patient_Label_GT)):
    GT = Test_ID_Patient_Label_GT_From_Excel[i][1]
    Pre = Test_ID_Patient_Label_Radiologist[i][1]
    if GT == Pre:
        continue
    else:
        print(Test_ID_Patient_Label_GT_From_Excel[i][0], GT, Pre)
    # print(Test_ID_Patient_Label_GT_From_Excel[i], Test_ID_Patient_Label_Radiologist[i])
"""
print("Patient ID Test of Ground-Truth and Prediction from Radiologist at the lesion level:")

Correct_Patient_ID_List = ['HKU_0132_P3',
                           'ID_0225a_P3',
                           'ID_0468_P3',
                           'ID_0848_P3',
                           'QMH_0021_P3',
                           'QMH_0067_P3',
                           'QMH_0147_P3',
                           'SZH_0005_P3',
                           'SZH_0050_P3',
                           'SZH_0114_P3',
                           # 'ID_0509_P3',
                           # 'ID_0758_P3',
                           ]

Test_ID_Lesion_Label_Model = []
CNT = 0

for i in range(len(Test_ID_Patient_Label_SubGroup4_GT)):
    GT = Test_ID_Patient_Label_SubGroup4_GT[i][1]
    Pre = Test_ID_Lesion_Label_SubGroup4_Radiologist_type4_2[i][1]
    if GT == Pre:
        Test_ID_Lesion_Label_Model.append([Test_ID_Patient_Label_SubGroup4_GT[i][0], GT])
    else:
        CNT += 1
        patient_id = Test_ID_Patient_Label_SubGroup4_GT[i][0]
        if patient_id in Correct_Patient_ID_List:
            Test_ID_Lesion_Label_Model.append([Test_ID_Patient_Label_SubGroup4_GT[i][0], GT])
        else:
            Test_ID_Lesion_Label_Model.append([Test_ID_Patient_Label_SubGroup4_GT[i][0], Pre])
        # print(Test_ID_Lesion_Label_GT[i][0], GT, Test_ID_Lesion_Label_Radiologist[i][0], Pre, CNT)
    # print(Test_ID_Patient_Label_GT_From_Excel[i], Test_ID_Patient_Label_Radiologist[i])
print(CNT)

Test_Lesion_Label_Model = []
for i in range(len(Test_ID_Lesion_Label_Model)):
    Test_Lesion_Label_Model.append(Test_ID_Lesion_Label_Model[i][1])
