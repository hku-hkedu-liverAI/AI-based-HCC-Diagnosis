import pandas as pd
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import nibabel as nib
from sklearn.metrics import roc_curve, auc, confusion_matrix
import os


def _bbox_3D(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))
    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    return rmin, rmax, cmin, cmax, zmin, zmax


def read_nifti(file_dir):
    file = nib.load(file_dir)
    file_voxels = file.get_data()
    file_voxels = np.transpose(file_voxels, (2, 0, 1))
    file_hdr = file.header
    file_affine = file._affine
    return file_voxels, file_hdr, file_affine


data_record_excel_file = '/home/ra1/original/Patient Statistics and Size of Observation_excluded OT cases.xlsx'
data_record_train_test = '/home/ra1/Documents/2D_3D_Classification_Segment_Split_Finalized_31_Oct_Check_3_Nov.txt'

mask_update_id_707_p3 = '/home/ra1/original/Finalized_Mask/Finalised Mask Update 2021_04_23/ID_0707_P3_2022_03_10.nii.gz'
mask_update_id_761_p3 = '/home/ra1/original/Finalized_Mask/Finalised Mask Update 2021_04_23/ID_0761_P3_2022_03_10.nii.gz'

patient_mask_fullpath_lists = []
patient_mask_fullpath_lists.append(mask_update_id_707_p3)
patient_mask_fullpath_lists.append(mask_update_id_761_p3)

# ID_0707_P3_2022_03_10 9 20.654481755994055 11.257999956607819 13.855999946594238 10.3863525390625 13.855999946594238
# ID_0761_P3_2022_03_10 2 253.99534445084 120.05600237846375 130.5260025858879 181.832763671875 181.832763671875

# file = open('/home/ra1/Desktop/Lesion_Size_Computation_Update_Internal.txt', "w")
mask_fullpath_list = [
    '/home/ra1/original/Finalized_Mask/Finalised Mask Update 2021_04_23/PYN_Part2_Mask/ID_0560_P3.nii.gz',
    '/home/ra1/original/Finalized_Mask/Finalised Mask Update 2021_04_23/PYN_Part2_Mask/ID_0848_P3.nii.gz'
]

Test_Record = []
record_lines = open(data_record_train_test, 'r')
records = record_lines.readlines()
print(len(records))

CNT = 0
Test_Case_ID_Label = []
Test_Case_ID = []
for line in records:
    CNT += 1
    if CNT > 1598:
        Case_ID_Label = line.strip().split('ata/')[1]
        Case_ID = Case_ID_Label.split(' ')[0]
        Case_Label = Case_ID_Label.split(' ')[1]
        Test_Case_ID_Label.append([Case_ID, Case_Label])   # Test_Case_ID_Label stores patient id and label
        Test_Case_ID.append(Case_ID)                       # Test_Case_ID only stores the patient id information

hcc_lesion_record = pd.read_excel(data_record_excel_file, 'HCC_lesions')
non_hcc_lesion_record = pd.read_excel(data_record_excel_file, 'NonHCC_lesions')
Test_Patient_ID_Radiologist_type8 = np.load('/home/ra1/original/Test_ID_Patient_Label_Radiologist_type8.npy')
Test_Lesion_ID_Radiologist_type8 = np.load('/home/ra1/original/Test_ID_Lesion_Label_Radiologist_type8.npy')

# Test_Case_Patient_Less_Than_Five: the test set at the patient level
# Test_Case_Lesion_Less_Than_Five: the test set at the lesion level
Test_Case_Patient_Less_Than_Five, Test_Case_Lesion_Less_Than_Five = [], []

Test_Record_Five_unique, Test_HCC_Record_Five_unique, Test_NonHCC_Record_Five_unique = [], [], []
Test_Record_Five, Test_HCC_Record_Five, Test_NonHCC_Record_Five = [], [], []
Test_Case_Patient_Less_Than_Two, Test_Case_Lesion_Less_Than_Two = [], []
Test_Record_Two, Test_HCC_Record_Two, Test_NonHCC_Record_Two = [], [], []

LiRads_Record_Five = []

for idx in hcc_lesion_record.index:
    case_id = hcc_lesion_record['ID'][idx]
    lesion_size = hcc_lesion_record['max'][idx]
    if case_id in Test_Case_ID and lesion_size < 20:  # 50:
        inndex = Test_Case_ID.index(case_id)

        for ii in range(len(Test_Lesion_ID_Radiologist_type8)):
            test_id = Test_Lesion_ID_Radiologist_type8[ii][0]
            if case_id == test_id:
                test_lesion_lirads = Test_Lesion_ID_Radiologist_type8[ii][1]

        Test_Case_Lesion_Less_Than_Five.append([case_id, Test_Case_ID_Label[inndex][1], test_lesion_lirads])

        # Test_Case_Patient_Less_Than_Five: [patient id, ground-truth label, LiRads--> Label]
        if case_id not in Test_Record_Five_unique:
            Test_Record_Five_unique.append(case_id)
            Test_Case_Patient_Less_Than_Five.append([case_id, Test_Case_ID_Label[inndex][1], test_lesion_lirads])
        if case_id not in Test_HCC_Record_Five_unique:
            Test_HCC_Record_Five_unique.append(case_id)

        Test_Record_Five.append(case_id)
        Test_HCC_Record_Five.append(case_id)
    else:
        continue

# There are 97 HCC patients with 117 HCC Lesions for lesion size < 50mm
print('The number of HCC Patient with lesion < 20mm and the number of HCC Lesion <20mm are: ')
print(len(Test_HCC_Record_Five), len(Test_HCC_Record_Five_unique), len(Test_Case_Lesion_Less_Than_Five),
      len(Test_Case_Patient_Less_Than_Five))

for idx in non_hcc_lesion_record.index:
    case_id = non_hcc_lesion_record['ID'][idx]
    lesion_size = non_hcc_lesion_record['max'][idx]
    if case_id in Test_Case_ID and lesion_size < 20: # 50:
        inndex = Test_Case_ID.index(case_id)
        type = non_hcc_lesion_record['type'][idx]
        for ii in range(len(Test_Lesion_ID_Radiologist_type8)):
            test_id = Test_Lesion_ID_Radiologist_type8[ii][0]
            if case_id == test_id:
                test_lesion_lirads = Test_Lesion_ID_Radiologist_type8[ii][1]

        Test_Case_Lesion_Less_Than_Five.append([case_id, Test_Case_ID_Label[inndex][1], test_lesion_lirads])

        if case_id not in Test_Record_Five_unique:
            Test_Record_Five_unique.append(case_id)
            Test_Case_Patient_Less_Than_Five.append([case_id, Test_Case_ID_Label[inndex][1], test_lesion_lirads])
        if case_id not in Test_NonHCC_Record_Five_unique:
            Test_NonHCC_Record_Five_unique.append(case_id)

        Test_Record_Five.append(case_id)
        Test_NonHCC_Record_Five.append(case_id)
    else:
        continue

# There are 428 Non-HCC with 751 lesions with lesion size < 50mm
print('The number of Non-HCC Patient with lesion < 20mm and the number of Non-HCC Lesion <20mm are: ')
print(len(Test_NonHCC_Record_Five), len(Test_NonHCC_Record_Five_unique),
      len(Test_Case_Lesion_Less_Than_Five)-len(Test_HCC_Record_Five),
      len(Test_Case_Patient_Less_Than_Five) - len(Test_HCC_Record_Five_unique), len(Test_Record_Five_unique))

Test_Record_Five_unique = np.array(Test_Record_Five_unique).reshape((len(Test_Record_Five_unique), ))

Test_5_Patient_ID = np.load('/home/ra1/original/Small_Lesion_5cm.npy')
Test_2_Patient_ID = np.load('/home/ra1/original/Small_Lesion_2cm.npy')
Test_2_5_Patient_ID = np.load('/home/ra1/original/Small_Lesion_2cm_5cm.npy')

print(len(Test_5_Patient_ID), len(Test_2_Patient_ID), len(Test_2_5_Patient_ID))
Test_5_Patient_List, Test_2_Patient_List, Test_Patient_2_5_List = [], [], []
for i in range(len(Test_2_Patient_ID)):
    Test_2_Patient_List.append(Test_2_Patient_ID[i])
for i in range(len(Test_5_Patient_ID)):
    Test_5_Patient_List.append(Test_5_Patient_ID[i])
for i in range(len(Test_2_5_Patient_ID)):
    Test_Patient_2_5_List.append(Test_2_5_Patient_ID[i])

print(len(Test_5_Patient_List), len(np.unique(np.array(Test_5_Patient_List))), "Line-155")
print(len(Test_2_Patient_List), len(np.unique(np.array(Test_2_Patient_List))), "Line-156")
print(len(Test_Patient_2_5_List), len(np.unique(np.array(Test_Patient_2_5_List))), "Line-157")

Test_Patient_GT_Five, Test_Patient_Radiologist_Five, Test_Lesion_GT_Five, Test_Lesion_Radiologist_Five = [], [], [], []
CNT = 0
print(len(Test_Case_Patient_Less_Than_Five), "Line-161")
for i in Test_Case_Patient_Less_Than_Five:

    Test_Patient_GT_Five.append(int(i[1]))
    Test_Patient_Radiologist_Five.append(int(i[2]))
    # if CNT > 21 and int(i[1]) == 1:
    #    print(i)

    if int(i[1]) == 1 and int(i[2]) == 0:
        CNT += 1
        print(i, CNT,  "Line-170")

for i in Test_Case_Lesion_Less_Than_Five:
    Test_Lesion_GT_Five.append(int(i[1]))
    Test_Lesion_Radiologist_Five.append(int(i[2]))

print(len(Test_Case_Patient_Less_Than_Five), len(Test_Case_Lesion_Less_Than_Five))
print(len(Test_Patient_GT_Five), len(Test_Patient_Radiologist_Five))
print(len(Test_Lesion_GT_Five), len(Test_Lesion_Radiologist_Five))

print('The confusion matrix and auc of testing set for lesion size < 50mm at the patient level')
confusion_test_patient_level_excel = confusion_matrix(Test_Patient_GT_Five, Test_Patient_Radiologist_Five)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_group1, tpr_group1, _ = roc_curve(Test_Patient_GT_Five, Test_Patient_Radiologist_Five)
auc_group1 = auc(fpr_group1, tpr_group1)
print(auc_group1)

print('The confusion matrix and auc of testing set for lesion size < 50mm at the lesion level')
confusion_test_patient_level_excel = confusion_matrix(Test_Lesion_GT_Five, Test_Lesion_Radiologist_Five)
print(confusion_test_patient_level_excel, np.sum(confusion_test_patient_level_excel), "Line-642")
fpr_group1, tpr_group1, _ = roc_curve(Test_Lesion_GT_Five, Test_Lesion_Radiologist_Five)
auc_group1 = auc(fpr_group1, tpr_group1)
print(auc_group1)


"""

for idx in hcc_lesion_record.index:
    case_id = hcc_lesion_record['ID'][idx]
    lesion_size = hcc_lesion_record['max'][idx]
    if case_id in Test_Case_ID and lesion_size < 20:
        inndex = Test_Case_ID.index(case_id)
        Test_Case_ID_Lesion_Less_Than_Two.append([case_id, Test_Case_ID_Label[inndex][1]])
        if case_id not in Test_Case_Lesion_Less_Than_Two:
            Test_Case_Lesion_Less_Than_Two.append([case_id, Test_Case_ID_Label[inndex][1]])
        else:
            print(case_id, Test_Case_ID_Label[inndex][1])
    else:
        continue

    if case_id in Test_Case_ID:
        Test_Record_Two.append(case_id)
        Test_HCC_Record_Two.append(case_id)

for idx in non_hcc_lesion_record.index:
    case_id = non_hcc_lesion_record['ID'][idx]
    lesion_size = non_hcc_lesion_record['max'][idx]
    if case_id in Test_Case_ID and lesion_size < 20:
        inndex = Test_Case_ID.index(case_id)
        Test_Case_ID_Lesion_Less_Than_Two.append([case_id, Test_Case_ID_Label[inndex][1]])
        if case_id not in Test_Case_Lesion_Less_Than_Two:
            Test_Case_Lesion_Less_Than_Two.append([case_id, Test_Case_ID_Label[inndex][1]])
        else:
            print(case_id, Test_Case_ID_Label[inndex][1])
    else:
        continue

    if case_id in Test_Case_ID:
        Test_Record_Two.append(case_id)
        Test_NonHCC_Record_Two.append(case_id)

print(len(Test_Case_ID_Lesion_Less_Than_Five), "The number of lesions < 50mm", len(Test_Case_ID_Lesion_Less_Than_Five))
print(len(Test_Case_ID_Lesion_Less_Than_Two), "The number of lesions < 20mm", len(Test_Case_Lesion_Less_Than_Two))

print(len(Test_Record_Five), len(Test_Record_Two))
print('The numbers of patients <5cm and 2cm are: ', len(np.unique(Test_Record_Five)), len(np.unique(Test_Record_Two)))
print(len(Test_HCC_Record_Five), len(Test_NonHCC_Record_Five))
print(len(Test_HCC_Record_Two), len(Test_NonHCC_Record_Two))

for case in Test_HCC_Record_Five:
    if case in Test_NonHCC_Record_Five:
        print(case, "Line-178")
    else:
        continue
print(len(np.unique(Test_HCC_Record_Five)), len(np.unique(Test_NonHCC_Record_Five)))
print(len(np.unique(Test_HCC_Record_Two)), len(np.unique(Test_NonHCC_Record_Two)))
"""