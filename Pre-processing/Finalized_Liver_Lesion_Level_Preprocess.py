import numpy as np
import pandas as pd
import os
from utils_library_finalized_preprocess import read_dicom, read_mask, obtain_hcc_nonhcc, vol_ct_mask_choose,\
    input_data_reshape, compute_maximum_nonzero_element
import SimpleITK as sitk
from PIL import Image
# ******************************************************************************************************************** #
# Excluding Cases according to Professor Seto.
ID_Exclude_List_by_Seto = ['ID_0862a_P3', 'ID_0862b_P3', 'ID_0584_P3', 'HKU_0340_P3', 'HKU_0503_P3', 'ID_0816_P3',
                           'QMH_0012_P3', 'QMH_0020a_P2']
data_id_excel_file_fullpath = '/home/ra1/original/LiRad lesions combine testing and training.xlsx'
xls = pd.ExcelFile(data_id_excel_file_fullpath)
ALL_Lesions = pd.read_excel(xls, 'All lesions')
All_Lesions_List = []
Patient_ID_List, Patient_ID_List_Phase2, Patient_ID_List_Phase3 = [], [], []
Patient_ID_List_Clean, Patient_ID_List_Clean_Phase2, Patient_ID_List_Clean_Phase3 = [], [], []
Patient_ID_List_Clean_seto, Patient_ID_List_Clean_Phase2_seto, Patient_ID_List_Clean_Phase3_seto = [], [], []
Exclude_Lesion_Counter, Include_Lesion_Counter, Include_Lesion_Counter_Clean_seto = 0, 0, 0
Patient_ID_Lesion_Type_LiRad_Label = []
Patient_ID_Tace_Record = []
Patient_ID_Tace_Record_Clean = []
Patient_ID_RFA_Record_Clean = []

for i in ALL_Lesions.index:
    id = ALL_Lesions['Case'][i]
    if not (id in Patient_ID_List):
        Patient_ID_List.append(id)
    if not (id in Patient_ID_List_Phase2) and 'P2' in id:
        Patient_ID_List_Phase2.append(id)
    if not (id in Patient_ID_List_Phase3) and 'P3' in id:
        Patient_ID_List_Phase3.append(id)

    if id in ID_Exclude_List_by_Seto and ('P3' in id or 'P2' in id):
        Include_Lesion_Counter_Clean_seto += 1

    Lesion_LiRad = ALL_Lesions['LiRad for Cao'][i]
    Lesion_Type = ALL_Lesions['type'][i]
    if type(Lesion_LiRad) == int or (type(Lesion_LiRad) == str and Lesion_LiRad == 'LR_M'):
        Include_Lesion_Counter += 1
        if not (id in Patient_ID_List_Clean) and not(id in ID_Exclude_List_by_Seto):
            Patient_ID_List_Clean.append(id)
        if not (id in Patient_ID_List_Clean_Phase2) and 'P2' in id:
            Patient_ID_List_Clean_Phase2.append(id)
        if not (id in Patient_ID_List_Clean_Phase3) and 'P3' in id:
            Patient_ID_List_Clean_Phase3.append(id)

        if not (id in Patient_ID_List_Clean_seto) and not (id in ID_Exclude_List_by_Seto):
            Patient_ID_List_Clean_seto.append(id)
        if not (id in Patient_ID_List_Clean_Phase2_seto) and 'P2' in id and not (id in ID_Exclude_List_by_Seto):
            Patient_ID_List_Clean_Phase2_seto.append(id)
        if not (id in Patient_ID_List_Clean_Phase3_seto) and 'P3' in id and not (id in ID_Exclude_List_by_Seto):
            Patient_ID_List_Clean_Phase3_seto.append(id)

        if not (id in ID_Exclude_List_by_Seto):
            Patient_ID_Lesion_Type_LiRad_Label.append([id, int(ALL_Lesions['type'][i]), Lesion_LiRad])
    else:
        Exclude_Lesion_Counter += 1

    if int(Lesion_Type) == 6 and type(Lesion_LiRad) == int:
        Patient_ID_Tace_Record.append(id)

    if int(Lesion_Type) == 6 and not(id in Patient_ID_Tace_Record_Clean):
        Patient_ID_Tace_Record_Clean.append(id)      # 89 cases

    if int(Lesion_Type) == 14 and not (id in Patient_ID_RFA_Record_Clean):
        Patient_ID_RFA_Record_Clean.append(id)       # 73 cases
print(Patient_ID_List_Clean.index('SZF_0004_P3'))
print(len(Patient_ID_Lesion_Type_LiRad_Label), len(Patient_ID_List_Clean), "Line-71")                              # 3765 Lesions
# First, Remove Tace
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
print(len(Patient_ID_Lesion_Type_LiRad_Label_Tace_Excluding), "Line-87")               # 3733 lesions
print(len(Patient_ID_List_Clean_Tace_Excluding), "Line-88")                            # 2325 patients
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

print(len(Patient_ID_Lesion_Type_LiRad_Label_Tace_RFA_Excluding), "Line-105")           # 3675 lesions
print(len(Patient_ID_List_Clean_Tace_RFA_Excluding), "Line-106")                       # 2288 patients
# The number of lesions before cleaning is 4060.
# After excluding, the number of lesions is 3773. For cases cleaned by seto's operation, there are 8 lesions in total
Patient_ID_Lesion_Type_LiRad_Label_Corrected = []
for case_info in Patient_ID_Lesion_Type_LiRad_Label_Tace_RFA_Excluding:
    case_info_id = case_info[0]
    # print(case_info_id, 'SZF_0004_P3', case_info_id == 'SZF_0004_P3')
    if case_info_id == 'SZF_0004_P3':
        print(case_info_id, case_info, case_info[0], case_info[1:])
    if case_info_id == 'SZF_0004_P3':
        Patient_ID_Lesion_Type_LiRad_Label_Corrected.append(['SZF_0004_P4', case_info[1], case_info[2]])
    elif case_info_id == 'SZF_0023_P3':
        Patient_ID_Lesion_Type_LiRad_Label_Corrected.append(['SZF_0023_P4', case_info[1], case_info[2]])
    else:
        Patient_ID_Lesion_Type_LiRad_Label_Corrected.append(case_info)
for case_info in Patient_ID_Lesion_Type_LiRad_Label_Corrected:
    case_id = case_info[0]
    # print(case_info, len(case_id), len('SZF_0004_P4'))
    if case_id == 'SZF_0004_P4':
        print(case_id, case_info)
print(len(Patient_ID_Lesion_Type_LiRad_Label_Corrected), len(Patient_ID_List_Clean_Tace_RFA_Excluding), "Line-121")

Patient_ID_Lesion_Type_LiRad_Label_Corrected_Check_ID = []
for i in range(len(Patient_ID_Lesion_Type_LiRad_Label_Corrected)):
    patient_id = Patient_ID_Lesion_Type_LiRad_Label_Corrected[i][0]
    if patient_id in Patient_ID_Lesion_Type_LiRad_Label_Corrected_Check_ID:
        continue
    else:
        Patient_ID_Lesion_Type_LiRad_Label_Corrected_Check_ID.append(patient_id)
print(len(Patient_ID_Lesion_Type_LiRad_Label_Corrected_Check_ID), len(Patient_ID_List_Clean_Tace_RFA_Excluding), "Line-135")
print(III)
# ******************************************************************************************************************** #
# ************* Here prepare the mask files that are updated by Homing, after referring to original report *********** #
updated_masks_finalized_rootpath = '/home/ra1/original/Finalized_Mask/Mask in lesions level'
file_folder = os.listdir(updated_masks_finalized_rootpath)
phase2_mask_file_list = []
phase3_mask_file_list = []
phase4_mask_file_list = []
Need_Recontoured_Cases_Train = []
Need_Recontoured_Cases_Test = []
for file_or_folder in file_folder:
    file_or_folder_path = os.path.join(updated_masks_finalized_rootpath, file_or_folder)
    if os.path.isfile(file_or_folder_path):
        if 'P2' in file_or_folder:
            phase2_mask_file_list.append(file_or_folder_path)
        elif 'P3' in file_or_folder:
            phase3_mask_file_list.append(file_or_folder_path)
        elif 'P4' in file_or_folder:
            phase4_mask_file_list.append(file_or_folder_path)
    elif os.path.isdir(file_or_folder_path):
        file_list = os.listdir(file_or_folder_path)
        if file_or_folder == 'Training':
            for file in file_list:
                file_fullpath = os.path.join(file_or_folder_path, file)
                if os.path.isfile(file_fullpath):
                    if 'P2' in file:
                        phase2_mask_file_list.append(file_fullpath)
                    elif 'P3' in file:
                        phase3_mask_file_list.append(file_fullpath)
                    elif 'P4' in file:
                        phase4_mask_file_list.append(file_fullpath)
        elif file_or_folder == 'Testing':
            for file in file_list:
                file_fullpath = os.path.join(file_or_folder_path, file)
                if os.path.isfile(file_fullpath):
                    if 'P2' in file:
                        phase2_mask_file_list.append(file_fullpath)
                    elif 'P3' in file:
                        phase3_mask_file_list.append(file_fullpath)
                    elif 'P4' in file:
                        phase4_mask_file_list.append(file_fullpath)
# Phase 2, Phase3 and Phase 4: 27, 175, 1 =>
print(len(phase2_mask_file_list), len(phase3_mask_file_list), len(phase4_mask_file_list))
# Update the names of mask files in Mask in lesions level
updated_mask_fullpath_file_list = []
for mask_path in phase2_mask_file_list:
    updated_mask_fullpath_file_list.append(mask_path)
for mask_path in phase3_mask_file_list:
    updated_mask_fullpath_file_list.append(mask_path)
for mask_path in phase4_mask_file_list:
    updated_mask_fullpath_file_list.append(mask_path)
print(len(updated_mask_fullpath_file_list), "The number of updated mask files!!!")   # There are 203 mask files in total

# ******************************************************************************************************************** #
# ************* Here prepare CT scan images and the corresponding mask files supported by Homing  ******************** #
mask_save_rootpath = '/home/ra1/original/Finalized_Mask/Finalised Mask Update 2021_04_23'
folder_list = os.listdir(mask_save_rootpath)
mask_folder_path_list = []
for i in folder_list:
    if 'Mask' in i:
        mask_folder_fullpath = os.path.join(mask_save_rootpath, i)
        mask_folder_path_list.append(mask_folder_fullpath)
    else:
        continue

all_mask_fullpath_list, mask_path_list_phase1, mask_path_list_phase2, mask_path_list_phase3,\
                                                                            mask_path_list_phase4 = [], [], [], [], []

for i in mask_folder_path_list:
    mask_list = os.listdir(i)
    for j in mask_list:
        mask_fullpath = os.path.join(i, j)
        if os.path.isfile(mask_fullpath):
            if 'P1' in mask_fullpath:
                mask_path_list_phase1.append(mask_fullpath)
                all_mask_fullpath_list.append(mask_fullpath)
            elif 'P2' in mask_fullpath:
                mask_path_list_phase2.append(mask_fullpath)
                all_mask_fullpath_list.append(mask_fullpath)
            elif 'P3' in mask_fullpath:
                mask_path_list_phase3.append(mask_fullpath)
                all_mask_fullpath_list.append(mask_fullpath)
            elif 'P4' in mask_fullpath:
                mask_path_list_phase4.append(mask_fullpath)
                all_mask_fullpath_list.append(mask_fullpath)

# Original CT Masks, 388 cases belong to P1, 2526 cases belong to P2, 2535 cases belong to P3 and 422 cases belong to P4
print("The total number of masked files is", len(all_mask_fullpath_list), len(mask_path_list_phase1) +
      len(mask_path_list_phase2) + len(mask_path_list_phase3) + len(mask_path_list_phase4))
print('The numbers of Phase 1, Phase 2, Phase 3 and Phase 4 mask files are %d, %d, %d and %d.'
      % (len(mask_path_list_phase1), len(mask_path_list_phase2), len(mask_path_list_phase3), len(mask_path_list_phase4)))

Mask_File_FullPath_List = []
for mask_path in mask_path_list_phase3:
    case_id = os.path.basename(mask_path).split('.nii')[0]
    for selected_case in Patient_ID_Lesion_Type_LiRad_Label_Corrected:
        case_id_clean = selected_case[0]
        # print(case_id, selected_case, case_id_clean)
        if case_id_clean == case_id and not(mask_path in Mask_File_FullPath_List):
            Mask_File_FullPath_List.append(mask_path)
        else:
            continue

for mask_path in mask_path_list_phase2:
    case_id = os.path.basename(mask_path).split('.nii')[0]
    for selected_case in Patient_ID_Lesion_Type_LiRad_Label_Corrected:
        case_id_clean = selected_case[0]
        if case_id_clean == case_id and not(mask_path in Mask_File_FullPath_List):
            Mask_File_FullPath_List.append(mask_path)
        else:
            continue

for mask_path in mask_path_list_phase4:
    case_id = os.path.basename(mask_path).split('.nii')[0]
    for selected_case in Patient_ID_Lesion_Type_LiRad_Label_Corrected:
        case_id_clean = selected_case[0]
        if case_id_clean == case_id and not(mask_path in Mask_File_FullPath_List):
            Mask_File_FullPath_List.append(mask_path)
        else:
            continue

# It includes 2345 cases in Mask_File_FullPath_List after cleaninng some cases
print(len(Mask_File_FullPath_List))

# ******************************************************************************************************************** #
# **************************    Here we replace the old mask files with new mask files   ***************************** #
Updated_Finalized_Mask_File_FullPath_List, Updated_Finalized_Mask_File_FullPath_List_ID = [], []

for old_mask_path in Mask_File_FullPath_List:
    old_mask_basename = os.path.basename(old_mask_path)
    old_mask_case_id = old_mask_basename.split('.nii')[0]
    for new_mask_path in updated_mask_fullpath_file_list:
        new_mask_basename = os.path.basename(new_mask_path)
        new_mask_case_id = new_mask_basename.split('.nii')[0]
        if new_mask_case_id == old_mask_case_id:
            Updated_Finalized_Mask_File_FullPath_List.append(new_mask_path)
            Updated_Finalized_Mask_File_FullPath_List_ID.append(new_mask_case_id)

for old_mask_path in Mask_File_FullPath_List:
    old_mask_basename = os.path.basename(old_mask_path)
    old_mask_case_id = old_mask_basename.split('.nii')[0]
    if old_mask_case_id in Updated_Finalized_Mask_File_FullPath_List_ID:
        continue
    else:
        Updated_Finalized_Mask_File_FullPath_List.append(old_mask_path)
        Updated_Finalized_Mask_File_FullPath_List_ID.append(old_mask_case_id)

# Obtain the final fullpath lists of mask files for the model training
print(len(Updated_Finalized_Mask_File_FullPath_List), len(Updated_Finalized_Mask_File_FullPath_List_ID))
CNT_PYN, CNT_HKU, CNT_SZH, CNT_KWH, CNT_QEH, CNT_QMH = 0, 0, 0, 0, 0, 0
CNT_PYN_0, CNT_HKU_0, CNT_SZH_0, CNT_KWH_0, CNT_QEH_0, CNT_QMH_0 = 0, 0, 0, 0, 0, 0
Updated_Finalized_CT_Image_File_FullPath_List = []
for mask_path in Updated_Finalized_Mask_File_FullPath_List:
    mask_case_id = os.path.basename(mask_path).split('.ni')[0]
    if "ID" in mask_case_id:
        CNT_PYN_0 += 1
        case_index_split = mask_case_id.split('_')
        if len(case_index_split[1]) == 4:
            case_index = int(case_index_split[1])
        else:
            case_index = int(case_index_split[1][0:-1])

        if case_index <= 547:
            if 'P1' in mask_case_id:
                ct_image_fullpath = os.path.join('/home/ra1/original/PreRegDataAII/Phase1_data', mask_case_id).strip()
            elif 'P2' in mask_case_id:
                ct_image_fullpath = os.path.join('/home/ra1/original/PreRegDataAII/Phase2_data', mask_case_id).strip()
            elif 'P3' in mask_case_id:
                ct_image_fullpath = os.path.join('/home/ra1/original/PreRegDataAII/Phase3_data', mask_case_id).strip()
            else:
                ct_image_fullpath = os.path.join('/home/ra1/original/PreRegDataAII/Phase4_data', mask_case_id).strip()
        else:
            if 'P1' in mask_case_id:
                ct_image_fullpath = os.path.join('/home/ra1/original/PYN_Part2/Phase1_data', mask_case_id).strip()
            elif 'P2' in mask_case_id:
                ct_image_fullpath = os.path.join('/home/ra1/original/PYN_Part2/Phase2_data', mask_case_id).strip()
            elif 'P3' in mask_case_id:
                ct_image_fullpath = os.path.join('/home/ra1/original/PYN_Part2/Phase3_data', mask_case_id).strip()
            else:
                ct_image_fullpath = os.path.join('/home/ra1/original/PYN_Part2/Phase4_data', mask_case_id).strip()

        # ct_image_fullpath = os.path.join(ct_image_fullpath_prefix, mask_case_id)
        if os.path.exists(ct_image_fullpath):
            Updated_Finalized_CT_Image_File_FullPath_List.append(ct_image_fullpath)
            CNT_PYN += 1
        else:
            print(ct_image_fullpath)

    elif 'HKU' in mask_case_id:
        CNT_HKU_0 += 1

        case_index_split = mask_case_id.split('_')
        if len(case_index_split[1]) == 4:
            case_index = int(case_index_split[1])
        else:
            case_index = int(case_index_split[1][0:-1])

        if case_index <= 650:
            if 'P1' in mask_case_id:
                ct_image_fullpath = os.path.join('/home/ra1/original/PreRegData_HKU/Phase1_data', mask_case_id)
            elif 'P2' in mask_case_id:
                ct_image_fullpath = os.path.join('/home/ra1/original/PreRegData_HKU/Phase2_data', mask_case_id)
            elif 'P3' in mask_case_id:
                ct_image_fullpath = os.path.join('/home/ra1/original/PreRegData_HKU/Phase3_data', mask_case_id)
            else:
                ct_image_fullpath = os.path.join('/home/ra1/original/PreRegData_HKU/Phase4_data', mask_case_id)
        else:
            if 'P1' in mask_case_id:
                ct_image_fullpath = os.path.join('/home/ra1/original/PreRegData_HKU/HKU_CholangioCA_CT/Phase1_Data', mask_case_id)
            elif 'P2' in mask_case_id:
                ct_image_fullpath = os.path.join('/home/ra1/original/PreRegData_HKU/HKU_CholangioCA_CT/Phase2_Data', mask_case_id)
            elif 'P3' in mask_case_id:
                ct_image_fullpath = os.path.join('/home/ra1/original/PreRegData_HKU/HKU_CholangioCA_CT/Phase3_Data', mask_case_id)
            else:
                ct_image_fullpath = os.path.join('/home/ra1/original/PreRegData_HKU/HKU_CholangioCA_CT/Phase4_Data', mask_case_id)

        if os.path.exists(ct_image_fullpath):
            Updated_Finalized_CT_Image_File_FullPath_List.append(ct_image_fullpath)
            CNT_HKU += 1
        else:
            print(ct_image_fullpath)

    elif 'SZ' in mask_case_id:
        """
        ct_image_fullpath_pre = '/home/ra1/original/PreRegData_SZH'
        
        if 'P1' in mask_case_id:
            ct_image_fullpath_prefix = os.path.join(ct_image_fullpath_pre, 'Phase1_Data')
        elif 'P2' in mask_case_id:
            ct_image_fullpath_prefix = os.path.join(ct_image_fullpath_pre, 'Phase2_Data')
        elif 'P3' in mask_case_id:
            ct_image_fullpath_prefix = os.path.join(ct_image_fullpath_pre, 'Phase3_Data')
        else:
            ct_image_fullpath_prefix = os.path.join(ct_image_fullpath_pre, 'Phase4_Data')
        ct_image_fullpath = os.path.join(ct_image_fullpath_prefix, mask_case_id)
        Updated_Finalized_CT_Image_File_FullPath_List.append(ct_image_fullpath)
        """
        CNT_SZH_0 += 1
        if 'P1' in mask_case_id:
            ct_image_fullpath = os.path.join('/home/ra1/original/PreRegData_SZH/Phase1_data', mask_case_id)
        elif 'P2' in mask_case_id:
            ct_image_fullpath = os.path.join('/home/ra1/original/PreRegData_SZH/Phase2_data', mask_case_id)
        elif 'P3' in mask_case_id:
            ct_image_fullpath = os.path.join('/home/ra1/original/PreRegData_SZH/Phase3_data', mask_case_id)
        else:
            ct_image_fullpath = os.path.join('/home/ra1/original/PreRegData_SZH/Phase4_data', mask_case_id)

        if os.path.exists(ct_image_fullpath):
            Updated_Finalized_CT_Image_File_FullPath_List.append(ct_image_fullpath)
            CNT_SZH += 1

    elif 'KW' in mask_case_id:
        """
        ct_image_fullpath_pre = '/home/ra1/original/KWH'
        if 'P1' in mask_case_id:
            ct_image_fullpath_prefix = os.path.join(ct_image_fullpath_pre, 'Phase1_Data')
        elif 'P2' in mask_case_id:
            ct_image_fullpath_prefix = os.path.join(ct_image_fullpath_pre, 'Phase2_Data')
        elif 'P3' in mask_case_id:
            ct_image_fullpath_prefix = os.path.join(ct_image_fullpath_pre, 'Phase3_Data')
        else:
            ct_image_fullpath_prefix = os.path.join(ct_image_fullpath_pre, 'Phase4_Data')

        ct_image_fullpath = os.path.join(ct_image_fullpath_prefix, mask_case_id)
        Updated_Finalized_CT_Image_File_FullPath_List.append(ct_image_fullpath)
        """
        CNT_KWH_0 += 1
        if 'P1' in mask_case_id:
            ct_image_fullpath = os.path.join('/home/ra1/original/KWH/Phase1_data', mask_case_id)
        elif 'P2' in mask_case_id:
            ct_image_fullpath = os.path.join('/home/ra1/original/KWH/Phase2_data', mask_case_id)
        elif 'P3' in mask_case_id:
            ct_image_fullpath = os.path.join('/home/ra1/original/KWH/Phase3_data', mask_case_id)
        else:
            ct_image_fullpath = os.path.join('/home/ra1/original/KWH/Phase4_data', mask_case_id)

        if os.path.exists(ct_image_fullpath):
            Updated_Finalized_CT_Image_File_FullPath_List.append(ct_image_fullpath)
            CNT_KWH += 1

    elif 'QE' in mask_case_id:
        """
        ct_image_fullpath_pre = '/home/ra1/original/QEH'
        if 'P1' in mask_case_id:
            ct_image_fullpath_prefix = os.path.join(ct_image_fullpath_pre, 'Phase1_Data')
        elif 'P2' in mask_case_id:
            ct_image_fullpath_prefix = os.path.join(ct_image_fullpath_pre, 'Phase2_Data')
        elif 'P3' in mask_case_id:
            ct_image_fullpath_prefix = os.path.join(ct_image_fullpath_pre, 'Phase3_Data')
        else:
            ct_image_fullpath_prefix = os.path.join(ct_image_fullpath_pre, 'Phase4_Data')
        
        ct_image_fullpath = os.path.join(ct_image_fullpath_prefix, mask_case_id)
        Updated_Finalized_CT_Image_File_FullPath_List.append(ct_image_fullpath)    
        """
        CNT_QEH_0 += 1
        if 'P1' in mask_case_id:
            ct_image_fullpath = os.path.join('/home/ra1/original/QEH/Phase1_data', mask_case_id)
        elif 'P2' in mask_case_id:
            ct_image_fullpath = os.path.join('/home/ra1/original/QEH/Phase2_data', mask_case_id)
        elif 'P3' in mask_case_id:
            ct_image_fullpath = os.path.join('/home/ra1/original/QEH/Phase3_data', mask_case_id)
        else:
            ct_image_fullpath = os.path.join('/home/ra1/original/QEH/Phase4_data', mask_case_id)

        if os.path.exists(ct_image_fullpath):
            Updated_Finalized_CT_Image_File_FullPath_List.append(ct_image_fullpath)
            CNT_QEH += 1
    elif 'QM' in mask_case_id:
        """
        ct_image_fullpath_pre = '/home/ra1/original/QMH'
        if 'P1' in mask_case_id:
            ct_image_fullpath_prefix = os.path.join(ct_image_fullpath_pre, 'Phase1_Data')
        elif 'P2' in mask_case_id:
            ct_image_fullpath_prefix = os.path.join(ct_image_fullpath_pre, 'Phase2_Data')
        elif 'P3' in mask_case_id:
            ct_image_fullpath_prefix = os.path.join(ct_image_fullpath_pre, 'Phase3_Data')
        else:
            ct_image_fullpath_prefix = os.path.join(ct_image_fullpath_pre, 'Phase4_Data')

        ct_image_fullpath = os.path.join(ct_image_fullpath_prefix, mask_case_id)
        Updated_Finalized_CT_Image_File_FullPath_List.append(ct_image_fullpath)
        """
        CNT_QMH_0 += 1
        if 'P1' in mask_case_id:
            ct_image_fullpath = os.path.join('/home/ra1/original/QMH/Phase1_data', mask_case_id)
        elif 'P2' in mask_case_id:
            ct_image_fullpath = os.path.join('/home/ra1/original/QMH/Phase2_data', mask_case_id)
        elif 'P3' in mask_case_id:
            ct_image_fullpath = os.path.join('/home/ra1/original/QMH/Phase3_data', mask_case_id)
        else:
            ct_image_fullpath = os.path.join('/home/ra1/original/QMH/Phase4_data', mask_case_id)

        if os.path.exists(ct_image_fullpath):
            Updated_Finalized_CT_Image_File_FullPath_List.append(ct_image_fullpath)
            CNT_QMH += 1

print(len(Updated_Finalized_CT_Image_File_FullPath_List), len(Updated_Finalized_Mask_File_FullPath_List), "Line-421")
for img_path in Updated_Finalized_CT_Image_File_FullPath_List:
    image_case_id = os.path.basename(img_path)
    if image_case_id == 'HKU_0598_P3':
        print(img_path, Updated_Finalized_Mask_File_FullPath_List_ID.count(image_case_id))
for msk_path in Updated_Finalized_Mask_File_FullPath_List:
    image_case_id = os.path.basename(msk_path).split('.ni')[0]
    if image_case_id == 'HKU_0598_P3':
        print(msk_path, Updated_Finalized_Mask_File_FullPath_List_ID.count(image_case_id))
# Label Value 25 ---- LiRad 1  ---- NonHCC
# Label Value 26 ---- LiRad 2  ---- NonHCC
# Label Value 27 ---- LiRad 3  ---- NonHCC
# Label Value 28 ---- LiRad 4  ---- HCC
# Label Value 29 ---- LiRad 5  ---- HCC
# Label Value 30 ---- LR_M     ---- NonHCC
Train_Idx = list(np.random.choice(list(range(len(Updated_Finalized_Mask_File_FullPath_List))),
                                         size=int(0.7 * len(Updated_Finalized_Mask_File_FullPath_List)), replace=False))
Test_Idx = list(set(list(range(len(Updated_Finalized_Mask_File_FullPath_List)))) - set(Train_Idx))
print(len(Train_Idx), len(Test_Idx))

file = open('/home/ra1/Documents/2D_3D_Classification_Segment_Split_Finalized_31_October.txt', "w")

Finalized_Classification_Segmentation_Save_Rootpath = '/home/ra1/original/Finalized_Classification_Segmentation_31_October'
if not os.path.exists(Finalized_Classification_Segmentation_Save_Rootpath):
    os.makedirs(Finalized_Classification_Segmentation_Save_Rootpath)
# For Train
Train_Save_Rootpath = os.path.join(Finalized_Classification_Segmentation_Save_Rootpath, 'Train')
if not os.path.exists(Train_Save_Rootpath):
    os.makedirs(Train_Save_Rootpath)

Train_Save_3D_Image_RootPath = os.path.join(Train_Save_Rootpath, '3D_Image')
if not os.path.exists(Train_Save_3D_Image_RootPath):
    os.makedirs(Train_Save_3D_Image_RootPath)

Train_Save_3D_Mask_RootPath = os.path.join(Train_Save_Rootpath, '3D_Mask')
if not os.path.exists(Train_Save_3D_Mask_RootPath):
    os.makedirs(Train_Save_3D_Mask_RootPath)

Train_Save_2D_Image_RootPath = os.path.join(Train_Save_Rootpath, '2D_Image')
if not os.path.exists(Train_Save_2D_Image_RootPath):
    os.makedirs(Train_Save_2D_Image_RootPath)

Train_Save_2D_Mask_RootPath = os.path.join(Train_Save_Rootpath, '2D_Mask')
if not os.path.exists(Train_Save_2D_Mask_RootPath):
    os.makedirs(Train_Save_2D_Mask_RootPath)

# For Test
Test_Save_Rootpath = os.path.join(Finalized_Classification_Segmentation_Save_Rootpath, 'Test')
if not os.path.exists(Test_Save_Rootpath):
    os.makedirs(Test_Save_Rootpath)

Test_Save_3D_Image_Rootpath = os.path.join(Test_Save_Rootpath, '3D_Image')
if not os.path.exists(Test_Save_3D_Image_Rootpath):
    os.makedirs(Test_Save_3D_Image_Rootpath)

Test_Save_3D_Mask_Rootpath = os.path.join(Test_Save_Rootpath, '3D_Mask')
if not os.path.exists(Test_Save_3D_Mask_Rootpath):
    os.makedirs(Test_Save_3D_Mask_Rootpath)

Test_Save_2D_Image_Rootpath = os.path.join(Test_Save_Rootpath, '2D_Image')
if not os.path.exists(Test_Save_2D_Image_Rootpath):
    os.makedirs(Test_Save_2D_Image_Rootpath)

Test_Save_2D_Mask_Rootpath = os.path.join(Test_Save_Rootpath, '2D_Mask')
if not os.path.exists(Test_Save_2D_Mask_Rootpath):
    os.makedirs(Test_Save_2D_Mask_Rootpath)

Train_Label_2D_Lesion, Test_Label_2D_Lesion = [], []
Train_Label_3D_Lesion, Test_Label_3D_Lesion = [], []
Train_Label_by_Radiologist_2D, Test_Label_by_Radiologist_2D = [], []
Train_Label_by_Radiologist_3D, Test_Label_by_Radiologist_3D = [], []
file.write('The list of patient cases in the training set is :' + '\n')  # HCC_or_NonHCC is the label
Train_CT_Image_Patient_ID_Record, Test_CT_Image_Patient_ID_Record = [], []
CNT_Train_Image_2D, CNT_Train_Mask_2D, CNT_Train_Image_3D, CNT_Train_Mask_3D = 0, 0, 0, 0
CNT_Test_Image_2D, CNT_Test_Mask_2D, CNT_Test_Image_3D, CNT_Test_Mask_3D = 0, 0, 0, 0
for idx in Train_Idx:
    mask_fullpath = Updated_Finalized_Mask_File_FullPath_List[idx]
    image_fullpath = Updated_Finalized_CT_Image_File_FullPath_List[idx]
    patient_id = os.path.basename(image_fullpath)
    Train_CT_Image_Patient_ID_Record.append([patient_id, mask_fullpath, image_fullpath])
    # mask_fullpath = '/home/ra1/original/Finalized_Mask/Finalised Mask Update 2021_04_23/2021_06_04_Mask/ID_0670_P3.nii.gz'
    # image_fullpath = '/home/ra1/original/PYN_Part2/Phase3_data/ID_0670_P3'
    image_volume, _ = read_dicom(image_fullpath)       # the shape of output is [N, 512, 512]
    if 'gz' in mask_fullpath:
        mask_volume = read_mask(mask_fullpath)         # the shape of output is [N, 512, 512]
    else:
        sikt_t1 = sitk.ReadImage(mask_fullpath)
        mask_volume = sitk.GetArrayFromImage(sikt_t1)  # the shape of output is [N, 512, 512]

    HCC_or_NonHCC = obtain_hcc_nonhcc(mask_volume)
    file.write(mask_fullpath + ' ' + image_fullpath + str(HCC_or_NonHCC) + '\n')       # HCC_or_NonHCC is the label
    unique_mask_value = np.unique(mask_volume)
    nonzeros_mask_value = unique_mask_value[1:]
    print(mask_fullpath, image_volume.shape, mask_volume.shape, unique_mask_value, nonzeros_mask_value)

    if image_volume.shape == mask_volume.shape:
        # Patient_ID_Lesion_Type_LiRad_Label_Corrected: stored in the format [Patient_ID, Lesion Type, LiRad Level]
        # Save in the 2D Image Format for 2D_Classification or Segmentation
        if len(nonzeros_mask_value) == 1:                           # For the case with one lesion or one type of lesions
            for depth_idx in range(mask_volume.shape[0]):
                slice_mask, slice_image = mask_volume[depth_idx, :, :], image_volume[depth_idx, :, :]
                slice_lesion_value = np.unique(slice_mask)

                if len(slice_lesion_value) > 1:      # mask_volume [0, lesion_value]
                    lesion_value = slice_lesion_value[1]
                    # Obtain the LiRad value
                    for case_id in Patient_ID_Lesion_Type_LiRad_Label:
                        if case_id[0] == patient_id and case_id[1] == lesion_value:
                            # print(case_id, patient_id)
                            LiRad_value = case_id[2]          # Lesion LiRad by Radiologist
                            img = Image.fromarray(np.uint8(slice_image))
                            img.save(os.path.join(Train_Save_2D_Image_RootPath, patient_id + "_{}.jpg".format(depth_idx+1)),
                                     quality=99, subsampling=0)

                            seg = Image.fromarray(np.uint8(slice_mask))
                            seg.save(os.path.join(Train_Save_2D_Mask_RootPath, patient_id + '_{}.jpg'.format(depth_idx+1)),
                                     quality=99, subsampling=0)
                            CNT_Train_Image_2D += 1
                            Train_Label_2D_Lesion.append([os.path.join(Train_Save_2D_Image_RootPath,
                                                                       patient_id + "_{}.jpg".format(depth_idx + 1)),
                                HCC_or_NonHCC])
                            CNT_Train_Image_2D += 1
                            Train_Label_by_Radiologist_2D.append([os.path.join(Train_Save_2D_Image_RootPath,
                                                                               patient_id + "_{}.jpg".format(depth_idx + 1)),
                                                                  LiRad_value])
                else:
                    continue
        else:                                                         # For the case with one lesion or one type of lesions
            for depth_idx in range(mask_volume.shape[0]):
                slice_mask, slice_image = mask_volume[depth_idx, :, :], image_volume[depth_idx, :, :]

                # when a slice with more than one type lesion, ensure the lesion type:
                slice_lesion_value_all = np.unique(slice_mask)
                if len(slice_lesion_value_all) > 1:
                    slice_lesion_value = slice_lesion_value_all[1:]
                    unique_nonzero_array = np.array(slice_lesion_value)

                    nonzeros_mask_value_times = []
                    for value in slice_lesion_value:
                        nonzeros_mask_value_times.append(np.sum(slice_mask == value))

                    num_unique_times = np.array(nonzeros_mask_value_times)
                    max_index = np.argmax(num_unique_times)
                    unique_nonzero_lesion_max = unique_nonzero_array[max_index]

                    for case_id in Patient_ID_Lesion_Type_LiRad_Label:
                        if case_id[0] == patient_id and (case_id[1] == unique_nonzero_lesion_max) \
                                and (unique_nonzero_lesion_max<25):
                            LiRad_value = case_id[2]
                            img = Image.fromarray(np.uint8(slice_image))
                            img.save(os.path.join(Train_Save_2D_Image_RootPath, patient_id + "_{}.jpg".format(depth_idx + 1)),
                                     quality=99, subsampling=0)

                            seg = Image.fromarray(np.uint8(slice_mask))
                            seg.save(os.path.join(Train_Save_2D_Mask_RootPath, patient_id + '_{}.jpg'.format(depth_idx + 1)),
                                     quality=99, subsampling=0)
                            CNT_Train_Image_2D += 1
                            Train_Label_2D_Lesion.append([os.path.join(Train_Save_2D_Image_RootPath,
                                                                       patient_id + "_{}.jpg".format(depth_idx + 1)),
                                                          HCC_or_NonHCC])
                            CNT_Train_Mask_2D += 1
                            Train_Label_by_Radiologist_2D.append([os.path.join(Train_Save_2D_Image_RootPath,
                                                                               patient_id + "_{}.jpg".format(depth_idx + 1)),
                                                                  LiRad_value])
                else:
                    continue

            # Save in the 3D Image Format for 2D_Classification or Segmentation
        vol_data, vol_mask, record_index = vol_ct_mask_choose(Vol_Data=image_volume, Vol_Mask=mask_volume)
        if len(record_index) == 1:
            # small lesion
            vol_mask_lesion_value = np.unique(vol_mask)[1:]
            if len(vol_mask_lesion_value) == 1:
                for case_id in Patient_ID_Lesion_Type_LiRad_Label:
                    if case_id[0] == patient_id and case_id[1] == vol_mask_lesion_value:
                        # print(case_id, patient_id)
                        LiRad_value = case_id[2]  # Lesion LiRad by Radiologist
                        vol_data_reshape = input_data_reshape(vol_data, expected_height=512, expected_width=512,
                                                              expected_depth=32)
                        vol_mask_reshape = input_data_reshape(vol_mask, expected_height=512, expected_width=512,
                                                              expected_depth=32)

                        np.save(os.path.join(Train_Save_3D_Image_RootPath, patient_id + '_' + str(record_index[0][0])
                                             + '_' + str(record_index[0][1]) + '.npy'), vol_data_reshape)
                        np.save(os.path.join(Train_Save_3D_Mask_RootPath, patient_id + '_' + str(record_index[0][0])
                                             + '_' + str(record_index[0][1]) + '.npy'), vol_mask_reshape)

                        CNT_Train_Image_3D += 1
                        Train_Label_3D_Lesion.append(
                            [os.path.join(Train_Save_3D_Image_RootPath, patient_id + '_' + str(record_index[0][0]) + '_'
                                          + str(record_index[0][1]) + '.npy'),
                             HCC_or_NonHCC])
                        CNT_Train_Mask_3D += 1
                        Train_Label_by_Radiologist_3D.append(
                            [os.path.join(Train_Save_3D_Image_RootPath, patient_id + '_' + str(record_index[0][0]) + '_'
                                          + str(record_index[0][1]) + '.npy'),
                             LiRad_value])
            else:
                mask_unique_value = compute_maximum_nonzero_element(vol_mask)
                # print(mask_unique_value, "Line-613")
                for case_id in Patient_ID_Lesion_Type_LiRad_Label:
                    if case_id[0] == patient_id and case_id[1] == mask_unique_value and mask_unique_value < 25:
                        # print(case_id, patient_id)
                        LiRad_value = case_id[2]  # Lesion LiRad by Radiologist
                        vol_data_reshape = input_data_reshape(vol_data, expected_height=512, expected_width=512,
                                                              expected_depth=32)
                        vol_mask_reshape = input_data_reshape(vol_mask, expected_height=512, expected_width=512,
                                                              expected_depth=32)

                        np.save(os.path.join(Train_Save_3D_Image_RootPath, patient_id + '_' + str(record_index[0][0])
                                             + '_' + str(record_index[0][1]) + '.npy'), vol_data_reshape)
                        np.save(os.path.join(Train_Save_3D_Mask_RootPath, patient_id + '_' + str(record_index[0][0])
                                             + '_' + str(record_index[0][1]) + '.npy'), vol_mask_reshape)
                        CNT_Train_Image_3D += 1
                        Train_Label_3D_Lesion.append(
                            [os.path.join(Train_Save_3D_Image_RootPath, patient_id + '_' + str(record_index[0][0])
                                          + '_' + str(record_index[0][1]) + '.npy'),
                             HCC_or_NonHCC])
                        CNT_Train_Mask_3D += 1
                        Train_Label_by_Radiologist_3D.append(
                            [os.path.join(Train_Save_3D_Image_RootPath, patient_id + '_' + str(record_index[0][0])
                                          + '_' + str(record_index[0][1]) + '.npy'),
                             LiRad_value])
        else:
            for item in range(len(vol_data)):
                select_vol_data = vol_data[item]
                select_vol_mask = vol_mask[item]
                # print(np.unique(select_vol_mask))
                # small lesion
                vol_mask_lesion_value = np.unique(select_vol_mask)[1:]
                if len(vol_mask_lesion_value) == 1:
                    for case_id in Patient_ID_Lesion_Type_LiRad_Label:
                        if case_id[0] == patient_id and case_id[1] == vol_mask_lesion_value:
                            # print(case_id, patient_id)
                            LiRad_value = case_id[2]  # Lesion LiRad by Radiologist
                            vol_data_reshape = input_data_reshape(select_vol_data, expected_height=512,
                                                                  expected_width=512,
                                                                  expected_depth=32)
                            vol_mask_reshape = input_data_reshape(select_vol_mask, expected_height=512,
                                                                  expected_width=512,
                                                                  expected_depth=32)

                            np.save(
                                os.path.join(Train_Save_3D_Image_RootPath, patient_id + '_' + str(record_index[item][0])
                                             + '_' + str(record_index[item][1]) + '.npy'), vol_data_reshape)
                            np.save(
                                os.path.join(Train_Save_3D_Mask_RootPath, patient_id + '_' + str(record_index[item][0])
                                             + '_' + str(record_index[item][1]) + '.npy'), vol_mask_reshape)

                            CNT_Train_Image_3D += 1
                            Train_Label_3D_Lesion.append(
                                [os.path.join(Train_Save_3D_Image_RootPath, patient_id + '_' + str(record_index[item][0])
                                              + '_' + str(record_index[item][1]) + '.npy'),
                                 HCC_or_NonHCC])
                            CNT_Train_Mask_3D += 1
                            Train_Label_by_Radiologist_3D.append(
                                [os.path.join(Train_Save_3D_Image_RootPath, patient_id + '_' + str(record_index[item][0])
                                              + '_' + str(record_index[item][1]) + '.npy'),
                                 LiRad_value])
                else:
                    # print(select_vol_mask.shape, np.unique(select_vol_mask), "Line-665")
                    mask_unique_value = compute_maximum_nonzero_element(select_vol_mask)
                    # print(np.unique(select_vol_mask))
                    for case_id in Patient_ID_Lesion_Type_LiRad_Label:
                        if case_id[0] == patient_id and case_id[1] == mask_unique_value and mask_unique_value < 25:
                            # print(case_id, patient_id)
                            LiRad_value = case_id[2]  # Lesion LiRad by Radiologist
                            vol_data_reshape = input_data_reshape(select_vol_data, expected_height=512,
                                                                  expected_width=512,
                                                                  expected_depth=32)
                            vol_mask_reshape = input_data_reshape(select_vol_mask, expected_height=512,
                                                                  expected_width=512,
                                                                  expected_depth=32)

                            np.save(
                                os.path.join(Train_Save_3D_Image_RootPath, patient_id + '_' + str(record_index[item][0])
                                             + '_' + str(record_index[item][1]) + '.npy'), vol_data_reshape)
                            np.save(
                                os.path.join(Train_Save_3D_Mask_RootPath, patient_id + '_' + str(record_index[item][0])
                                             + '_' + str(record_index[item][1]) + '.npy'), vol_mask_reshape)
                            CNT_Train_Image_3D += 1
                            Train_Label_3D_Lesion.append(
                                [os.path.join(Train_Save_3D_Image_RootPath, patient_id + '_' + str(record_index[item][0])
                                              + '_' + str(record_index[item][1]) + '.npy'),
                                 HCC_or_NonHCC])
                            CNT_Train_Mask_3D += 1
                            Train_Label_by_Radiologist_3D.append(
                                [os.path.join(Train_Save_3D_Image_RootPath, patient_id + '_' + str(record_index[item][0])
                                              + '_' + str(record_index[item][1]) + '.npy'),
                                 LiRad_value])
    else:
        Need_Recontoured_Cases_Train.append([image_fullpath, mask_volume, image_volume.shape, mask_volume.shape])

Train_Label_2D_Lesion = np.array(Train_Label_2D_Lesion).reshape((len(Train_Label_2D_Lesion), 2))
Train_Label_3D_Lesion = np.array(Train_Label_3D_Lesion).reshape((len(Train_Label_3D_Lesion), 2))
Train_Label_by_Radiologist_2D = np.array(Train_Label_by_Radiologist_2D).reshape((len(Train_Label_by_Radiologist_2D), 2))
Train_Label_by_Radiologist_3D = np.array(Train_Label_by_Radiologist_3D).reshape((len(Train_Label_by_Radiologist_3D), 2))
np.save(os.path.join(Train_Save_Rootpath, 'Train_Label_2D_Lesion.npy'), Train_Label_2D_Lesion)
np.save(os.path.join(Train_Save_Rootpath, 'Train_Label_3D_Lesion.npy'), Train_Label_3D_Lesion)
np.save(os.path.join(Train_Save_Rootpath, 'Train_Label_by_Radiologist_2D.npy'), Train_Label_by_Radiologist_2D)
np.save(os.path.join(Train_Save_Rootpath, 'Train_Label_by_Radiologist_3D.npy'), Train_Label_by_Radiologist_3D)

file.write('The list of patient cases in the test set is :' + '\n')  # HCC_or_NonHCC is the label

for idx in Test_Idx:
    mask_fullpath = Updated_Finalized_Mask_File_FullPath_List[idx]
    image_fullpath = Updated_Finalized_CT_Image_File_FullPath_List[idx]
    patient_id = os.path.basename(image_fullpath)
    Test_CT_Image_Patient_ID_Record.append([patient_id, image_fullpath, mask_fullpath])
    image_volume, _ = read_dicom(image_fullpath)       # the shape of output is [N, 512, 512]
    if 'gz' in mask_fullpath:
        mask_volume = read_mask(mask_fullpath)         # the shape of output is [N, 512, 512]
    else:
        sikt_t1 = sitk.ReadImage(mask_fullpath)
        mask_volume = sitk.GetArrayFromImage(sikt_t1)  # the shape of output is [N, 512, 512]

    HCC_or_NonHCC = obtain_hcc_nonhcc(mask_volume)
    file.write(mask_fullpath + ' ' + image_fullpath + str(HCC_or_NonHCC) + '\n')       # HCC_or_NonHCC is the label
    unique_mask_value = np.unique(mask_volume)
    nonzeros_mask_value = unique_mask_value[1:]
    print(mask_fullpath, image_volume.shape, mask_volume.shape, unique_mask_value, nonzeros_mask_value)

    # Patient_ID_Lesion_Type_LiRad_Label_Corrected: stored in the format [Patient_ID, Lesion Type, LiRad Level]
    if mask_volume.shape == image_volume.shape:
        # Save in the 2D Image Format for 2D_Classification or Segmentation
        if len(nonzeros_mask_value) == 1:                           # For the case with one lesion or one type of lesions
            for depth_idx in range(mask_volume.shape[0]):
                slice_mask, slice_image = mask_volume[depth_idx, :, :], image_volume[depth_idx, :, :]
                slice_lesion_value = np.unique(slice_mask)

                if len(slice_lesion_value) > 1:      # mask_volume [0, lesion_value]
                    lesion_value = slice_lesion_value[1]
                    # Obtain the LiRad value
                    for case_id in Patient_ID_Lesion_Type_LiRad_Label:
                        if case_id[0] == patient_id and case_id[1] == lesion_value:
                            # print(case_id, patient_id)
                            LiRad_value = case_id[2]          # Lesion LiRad by Radiologist
                            img = Image.fromarray(np.uint8(slice_image))
                            img.save(os.path.join(Test_Save_2D_Image_Rootpath, patient_id + "_{}.jpg".format(depth_idx+1)),
                                     quality=99, subsampling=0)

                            seg = Image.fromarray(np.uint8(slice_mask))
                            seg.save(os.path.join(Test_Save_2D_Mask_Rootpath, patient_id + '_{}.jpg'.format(depth_idx+1)),
                                     quality=99, subsampling=0)

                            CNT_Test_Image_2D += 1
                            Test_Label_2D_Lesion.append([os.path.join(Test_Save_2D_Image_Rootpath,
                                                                       patient_id + "_{}.jpg".format(depth_idx + 1)),
                                HCC_or_NonHCC])
                            CNT_Test_Mask_2D += 1
                            Test_Label_by_Radiologist_2D.append([os.path.join(Test_Save_2D_Image_Rootpath,
                                                                               patient_id + "_{}.jpg".format(depth_idx + 1)),
                                                                  LiRad_value])
                else:
                    continue
        else:                                                         # For the case with one lesion or one type of lesions
            for depth_idx in range(mask_volume.shape[0]):
                slice_mask, slice_image = mask_volume[depth_idx, :, :], image_volume[depth_idx, :, :]

                # when a slice with more than one type lesion, ensure the lesion type:
                slice_lesion_value_all = np.unique(slice_mask)
                if len(slice_lesion_value_all) > 1:
                    slice_lesion_value = slice_lesion_value_all[1:]
                    unique_nonzero_array = np.array(slice_lesion_value)

                    nonzeros_mask_value_times = []
                    for value in slice_lesion_value:
                        nonzeros_mask_value_times.append(np.sum(slice_mask == value))

                    num_unique_times = np.array(nonzeros_mask_value_times)
                    max_index = np.argmax(num_unique_times)
                    unique_nonzero_lesion_max = unique_nonzero_array[max_index]

                    for case_id in Patient_ID_Lesion_Type_LiRad_Label:
                        if case_id[0] == patient_id and (case_id[1] == unique_nonzero_lesion_max) \
                                and (unique_nonzero_lesion_max<25):
                            LiRad_value = case_id[2]
                            img = Image.fromarray(np.uint8(slice_image))
                            img.save(os.path.join(Test_Save_2D_Image_Rootpath, patient_id + "_{}.jpg".format(depth_idx + 1)),
                                     quality=99, subsampling=0)

                            seg = Image.fromarray(np.uint8(slice_mask))
                            seg.save(os.path.join(Test_Save_2D_Mask_Rootpath, patient_id + '_{}.jpg'.format(depth_idx + 1)),
                                     quality=99, subsampling=0)

                            CNT_Test_Image_2D += 1
                            Test_Label_2D_Lesion.append([os.path.join(Test_Save_2D_Image_Rootpath,
                                                                       patient_id + "_{}.jpg".format(depth_idx + 1)),
                                                          HCC_or_NonHCC])
                            CNT_Test_Mask_2D += 1
                            Test_Label_by_Radiologist_2D.append([os.path.join(Test_Save_2D_Image_Rootpath,
                                                                               patient_id + "_{}.jpg".format(depth_idx + 1)),
                                                                  LiRad_value])
                else:
                    continue

        # Save in the 3D Image Format for 2D_Classification or Segmentation
        vol_data, vol_mask, record_index = vol_ct_mask_choose(Vol_Data=image_volume, Vol_Mask=mask_volume)
        if len(record_index) == 1:
            # small lesion
            vol_mask_lesion_value = np.unique(vol_mask)[1:]
            if len(vol_mask_lesion_value) == 1:
                for case_id in Patient_ID_Lesion_Type_LiRad_Label:
                    if case_id[0] == patient_id and case_id[1] == vol_mask_lesion_value:
                        # print(case_id, patient_id)
                        LiRad_value = case_id[2]  # Lesion LiRad by Radiologist
                        vol_data_reshape = input_data_reshape(vol_data, expected_height=512, expected_width=512,
                                                              expected_depth=32)
                        vol_mask_reshape = input_data_reshape(vol_mask, expected_height=512, expected_width=512,
                                                              expected_depth=32)

                        np.save(os.path.join(Test_Save_3D_Image_Rootpath, patient_id + '_' + str(record_index[0][0])
                                             + '_' + str(record_index[0][1]) + '.npy'), vol_data_reshape)
                        np.save(os.path.join(Test_Save_3D_Mask_Rootpath, patient_id + '_' + str(record_index[0][0])
                                             + '_' + str(record_index[0][1]) + '.npy'), vol_mask_reshape)

                        CNT_Test_Image_3D += 1
                        Test_Label_3D_Lesion.append(
                            [os.path.join(Test_Save_3D_Image_Rootpath,patient_id + '_' + str(record_index[0][0])
                                          + '_' + str(record_index[0][1]) + '.npy'),
                             HCC_or_NonHCC])
                        CNT_Test_Mask_3D += 1
                        Test_Label_by_Radiologist_3D.append(
                            [os.path.join(Test_Save_3D_Image_Rootpath, patient_id + '_' + str(record_index[0][0])
                                          + '_' + str(record_index[0][1]) + '.npy'),
                             LiRad_value])
            else:
                mask_unique_value = compute_maximum_nonzero_element(vol_mask)
                for case_id in Patient_ID_Lesion_Type_LiRad_Label:
                    if case_id[0] == patient_id and case_id[1] == mask_unique_value and mask_unique_value < 25:
                        # print(case_id, patient_id)
                        LiRad_value = case_id[2]  # Lesion LiRad by Radiologist
                        vol_data_reshape = input_data_reshape(vol_data, expected_height=512, expected_width=512,
                                                              expected_depth=32)
                        vol_mask_reshape = input_data_reshape(vol_mask, expected_height=512, expected_width=512,
                                                              expected_depth=32)

                        np.save(os.path.join(Test_Save_3D_Image_Rootpath, patient_id + '_' + str(record_index[0][0])
                                             + '_' + str(record_index[0][1]) + '.npy'), vol_data_reshape)
                        np.save(os.path.join(Test_Save_3D_Mask_Rootpath, patient_id + '_' + str(record_index[0][0])
                                             + '_' + str(record_index[0][1]) + '.npy'), vol_mask_reshape)

                        CNT_Test_Image_3D += 1
                        Test_Label_3D_Lesion.append(
                            [os.path.join(Test_Save_3D_Image_Rootpath, patient_id + '_' + str(record_index[0][0])
                                          + '_' + str(record_index[0][1]) + '.npy'),
                             HCC_or_NonHCC])
                        CNT_Test_Mask_3D += 1
                        Test_Label_by_Radiologist_3D.append(
                            [os.path.join(Test_Save_3D_Image_Rootpath, patient_id + '_' + str(record_index[0][0])
                                          + '_' + str(record_index[0][1]) + '.npy'),
                             LiRad_value])
        else:
            for item in range(len(vol_data)):
                select_vol_data = vol_data[item]
                select_vol_mask = vol_mask[item]
                # small lesion
                vol_mask_lesion_value = np.unique(select_vol_mask)[1:]
                if len(vol_mask_lesion_value) == 1:
                    for case_id in Patient_ID_Lesion_Type_LiRad_Label:
                        if case_id[0] == patient_id and case_id[1] == vol_mask_lesion_value:
                            # print(case_id, patient_id)
                            LiRad_value = case_id[2]  # Lesion LiRad by Radiologist
                            vol_data_reshape = input_data_reshape(select_vol_data, expected_height=512, expected_width=512,
                                                                  expected_depth=32)
                            vol_mask_reshape = input_data_reshape(select_vol_mask, expected_height=512, expected_width=512,
                                                                  expected_depth=32)

                            np.save(os.path.join(Test_Save_3D_Image_Rootpath, patient_id + '_' + str(record_index[item][0])
                                                 + '_' + str(record_index[item][1]) + '.npy'), vol_data_reshape)
                            np.save(os.path.join(Test_Save_3D_Mask_Rootpath, patient_id + '_' + str(record_index[item][0])
                                                 + '_' + str(record_index[item][1]) + '.npy'), vol_mask_reshape)

                            CNT_Test_Image_3D += 1
                            Test_Label_3D_Lesion.append(
                                [os.path.join(Test_Save_3D_Image_Rootpath, patient_id + '_' + str(record_index[item][0])
                                              + '_' + str(record_index[item][1]) + '.npy'),
                                 HCC_or_NonHCC])
                            CNT_Test_Mask_3D += 1
                            Test_Label_by_Radiologist_3D.append(
                                [os.path.join(Test_Save_3D_Image_Rootpath, patient_id + '_' + str(record_index[item][0])
                                              + '_' + str(record_index[item][1]) + '.npy'),
                                 LiRad_value])
                else:
                    mask_unique_value = compute_maximum_nonzero_element(select_vol_mask)
                    for case_id in Patient_ID_Lesion_Type_LiRad_Label:
                        if case_id[0] == patient_id and case_id[1] == mask_unique_value and mask_unique_value < 25:
                            # print(case_id, patient_id)
                            LiRad_value = case_id[2]  # Lesion LiRad by Radiologist
                            vol_data_reshape = input_data_reshape(select_vol_data, expected_height=512, expected_width=512,
                                                                      expected_depth=32)
                            vol_mask_reshape = input_data_reshape(select_vol_mask, expected_height=512, expected_width=512,
                                                                      expected_depth=32)

                            np.save(os.path.join(Test_Save_3D_Image_Rootpath, patient_id + '_' + str(record_index[item][0])
                                                 + '_' + str(record_index[item][1]) + '.npy'), vol_data_reshape)
                            np.save(os.path.join(Test_Save_3D_Mask_Rootpath, patient_id + '_' + str(record_index[item][0])
                                                 + '_' + str(record_index[item][1]) + '.npy'), vol_mask_reshape)

                            CNT_Test_Image_3D += 1
                            Test_Label_3D_Lesion.append(
                                [os.path.join(Test_Save_3D_Image_Rootpath, patient_id + '_' + str(record_index[item][0])
                                              + '_' + str(record_index[item][1]) + '.npy'),
                                 HCC_or_NonHCC])
                            CNT_Test_Mask_3D += 1
                            Test_Label_by_Radiologist_3D.append(
                                [os.path.join(Test_Save_3D_Image_Rootpath, patient_id + '_' + str(record_index[item][0])
                                              + '_' + str(record_index[item][1]) + '.npy'),
                                 LiRad_value])
    else:
        Need_Recontoured_Cases_Test.append([image_fullpath, mask_fullpath, image_volume.shape, mask_volume.shape])
file.close()

Test_Label_2D_Lesion = np.array(Test_Label_2D_Lesion).reshape((len(Test_Label_2D_Lesion), 2))
Test_Label_3D_Lesion = np.array(Test_Label_3D_Lesion).reshape((len(Test_Label_3D_Lesion), 2))
Test_Label_by_Radiologist_2D = np.array(Test_Label_by_Radiologist_2D).reshape((len(Test_Label_by_Radiologist_2D), 2))
Test_Label_by_Radiologist_3D = np.array(Test_Label_by_Radiologist_3D).reshape((len(Test_Label_by_Radiologist_3D), 2))
np.save(os.path.join(Test_Save_Rootpath, 'Test_Label_2D_Lesion.npy'), Test_Label_2D_Lesion)
np.save(os.path.join(Test_Save_Rootpath, 'Test_Label_3D_Lesion.npy'), Test_Label_3D_Lesion)
np.save(os.path.join(Test_Save_Rootpath, 'Test_Label_by_Radiologist_2D.npy'), Test_Label_by_Radiologist_2D)
np.save(os.path.join(Test_Save_Rootpath, 'Test_Label_by_Radiologist_3D.npy'), Test_Label_by_Radiologist_3D)

Need_Recontoured_Cases_Train = np.array(Need_Recontoured_Cases_Train)
Need_Recontoured_Cases_Test = np.array(Need_Recontoured_Cases_Test)
np.save(os.path.join(Train_Save_Rootpath, 'Need_Recontoured_Cases_Train.npy'), Need_Recontoured_Cases_Train)
np.save(os.path.join(Test_Save_Rootpath, 'Need_Recontoured_Cases_Test.npy'), Need_Recontoured_Cases_Test)
Train_CT_Image_Patient_ID_Record = np.array(Train_CT_Image_Patient_ID_Record)
Test_CT_Image_Patient_ID_Record = np.array(Test_CT_Image_Patient_ID_Record)
np.save(os.path.join(Train_Save_Rootpath, 'Train_Image_Mask_Path_ID_Record.npy'), Train_CT_Image_Patient_ID_Record)
np.save(os.path.join(Test_Save_Rootpath, 'Test_Image_Mask_Path_ID_Record.npy'), Test_CT_Image_Patient_ID_Record)