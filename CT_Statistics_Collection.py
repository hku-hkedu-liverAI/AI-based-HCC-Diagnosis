import os
import numpy as np
import glob
import pydicom
import datetime
import time

pyn_part1_rootpath = '/home/ra1/original/PreRegDataAII'
pyn_part2_rootpath = '/home/ra1/original/PYN_Part2'
hku_save_rootpath = '/home/ra1/original/PreRegData_HKU'
szh_save_rootpath = '/home/ra1/original/PreRegData_SZH'
qmh_save_rootpath = '/home/ra1/original/QMH'
kwh_save_rootpath = '/home/ra1/original/KWH'
qeh_save_rootpath = '/home/ra1/original/QEH'
gzh_hcc_rootpath = '/home/ra1/original/GZH_Output/HCC_Data'
gzh_nonhcc_rootpath = '/home/ra1/original/GZH_Output/NonHCC_Data'
gzh_meta_rootpath = '/home/ra1/original/GZH_Output/Metastase_Data'
gzh_chol_rootpath = '/home/ra1/original/GZH_Output/CholangCA_Data'

pyn_part1_phase1 = os.listdir(os.path.join(pyn_part1_rootpath, 'Phase1_data'))
pyn_part1_phase2 = os.listdir(os.path.join(pyn_part1_rootpath, 'Phase2_data'))
pyn_part1_phase3 = os.listdir(os.path.join(pyn_part1_rootpath, 'Phase3_data'))
pyn_part1_phase4 = os.listdir(os.path.join(pyn_part1_rootpath, 'Phase4_data'))
print('The number of patients in PYN Part1 is',
      len(pyn_part1_phase1), len(pyn_part1_phase2), len(pyn_part1_phase3), len(pyn_part1_phase4))

pyn_part2_phase1 = os.listdir(os.path.join(pyn_part2_rootpath, 'Phase1_data'))
pyn_part2_phase2 = os.listdir(os.path.join(pyn_part2_rootpath, 'Phase2_data'))
pyn_part2_phase3 = os.listdir(os.path.join(pyn_part2_rootpath, 'Phase3_data'))
pyn_part2_phase4 = os.listdir(os.path.join(pyn_part2_rootpath, 'Phase4_data'))
print('The number of patients in PYN Part2 is',
      len(pyn_part2_phase1), len(pyn_part2_phase2), len(pyn_part2_phase3), len(pyn_part2_phase4))

hku_phase1 = os.listdir(os.path.join(hku_save_rootpath, 'Phase1_data'))
hku_phase2 = os.listdir(os.path.join(hku_save_rootpath, 'Phase2_data'))
hku_phase3 = os.listdir(os.path.join(hku_save_rootpath, 'Phase3_data'))
hku_phase4 = os.listdir(os.path.join(hku_save_rootpath, 'Phase4_data'))
print('The number of patients in HKU is', len(hku_phase1), len(hku_phase2), len(hku_phase3), len(hku_phase4))

szh_phase1 = os.listdir(os.path.join(szh_save_rootpath, 'Phase1_data'))
szh_phase2 = os.listdir(os.path.join(szh_save_rootpath, 'Phase2_data'))
szh_phase3 = os.listdir(os.path.join(szh_save_rootpath, 'Phase3_data'))
szh_phase4 = os.listdir(os.path.join(szh_save_rootpath, 'Phase4_data'))
print('The number of patients in SZH is', len(szh_phase1), len(szh_phase2), len(szh_phase3), len(szh_phase4))

qmh_phase1 = os.listdir(os.path.join(qmh_save_rootpath, 'Phase1_data'))
qmh_phase2 = os.listdir(os.path.join(qmh_save_rootpath, 'Phase2_data'))
qmh_phase3 = os.listdir(os.path.join(qmh_save_rootpath, 'Phase3_data'))
qmh_phase4 = os.listdir(os.path.join(qmh_save_rootpath, 'Phase4_data'))
print('The number of patients in QMH is', len(qmh_phase1), len(qmh_phase2), len(qmh_phase3), len(qmh_phase4))

kwh_phase1 = os.listdir(os.path.join(kwh_save_rootpath, 'Phase1_data'))
kwh_phase2 = os.listdir(os.path.join(kwh_save_rootpath, 'Phase2_data'))
kwh_phase3 = os.listdir(os.path.join(kwh_save_rootpath, 'Phase3_data'))
kwh_phase4 = os.listdir(os.path.join(kwh_save_rootpath, 'Phase4_data'))
print('The number of patients in KWH is', len(kwh_phase1), len(kwh_phase2), len(kwh_phase3), len(kwh_phase4))

qeh_phase1 = os.listdir(os.path.join(qeh_save_rootpath, 'Phase1_data'))
qeh_phase2 = os.listdir(os.path.join(qeh_save_rootpath, 'Phase2_data'))
qeh_phase3 = os.listdir(os.path.join(qeh_save_rootpath, 'Phase3_data'))
qeh_phase4 = os.listdir(os.path.join(qeh_save_rootpath, 'Phase4_data'))
print('The number of patients in QEH is', len(qeh_phase1), len(qeh_phase2), len(qeh_phase3), len(qeh_phase4))

gzh_hcc_phase1 = os.listdir(os.path.join(gzh_hcc_rootpath, 'Phase1_Data'))
gzh_hcc_phase2 = os.listdir(os.path.join(gzh_hcc_rootpath, 'Phase2_Data'))
gzh_hcc_phase3 = os.listdir(os.path.join(gzh_hcc_rootpath, 'Phase3_Data'))
gzh_hcc_phase4 = os.listdir(os.path.join(gzh_hcc_rootpath, 'Phase4_Data'))
print('The number of patients in GZH_HCC is',
      len(gzh_hcc_phase1), len(gzh_hcc_phase2), len(gzh_hcc_phase3), len(gzh_hcc_phase4))

gzh_nonhcc_phase1 = os.listdir(os.path.join(gzh_nonhcc_rootpath, 'Phase1_Data'))
gzh_nonhcc_phase2 = os.listdir(os.path.join(gzh_nonhcc_rootpath, 'Phase2_Data'))
gzh_nonhcc_phase3 = os.listdir(os.path.join(gzh_nonhcc_rootpath, 'Phase3_Data'))
gzh_nonhcc_phase4 = os.listdir(os.path.join(gzh_nonhcc_rootpath, 'Phase4_Data'))
print('The number of patients in GZH_NonHCC is',
      len(gzh_nonhcc_phase1), len(gzh_nonhcc_phase2), len(gzh_nonhcc_phase3), len(gzh_nonhcc_phase4))

gzh_meta_phase1 = os.listdir(os.path.join(gzh_meta_rootpath, 'Phase1_Data'))
gzh_meta_phase2 = os.listdir(os.path.join(gzh_meta_rootpath, 'Phase2_Data'))
gzh_meta_phase3 = os.listdir(os.path.join(gzh_meta_rootpath, 'Phase3_Data'))
gzh_meta_phase4 = os.listdir(os.path.join(gzh_meta_rootpath, 'Phase4_Data'))
print('The number of patients in GZH_Meta is',
      len(gzh_meta_phase1), len(gzh_meta_phase2), len(gzh_meta_phase3), len(gzh_meta_phase4))

gzh_chol_phase1 = os.listdir(os.path.join(gzh_chol_rootpath, 'Phase1_Data'))
gzh_chol_phase2 = os.listdir(os.path.join(gzh_chol_rootpath, 'Phase2_Data'))
gzh_chol_phase3 = os.listdir(os.path.join(gzh_chol_rootpath, 'Phase3_Data'))
gzh_chol_phase4 = os.listdir(os.path.join(gzh_chol_rootpath, 'Phase4_Data'))
print('The number of patients in GZH_Chol is',
      len(gzh_chol_phase1), len(gzh_chol_phase2), len(gzh_chol_phase3), len(gzh_chol_phase4))

PYN_Part1_Phase1_List, PYN_Part1_Phase2_List, PYN_Part1_Phase3_List, PYN_Part1_Phase4_List = [], [], [], []
PYN_Part2_Phase1_List, PYN_Part2_Phase2_List, PYN_Part2_Phase3_List, PYN_Part2_Phase4_List = [], [], [], []
HKU_Phase1_List, HKU_Phase2_List, HKU_Phase3_List, HKU_Phase4_List = [], [], [], []
SZH_Phase1_List, SZH_Phase2_List, SZH_Phase3_List, SZH_Phase4_List = [], [], [], []
QMH_Phase1_List, QMH_Phase2_List, QMH_Phase3_List, QMH_Phase4_List = [], [], [], []
KWH_Phase1_List, KWH_Phase2_List, KWH_Phase3_List, KWH_Phase4_List = [], [], [], []
QEH_Phase1_List, QEH_Phase2_List, QEH_Phase3_List, QEH_Phase4_List = [], [], [], []

for case in pyn_part1_phase1:
    PYN_Part1_Phase1_List.append(case.split('_P1')[0])
for case in pyn_part1_phase2:
    PYN_Part1_Phase2_List.append(case.split('_P2')[0])
for case in pyn_part1_phase3:
    PYN_Part1_Phase3_List.append(case.split('_P3')[0])
for case in pyn_part1_phase4:
    PYN_Part1_Phase4_List.append(case.split('_P4')[0])
print(len(PYN_Part1_Phase1_List), len(PYN_Part1_Phase2_List), len(PYN_Part1_Phase3_List), len(PYN_Part1_Phase4_List))
print(len(np.unique(PYN_Part1_Phase1_List)), len(np.unique(PYN_Part1_Phase2_List)),
      len(np.unique(PYN_Part1_Phase3_List)), len(np.unique(PYN_Part1_Phase4_List)))

Shared_Phase1_2_PYN_Part1, Shared_Phase1_2_3_PYN_Part1, Shared_Four_Phase_PYN_Part1 = [], [], []
for case in PYN_Part1_Phase1_List:
    if case in PYN_Part1_Phase2_List:
        Shared_Phase1_2_PYN_Part1.append(case)
    else:
        continue
for case in Shared_Phase1_2_PYN_Part1:
    if case in PYN_Part1_Phase3_List:
        Shared_Phase1_2_3_PYN_Part1.append(case)
    else:
        continue
for case in Shared_Phase1_2_3_PYN_Part1:
    if case in PYN_Part1_Phase4_List:
        Shared_Four_Phase_PYN_Part1.append(case)
    else:
        continue
print(len(Shared_Four_Phase_PYN_Part1))
# There are 567 Phase 1, 564 Phase 2, 567 Phase 3 and 563 Phase 4. =====> the number of cases with four phases is 560

for case in pyn_part2_phase1:
    PYN_Part2_Phase1_List.append(case.split('_P1')[0])
for case in pyn_part2_phase2:
    PYN_Part2_Phase2_List.append(case.split('_P2')[0])
for case in pyn_part2_phase3:
    PYN_Part2_Phase3_List.append(case.split('_P3')[0])
for case in pyn_part2_phase4:
    PYN_Part2_Phase4_List.append(case.split('_P4')[0])
print(len(PYN_Part2_Phase1_List), len(PYN_Part2_Phase2_List), len(PYN_Part2_Phase3_List), len(PYN_Part2_Phase4_List))

Shared_Phase1_2_PYN_Part2, Shared_Phase1_2_3_PYN_Part2, Shared_Four_Phase_PYN_Part2 = [], [], []
for case in PYN_Part2_Phase1_List:
    if case in PYN_Part2_Phase2_List:
        Shared_Phase1_2_PYN_Part2.append(case)
for case in Shared_Phase1_2_PYN_Part2:
    if case in PYN_Part2_Phase3_List:
        Shared_Phase1_2_3_PYN_Part2.append(case)
for case in Shared_Phase1_2_3_PYN_Part2:
    if case in PYN_Part2_Phase4_List:
        Shared_Four_Phase_PYN_Part2.append(case)
print(len(Shared_Four_Phase_PYN_Part2))
# There are 380 Phase 1, 379 Phase 2, 380 Phase 3 and 377 Phase 4. =====> the number of cases with four phases is 375

for case in hku_phase1:
    HKU_Phase1_List.append(case.split('_P1')[0])
for case in hku_phase2:
    HKU_Phase2_List.append(case.split('_P2')[0])
for case in hku_phase3:
    HKU_Phase3_List.append(case.split('_P3')[0])
for case in hku_phase4:
    HKU_Phase4_List.append(case.split('_P4')[0])
print(len(HKU_Phase1_List), len(HKU_Phase2_List), len(HKU_Phase3_List), len(HKU_Phase4_List))
print(len(np.unique(HKU_Phase1_List)), len(np.unique(HKU_Phase2_List)), len(np.unique(HKU_Phase3_List)),
      len(np.unique(HKU_Phase4_List)))
Shared_Phase1_2_HKU, Shared_Phase1_2_3_HKU, Shared_Four_Phase_HKU = [], [], []
for case in HKU_Phase1_List:
    if case in HKU_Phase2_List:
        Shared_Phase1_2_HKU.append(case)
for case in Shared_Phase1_2_HKU:
    if case in HKU_Phase3_List:
        Shared_Phase1_2_3_HKU.append(case)
for case in Shared_Phase1_2_3_HKU:
    if case in HKU_Phase4_List:
        Shared_Four_Phase_HKU.append(case)
print(len(Shared_Four_Phase_HKU))
# There are 538 Phase 1, 538 Phase 2, 537 Phase 3 and 529 Phase 4. =====> the number of cases with four phases is 524

for case in szh_phase1:
    SZH_Phase1_List.append(case.split('_P1')[0])
for case in szh_phase2:
    SZH_Phase2_List.append(case.split('_P2')[0])
for case in szh_phase3:
    SZH_Phase3_List.append(case.split('_P3')[0])
for case in szh_phase4:
    SZH_Phase4_List.append(case.split('_P4')[0])
print(len(SZH_Phase1_List), len(SZH_Phase2_List), len(SZH_Phase3_List), len(SZH_Phase4_List))
print(len(np.unique(SZH_Phase1_List)), len(np.unique(SZH_Phase2_List)), len(np.unique(SZH_Phase3_List)),
      len(np.unique(SZH_Phase4_List)))
Shared_Phase1_2_SZH, Shared_Phase1_2_3_SZH, Shared_Four_Phase_SZH = [], [], []
for case in SZH_Phase1_List:
    if case in SZH_Phase2_List:
        Shared_Phase1_2_SZH.append(case)
for case in Shared_Phase1_2_SZH:
    if case in SZH_Phase3_List:
        Shared_Phase1_2_3_SZH.append(case)
for case in Shared_Phase1_2_3_SZH:
    if case in SZH_Phase4_List:
        Shared_Four_Phase_SZH.append(case)
print(len(Shared_Four_Phase_SZH))
# There are 999 Phase 1, 983 Phase 2, 981 Phase 3 and 963 Phase 4. =====> the number of cases with four phases is 939

for case in qmh_phase1:
    QMH_Phase1_List.append(case.split('_P1')[0])
for case in qmh_phase2:
    QMH_Phase2_List.append(case.split('_P2')[0])
for case in qmh_phase3:
    QMH_Phase3_List.append(case.split('_P3')[0])
for case in qmh_phase4:
    QMH_Phase4_List.append(case.split('_P4')[0])
print(len(QMH_Phase1_List), len(QMH_Phase2_List), len(QMH_Phase3_List), len(QMH_Phase4_List))
# print(len(np.unique(QMH_Phase1_List)), len(np.unique(QMH_Phase2_List)), len(np.unique(QMH_Phase3_List)),
#       len(np.unique(QMH_Phase4_List)))
Shared_Phase1_2_QMH, Shared_Phase1_2_3_QMH, Shared_Four_Phase_QMH = [], [], []
for case in QMH_Phase1_List:
    if case in QMH_Phase2_List:
        Shared_Phase1_2_QMH.append(case)
for case in Shared_Phase1_2_QMH:
    if case in QMH_Phase3_List:
        Shared_Phase1_2_3_QMH.append(case)
for case in Shared_Phase1_2_3_QMH:
    if case in QMH_Phase4_List:
        Shared_Four_Phase_QMH.append(case)
print(len(Shared_Four_Phase_QMH))
# There are 139 Phase 1, 151 Phase 2, 150 Phase 3 and 127 Phase 4. =====> the number of cases with four phases is 116

for case in kwh_phase1:
    KWH_Phase1_List.append(case.split('_P1')[0])
for case in kwh_phase2:
    KWH_Phase2_List.append(case.split('_P2')[0])
for case in kwh_phase3:
    KWH_Phase3_List.append(case.split('_P3')[0])
for case in kwh_phase4:
    KWH_Phase4_List.append(case.split('_P4')[0])
print(len(KWH_Phase1_List), len(KWH_Phase2_List), len(KWH_Phase3_List), len(KWH_Phase4_List))
# print(len(np.unique(QMH_Phase1_List)), len(np.unique(QMH_Phase2_List)), len(np.unique(QMH_Phase3_List)),
#       len(np.unique(QMH_Phase4_List)))
Shared_Phase1_2_KWH, Shared_Phase1_2_3_KWH, Shared_Four_Phase_KWH = [], [], []
for case in KWH_Phase1_List:
    if case in KWH_Phase2_List:
        Shared_Phase1_2_KWH.append(case)
for case in Shared_Phase1_2_KWH:
    if case in KWH_Phase3_List:
        Shared_Phase1_2_3_KWH.append(case)
for case in Shared_Phase1_2_3_KWH:
    if case in KWH_Phase4_List:
        Shared_Four_Phase_KWH.append(case)
print(len(Shared_Four_Phase_KWH))
# There are 53 Phase 1, 54 Phase 2, 53 Phase 3 and 53 Phase 4. =====> the number of cases with four phases is 51

for case in qeh_phase1:
    QEH_Phase1_List.append(case.split('_P1')[0])
for case in qeh_phase2:
    QEH_Phase2_List.append(case.split('_P2')[0])
for case in qeh_phase3:
    QEH_Phase3_List.append(case.split('_P3')[0])
for case in qeh_phase4:
    QEH_Phase4_List.append(case.split('_P4')[0])
print(len(QEH_Phase1_List), len(QEH_Phase2_List), len(QEH_Phase3_List), len(QEH_Phase4_List))
# print(len(np.unique(QMH_Phase1_List)), len(np.unique(QMH_Phase2_List)), len(np.unique(QMH_Phase3_List)),
#       len(np.unique(QMH_Phase4_List)))
Shared_Phase1_2_QEH, Shared_Phase1_2_3_QEH, Shared_Four_Phase_QEH = [], [], []
for case in QEH_Phase1_List:
    if case in QEH_Phase2_List:
        Shared_Phase1_2_QEH.append(case)
for case in Shared_Phase1_2_QEH:
    if case in QEH_Phase3_List:
        Shared_Phase1_2_3_QEH.append(case)
for case in Shared_Phase1_2_3_QEH:
    if case in QEH_Phase4_List:
        Shared_Four_Phase_QEH.append(case)
print(len(Shared_Four_Phase_QEH))
# There are 53 Phase 1, 54 Phase 2, 53 Phase 3 and 53 Phase 4. =====> the number of cases with four phases is 51

Share_Four_Phases_PYN_Part1_Phase1, Share_Four_Phases_PYN_Part1_Phase2, \
Share_Four_Phases_PYN_Part1_Phase3, Share_Four_Phases_PYN_Part1_Phase4 = [], [], [], []
# There are 560 patients in PYN_Part1
for case in pyn_part1_phase1:
    fullpath = os.path.join(pyn_part1_rootpath + '/Phase1_data', case)
    patient_id_phase = os.path.basename(fullpath)
    for shared_case in Shared_Four_Phase_PYN_Part1:
        if shared_case in patient_id_phase:
            Share_Four_Phases_PYN_Part1_Phase1.append(fullpath)
for case in pyn_part1_phase2:
    fullpath = os.path.join(pyn_part1_rootpath + '/Phase2_data', case)
    patient_id_phase = os.path.basename(fullpath)
    for shared_case in Shared_Four_Phase_PYN_Part1:
        if shared_case in patient_id_phase:
            Share_Four_Phases_PYN_Part1_Phase2.append(fullpath)
for case in pyn_part1_phase3:
    fullpath = os.path.join(pyn_part1_rootpath + '/Phase3_data', case)
    patient_id_phase = os.path.basename(fullpath)
    for shared_case in Shared_Four_Phase_PYN_Part1:
        if shared_case in patient_id_phase:
            Share_Four_Phases_PYN_Part1_Phase3.append(fullpath)
for case in pyn_part1_phase4:
    fullpath = os.path.join(pyn_part1_rootpath + '/Phase4_data', case)
    patient_id_phase = os.path.basename(fullpath)
    for shared_case in Shared_Four_Phase_PYN_Part1:
        if shared_case in patient_id_phase and len(patient_id_phase) < 12:
            Share_Four_Phases_PYN_Part1_Phase4.append(fullpath)
Sorted_Shared_PYN_Part1_Phase1 = sorted(Share_Four_Phases_PYN_Part1_Phase1)
Sorted_Shared_PYN_Part1_Phase2 = sorted(Share_Four_Phases_PYN_Part1_Phase2)
Sorted_Shared_PYN_Part1_Phase3 = sorted(Share_Four_Phases_PYN_Part1_Phase3)
Sorted_Shared_PYN_Part1_Phase4 = sorted(Share_Four_Phases_PYN_Part1_Phase4)

CNT = 0
time_range_phase1_2, time_range_phase2_3, time_range_phase3_4 = [], [], []
scan_machine, scan_model, kilovoltage_phase1, kilovoltage_phase2, kilovoltage_phase3, \
kilovoltage_phase4, = [], [], [], [], [], []
for idx in range(len(Share_Four_Phases_PYN_Part1_Phase1)):
    fullpath_phase1 = Sorted_Shared_PYN_Part1_Phase1[idx]
    fullpath_phase2 = Sorted_Shared_PYN_Part1_Phase2[idx]
    fullpath_phase3 = Sorted_Shared_PYN_Part1_Phase3[idx]
    fullpath_phase4 = Sorted_Shared_PYN_Part1_Phase4[idx]
    phase1_basename = os.path.basename(fullpath_phase1).split('_P1')[0]
    phase2_basename = os.path.basename(fullpath_phase2).split('_P2')[0]
    phase3_basename = os.path.basename(fullpath_phase3).split('_P3')[0]
    phase4_basename = os.path.basename(fullpath_phase4).split('_P4')[0]
    if phase1_basename == phase2_basename and phase2_basename == phase3_basename and phase3_basename == phase4_basename:
        CNT += 1
        phase1_slice_list = os.listdir(fullpath_phase1)
        phase2_slice_list = os.listdir(fullpath_phase2)
        phase3_slice_list = os.listdir(fullpath_phase3)
        phase4_slice_list = os.listdir(fullpath_phase4)

        phase1_slice1_fullpath = os.path.join(fullpath_phase1, phase1_slice_list[0])
        phase2_slice1_fullpath = os.path.join(fullpath_phase2, phase2_slice_list[0])
        phase3_slice1_fullpath = os.path.join(fullpath_phase3, phase3_slice_list[0])
        phase4_slice1_fullpath = os.path.join(fullpath_phase4, phase4_slice_list[0])

        file_reader_phase1 = pydicom.dcmread(phase1_slice1_fullpath, force=True)
        file_reader_phase2 = pydicom.dcmread(phase2_slice1_fullpath, force=True)
        file_reader_phase3 = pydicom.dcmread(phase3_slice1_fullpath, force=True)
        file_reader_phase4 = pydicom.dcmread(phase4_slice1_fullpath, force=True)

        scanned_date_phase1 = file_reader_phase1.get_item('AcquisitionDate').value.decode('utf-8')
        scanned_date_formatted_phase1 = scanned_date_phase1[0:4] + '-' + scanned_date_phase1[
                                                                         4:6] + '-' + scanned_date_phase1[6:]
        scanned_time_phase1 = file_reader_phase1.get_item('AcquisitionTime').value.decode('utf-8')
        scanned_time_formatted_phase1 = scanned_time_phase1[0:2] + ':' + scanned_time_phase1[
                                                                         2:4] + ':' + scanned_time_phase1[4:]
        manufacturer_phase1 = file_reader_phase1.get_item('Manufacturer').value.decode('utf-8')
        manufacturer_model_name_phase1 = file_reader_phase1.get_item('ManufacturerModelName').value.decode('utf-8')
        kilovoltage_phase1.append(file_reader_phase1.get_item('KVP').value.decode('utf-8'))
        # print(scanned_date_formatted_phase1, scanned_time_formatted_phase1)

        scanned_date_phase2 = file_reader_phase2.get_item('AcquisitionDate').value.decode('utf-8')
        scanned_date_formatted_phase2 = scanned_date_phase2[0:4] + '-' + scanned_date_phase2[
                                                                         4:6] + '-' + scanned_date_phase2[6:]
        scanned_time_phase2 = file_reader_phase2.get_item('AcquisitionTime').value.decode('utf-8')
        scanned_time_formatted_phase2 = scanned_time_phase2[0:2] + ':' + scanned_time_phase2[
                                                                         2:4] + ':' + scanned_time_phase2[4:]
        manufacturer_phase2 = file_reader_phase2.get_item('Manufacturer').value.decode('utf-8')
        manufacturer_model_name_phase2 = file_reader_phase2.get_item('ManufacturerModelName').value.decode('utf-8')
        kilovoltage_phase2.append(file_reader_phase2.get_item('KVP').value.decode('utf-8'))

        scanned_date_phase3 = file_reader_phase3.get_item('AcquisitionDate').value.decode('utf-8')
        scanned_date_formatted_phase3 = scanned_date_phase3[0:4] + '-' + scanned_date_phase3[
                                                                         4:6] + '-' + scanned_date_phase3[6:]
        scanned_time_phase3 = file_reader_phase3.get_item('AcquisitionTime').value.decode('utf-8')
        scanned_time_formatted_phase3 = scanned_time_phase3[0:2] + ':' + scanned_time_phase3[
                                                                         2:4] + ':' + scanned_time_phase3[4:]
        manufacturer_phase3 = file_reader_phase3.get_item('Manufacturer').value.decode('utf-8')
        manufacturer_model_name_phase3 = file_reader_phase3.get_item('ManufacturerModelName').value.decode('utf-8')
        kilovoltage_phase3.append(file_reader_phase3.get_item('KVP').value.decode('utf-8'))

        scanned_date_phase4 = file_reader_phase4.get_item('AcquisitionDate').value.decode('utf-8')
        scanned_date_formatted_phase4 = scanned_date_phase4[0:4] + '-' + scanned_date_phase4[
                                                                         4:6] + '-' + scanned_date_phase4[6:]
        scanned_time_phase4 = file_reader_phase4.get_item('AcquisitionTime').value.decode('utf-8')
        scanned_time_formatted_phase4 = scanned_time_phase4[0:2] + ':' + scanned_time_phase4[
                                                                         2:4] + ':' + scanned_time_phase4[4:]
        manufacturer_phase4 = file_reader_phase4.get_item('Manufacturer').value.decode('utf-8')
        manufacturer_model_name_phase4 = file_reader_phase4.get_item('ManufacturerModelName').value.decode('utf-8')
        kilovoltage_phase4.append(file_reader_phase4.get_item('KVP').value.decode('utf-8'))

        # print(scanned_date_formatted_phase1, scanned_date_formatted_phase2, scanned_date_formatted_phase3,
        #       scanned_date_formatted_phase4)
        # print(scanned_time_formatted_phase1, scanned_time_formatted_phase2, scanned_time_formatted_phase3,
        #       scanned_time_formatted_phase4)
        # scan_machine.append([manufacturer_phase1, manufacturer_phase2, manufacturer_phase3, manufacturer_phase4])
        # scan_model.append([manufacturer_model_name_phase1, manufacturer_model_name_phase2,
        #                    manufacturer_model_name_phase3, manufacturer_model_name_phase4])
        # print(kilovoltage_phase1, kilovoltage_phase2, kilovoltage_phase3, kilovoltage_phase4)
        scan_machine.append(manufacturer_phase1)
        scan_model.append(manufacturer_model_name_phase1)

        x1 = time.strptime(scanned_time_formatted_phase1.split('.')[0], '%H:%M:%S')
        x2 = time.strptime(scanned_time_formatted_phase2.split('.')[0], '%H:%M:%S')
        x3 = time.strptime(scanned_time_formatted_phase3.split('.')[0], '%H:%M:%S')
        x4 = time.strptime(scanned_time_formatted_phase4.split('.')[0], '%H:%M:%S')
        # print(scanned_time1_formatted, scanned_time2_formatted, scanned_time3_formatted, scanned_time4_formatted, x1, x2, x3, x4)

        second1 = datetime.timedelta(hours=x1.tm_hour, minutes=x1.tm_min, seconds=x1.tm_sec).total_seconds()
        second2 = datetime.timedelta(hours=x2.tm_hour, minutes=x2.tm_min, seconds=x2.tm_sec).total_seconds()
        second3 = datetime.timedelta(hours=x3.tm_hour, minutes=x3.tm_min, seconds=x3.tm_sec).total_seconds()
        second4 = datetime.timedelta(hours=x4.tm_hour, minutes=x4.tm_min, seconds=x4.tm_sec).total_seconds()

        time_range_phase1_2.append(second2 - second1)
        time_range_phase2_3.append(second3 - second2)
        time_range_phase3_4.append(second4 - second3)

# Scan Machine Name [TOSHIBA, SIEMENS]
# Scan Model Name [Aquilion, SOMATOM Definition AS+]
print(time_range_phase1_2)
print(time_range_phase2_3)
print(time_range_phase3_4)
print(scan_machine)
print(scan_model)
print(len(np.unique(scan_model)), len(np.unique(scan_machine)))
print(np.unique(scan_machine), np.unique(scan_model))
print(np.mean(np.array(time_range_phase1_2)), np.std(np.array(time_range_phase1_2)),
      np.median(np.array(time_range_phase1_2)))
print(np.mean(np.array(time_range_phase2_3)), np.std(np.array(time_range_phase2_3)),
      np.median(np.array(time_range_phase2_3)))
print(np.mean(np.array(time_range_phase3_4)), np.std(np.array(time_range_phase3_4)),
      np.median(np.array(time_range_phase3_4)))
print(len(kilovoltage_phase1), len(np.unique(kilovoltage_phase1)), np.unique(kilovoltage_phase1), "Phase_1")
print(len(kilovoltage_phase2), len(np.unique(kilovoltage_phase2)), np.unique(kilovoltage_phase2), "Phase_2")
print(len(kilovoltage_phase3), len(np.unique(kilovoltage_phase3)), np.unique(kilovoltage_phase3), "Phase_3")
print(len(kilovoltage_phase4), len(np.unique(kilovoltage_phase4)), np.unique(kilovoltage_phase4), "Phase_4")
machine_name_unique = np.unique(scan_machine)
machine_model_unique = np.unique(scan_model)

CNT_1, CNT_2 = 0, 0
for case in scan_machine:
    if case == machine_name_unique[0]:
        CNT_1 += 1
    elif case == machine_name_unique[1]:
        CNT_2 += 1
    else:
        continue
print(np.unique(machine_name_unique))
print(CNT_1, CNT_2)

cnt_1, cnt_2 = 0, 0
for case in scan_model:
    if case == machine_model_unique[0]:
        cnt_1 += 1
    elif case == machine_model_unique[1]:
        cnt_2 += 1
    else:
        continue
print(np.unique(machine_model_unique))
print(cnt_1, cnt_2)
print('*#*' * 50)
Share_Four_Phases_PYN_Part2_Phase1, Share_Four_Phases_PYN_Part2_Phase2, \
Share_Four_Phases_PYN_Part2_Phase3, Share_Four_Phases_PYN_Part2_Phase4 = [], [], [], []
# There are 375 patients in PYN_Part1
for case in pyn_part2_phase1:
    fullpath = os.path.join(pyn_part2_rootpath + '/Phase1_data', case)
    patient_id_phase = os.path.basename(fullpath)
    for shared_case in Shared_Four_Phase_PYN_Part2:
        if shared_case in patient_id_phase:
            Share_Four_Phases_PYN_Part2_Phase1.append(fullpath)
for case in pyn_part2_phase2:
    fullpath = os.path.join(pyn_part2_rootpath + '/Phase2_data', case)
    patient_id_phase = os.path.basename(fullpath)
    for shared_case in Shared_Four_Phase_PYN_Part2:
        if shared_case in patient_id_phase:
            Share_Four_Phases_PYN_Part2_Phase2.append(fullpath)
for case in pyn_part2_phase3:
    fullpath = os.path.join(pyn_part2_rootpath + '/Phase3_data', case)
    patient_id_phase = os.path.basename(fullpath)
    for shared_case in Shared_Four_Phase_PYN_Part2:
        if shared_case in patient_id_phase:
            Share_Four_Phases_PYN_Part2_Phase3.append(fullpath)
for case in pyn_part2_phase4:
    fullpath = os.path.join(pyn_part2_rootpath + '/Phase4_data', case)
    patient_id_phase = os.path.basename(fullpath)
    for shared_case in Shared_Four_Phase_PYN_Part2:
        if shared_case in patient_id_phase and len(patient_id_phase) < 12:
            Share_Four_Phases_PYN_Part2_Phase4.append(fullpath)
Sorted_Shared_PYN_Part2_Phase1 = sorted(Share_Four_Phases_PYN_Part2_Phase1)
Sorted_Shared_PYN_Part2_Phase2 = sorted(Share_Four_Phases_PYN_Part2_Phase2)
Sorted_Shared_PYN_Part2_Phase3 = sorted(Share_Four_Phases_PYN_Part2_Phase3)
Sorted_Shared_PYN_Part2_Phase4 = sorted(Share_Four_Phases_PYN_Part2_Phase4)
print(len(Sorted_Shared_PYN_Part2_Phase1), len(Sorted_Shared_PYN_Part2_Phase2), len(Sorted_Shared_PYN_Part2_Phase3),
      len(Sorted_Shared_PYN_Part2_Phase4))
CNT = 0
time_range_phase1_2, time_range_phase2_3, time_range_phase3_4 = [], [], []
scan_machine, scan_model, kilovoltage_phase1, kilovoltage_phase2, kilovoltage_phase3, \
kilovoltage_phase4, = [], [], [], [], [], []
for idx in range(len(Share_Four_Phases_PYN_Part2_Phase1)):
    fullpath_phase1 = Sorted_Shared_PYN_Part2_Phase1[idx]
    fullpath_phase2 = Sorted_Shared_PYN_Part2_Phase2[idx]
    fullpath_phase3 = Sorted_Shared_PYN_Part2_Phase3[idx]
    fullpath_phase4 = Sorted_Shared_PYN_Part2_Phase4[idx]
    phase1_basename = os.path.basename(fullpath_phase1).split('_P1')[0]
    phase2_basename = os.path.basename(fullpath_phase2).split('_P2')[0]
    phase3_basename = os.path.basename(fullpath_phase3).split('_P3')[0]
    phase4_basename = os.path.basename(fullpath_phase4).split('_P4')[0]
    if phase1_basename == phase2_basename and phase2_basename == phase3_basename and phase3_basename == phase4_basename:
        CNT += 1
        phase1_slice_list = os.listdir(fullpath_phase1)
        phase2_slice_list = os.listdir(fullpath_phase2)
        phase3_slice_list = os.listdir(fullpath_phase3)
        phase4_slice_list = os.listdir(fullpath_phase4)

        phase1_slice1_fullpath = os.path.join(fullpath_phase1, phase1_slice_list[0])
        phase2_slice1_fullpath = os.path.join(fullpath_phase2, phase2_slice_list[0])
        phase3_slice1_fullpath = os.path.join(fullpath_phase3, phase3_slice_list[0])
        phase4_slice1_fullpath = os.path.join(fullpath_phase4, phase4_slice_list[0])

        file_reader_phase1 = pydicom.dcmread(phase1_slice1_fullpath, force=True)
        file_reader_phase2 = pydicom.dcmread(phase2_slice1_fullpath, force=True)
        file_reader_phase3 = pydicom.dcmread(phase3_slice1_fullpath, force=True)
        file_reader_phase4 = pydicom.dcmread(phase4_slice1_fullpath, force=True)

        # print(phase1_slice1_fullpath, phase3_slice1_fullpath, phase3_slice1_fullpath, phase4_slice1_fullpath)
        scanned_date_phase1 = file_reader_phase1.get_item('AcquisitionDate').value.decode('utf-8')
        scanned_date_formatted_phase1 = scanned_date_phase1[0:4] + '-' + scanned_date_phase1[
                                                                         4:6] + '-' + scanned_date_phase1[6:]
        scanned_time_phase1 = file_reader_phase1.get_item('AcquisitionTime').value.decode('utf-8')
        scanned_time_formatted_phase1 = scanned_time_phase1[0:2] + ':' + scanned_time_phase1[
                                                                         2:4] + ':' + scanned_time_phase1[4:]
        manufacturer_phase1 = file_reader_phase1.get_item('Manufacturer').value.decode('utf-8')
        manufacturer_model_name_phase1 = file_reader_phase1.get_item('ManufacturerModelName').value.decode('utf-8')
        kilovoltage_phase1.append(file_reader_phase1.get_item('KVP').value.decode('utf-8'))
        # print(scanned_date_formatted_phase1, scanned_time_formatted_phase1)

        scanned_date_phase2 = file_reader_phase2.get_item('AcquisitionDate').value.decode('utf-8')
        scanned_date_formatted_phase2 = scanned_date_phase2[0:4] + '-' + scanned_date_phase2[
                                                                         4:6] + '-' + scanned_date_phase2[6:]
        scanned_time_phase2 = file_reader_phase2.get_item('AcquisitionTime').value.decode('utf-8')
        scanned_time_formatted_phase2 = scanned_time_phase2[0:2] + ':' + scanned_time_phase2[
                                                                         2:4] + ':' + scanned_time_phase2[4:]
        manufacturer_phase2 = file_reader_phase2.get_item('Manufacturer').value.decode('utf-8')
        manufacturer_model_name_phase2 = file_reader_phase2.get_item('ManufacturerModelName').value.decode('utf-8')
        kilovoltage_phase2.append(file_reader_phase2.get_item('KVP').value.decode('utf-8'))

        scanned_date_phase3 = file_reader_phase3.get_item('AcquisitionDate').value.decode('utf-8')
        scanned_date_formatted_phase3 = scanned_date_phase3[0:4] + '-' + scanned_date_phase3[
                                                                         4:6] + '-' + scanned_date_phase3[6:]
        scanned_time_phase3 = file_reader_phase3.get_item('AcquisitionTime').value.decode('utf-8')
        scanned_time_formatted_phase3 = scanned_time_phase3[0:2] + ':' + scanned_time_phase3[
                                                                         2:4] + ':' + scanned_time_phase3[4:]
        manufacturer_phase3 = file_reader_phase3.get_item('Manufacturer').value.decode('utf-8')
        manufacturer_model_name_phase3 = file_reader_phase3.get_item('ManufacturerModelName').value.decode('utf-8')
        kilovoltage_phase3.append(file_reader_phase3.get_item('KVP').value.decode('utf-8'))

        scanned_date_phase4 = file_reader_phase4.get_item('AcquisitionDate').value.decode('utf-8')
        scanned_date_formatted_phase4 = scanned_date_phase4[0:4] + '-' + scanned_date_phase4[
                                                                         4:6] + '-' + scanned_date_phase4[6:]
        scanned_time_phase4 = file_reader_phase4.get_item('AcquisitionTime').value.decode('utf-8')
        scanned_time_formatted_phase4 = scanned_time_phase4[0:2] + ':' + scanned_time_phase4[
                                                                         2:4] + ':' + scanned_time_phase4[4:]
        manufacturer_phase4 = file_reader_phase4.get_item('Manufacturer').value.decode('utf-8')
        manufacturer_model_name_phase4 = file_reader_phase4.get_item('ManufacturerModelName').value.decode('utf-8')
        kilovoltage_phase4.append(file_reader_phase4.get_item('KVP').value.decode('utf-8'))

        # print(scanned_date_formatted_phase1, scanned_date_formatted_phase2, scanned_date_formatted_phase3,
        #       scanned_date_formatted_phase4)
        # print(scanned_time_formatted_phase1, scanned_time_formatted_phase2, scanned_time_formatted_phase3,
        #       scanned_time_formatted_phase4)
        # scan_machine.append([manufacturer_phase1, manufacturer_phase2, manufacturer_phase3, manufacturer_phase4])
        # scan_model.append([manufacturer_model_name_phase1, manufacturer_model_name_phase2,
        #                    manufacturer_model_name_phase3, manufacturer_model_name_phase4])
        # print(kilovoltage_phase1, kilovoltage_phase2, kilovoltage_phase3, kilovoltage_phase4)
        scan_machine.append(manufacturer_phase1)
        scan_model.append(manufacturer_model_name_phase1)

        x1 = time.strptime(scanned_time_formatted_phase1.split('.')[0], '%H:%M:%S')
        x2 = time.strptime(scanned_time_formatted_phase2.split('.')[0], '%H:%M:%S')
        x3 = time.strptime(scanned_time_formatted_phase3.split('.')[0], '%H:%M:%S')
        x4 = time.strptime(scanned_time_formatted_phase4.split('.')[0], '%H:%M:%S')
        # print(scanned_time1_formatted, scanned_time2_formatted, scanned_time3_formatted, scanned_time4_formatted, x1, x2, x3, x4)

        second1 = datetime.timedelta(hours=x1.tm_hour, minutes=x1.tm_min, seconds=x1.tm_sec).total_seconds()
        second2 = datetime.timedelta(hours=x2.tm_hour, minutes=x2.tm_min, seconds=x2.tm_sec).total_seconds()
        second3 = datetime.timedelta(hours=x3.tm_hour, minutes=x3.tm_min, seconds=x3.tm_sec).total_seconds()
        second4 = datetime.timedelta(hours=x4.tm_hour, minutes=x4.tm_min, seconds=x4.tm_sec).total_seconds()

        time_range_phase1_2.append(second2 - second1)
        time_range_phase2_3.append(second3 - second2)
        time_range_phase3_4.append(second4 - second3)

# Scan Machine Name [TOSHIBA, SIEMENS]
# Scan Model Name [Aquilion, SOMATOM Definition AS+]
print(time_range_phase1_2)
print(time_range_phase2_3)
print(time_range_phase3_4)
print(scan_machine)
print(scan_model)
print(len(np.unique(scan_model)), len(np.unique(scan_machine)))
print(np.unique(scan_machine), np.unique(scan_model))
print(np.mean(np.array(time_range_phase1_2)), np.std(np.array(time_range_phase1_2)),
      np.median(np.array(time_range_phase1_2)))
print(np.mean(np.array(time_range_phase2_3)), np.std(np.array(time_range_phase2_3)),
      np.median(np.array(time_range_phase2_3)))
print(np.mean(np.array(time_range_phase3_4)), np.std(np.array(time_range_phase3_4)),
      np.median(np.array(time_range_phase3_4)))
print(len(kilovoltage_phase1), len(np.unique(kilovoltage_phase1)), np.unique(kilovoltage_phase1), "Phase_1")
print(len(kilovoltage_phase2), len(np.unique(kilovoltage_phase2)), np.unique(kilovoltage_phase2), "Phase_2")
print(len(kilovoltage_phase3), len(np.unique(kilovoltage_phase3)), np.unique(kilovoltage_phase3), "Phase_3")
print(len(kilovoltage_phase4), len(np.unique(kilovoltage_phase4)), np.unique(kilovoltage_phase4), "Phase_4")
machine_name_unique = np.unique(scan_machine)
machine_model_unique = np.unique(scan_model)

CNT_1, CNT_2 = 0, 0
for case in scan_machine:
    if case == machine_name_unique[0]:
        CNT_1 += 1
    elif case == machine_name_unique[1]:
        CNT_2 += 1
    else:
        continue
print(np.unique(machine_name_unique))
print(CNT_1, CNT_2)

cnt_1, cnt_2 = 0, 0
for case in scan_model:
    if case == machine_model_unique[0]:
        cnt_1 += 1
    elif case == machine_model_unique[1]:
        cnt_2 += 1
    else:
        continue
print(np.unique(machine_model_unique))
print(cnt_1, cnt_2)
print('*#*' * 50)
Share_Four_Phases_HKU_Phase1, Share_Four_Phases_HKU_Phase2, Share_Four_Phases_HKU_Phase3, \
Share_Four_Phases_HKU_Phase4 = [], [], [], []
for case in hku_phase1:
    fullpath = os.path.join(hku_save_rootpath + '/Phase1_data', case)
    patient_id_phase = os.path.basename(fullpath)
    for shared_case in Shared_Four_Phase_HKU:
        if shared_case in patient_id_phase:
            Share_Four_Phases_HKU_Phase1.append(fullpath)
for case in hku_phase2:
    fullpath = os.path.join(hku_save_rootpath + '/Phase2_data', case)
    patient_id_phase = os.path.basename(fullpath)
    for shared_case in Shared_Four_Phase_HKU:
        if shared_case in patient_id_phase:
            Share_Four_Phases_HKU_Phase2.append(fullpath)
for case in hku_phase3:
    fullpath = os.path.join(hku_save_rootpath + '/Phase3_data', case)
    patient_id_phase = os.path.basename(fullpath)
    for shared_case in Shared_Four_Phase_HKU:
        if shared_case in patient_id_phase:
            Share_Four_Phases_HKU_Phase3.append(fullpath)
for case in hku_phase4:
    fullpath = os.path.join(hku_save_rootpath + '/Phase4_data', case)
    patient_id_phase = os.path.basename(fullpath)
    for shared_case in Shared_Four_Phase_HKU:
        if shared_case in patient_id_phase and len(patient_id_phase) < 12:
            Share_Four_Phases_HKU_Phase4.append(fullpath)
Sorted_Shared_HKU_Phase1 = sorted(Share_Four_Phases_HKU_Phase1)
Sorted_Shared_HKU_Phase2 = sorted(Share_Four_Phases_HKU_Phase2)
Sorted_Shared_HKU_Phase3 = sorted(Share_Four_Phases_HKU_Phase3)
Sorted_Shared_HKU_Phase4 = sorted(Share_Four_Phases_HKU_Phase4)
print(len(Sorted_Shared_HKU_Phase1), len(Sorted_Shared_HKU_Phase2), len(Sorted_Shared_HKU_Phase3),
      len(Sorted_Shared_HKU_Phase4))
CNT = 0
time_range_phase1_2, time_range_phase2_3, time_range_phase3_4 = [], [], []
scan_machine, scan_model, kilovoltage_phase1, kilovoltage_phase2, kilovoltage_phase3, \
kilovoltage_phase4, = [], [], [], [], [], []

manufacturer_phase1, manufacturer_phase2, manufacturer_phase3, manufacturer_phase4 = None, None, None, None
manufacturer_model_name_phase1, manufacturer_model_name_phase2, manufacturer_model_name_phase3, \
manufacturer_model_name_phase4 = None, None, None, None
toshiba_hku_patient_id, ge_hku_patient_id = [], []
nonge_hku_patient_id = []
for idx in range(len(Share_Four_Phases_HKU_Phase1)):
    fullpath_phase1 = Sorted_Shared_HKU_Phase1[idx]
    fullpath_phase2 = Sorted_Shared_HKU_Phase2[idx]
    fullpath_phase3 = Sorted_Shared_HKU_Phase3[idx]
    fullpath_phase4 = Sorted_Shared_HKU_Phase4[idx]
    phase1_basename = os.path.basename(fullpath_phase1).split('_P1')[0]
    phase2_basename = os.path.basename(fullpath_phase2).split('_P2')[0]
    phase3_basename = os.path.basename(fullpath_phase3).split('_P3')[0]
    phase4_basename = os.path.basename(fullpath_phase4).split('_P4')[0]
    # print(fullpath_phase1, fullpath_phase2, fullpath_phase3, fullpath_phase4)
    if phase1_basename == phase2_basename and phase2_basename == phase3_basename and phase3_basename == phase4_basename:
        CNT += 1
        phase1_slice_list = os.listdir(fullpath_phase1)
        phase2_slice_list = os.listdir(fullpath_phase2)
        phase3_slice_list = os.listdir(fullpath_phase3)
        phase4_slice_list = os.listdir(fullpath_phase4)

        phase1_slice1_fullpath = os.path.join(fullpath_phase1, phase1_slice_list[0])
        phase2_slice1_fullpath = os.path.join(fullpath_phase2, phase2_slice_list[0])
        phase3_slice1_fullpath = os.path.join(fullpath_phase3, phase3_slice_list[0])
        phase4_slice1_fullpath = os.path.join(fullpath_phase4, phase4_slice_list[0])

        file_reader_phase1 = pydicom.dcmread(phase1_slice1_fullpath, force=True)
        file_reader_phase2 = pydicom.dcmread(phase2_slice1_fullpath, force=True)
        file_reader_phase3 = pydicom.dcmread(phase3_slice1_fullpath, force=True)
        file_reader_phase4 = pydicom.dcmread(phase4_slice1_fullpath, force=True)

        scanned_date_phase1 = file_reader_phase1.get_item('AcquisitionDate').value.decode('utf-8')
        scanned_date_formatted_phase1 = scanned_date_phase1[0:4] + '-' + scanned_date_phase1[
                                                                         4:6] + '-' + scanned_date_phase1[6:]
        scanned_time_phase1 = file_reader_phase1.get_item('AcquisitionTime').value.decode('utf-8')
        scanned_time_formatted_phase1 = scanned_time_phase1[0:2] + ':' + scanned_time_phase1[
                                                                         2:4] + ':' + scanned_time_phase1[4:]

        if file_reader_phase1.get_item('Manufacturer'):
            manufacturer_phase1 = file_reader_phase1.get_item('Manufacturer').value.decode('utf-8')

        if file_reader_phase1.get_item('ManufacturerModelName'):
            manufacturer_model_name_phase1 = file_reader_phase1.get_item('ManufacturerModelName').value.decode('utf-8')
        kilovoltage_phase1.append(file_reader_phase1.get_item('KVP').value.decode('utf-8'))
        # print(scanned_date_formatted_phase1, scanned_time_formatted_phase1)

        scanned_date_phase2 = file_reader_phase2.get_item('AcquisitionDate').value.decode('utf-8')
        scanned_date_formatted_phase2 = scanned_date_phase2[0:4] + '-' + scanned_date_phase2[
                                                                         4:6] + '-' + scanned_date_phase2[6:]
        scanned_time_phase2 = file_reader_phase2.get_item('AcquisitionTime').value.decode('utf-8')
        scanned_time_formatted_phase2 = scanned_time_phase2[0:2] + ':' + scanned_time_phase2[
                                                                         2:4] + ':' + scanned_time_phase2[4:]
        if file_reader_phase2.get_item('Manufacturer'):
            manufacturer_phase2 = file_reader_phase2.get_item('Manufacturer').value.decode('utf-8')
        if file_reader_phase2.get_item('ManufacturerModelName'):
            manufacturer_model_name_phase2 = file_reader_phase2.get_item('ManufacturerModelName').value.decode('utf-8')
        kilovoltage_phase2.append(file_reader_phase2.get_item('KVP').value.decode('utf-8'))

        scanned_date_phase3 = file_reader_phase3.get_item('AcquisitionDate').value.decode('utf-8')
        scanned_date_formatted_phase3 = scanned_date_phase3[0:4] + '-' + scanned_date_phase3[
                                                                         4:6] + '-' + scanned_date_phase3[6:]
        scanned_time_phase3 = file_reader_phase3.get_item('AcquisitionTime').value.decode('utf-8')
        scanned_time_formatted_phase3 = scanned_time_phase3[0:2] + ':' + scanned_time_phase3[
                                                                         2:4] + ':' + scanned_time_phase3[4:]
        if file_reader_phase3.get_item('Manufacturer'):
            manufacturer_phase3 = file_reader_phase3.get_item('Manufacturer').value.decode('utf-8')
        if file_reader_phase3.get_item('ManufacturerModelName'):
            manufacturer_model_name_phase3 = file_reader_phase3.get_item('ManufacturerModelName').value.decode('utf-8')
        kilovoltage_phase3.append(file_reader_phase3.get_item('KVP').value.decode('utf-8'))

        scanned_date_phase4 = file_reader_phase4.get_item('AcquisitionDate').value.decode('utf-8')
        scanned_date_formatted_phase4 = scanned_date_phase4[0:4] + '-' + scanned_date_phase4[
                                                                         4:6] + '-' + scanned_date_phase4[6:]
        scanned_time_phase4 = file_reader_phase4.get_item('AcquisitionTime').value.decode('utf-8')
        scanned_time_formatted_phase4 = scanned_time_phase4[0:2] + ':' + scanned_time_phase4[
                                                                         2:4] + ':' + scanned_time_phase4[4:]
        if file_reader_phase4.get_item('Manufacturer'):
            manufacturer_phase4 = file_reader_phase4.get_item('Manufacturer').value.decode('utf-8')
        if file_reader_phase4.get_item('ManufacturerModelName'):
            manufacturer_model_name_phase4 = file_reader_phase4.get_item('ManufacturerModelName').value.decode('utf-8')
        kilovoltage_phase4.append(file_reader_phase4.get_item('KVP').value.decode('utf-8'))

        # print(scanned_date_formatted_phase1, scanned_date_formatted_phase2, scanned_date_formatted_phase3,
        #       scanned_date_formatted_phase4)
        # print(scanned_time_formatted_phase1, scanned_time_formatted_phase2, scanned_time_formatted_phase3,
        #       scanned_time_formatted_phase4)
        # scan_machine.append([manufacturer_phase1, manufacturer_phase2, manufacturer_phase3, manufacturer_phase4])
        # scan_model.append([manufacturer_model_name_phase1, manufacturer_model_name_phase2,
        #                    manufacturer_model_name_phase3, manufacturer_model_name_phase4])
        # print(kilovoltage_phase1, kilovoltage_phase2, kilovoltage_phase3, kilovoltage_phase4)
        if manufacturer_phase1 and manufacturer_phase2 and manufacturer_phase3 and manufacturer_phase4:
            scan_machine.append([manufacturer_phase1 + manufacturer_model_name_phase1,
                                 manufacturer_phase2 + manufacturer_model_name_phase2,
                                 manufacturer_phase3 + manufacturer_model_name_phase3,
                                 manufacturer_phase4 + manufacturer_model_name_phase4])
            scan_model.append(manufacturer_model_name_phase1)

        x1 = time.strptime(scanned_time_formatted_phase1.split('.')[0], '%H:%M:%S')
        x2 = time.strptime(scanned_time_formatted_phase2.split('.')[0], '%H:%M:%S')
        x3 = time.strptime(scanned_time_formatted_phase3.split('.')[0], '%H:%M:%S')
        x4 = time.strptime(scanned_time_formatted_phase4.split('.')[0], '%H:%M:%S')
        # print(scanned_time1_formatted, scanned_time2_formatted, scanned_time3_formatted, scanned_time4_formatted, x1, x2, x3, x4)

        second1 = datetime.timedelta(hours=x1.tm_hour, minutes=x1.tm_min, seconds=x1.tm_sec).total_seconds()
        second2 = datetime.timedelta(hours=x2.tm_hour, minutes=x2.tm_min, seconds=x2.tm_sec).total_seconds()
        second3 = datetime.timedelta(hours=x3.tm_hour, minutes=x3.tm_min, seconds=x3.tm_sec).total_seconds()
        second4 = datetime.timedelta(hours=x4.tm_hour, minutes=x4.tm_min, seconds=x4.tm_sec).total_seconds()

        time_range_phase1_2.append(second2 - second1)
        time_range_phase2_3.append(second3 - second2)
        time_range_phase3_4.append(second4 - second3)

        if manufacturer_phase1 and 'GE' in manufacturer_phase1:
            ge_hku_patient_id.append([phase1_basename, manufacturer_phase1, manufacturer_model_name_phase1, scanned_date_formatted_phase1])
        else:
            toshiba_hku_patient_id.append(phase1_basename)
            nonge_hku_patient_id.append([phase1_basename, manufacturer_phase1,
                                         manufacturer_model_name_phase1, scanned_date_formatted_phase1])
            continue

print(scan_machine)

print('**>'*50)

print(nonge_hku_patient_id)
for i in nonge_hku_patient_id:
    print(i)
print(iii)
# Scan Machine Name [TOSHIBA, SIEMENS]
# Scan Model Name [Aquilion, SOMATOM Definition AS+]
print(time_range_phase1_2)
print(time_range_phase2_3)
print(time_range_phase3_4)
print(scan_machine)
print(scan_model)
print(len(np.unique(scan_model)), "Line-768", len(np.unique(scan_machine)))
print(np.unique(scan_machine),  'Line-769',
      np.unique(scan_model))
print(np.mean(np.array(time_range_phase1_2)), np.std(np.array(time_range_phase1_2)),
      np.median(np.array(time_range_phase1_2)))
print(np.mean(np.array(time_range_phase2_3)), np.std(np.array(time_range_phase2_3)),
      np.median(np.array(time_range_phase2_3)))
print(np.mean(np.array(time_range_phase3_4)), np.std(np.array(time_range_phase3_4)),
      np.median(np.array(time_range_phase3_4)))
print(len(kilovoltage_phase1), len(np.unique(kilovoltage_phase1)), np.unique(kilovoltage_phase1), "Phase_1")
print(len(kilovoltage_phase2), len(np.unique(kilovoltage_phase2)), np.unique(kilovoltage_phase2), "Phase_2")
print(len(kilovoltage_phase3), len(np.unique(kilovoltage_phase3)), np.unique(kilovoltage_phase3), "Phase_3")
print(len(kilovoltage_phase4), len(np.unique(kilovoltage_phase4)), np.unique(kilovoltage_phase4), "Phase_4")
machine_name_unique = np.unique(scan_machine)
machine_model_unique = np.unique(scan_model)
print('HKU_Scan_Model')
CNT_1, CNT_2 = 0, 0
for case in scan_machine:
    if case == machine_name_unique[0]:
        CNT_1 += 1
    elif case == machine_name_unique[1]:
        CNT_2 += 1
    else:
        continue
print(np.unique(machine_name_unique))
print(CNT_1, CNT_2)

cnt_1, cnt_2, cnt_3, cnt_4 = 0, 0, 0, 0
for case in scan_model:
    if case == machine_model_unique[0]:
        cnt_1 += 1
    elif case == machine_model_unique[1]:
        cnt_2 += 1
    elif case == machine_model_unique[2]:
        cnt_3 += 1
    elif case == machine_model_unique[3]:
        cnt_4 += 1
    else:
        continue
print(np.unique(machine_model_unique))
print(cnt_1, cnt_2, cnt_3, cnt_4)

print('*#*' * 50)
Share_Four_Phases_SZH_Phase1, Share_Four_Phases_SZH_Phase2, Share_Four_Phases_SZH_Phase3, \
Share_Four_Phases_SZH_Phase4 = [], [], [], []
# There are 375 patients in PYN_Part1
for case in szh_phase1:
    fullpath = os.path.join(szh_save_rootpath + '/Phase1_data', case)
    patient_id_phase = os.path.basename(fullpath)
    for shared_case in Shared_Four_Phase_SZH:
        if shared_case in patient_id_phase:
            Share_Four_Phases_SZH_Phase1.append(fullpath)
for case in szh_phase2:
    fullpath = os.path.join(szh_save_rootpath + '/Phase2_data', case)
    patient_id_phase = os.path.basename(fullpath)
    for shared_case in Shared_Four_Phase_SZH:
        if shared_case in patient_id_phase:
            Share_Four_Phases_SZH_Phase2.append(fullpath)
for case in szh_phase3:
    fullpath = os.path.join(szh_save_rootpath + '/Phase3_data', case)
    patient_id_phase = os.path.basename(fullpath)
    for shared_case in Shared_Four_Phase_SZH:
        if shared_case in patient_id_phase:
            Share_Four_Phases_SZH_Phase3.append(fullpath)
for case in szh_phase4:
    fullpath = os.path.join(szh_save_rootpath + '/Phase4_data', case)
    patient_id_phase = os.path.basename(fullpath)
    for shared_case in Shared_Four_Phase_SZH:
        if shared_case in patient_id_phase:
            Share_Four_Phases_SZH_Phase4.append(fullpath)
Sorted_Shared_SZH_Phase1 = sorted(Share_Four_Phases_SZH_Phase1)
Sorted_Shared_SZH_Phase2 = sorted(Share_Four_Phases_SZH_Phase2)
Sorted_Shared_SZH_Phase3 = sorted(Share_Four_Phases_SZH_Phase3)
Sorted_Shared_SZH_Phase4 = sorted(Share_Four_Phases_SZH_Phase4)
print(len(Sorted_Shared_SZH_Phase1), len(Sorted_Shared_SZH_Phase2), len(Sorted_Shared_SZH_Phase3),
      len(Sorted_Shared_SZH_Phase4))
print(Sorted_Shared_SZH_Phase1)
Sorted_Shared_SZH_Phase1_Clean, Sorted_Shared_SZH_Phase2_Clean, Sorted_Shared_SZH_Phase3_Clean, \
Sorted_Shared_SZH_Phase4_Clean = [], [], [], []

Case_Removed = ['SZF_0291', 'SZF_0735', 'SZH_0002', 'SZH_0345', 'SZF_0042', 'SZH_0223'] #, 'SZF_0265', 'SZF_0531']
for case in Sorted_Shared_SZH_Phase1:
    case_id = os.path.basename(case).split('_P1')[0]
    if case_id in Case_Removed:
        continue
    else:
        if case not in Sorted_Shared_SZH_Phase1_Clean:
            Sorted_Shared_SZH_Phase1_Clean.append(case)

for case in Sorted_Shared_SZH_Phase2:
    case_id = os.path.basename(case).split('_P2')[0]
    if case_id in Case_Removed:
        continue
    else:
        if case not in Sorted_Shared_SZH_Phase2_Clean:
            Sorted_Shared_SZH_Phase2_Clean.append(case)

for case in Sorted_Shared_SZH_Phase3:
    case_id = os.path.basename(case).split('_P3')[0]
    if case_id in Case_Removed:
        continue
    else:
        if case not in Sorted_Shared_SZH_Phase3_Clean:
            Sorted_Shared_SZH_Phase3_Clean.append(case)

for case in Sorted_Shared_SZH_Phase4:
    case_id = os.path.basename(case).split('_P4')[0]
    if case_id in Case_Removed:
        continue
    else:
        if case not in Sorted_Shared_SZH_Phase4_Clean:
            Sorted_Shared_SZH_Phase4_Clean.append(case)


print(len(Sorted_Shared_SZH_Phase1_Clean))

CNT = 0
time_range_phase1_2, time_range_phase2_3, time_range_phase3_4 = [], [], []
scan_machine, scan_model, kilovoltage_phase1, kilovoltage_phase2, kilovoltage_phase3, \
kilovoltage_phase4, = [], [], [], [], [], []
manufacturer_phase1, manufacturer_phase2, manufacturer_phase3, manufacturer_phase4 = None, None, None, None
manufacturer_model_name_phase1, manufacturer_model_name_phase2, manufacturer_model_name_phase3, manufacturer_model_name_phase4 = None, None, None, None
for idx in range(len(Sorted_Shared_SZH_Phase1_Clean)):
    fullpath_phase1 = Sorted_Shared_SZH_Phase1_Clean[idx]
    fullpath_phase2 = Sorted_Shared_SZH_Phase2_Clean[idx]
    fullpath_phase3 = Sorted_Shared_SZH_Phase3_Clean[idx]
    fullpath_phase4 = Sorted_Shared_SZH_Phase4_Clean[idx]
    phase1_basename = os.path.basename(fullpath_phase1).split('_P1')[0]
    phase2_basename = os.path.basename(fullpath_phase2).split('_P2')[0]
    phase3_basename = os.path.basename(fullpath_phase3).split('_P3')[0]
    phase4_basename = os.path.basename(fullpath_phase4).split('_P4')[0]
    if phase1_basename == phase2_basename and phase2_basename == phase3_basename and phase3_basename == phase4_basename:
        CNT += 1
        # phase1_slice_list = sorted(os.listdir(fullpath_phase1))
        # phase2_slice_list = sorted(os.listdir(fullpath_phase2))
        # phase3_slice_list = sorted(os.listdir(fullpath_phase3))
        # phase4_slice_list = sorted(os.listdir(fullpath_phase4))

        phase1_slice_list = [slice for slice in os.listdir(fullpath_phase1) if not slice.startswith('.')]
        phase2_slice_list = [slice for slice in os.listdir(fullpath_phase2) if not slice.startswith('.')]
        phase3_slice_list = [slice for slice in os.listdir(fullpath_phase3) if not slice.startswith('.')]
        phase4_slice_list = [slice for slice in os.listdir(fullpath_phase4) if not slice.startswith('.')]
        phase1_slice1_fullpath = os.path.join(fullpath_phase1, phase1_slice_list[0])
        phase2_slice1_fullpath = os.path.join(fullpath_phase2, phase2_slice_list[0])
        phase3_slice1_fullpath = os.path.join(fullpath_phase3, phase3_slice_list[0])
        phase4_slice1_fullpath = os.path.join(fullpath_phase4, phase4_slice_list[0])

        file_reader_phase1 = pydicom.dcmread(phase1_slice1_fullpath, force=True)
        file_reader_phase2 = pydicom.dcmread(phase2_slice1_fullpath, force=True)
        file_reader_phase3 = pydicom.dcmread(phase3_slice1_fullpath, force=True)
        file_reader_phase4 = pydicom.dcmread(phase4_slice1_fullpath, force=True)

        # print(phase1_slice1_fullpath, phase2_slice1_fullpath, phase3_slice1_fullpath, phase4_slice1_fullpath)
        scanned_date_phase1 = file_reader_phase1.get_item('AcquisitionDate').value.decode('utf-8')
        scanned_date_formatted_phase1 = scanned_date_phase1[0:4] + '-' + scanned_date_phase1[
                                                                         4:6] + '-' + scanned_date_phase1[6:]
        scanned_time_phase1 = file_reader_phase1.get_item('AcquisitionTime').value.decode('utf-8')
        scanned_time_formatted_phase1 = scanned_time_phase1[0:2] + ':' + scanned_time_phase1[
                                                                         2:4] + ':' + scanned_time_phase1[4:]
        if file_reader_phase1.get_item('Manufacturer'):
            manufacturer_phase1 = file_reader_phase1.get_item('Manufacturer').value.decode('utf-8')
        if file_reader_phase1.get_item('ManufacturerModelName'):
            manufacturer_model_name_phase1 = file_reader_phase1.get_item('ManufacturerModelName').value.decode('utf-8')
        kilovoltage_phase1.append(file_reader_phase1.get_item('KVP').value.decode('utf-8'))
        # print(scanned_date_formatted_phase1, scanned_time_formatted_phase1)

        scanned_date_phase2 = file_reader_phase2.get_item('AcquisitionDate').value.decode('utf-8')
        scanned_date_formatted_phase2 = scanned_date_phase2[0:4] + '-' + scanned_date_phase2[
                                                                         4:6] + '-' + scanned_date_phase2[6:]
        scanned_time_phase2 = file_reader_phase2.get_item('AcquisitionTime').value.decode('utf-8')
        scanned_time_formatted_phase2 = scanned_time_phase2[0:2] + ':' + scanned_time_phase2[
                                                                         2:4] + ':' + scanned_time_phase2[4:]
        if file_reader_phase2.get_item('Manufacturer'):
            manufacturer_phase2 = file_reader_phase2.get_item('Manufacturer').value.decode('utf-8')
        if file_reader_phase2.get_item('ManufacturerModelName'):
            manufacturer_model_name_phase2 = file_reader_phase2.get_item('ManufacturerModelName').value.decode('utf-8')
        kilovoltage_phase2.append(file_reader_phase2.get_item('KVP').value.decode('utf-8'))

        scanned_date_phase3 = file_reader_phase3.get_item('AcquisitionDate').value.decode('utf-8')
        scanned_date_formatted_phase3 = scanned_date_phase3[0:4] + '-' + scanned_date_phase3[
                                                                         4:6] + '-' + scanned_date_phase3[6:]
        scanned_time_phase3 = file_reader_phase3.get_item('AcquisitionTime').value.decode('utf-8')
        scanned_time_formatted_phase3 = scanned_time_phase3[0:2] + ':' + scanned_time_phase3[
                                                                         2:4] + ':' + scanned_time_phase3[4:]
        if file_reader_phase3.get_item('Manufacturer'):
            manufacturer_phase3 = file_reader_phase3.get_item('Manufacturer').value.decode('utf-8')
        if file_reader_phase3.get_item('ManufacturerModelName'):
            manufacturer_model_name_phase3 = file_reader_phase3.get_item('ManufacturerModelName').value.decode('utf-8')
        kilovoltage_phase3.append(file_reader_phase3.get_item('KVP').value.decode('utf-8'))

        scanned_date_phase4 = file_reader_phase4.get_item('AcquisitionDate').value.decode('utf-8')
        scanned_date_formatted_phase4 = scanned_date_phase4[0:4] + '-' + scanned_date_phase4[
                                                                         4:6] + '-' + scanned_date_phase4[6:]
        scanned_time_phase4 = file_reader_phase4.get_item('AcquisitionTime').value.decode('utf-8')
        scanned_time_formatted_phase4 = scanned_time_phase4[0:2] + ':' + scanned_time_phase4[
                                                                         2:4] + ':' + scanned_time_phase4[4:]
        if file_reader_phase4.get_item('Manufacturer'):
            manufacturer_phase4 = file_reader_phase4.get_item('Manufacturer').value.decode('utf-8')
        if file_reader_phase4.get_item('ManufacturerModelName'):
            manufacturer_model_name_phase4 = file_reader_phase4.get_item('ManufacturerModelName').value.decode('utf-8')
        kilovoltage_phase4.append(file_reader_phase4.get_item('KVP').value.decode('utf-8'))

        # print(scanned_date_formatted_phase1, scanned_date_formatted_phase2, scanned_date_formatted_phase3,
        #       scanned_date_formatted_phase4)
        # print(scanned_time_formatted_phase1, scanned_time_formatted_phase2, scanned_time_formatted_phase3,
        #       scanned_time_formatted_phase4)
        # scan_machine.append([manufacturer_phase1, manufacturer_phase2, manufacturer_phase3, manufacturer_phase4])
        # scan_model.append([manufacturer_model_name_phase1, manufacturer_model_name_phase2,
        #                    manufacturer_model_name_phase3, manufacturer_model_name_phase4])
        # print(kilovoltage_phase1, kilovoltage_phase2, kilovoltage_phase3, kilovoltage_phase4)
        if manufacturer_phase1 and manufacturer_model_name_phase1:
            scan_machine.append([manufacturer_phase1 + manufacturer_model_name_phase1])

        if manufacturer_model_name_phase1:
            scan_model.append(manufacturer_model_name_phase1)

        x1 = time.strptime(scanned_time_formatted_phase1.split('.')[0], '%H:%M:%S')
        x2 = time.strptime(scanned_time_formatted_phase2.split('.')[0], '%H:%M:%S')
        x3 = time.strptime(scanned_time_formatted_phase3.split('.')[0], '%H:%M:%S')
        x4 = time.strptime(scanned_time_formatted_phase4.split('.')[0], '%H:%M:%S')
        # print(scanned_time1_formatted, scanned_time2_formatted, scanned_time3_formatted, scanned_time4_formatted, x1, x2, x3, x4)

        second1 = datetime.timedelta(hours=x1.tm_hour, minutes=x1.tm_min, seconds=x1.tm_sec).total_seconds()
        second2 = datetime.timedelta(hours=x2.tm_hour, minutes=x2.tm_min, seconds=x2.tm_sec).total_seconds()
        second3 = datetime.timedelta(hours=x3.tm_hour, minutes=x3.tm_min, seconds=x3.tm_sec).total_seconds()
        second4 = datetime.timedelta(hours=x4.tm_hour, minutes=x4.tm_min, seconds=x4.tm_sec).total_seconds()

        time_range_phase1_2.append(second2 - second1)
        time_range_phase2_3.append(second3 - second2)
        time_range_phase3_4.append(second4 - second3)
        if second2 - second1 == 1084.0:
            print(phase1_slice1_fullpath, phase2_slice1_fullpath, phase3_slice1_fullpath, phase4_slice1_fullpath, second2-second1)
        if second4 - second3 == 843.0:
            print(phase1_slice1_fullpath, phase2_slice1_fullpath, phase3_slice1_fullpath, phase4_slice1_fullpath, second4-second3)
        if second3 - second2 == 305.0:
            print(phase1_slice1_fullpath, phase2_slice1_fullpath, phase3_slice1_fullpath, phase4_slice1_fullpath, second3-second2)

# Scan Machine Name [TOSHIBA, SIEMENS]
# Scan Model Name [Aquilion, SOMATOM Definition AS+]
print(scan_machine)
print(scan_model)
print(len(np.unique(scan_model)), "Line-768", len(np.unique(scan_machine)))
print(np.unique(scan_machine),  'Line-769',
      np.unique(scan_model))

print(np.mean(np.array(time_range_phase1_2)), np.std(np.array(time_range_phase1_2)),
      np.median(np.array(time_range_phase1_2)), np.max(np.array(time_range_phase1_2)))
print(np.mean(np.array(time_range_phase2_3)), np.std(np.array(time_range_phase2_3)),
      np.median(np.array(time_range_phase2_3)), np.max(np.array(time_range_phase2_3)))
print(np.mean(np.array(time_range_phase3_4)), np.std(np.array(time_range_phase3_4)),
      np.median(np.array(time_range_phase3_4)), np.max(np.array(time_range_phase3_4)))
idx_1 = np.where(np.array(time_range_phase1_2) == np.max(np.array(time_range_phase1_2)))
idx_2 = np.where(np.array(time_range_phase2_3) == np.max(np.array(time_range_phase2_3)))
idx_3 = np.where(np.array(time_range_phase3_4) == np.max(np.array(time_range_phase3_4)))
print(idx_1[0], idx_2[0], idx_3[0])
print(len(kilovoltage_phase1), len(np.unique(kilovoltage_phase1)), np.unique(kilovoltage_phase1), "Phase_1")
print(len(kilovoltage_phase2), len(np.unique(kilovoltage_phase2)), np.unique(kilovoltage_phase2), "Phase_2")
print(len(kilovoltage_phase3), len(np.unique(kilovoltage_phase3)), np.unique(kilovoltage_phase3), "Phase_3")
print(len(kilovoltage_phase4), len(np.unique(kilovoltage_phase4)), np.unique(kilovoltage_phase4), "Phase_4")
machine_name_unique = np.unique(scan_machine)
machine_model_unique = np.unique(scan_model)
print('SZH_Scan_Model')
CNT_1, CNT_2 = 0, 0
for case in scan_machine:
    if case == machine_name_unique[0]:
        CNT_1 += 1
    elif case == machine_name_unique[1]:
        CNT_2 += 1
    else:
        continue
print(np.unique(machine_name_unique))
print(CNT_1, CNT_2)

cnt_1, cnt_2, cnt_3, cnt_4 = 0, 0, 0, 0
for case in scan_model:
    if case == machine_model_unique[0]:
        cnt_1 += 1
    elif case == machine_model_unique[1]:
        cnt_2 += 1
    elif case == machine_model_unique[2]:
        cnt_3 += 1
    elif case == machine_model_unique[3]:
        cnt_4 += 1
    else:
        continue
print(np.unique(machine_model_unique))
print(cnt_1, cnt_2, cnt_3, cnt_4)
print(cnt_1, cnt_2)

print('*#*' * 50)
Share_Four_Phases_QMH_Phase1, Share_Four_Phases_QMH_Phase2, Share_Four_Phases_QMH_Phase3, \
Share_Four_Phases_QMH_Phase4 = [], [], [], []
# There are 375 patients in PYN_Part1
for case in qmh_phase1:
    fullpath = os.path.join(qmh_save_rootpath + '/Phase1_data', case)
    patient_id_phase = os.path.basename(fullpath)
    for shared_case in Shared_Four_Phase_QMH:
        if shared_case in patient_id_phase:
            Share_Four_Phases_QMH_Phase1.append(fullpath)
for case in qmh_phase2:
    fullpath = os.path.join(qmh_save_rootpath + '/Phase2_data', case)
    patient_id_phase = os.path.basename(fullpath)
    for shared_case in Shared_Four_Phase_QMH:
        if shared_case in patient_id_phase:
            Share_Four_Phases_QMH_Phase2.append(fullpath)
for case in qmh_phase3:
    fullpath = os.path.join(qmh_save_rootpath + '/Phase3_data', case)
    patient_id_phase = os.path.basename(fullpath)
    for shared_case in Shared_Four_Phase_QMH:
        if shared_case in patient_id_phase:
            Share_Four_Phases_QMH_Phase3.append(fullpath)
for case in qmh_phase4:
    fullpath = os.path.join(qmh_save_rootpath + '/Phase4_data', case)
    patient_id_phase = os.path.basename(fullpath)
    for shared_case in Shared_Four_Phase_QMH:
        if shared_case in patient_id_phase:
            Share_Four_Phases_QMH_Phase4.append(fullpath)
Sorted_Shared_QMH_Phase1 = sorted(Share_Four_Phases_QMH_Phase1)
Sorted_Shared_QMH_Phase2 = sorted(Share_Four_Phases_QMH_Phase2)
Sorted_Shared_QMH_Phase3 = sorted(Share_Four_Phases_QMH_Phase3)
Sorted_Shared_QMH_Phase4 = sorted(Share_Four_Phases_QMH_Phase4)
print(len(Sorted_Shared_QMH_Phase1), len(Sorted_Shared_QMH_Phase2), len(Sorted_Shared_QMH_Phase3),
      len(Sorted_Shared_QMH_Phase4), "Line-1036")
print(Sorted_Shared_QMH_Phase1)
print(Sorted_Shared_QMH_Phase2)
print(Sorted_Shared_QMH_Phase3)
print(Sorted_Shared_QMH_Phase4)

CNT = 0
time_range_phase1_2, time_range_phase2_3, time_range_phase3_4 = [], [], []
scan_machine, scan_model, kilovoltage_phase1, kilovoltage_phase2, kilovoltage_phase3, \
kilovoltage_phase4, = [], [], [], [], [], []
manufacturer_phase1, manufacturer_phase2, manufacturer_phase3, manufacturer_phase4 = None, None, None, None
manufacturer_model_name_phase1, manufacturer_model_name_phase2, manufacturer_model_name_phase3, \
manufacturer_model_name_phase4 = None, None, None, None
for idx in range(len(Share_Four_Phases_QMH_Phase1)):
    fullpath_phase1 = Sorted_Shared_QMH_Phase1[idx]
    fullpath_phase2 = Sorted_Shared_QMH_Phase2[idx]
    fullpath_phase3 = Sorted_Shared_QMH_Phase3[idx]
    fullpath_phase4 = Sorted_Shared_QMH_Phase4[idx]
    phase1_basename = os.path.basename(fullpath_phase1).split('_P1')[0]
    phase2_basename = os.path.basename(fullpath_phase2).split('_P2')[0]
    phase3_basename = os.path.basename(fullpath_phase3).split('_P3')[0]
    phase4_basename = os.path.basename(fullpath_phase4).split('_P4')[0]
    # print(phase1_basename, phase2_basename, phase3_basename, phase4_basename)
    if phase1_basename == phase2_basename and phase2_basename == phase3_basename and phase3_basename == phase4_basename:
        CNT += 1
        phase1_slice_list = [slice for slice in os.listdir(fullpath_phase1) if not slice.startswith('.')]
        # phase1_slice_list = os.listdir(fullpath_phase1)
        phase2_slice_list = [slice for slice in os.listdir(fullpath_phase2) if not slice.startswith('.')]
        # phase2_slice_list = os.listdir(fullpath_phase2)
        phase3_slice_list = [slice for slice in os.listdir(fullpath_phase3) if not slice.startswith('.')]
        # phase3_slice_list = os.listdir(fullpath_phase3)
        phase4_slice_list = [slice for slice in os.listdir(fullpath_phase4) if not slice.startswith('.')]
        # phase4_slice_list = os.listdir(fullpath_phase4)

        phase1_slice1_fullpath = os.path.join(fullpath_phase1, phase1_slice_list[0])
        phase2_slice1_fullpath = os.path.join(fullpath_phase2, phase2_slice_list[0])
        phase3_slice1_fullpath = os.path.join(fullpath_phase3, phase3_slice_list[0])
        phase4_slice1_fullpath = os.path.join(fullpath_phase4, phase4_slice_list[0])

        file_reader_phase1 = pydicom.dcmread(phase1_slice1_fullpath, force=True)
        file_reader_phase2 = pydicom.dcmread(phase2_slice1_fullpath, force=True)
        file_reader_phase3 = pydicom.dcmread(phase3_slice1_fullpath, force=True)
        file_reader_phase4 = pydicom.dcmread(phase4_slice1_fullpath, force=True)

        # print(phase1_slice1_fullpath, phase3_slice1_fullpath, phase3_slice1_fullpath, phase4_slice1_fullpath)
        scanned_date_phase1 = file_reader_phase1.get_item('AcquisitionDate').value.decode('utf-8')
        scanned_date_formatted_phase1 = scanned_date_phase1[0:4] + '-' + scanned_date_phase1[
                                                                         4:6] + '-' + scanned_date_phase1[6:]
        scanned_time_phase1 = file_reader_phase1.get_item('AcquisitionTime').value.decode('utf-8')
        scanned_time_formatted_phase1 = scanned_time_phase1[0:2] + ':' + scanned_time_phase1[
                                                                         2:4] + ':' + scanned_time_phase1[4:]
        if file_reader_phase1.get_item('Manufacturer'):
            manufacturer_phase1 = file_reader_phase1.get_item('Manufacturer').value.decode('utf-8')
        if file_reader_phase1.get_item('ManufacturerModelName'):
            manufacturer_model_name_phase1 = file_reader_phase1.get_item('ManufacturerModelName').value.decode('utf-8')
        kilovoltage_phase1.append(file_reader_phase1.get_item('KVP').value.decode('utf-8'))
        # print(scanned_date_formatted_phase1, scanned_time_formatted_phase1)

        scanned_date_phase2 = file_reader_phase2.get_item('AcquisitionDate').value.decode('utf-8')
        scanned_date_formatted_phase2 = scanned_date_phase2[0:4] + '-' + scanned_date_phase2[
                                                                         4:6] + '-' + scanned_date_phase2[6:]
        scanned_time_phase2 = file_reader_phase2.get_item('AcquisitionTime').value.decode('utf-8')
        scanned_time_formatted_phase2 = scanned_time_phase2[0:2] + ':' + scanned_time_phase2[
                                                                         2:4] + ':' + scanned_time_phase2[4:]
        if file_reader_phase2.get_item('Manufacturer') :
            manufacturer_phase2 = file_reader_phase2.get_item('Manufacturer').value.decode('utf-8')
        if file_reader_phase2.get_item('ManufacturerModelName'):
            manufacturer_model_name_phase2 = file_reader_phase2.get_item('ManufacturerModelName').value.decode('utf-8')
        kilovoltage_phase2.append(file_reader_phase2.get_item('KVP').value.decode('utf-8'))

        scanned_date_phase3 = file_reader_phase3.get_item('AcquisitionDate').value.decode('utf-8')
        scanned_date_formatted_phase3 = scanned_date_phase3[0:4] + '-' + scanned_date_phase3[
                                                                         4:6] + '-' + scanned_date_phase3[6:]
        scanned_time_phase3 = file_reader_phase3.get_item('AcquisitionTime').value.decode('utf-8')
        scanned_time_formatted_phase3 = scanned_time_phase3[0:2] + ':' + scanned_time_phase3[
                                                                         2:4] + ':' + scanned_time_phase3[4:]
        if file_reader_phase3.get_item('Manufacturer'):
            manufacturer_phase3 = file_reader_phase3.get_item('Manufacturer').value.decode('utf-8')
        if file_reader_phase3.get_item('ManufacturerModelName'):
            manufacturer_model_name_phase3 = file_reader_phase3.get_item('ManufacturerModelName').value.decode('utf-8')
        kilovoltage_phase3.append(file_reader_phase3.get_item('KVP').value.decode('utf-8'))

        scanned_date_phase4 = file_reader_phase4.get_item('AcquisitionDate').value.decode('utf-8')
        scanned_date_formatted_phase4 = scanned_date_phase4[0:4] + '-' + scanned_date_phase4[
                                                                         4:6] + '-' + scanned_date_phase4[6:]
        scanned_time_phase4 = file_reader_phase4.get_item('AcquisitionTime').value.decode('utf-8')
        scanned_time_formatted_phase4 = scanned_time_phase4[0:2] + ':' + scanned_time_phase4[
                                                                         2:4] + ':' + scanned_time_phase4[4:]
        if file_reader_phase4.get_item('Manufacturer'):
            manufacturer_phase4 = file_reader_phase4.get_item('Manufacturer').value.decode('utf-8')
        if file_reader_phase4.get_item('ManufacturerModelName'):
            manufacturer_model_name_phase4 = file_reader_phase4.get_item('ManufacturerModelName').value.decode('utf-8')
        kilovoltage_phase4.append(file_reader_phase4.get_item('KVP').value.decode('utf-8'))

        # print(scanned_date_formatted_phase1, scanned_date_formatted_phase2, scanned_date_formatted_phase3,
        #       scanned_date_formatted_phase4)
        # print(scanned_time_formatted_phase1, scanned_time_formatted_phase2, scanned_time_formatted_phase3,
        #       scanned_time_formatted_phase4)
        # scan_machine.append([manufacturer_phase1, manufacturer_phase2, manufacturer_phase3, manufacturer_phase4])
        # scan_model.append([manufacturer_model_name_phase1, manufacturer_model_name_phase2,
        #                    manufacturer_model_name_phase3, manufacturer_model_name_phase4])
        # print(kilovoltage_phase1, kilovoltage_phase2, kilovoltage_phase3, kilovoltage_phase4)
        if manufacturer_phase1:
            scan_machine.append([manufacturer_phase1 + manufacturer_model_name_phase1])
        if manufacturer_model_name_phase1:
            scan_model.append(manufacturer_model_name_phase1)

        x1 = time.strptime(scanned_time_formatted_phase1.split('.')[0], '%H:%M:%S')
        x2 = time.strptime(scanned_time_formatted_phase2.split('.')[0], '%H:%M:%S')
        x3 = time.strptime(scanned_time_formatted_phase3.split('.')[0], '%H:%M:%S')
        x4 = time.strptime(scanned_time_formatted_phase4.split('.')[0], '%H:%M:%S')
        # print(scanned_time1_formatted, scanned_time2_formatted, scanned_time3_formatted, scanned_time4_formatted, x1, x2, x3, x4)

        second1 = datetime.timedelta(hours=x1.tm_hour, minutes=x1.tm_min, seconds=x1.tm_sec).total_seconds()
        second2 = datetime.timedelta(hours=x2.tm_hour, minutes=x2.tm_min, seconds=x2.tm_sec).total_seconds()
        second3 = datetime.timedelta(hours=x3.tm_hour, minutes=x3.tm_min, seconds=x3.tm_sec).total_seconds()
        second4 = datetime.timedelta(hours=x4.tm_hour, minutes=x4.tm_min, seconds=x4.tm_sec).total_seconds()

        time_range_phase1_2.append(second2 - second1)
        time_range_phase2_3.append(second3 - second2)
        time_range_phase3_4.append(second4 - second3)

print(np.unique(scan_machine), np.unique(scan_model))
print(time_range_phase1_2)
print(time_range_phase2_3)
print(time_range_phase3_4)
print(scan_machine)
print(scan_model)
print(len(np.unique(scan_model)), "Line-1154", len(np.unique(scan_machine)))
print(np.unique(scan_machine),  'Line-1155',
      np.unique(scan_model))
print(np.mean(np.array(time_range_phase1_2)), np.std(np.array(time_range_phase1_2)),
      np.median(np.array(time_range_phase1_2)))
print(np.mean(np.array(time_range_phase2_3)), np.std(np.array(time_range_phase2_3)),
      np.median(np.array(time_range_phase2_3)))
print(np.mean(np.array(time_range_phase3_4)), np.std(np.array(time_range_phase3_4)),
      np.median(np.array(time_range_phase3_4)))
print(len(kilovoltage_phase1), len(np.unique(kilovoltage_phase1)), np.unique(kilovoltage_phase1), "Phase_1")
print(len(kilovoltage_phase2), len(np.unique(kilovoltage_phase2)), np.unique(kilovoltage_phase2), "Phase_2")
print(len(kilovoltage_phase3), len(np.unique(kilovoltage_phase3)), np.unique(kilovoltage_phase3), "Phase_3")
print(len(kilovoltage_phase4), len(np.unique(kilovoltage_phase4)), np.unique(kilovoltage_phase4), "Phase_4")
machine_name_unique = np.unique(scan_machine)
machine_model_unique = np.unique(scan_model)
print('QMH_Scan_Model', len(time_range_phase1_2), len(scan_machine), len(scan_model), "Line-1169")
CNT_1, CNT_2 = 0, 0
for case in scan_machine:
    if case == machine_name_unique[0]:
        CNT_1 += 1
    elif case == machine_name_unique[1]:
        CNT_2 += 1
    else:
        continue
print(np.unique(machine_name_unique))
print(CNT_1, CNT_2)

cnt_1, cnt_2, cnt_3, cnt_4 = 0, 0, 0, 0
for case in scan_model:
    if case == machine_model_unique[0]:
        cnt_1 += 1
    elif case == machine_model_unique[1]:
        cnt_2 += 1
    elif case == machine_model_unique[2]:
        cnt_3 += 1
    elif case == machine_model_unique[3]:
        cnt_4 += 1
    else:
        continue
print(np.unique(machine_model_unique))
print(cnt_1, cnt_2, cnt_3, cnt_4)

print(cnt_1, cnt_2)
print('*#*' * 50)
Share_Four_Phases_KWH_Phase1, Share_Four_Phases_KWH_Phase2, Share_Four_Phases_KWH_Phase3, \
Share_Four_Phases_KWH_Phase4 = [], [], [], []
# There are 375 patients in PYN_Part1
for case in kwh_phase1:
    fullpath = os.path.join(kwh_save_rootpath + '/Phase1_data', case)
    patient_id_phase = os.path.basename(fullpath)
    for shared_case in Shared_Four_Phase_KWH:
        if shared_case in patient_id_phase:
            Share_Four_Phases_KWH_Phase1.append(fullpath)
for case in kwh_phase2:
    fullpath = os.path.join(kwh_save_rootpath + '/Phase2_data', case)
    patient_id_phase = os.path.basename(fullpath)
    for shared_case in Shared_Four_Phase_KWH:
        if shared_case in patient_id_phase:
            Share_Four_Phases_KWH_Phase2.append(fullpath)
for case in kwh_phase3:
    fullpath = os.path.join(kwh_save_rootpath + '/Phase3_data', case)
    patient_id_phase = os.path.basename(fullpath)
    for shared_case in Shared_Four_Phase_KWH:
        if shared_case in patient_id_phase:
            Share_Four_Phases_KWH_Phase3.append(fullpath)
for case in kwh_phase4:
    fullpath = os.path.join(kwh_save_rootpath + '/Phase4_data', case)
    patient_id_phase = os.path.basename(fullpath)
    for shared_case in Shared_Four_Phase_KWH:
        if shared_case in patient_id_phase:
            Share_Four_Phases_KWH_Phase4.append(fullpath)
Sorted_Shared_KWH_Phase1 = sorted(Share_Four_Phases_KWH_Phase1)
Sorted_Shared_KWH_Phase2 = sorted(Share_Four_Phases_KWH_Phase2)
Sorted_Shared_KWH_Phase3 = sorted(Share_Four_Phases_KWH_Phase3)
Sorted_Shared_KWH_Phase4 = sorted(Share_Four_Phases_KWH_Phase4)
print(len(Sorted_Shared_KWH_Phase1), len(Sorted_Shared_KWH_Phase2), len(Sorted_Shared_KWH_Phase3),
      len(Sorted_Shared_KWH_Phase4))
CNT = 0
time_range_phase1_2, time_range_phase2_3, time_range_phase3_4 = [], [], []
scan_machine, scan_model, kilovoltage_phase1, kilovoltage_phase2, kilovoltage_phase3, \
kilovoltage_phase4, = [], [], [], [], [], []
manufacturer_phase1, manufacturer_phase2, manufacturer_phase3, manufacturer_phase4 = None, None, None, None
manufacturer_model_name_phase1, manufacturer_model_name_phase2, manufacturer_model_name_phase3, \
manufacturer_model_name_phase4 = None, None, None, None
for idx in range(len(Share_Four_Phases_KWH_Phase1)):
    fullpath_phase1 = Sorted_Shared_KWH_Phase1[idx]
    fullpath_phase2 = Sorted_Shared_KWH_Phase2[idx]
    fullpath_phase3 = Sorted_Shared_KWH_Phase3[idx]
    fullpath_phase4 = Sorted_Shared_KWH_Phase4[idx]
    phase1_basename = os.path.basename(fullpath_phase1).split('_P1')[0]
    phase2_basename = os.path.basename(fullpath_phase2).split('_P2')[0]
    phase3_basename = os.path.basename(fullpath_phase3).split('_P3')[0]
    phase4_basename = os.path.basename(fullpath_phase4).split('_P4')[0]
    if phase1_basename == phase2_basename and phase2_basename == phase3_basename and phase3_basename == phase4_basename:
        CNT += 1
        phase1_slice_list = [slice for slice in os.listdir(fullpath_phase1) if not slice.startswith('.')]
        phase2_slice_list = [slice for slice in os.listdir(fullpath_phase2) if not slice.startswith('.')]
        phase3_slice_list = [slice for slice in os.listdir(fullpath_phase3) if not slice.startswith('.')]
        phase4_slice_list = [slice for slice in os.listdir(fullpath_phase4) if not slice.startswith('.')]
        # phase1_slice_list = os.listdir(fullpath_phase1)
        # phase2_slice_list = os.listdir(fullpath_phase2)
        # phase3_slice_list = os.listdir(fullpath_phase3)
        # phase4_slice_list = os.listdir(fullpath_phase4)

        phase1_slice1_fullpath = os.path.join(fullpath_phase1, phase1_slice_list[0])
        phase2_slice1_fullpath = os.path.join(fullpath_phase2, phase2_slice_list[0])
        phase3_slice1_fullpath = os.path.join(fullpath_phase3, phase3_slice_list[0])
        phase4_slice1_fullpath = os.path.join(fullpath_phase4, phase4_slice_list[0])

        file_reader_phase1 = pydicom.dcmread(phase1_slice1_fullpath, force=True)
        file_reader_phase2 = pydicom.dcmread(phase2_slice1_fullpath, force=True)
        file_reader_phase3 = pydicom.dcmread(phase3_slice1_fullpath, force=True)
        file_reader_phase4 = pydicom.dcmread(phase4_slice1_fullpath, force=True)

        # print(phase1_slice1_fullpath, phase3_slice1_fullpath, phase3_slice1_fullpath, phase4_slice1_fullpath)
        scanned_date_phase1 = file_reader_phase1.get_item('AcquisitionDate').value.decode('utf-8')
        scanned_date_formatted_phase1 = scanned_date_phase1[0:4] + '-' + scanned_date_phase1[
                                                                         4:6] + '-' + scanned_date_phase1[6:]
        scanned_time_phase1 = file_reader_phase1.get_item('AcquisitionTime').value.decode('utf-8')
        scanned_time_formatted_phase1 = scanned_time_phase1[0:2] + ':' + scanned_time_phase1[
                                                                         2:4] + ':' + scanned_time_phase1[4:]
        if file_reader_phase1.get_item('Manufacturer'):
            manufacturer_phase1 = file_reader_phase1.get_item('Manufacturer').value.decode('utf-8')
        if file_reader_phase1.get_item('ManufacturerModelName'):
            manufacturer_model_name_phase1 = file_reader_phase1.get_item('ManufacturerModelName').value.decode('utf-8')
        kilovoltage_phase1.append(file_reader_phase1.get_item('KVP').value.decode('utf-8'))
        # print(scanned_date_formatted_phase1, scanned_time_formatted_phase1)

        scanned_date_phase2 = file_reader_phase2.get_item('AcquisitionDate').value.decode('utf-8')
        scanned_date_formatted_phase2 = scanned_date_phase2[0:4] + '-' + scanned_date_phase2[
                                                                         4:6] + '-' + scanned_date_phase2[6:]
        scanned_time_phase2 = file_reader_phase2.get_item('AcquisitionTime').value.decode('utf-8')
        scanned_time_formatted_phase2 = scanned_time_phase2[0:2] + ':' + scanned_time_phase2[
                                                                         2:4] + ':' + scanned_time_phase2[4:]
        if file_reader_phase2.get_item('Manufacturer'):
            manufacturer_phase2 = file_reader_phase2.get_item('Manufacturer').value.decode('utf-8')
        if file_reader_phase2.get_item('ManufacturerModelName'):
            manufacturer_model_name_phase2 = file_reader_phase2.get_item('ManufacturerModelName').value.decode('utf-8')
        kilovoltage_phase2.append(file_reader_phase2.get_item('KVP').value.decode('utf-8'))

        scanned_date_phase3 = file_reader_phase3.get_item('AcquisitionDate').value.decode('utf-8')
        scanned_date_formatted_phase3 = scanned_date_phase3[0:4] + '-' + scanned_date_phase3[
                                                                         4:6] + '-' + scanned_date_phase3[6:]
        scanned_time_phase3 = file_reader_phase3.get_item('AcquisitionTime').value.decode('utf-8')
        scanned_time_formatted_phase3 = scanned_time_phase3[0:2] + ':' + scanned_time_phase3[
                                                                         2:4] + ':' + scanned_time_phase3[4:]
        if file_reader_phase3.get_item('Manufacturer'):
            manufacturer_phase3 = file_reader_phase3.get_item('Manufacturer').value.decode('utf-8')
        if file_reader_phase3.get_item('ManufacturerModelName'):
            manufacturer_model_name_phase3 = file_reader_phase3.get_item('ManufacturerModelName').value.decode('utf-8')
        kilovoltage_phase3.append(file_reader_phase3.get_item('KVP').value.decode('utf-8'))

        scanned_date_phase4 = file_reader_phase4.get_item('AcquisitionDate').value.decode('utf-8')
        scanned_date_formatted_phase4 = scanned_date_phase4[0:4] + '-' + scanned_date_phase4[
                                                                         4:6] + '-' + scanned_date_phase4[6:]
        scanned_time_phase4 = file_reader_phase4.get_item('AcquisitionTime').value.decode('utf-8')
        scanned_time_formatted_phase4 = scanned_time_phase4[0:2] + ':' + scanned_time_phase4[
                                                                         2:4] + ':' + scanned_time_phase4[4:]
        if file_reader_phase4.get_item('Manufacturer'):
            manufacturer_phase4 = file_reader_phase4.get_item('Manufacturer').value.decode('utf-8')
        if file_reader_phase4.get_item('ManufacturerModelName'):
            manufacturer_model_name_phase4 = file_reader_phase4.get_item('ManufacturerModelName').value.decode('utf-8')
        kilovoltage_phase4.append(file_reader_phase4.get_item('KVP').value.decode('utf-8'))

        # print(scanned_date_formatted_phase1, scanned_date_formatted_phase2, scanned_date_formatted_phase3,
        #       scanned_date_formatted_phase4)
        # print(scanned_time_formatted_phase1, scanned_time_formatted_phase2, scanned_time_formatted_phase3,
        #       scanned_time_formatted_phase4)
        # scan_machine.append([manufacturer_phase1, manufacturer_phase2, manufacturer_phase3, manufacturer_phase4])
        # scan_model.append([manufacturer_model_name_phase1, manufacturer_model_name_phase2,
        #                    manufacturer_model_name_phase3, manufacturer_model_name_phase4])
        # print(kilovoltage_phase1, kilovoltage_phase2, kilovoltage_phase3, kilovoltage_phase4)
        if manufacturer_phase1 and manufacturer_model_name_phase1:
            scan_machine.append([manufacturer_phase1 + manufacturer_model_name_phase1])

        if manufacturer_model_name_phase1:
            scan_model.append(manufacturer_model_name_phase1)

        x1 = time.strptime(scanned_time_formatted_phase1.split('.')[0], '%H:%M:%S')
        x2 = time.strptime(scanned_time_formatted_phase2.split('.')[0], '%H:%M:%S')
        x3 = time.strptime(scanned_time_formatted_phase3.split('.')[0], '%H:%M:%S')
        x4 = time.strptime(scanned_time_formatted_phase4.split('.')[0], '%H:%M:%S')
        # print(scanned_time1_formatted, scanned_time2_formatted, scanned_time3_formatted, scanned_time4_formatted, x1, x2, x3, x4)

        second1 = datetime.timedelta(hours=x1.tm_hour, minutes=x1.tm_min, seconds=x1.tm_sec).total_seconds()
        second2 = datetime.timedelta(hours=x2.tm_hour, minutes=x2.tm_min, seconds=x2.tm_sec).total_seconds()
        second3 = datetime.timedelta(hours=x3.tm_hour, minutes=x3.tm_min, seconds=x3.tm_sec).total_seconds()
        second4 = datetime.timedelta(hours=x4.tm_hour, minutes=x4.tm_min, seconds=x4.tm_sec).total_seconds()

        time_range_phase1_2.append(second2 - second1)
        time_range_phase2_3.append(second3 - second2)
        time_range_phase3_4.append(second4 - second3)

# Scan Machine Name [TOSHIBA, SIEMENS]
# Scan Model Name [Aquilion, SOMATOM Definition AS+]
print(time_range_phase1_2)
print(time_range_phase2_3)
print(time_range_phase3_4)
print(scan_machine)
print(scan_model)
print(len(np.unique(scan_model)), "Line-768", len(np.unique(scan_machine)))
print(np.unique(scan_machine),  'Line-769',
      np.unique(scan_model))
print(np.mean(np.array(time_range_phase1_2)), np.std(np.array(time_range_phase1_2)),
      np.median(np.array(time_range_phase1_2)))
print(np.mean(np.array(time_range_phase2_3)), np.std(np.array(time_range_phase2_3)),
      np.median(np.array(time_range_phase2_3)))
print(np.mean(np.array(time_range_phase3_4)), np.std(np.array(time_range_phase3_4)),
      np.median(np.array(time_range_phase3_4)))
print(len(kilovoltage_phase1), len(np.unique(kilovoltage_phase1)), np.unique(kilovoltage_phase1), "Phase_1")
print(len(kilovoltage_phase2), len(np.unique(kilovoltage_phase2)), np.unique(kilovoltage_phase2), "Phase_2")
print(len(kilovoltage_phase3), len(np.unique(kilovoltage_phase3)), np.unique(kilovoltage_phase3), "Phase_3")
print(len(kilovoltage_phase4), len(np.unique(kilovoltage_phase4)), np.unique(kilovoltage_phase4), "Phase_4")
machine_name_unique = np.unique(scan_machine)
machine_model_unique = np.unique(scan_model)
print('HKU_Scan_Model')
CNT_1, CNT_2 = 0, 0
for case in scan_machine:
    if case == machine_name_unique[0]:
        CNT_1 += 1
    elif case == machine_name_unique[1]:
        CNT_2 += 1
    else:
        continue
print(np.unique(machine_name_unique))
print(CNT_1, CNT_2)

cnt_1, cnt_2, cnt_3, cnt_4 = 0, 0, 0, 0
for case in scan_model:
    if case == machine_model_unique[0]:
        cnt_1 += 1
    elif case == machine_model_unique[1]:
        cnt_2 += 1
    elif case == machine_model_unique[2]:
        cnt_3 += 1
    elif case == machine_model_unique[3]:
        cnt_4 += 1
    else:
        continue
print(np.unique(machine_model_unique))
print(cnt_1, cnt_2, cnt_3, cnt_4)