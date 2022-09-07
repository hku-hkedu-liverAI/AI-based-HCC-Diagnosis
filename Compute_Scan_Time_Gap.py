import os
import numpy
import glob
import pydicom
import datetime
import time
import numpy as np


data_rootpath_pyn_part2 = '/home/ra1/original/PYN_Part2'
data_rootpath_pyn_part1 = '/home/ra1/original/PreRegDataAII'
data_rootpath_hku = '/home/ra1/original/PreRegData_HKU'
data_rootpath_szh_hku = '/home/ra1/original/PreRegData_SZH'

data_phases_rootpath_pyn_part2 = glob.glob(data_rootpath_pyn_part2+'/Phase*_data')
data_phases_rootpath_pyn_part1 = glob.glob(data_rootpath_pyn_part1+'/Phase*_data')
data_phases_rootpath_hku = glob.glob(data_rootpath_hku+'/Phase*_data')
data_phases_rootpath_szh_hku = glob.glob(data_rootpath_szh_hku+'/Phase*_data')

print(len(data_phases_rootpath_pyn_part2), len(data_phases_rootpath_pyn_part1), len(data_phases_rootpath_hku),
      len(data_phases_rootpath_szh_hku), "Line-16")
"""
phase_cases_pyn_part2 = []
phase_list_list = []
for i in data_phases_rootpath_pyn_part2:
    phase_list = glob.glob(i+'/ID*')
    phase_list_list.append(phase_list)
    phase_cases_pyn_part2.append(len(phase_list))
"""
phase_cases_pyn_part2 = []
phase_list_list = []
for i in data_phases_rootpath_szh_hku:
    phase_list = sorted(glob.glob(i+'/SZ*'))
    phase_list_list.append(phase_list)
    phase_cases_pyn_part2.append(len(phase_list))
"""
for i in data_phases_rootpath_pyn_part2:
    phase_list = sorted(glob.glob(i+'/ID*'))
    phase_list_list.append(phase_list)
    phase_cases_pyn_part2.append(len(phase_list))
"""

Count1, Count2 = 0, 0
print(phase_cases_pyn_part2, len(phase_list_list), len(phase_list), "Line-37")
gap1_pyn_part2, gap2_pyn_part2, gap3_pyn_part2 = [], [], []
for i in range(len(phase_list)):
    phase1_fullpath = phase_list_list[0][i]
    phase2_fullpath = phase_list_list[1][i]
    phase3_fullpath = phase_list_list[2][i]
    phase4_fullpath = phase_list_list[3][i]
    # print(phase1_fullpath, phase2_fullpath, phase3_fullpath, phase4_fullpath)

    phase1_fullpath_slide = glob.glob(phase1_fullpath+'/*')   # sorted(glob.glob(phase1_fullpath+'/*'))
    phase2_fullpath_slide = glob.glob(phase2_fullpath+'/*')   # sorted(glob.glob(phase2_fullpath+'/*'))
    phase3_fullpath_slide = glob.glob(phase3_fullpath+'/*')   # sorted(glob.glob(phase3_fullpath+'/*'))
    phase4_fullpath_slide = glob.glob(phase4_fullpath+'/*')   # sorted(glob.glob(phase4_fullpath+'/*'))
    # print(phase1_fullpath_slide[0], phase2_fullpath_slide[0], phase3_fullpath_slide[0], phase4_fullpath_slide[0])
    phase1_slide = pydicom.dcmread(phase1_fullpath_slide[0])
    phase2_slide = pydicom.dcmread(phase2_fullpath_slide[0])
    phase3_slide = pydicom.dcmread(phase3_fullpath_slide[0])
    phase4_slide = pydicom.dcmread(phase4_fullpath_slide[0])
    scanned_time1 = phase1_slide.get_item('AcquisitionTime').value.decode('utf-8')   # 'AcquisitionTime'
    scanned_time2 = phase2_slide.get_item('AcquisitionTime').value.decode('utf-8')
    scanned_time3 = phase3_slide.get_item('AcquisitionTime').value.decode('utf-8')   # 'AcquisitionTime'
    scanned_time4 = phase4_slide.get_item('AcquisitionTime').value.decode('utf-8')
    scanned_time1_formatted = scanned_time1[0:2] + ':' + scanned_time1[2:4] + ':' + scanned_time1[4:]
    scanned_time2_formatted = scanned_time2[0:2] + ':' + scanned_time2[2:4] + ':' + scanned_time2[4:]
    scanned_time3_formatted = scanned_time3[0:2] + ':' + scanned_time3[2:4] + ':' + scanned_time3[4:]
    scanned_time4_formatted = scanned_time4[0:2] + ':' + scanned_time4[2:4] + ':' + scanned_time4[4:]
    # print()

    x1 = time.strptime(scanned_time1_formatted.split('.')[0], '%H:%M:%S')
    x2 = time.strptime(scanned_time2_formatted.split('.')[0], '%H:%M:%S')
    x3 = time.strptime(scanned_time3_formatted.split('.')[0], '%H:%M:%S')
    x4 = time.strptime(scanned_time4_formatted.split('.')[0], '%H:%M:%S')
    # print(scanned_time1_formatted, scanned_time2_formatted, scanned_time3_formatted, scanned_time4_formatted, x1, x2, x3, x4)

    second1 = datetime.timedelta(hours=x1.tm_hour, minutes=x1.tm_min, seconds=x1.tm_sec).total_seconds()
    second2 = datetime.timedelta(hours=x2.tm_hour, minutes=x2.tm_min, seconds=x2.tm_sec).total_seconds()
    second3 = datetime.timedelta(hours=x3.tm_hour, minutes=x3.tm_min, seconds=x3.tm_sec).total_seconds()
    second4 = datetime.timedelta(hours=x4.tm_hour, minutes=x4.tm_min, seconds=x4.tm_sec).total_seconds()
    # print(phase1_fullpath_slide[0], scanned_time1_formatted, scanned_time2_formatted, scanned_time3_formatted, scanned_time4_formatted, second1, second2, second3, second4)

    if second2-second1 > 900:
        # print(second1, second2, phase_list_list[1][i])
        # print(phase1_fullpath, phase2_fullpath, phase3_fullpath, phase4_fullpath)
        # print(scanned_time1_formatted, scanned_time2_formatted, scanned_time3_formatted, scanned_time4_formatted, 'Type1')
        Count1 += 1
    elif second2 - second1 < 0:
        Count2 += 1
        # print(second2, second1, phase_list_list[1][i], phase_list_list[0][i])
        # print(phase1_fullpath, phase2_fullpath, phase3_fullpath, phase4_fullpath)
        # print(phase1_fullpath, scanned_time1_formatted, scanned_time2_formatted, scanned_time3_formatted, scanned_time4_formatted, 'type2')
        # print(phase1_fullpath_slide[0], phase2_fullpath_slide[0], phase3_fullpath_slide[0], phase4_fullpath_slide[0])
    # if second3 - second2 < 0:
    #    print(phase1_fullpath_slide[0], phase2_fullpath_slide[0], phase3_fullpath_slide[0], phase4_fullpath_slide[0], scanned_time1_formatted, scanned_time2_formatted, scanned_time3_formatted, scanned_time4_formatted)
    if 900>second2-second1 > 0:
        gap1_pyn_part2.append(second2-second1)

    if second3-second2>0:
       gap2_pyn_part2.append(second3-second2)
    # gap2_pyn_part2.append(second3 - second2)

    if 0 < second4 - second3<900:
        gap3_pyn_part2.append(second4-second3)
    else:
        print(os.path.dirname(phase3_fullpath_slide[0]),
              scanned_time1_formatted, scanned_time2_formatted, scanned_time3_formatted, scanned_time4_formatted)

# SZH_0035_P3, SZH_0080_P3
print(len(gap1_pyn_part2), len(gap2_pyn_part2), len(gap3_pyn_part2), "Line-83")
print(Count1, Count2, "Line-84")
print(len(phase_cases_pyn_part2), len(gap2_pyn_part2), "Line-109")
gap1_pyn_part2 = np.array(gap1_pyn_part2)
gap2_pyn_part2 = np.array(gap2_pyn_part2)
gap3_pyn_part2 = np.array(gap3_pyn_part2)

mean_gap1_pyn_part2 = np.mean(gap1_pyn_part2)
mean_gap2_pyn_part2 = np.mean(gap2_pyn_part2)
mean_gap3_pyn_part2 = np.mean(gap3_pyn_part2)

max_gap1_pyn_part2 = np.max(gap1_pyn_part2)
max_gap2_pyn_part2 = np.max(gap2_pyn_part2)
max_gap3_pyn_part2 = np.max(gap3_pyn_part2)

min_gap1_pyn_part2 = np.min(gap1_pyn_part2)
min_gap2_pyn_part2 = np.min(gap2_pyn_part2)
min_gap3_pyn_part2 = np.min(gap3_pyn_part2)

std1_gap1_pyn_part2 = np.std(gap1_pyn_part2)
std2_gap2_pyn_part2 = np.std(gap2_pyn_part2)
std3_gap3_pyn_part2 = np.std(gap3_pyn_part2)

print(mean_gap1_pyn_part2, mean_gap2_pyn_part2, mean_gap3_pyn_part2, "Line-87")
print(max_gap1_pyn_part2, max_gap2_pyn_part2, max_gap3_pyn_part2, "Line-88")
print(min_gap1_pyn_part2, min_gap2_pyn_part2, min_gap3_pyn_part2, "Line-89")
print(std1_gap1_pyn_part2, std2_gap2_pyn_part2, std3_gap3_pyn_part2, "Line-90")

"""
phase_cases_pyn_part1 = []
for i in data_phases_rootpath_pyn_part1:
    phase_list = glob.glob(i+'/ID*')
    phase_cases_pyn_part1.append(len(phase_list))

phase_cases_hku = []
for i in data_phases_rootpath_hku:
    phase_list = glob.glob(i+'/HKU*')
    phase_cases_hku.append(len(phase_list))

phase_cases_szh_hku = []
for i in data_phases_rootpath_szh_hku:
    phase_list = glob.glob(i + '/SZ*')
    phase_cases_szh_hku.append(len(phase_list))
"""


