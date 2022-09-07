import os
import numpy as np
from sklearn.metrics import roc_curve, auc, confusion_matrix, cohen_kappa_score
from FleissKappa import fleissKappa


patient_lesion_first_radiologist = np.load('/home/ra1/original/GZH_Output/Patient_ID_Lesion_LiRad_First_Radiologist.npy')
patient_label_first_radiologist = np.load('/home/ra1/original/GZH_Output/Patient_ID_LiRad_First_Radiologist.npy')

patient_lesion_second_radiologist = np.load('/home/ra1/original/GZH_Output/Patient_ID_Lesion_LiRad_Second_Radiologist.npy')
patient_label_second_radiologist = np.load('/home/ra1/original/GZH_Output/Patient_ID_LiRad_Second_Radiologist.npy')

patient_lesion_third_radiologist = np.load('/home/ra1/original/GZH_Output/Patient_ID_Lesion_LiRad_Third_Radiologist.npy')
patient_label_third_radiologist = np.load('/home/ra1/original/GZH_Output/Patient_ID_LiRad_Third_Radiologist.npy')


print(len(patient_lesion_first_radiologist), len(patient_lesion_second_radiologist), len(patient_lesion_third_radiologist))
print(len(patient_label_first_radiologist), len(patient_label_second_radiologist), len(patient_label_third_radiologist))


Patient_Label_First, Patient_Label_Second, Patient_Label_Third = [], [], []
for i in range(len(patient_label_first_radiologist)):

    case_first = patient_label_first_radiologist[i][0]
    case_second = patient_label_second_radiologist[i][0]
    case_third = patient_label_third_radiologist[i][0]
    # print(patient_label_first_radiologist[i], patient_label_second_radiologist[i], patient_label_third_radiologist[i])
    # print(len(patient_label_first_radiologist[i]), patient_label_first_radiologist[i][1])

    if case_first == case_second and case_first == case_third:
        # continue
        if len(patient_label_first_radiologist[i][1]) == 1:
            Patient_Label_First.append(int(patient_label_first_radiologist[i][1]))
        else:
            Patient_Label_First.append(0)
        if len(patient_label_second_radiologist[i][1]) == 1:
            Patient_Label_Second.append(int(patient_label_second_radiologist[i][1]))
        else:
            Patient_Label_Second.append(0)
        if len(patient_label_third_radiologist[i][1]) == 1:
            Patient_Label_Third.append(int(patient_label_third_radiologist[i][1]))
        else:
            Patient_Label_Third.append(0)
    else:
        print(i, patient_label_first_radiologist[i], patient_label_second_radiologist[i],
              patient_label_third_radiologist[i])
    # print(patient_label_first_radiologist[i])

print(Patient_Label_First==Patient_Label_Second, Patient_Label_First==Patient_Label_Third, Patient_Label_Second==Patient_Label_Third)
print(np.unique(Patient_Label_First), len(Patient_Label_First))
print(np.unique(Patient_Label_Second))
print(np.unique(Patient_Label_Third))

# Confusion_Matrix_First_Second = np.zeros((len(np.unique(Patient_Label_First)), len(np.unique(Patient_Label_Second))))
A11, A12, A13, A14, A15, A16 = 0, 0, 0, 0, 0, 0
A21, A22, A23, A24, A25, A26 = 0, 0, 0, 0, 0, 0
A31, A32, A33, A34, A35, A36 = 0, 0, 0, 0, 0, 0
A41, A42, A43, A44, A45, A46 = 0, 0, 0, 0, 0, 0
A51, A52, A53, A54, A55, A56 = 0, 0, 0, 0, 0, 0
A61, A62, A63, A64, A65, A66 = 0, 0, 0, 0, 0, 0
for i in range(len(Patient_Label_First)):
    j = i
    if Patient_Label_Second[i] == 0 and Patient_Label_Third[j] == 0:
        A11 += 1
    elif Patient_Label_Second[i] == 0 and Patient_Label_Third[j] == 1:
        A12 += 1
    elif Patient_Label_Second[i] == 0 and Patient_Label_Third[j] == 2:
        A13 += 1
    elif Patient_Label_Second[i] == 0 and Patient_Label_Third[j] == 3:
        A14 += 1
    elif Patient_Label_Second[i] == 0 and Patient_Label_Third[j] == 4:
        A15 += 1
    elif Patient_Label_Second[i] == 0 and Patient_Label_Third[j] == 5:
        A16 += 1
    elif Patient_Label_Second[i] == 1 and Patient_Label_Third[j] == 0:
        A21 += 1
    elif Patient_Label_Second[i] == 1 and Patient_Label_Third[j] == 1:
        A22 += 1
    elif Patient_Label_Second[i] == 1 and Patient_Label_Third[j] == 2:
        A23 += 1
    elif Patient_Label_Second[i] == 1 and Patient_Label_Third[j] == 3:
        A24 += 1
    elif Patient_Label_Second[i] == 1 and Patient_Label_Third[j] == 4:
        A25 += 1
    elif Patient_Label_Second[i] == 1 and Patient_Label_Third[j] == 5:
        A26 += 1
    elif Patient_Label_Second[i] == 2 and Patient_Label_Third[j] == 0:
        A31 += 1
    elif Patient_Label_Second[i] == 2 and Patient_Label_Third[j] == 1:
        A32 += 1
    elif Patient_Label_Second[i] == 2 and Patient_Label_Third[j] == 2:
        A33 += 1
    elif Patient_Label_Second[i] == 2 and Patient_Label_Third[j] == 3:
        A34 += 1
    elif Patient_Label_Second[i] == 2 and Patient_Label_Third[j] == 4:
        A35 += 1
    elif Patient_Label_Second[i] == 2 and Patient_Label_Third[j] == 5:
        A36 += 1
    elif Patient_Label_Second[i] == 3 and Patient_Label_Third[j] == 0:
        A41 += 1
    elif Patient_Label_Second[i] == 3 and Patient_Label_Third[j] == 1:
        A42 += 1
    elif Patient_Label_Second[i] == 3 and Patient_Label_Third[j] == 2:
        A43 += 1
    elif Patient_Label_Second[i] == 3 and Patient_Label_Third[j] == 3:
        A44 += 1
    elif Patient_Label_Second[i] == 3 and Patient_Label_Third[j] == 4:
        A45 += 1
    elif Patient_Label_Second[i] == 3 and Patient_Label_Third[j] == 5:
        A46 += 1
    elif Patient_Label_Second[i] == 4 and Patient_Label_Third[j] == 0:
        A51 += 1
    elif Patient_Label_Second[i] == 4 and Patient_Label_Third[j] == 1:
        A52 += 1
    elif Patient_Label_Second[i] == 4 and Patient_Label_Third[j] == 2:
        A53 += 1
    elif Patient_Label_Second[i] == 4 and Patient_Label_Third[j] == 3:
        A54 += 1
    elif Patient_Label_Second[i] == 4 and Patient_Label_Third[j] == 4:
        A55 += 1
    elif Patient_Label_Second[i] == 4 and Patient_Label_Third[j] == 5:
        A56 += 1
    elif Patient_Label_Second[i] == 5 and Patient_Label_Third[j] == 0:
        A61 += 1
    elif Patient_Label_Second[i] == 5 and Patient_Label_Third[j] == 1:
        A62 += 1
    elif Patient_Label_Second[i] == 5 and Patient_Label_Third[j] == 2:
        A63 += 1
    elif Patient_Label_Second[i] == 5 and Patient_Label_Third[j] == 3:
        A64 += 1
    elif Patient_Label_Second[i] == 5 and Patient_Label_Third[j] == 4:
        A65 += 1
    elif Patient_Label_Second[i] == 5 and Patient_Label_Third[j] == 5:
        A66 += 1

Confusion_Matrix_First_Second = np.array([[A11, A12, A13, A14, A15, A16],
                                          [A21, A22, A23, A24, A25, A26],
                                          [A31, A32, A33, A34, A35, A36],
                                          [A41, A42, A43, A44, A45, A46],
                                          [A51, A52, A53, A54, A55, A56],
                                          [A61, A62, A63, A64, A65, A66]])

print(Confusion_Matrix_First_Second)

print(cohen_kappa_score(Patient_Label_First, Patient_Label_Second), "Line-142")
print(cohen_kappa_score(Patient_Label_First, Patient_Label_Third), "Line-143")
print(cohen_kappa_score(Patient_Label_Second, Patient_Label_Third), "Line-144")

Patient_Lesion_First, Patient_Lesion_Second, Patient_Lesion_Third = [], [], []
for i in range(len(patient_lesion_first_radiologist)):
    case_first = patient_lesion_first_radiologist[i][0]
    case_second = patient_lesion_second_radiologist[i][0]
    case_third = patient_lesion_third_radiologist[i][0]
    if case_first == case_second and case_first == case_third:
        if len(patient_lesion_first_radiologist[i][1]) == 1:
            Patient_Lesion_First.append(int(patient_lesion_first_radiologist[i][1]))
        else:
            Patient_Lesion_First.append(0)
        if len(patient_lesion_second_radiologist[i][1]) == 1:
            Patient_Lesion_Second.append(int(patient_lesion_second_radiologist[i][1]))
        else:
            Patient_Lesion_Second.append(0)
        if len(patient_lesion_third_radiologist[i][1]) == 1:
            Patient_Lesion_Third.append(int(patient_lesion_third_radiologist[i][1]))
        else:
            Patient_Lesion_Third.append(0)
    else:
        print(i, patient_lesion_first_radiologist[i], patient_lesion_second_radiologist[i],
              patient_lesion_third_radiologist[i])
    # Patient_Lesion_First.append(patient_lesion_first_radiologist[i][1])
    # Patient_Lesion_Second.append(patient_lesion_second_radiologist[i][1])
    # Patient_Lesion_Third.append(patient_lesion_third_radiologist[i][1])

# Confusion_Matrix_First_Second = np.zeros((len(np.unique(Patient_Label_First)), len(np.unique(Patient_Label_Second))))
A11, A12, A13, A14, A15, A16 = 0, 0, 0, 0, 0, 0
A21, A22, A23, A24, A25, A26 = 0, 0, 0, 0, 0, 0
A31, A32, A33, A34, A35, A36 = 0, 0, 0, 0, 0, 0
A41, A42, A43, A44, A45, A46 = 0, 0, 0, 0, 0, 0
A51, A52, A53, A54, A55, A56 = 0, 0, 0, 0, 0, 0
A61, A62, A63, A64, A65, A66 = 0, 0, 0, 0, 0, 0
for i in range(len(Patient_Lesion_First)):
    j = i
    if Patient_Lesion_Second[i] == 0 and Patient_Lesion_Third[j] == 0:
        A11 += 1
    elif Patient_Lesion_Second[i] == 0 and Patient_Lesion_Third[j] == 1:
        A12 += 1
    elif Patient_Lesion_Second[i] == 0 and Patient_Lesion_Third[j] == 2:
        A13 += 1
    elif Patient_Lesion_Second[i] == 0 and Patient_Lesion_Third[j] == 3:
        A14 += 1
    elif Patient_Lesion_Second[i] == 0 and Patient_Lesion_Third[j] == 4:
        A15 += 1
    elif Patient_Lesion_Second[i] == 0 and Patient_Lesion_Third[j] == 5:
        A16 += 1
    elif Patient_Lesion_Second[i] == 1 and Patient_Lesion_Third[j] == 0:
        A21 += 1
    elif Patient_Lesion_Second[i] == 1 and Patient_Lesion_Third[j] == 1:
        A22 += 1
    elif Patient_Lesion_Second[i] == 1 and Patient_Lesion_Third[j] == 2:
        A23 += 1
    elif Patient_Lesion_Second[i] == 1 and Patient_Lesion_Third[j] == 3:
        A24 += 1
    elif Patient_Lesion_Second[i] == 1 and Patient_Lesion_Third[j] == 4:
        A25 += 1
    elif Patient_Lesion_Second[i] == 1 and Patient_Lesion_Third[j] == 5:
        A26 += 1
    elif Patient_Lesion_Second[i] == 2 and Patient_Lesion_Third[j] == 0:
        A31 += 1
    elif Patient_Lesion_Second[i] == 2 and Patient_Lesion_Third[j] == 1:
        A32 += 1
    elif Patient_Lesion_Second[i] == 2 and Patient_Lesion_Third[j] == 2:
        A33 += 1
    elif Patient_Lesion_Second[i] == 2 and Patient_Lesion_Third[j] == 3:
        A34 += 1
    elif Patient_Lesion_Second[i] == 2 and Patient_Lesion_Third[j] == 4:
        A35 += 1
    elif Patient_Lesion_Second[i] == 2 and Patient_Lesion_Third[j] == 5:
        A36 += 1
    elif Patient_Lesion_Second[i] == 3 and Patient_Lesion_Third[j] == 0:
        A41 += 1
    elif Patient_Lesion_Second[i] == 3 and Patient_Lesion_Third[j] == 1:
        A42 += 1
    elif Patient_Lesion_Second[i] == 3 and Patient_Lesion_Third[j] == 2:
        A43 += 1
    elif Patient_Lesion_Second[i] == 3 and Patient_Lesion_Third[j] == 3:
        A44 += 1
    elif Patient_Lesion_Second[i] == 3 and Patient_Lesion_Third[j] == 4:
        A45 += 1
    elif Patient_Lesion_Second[i] == 3 and Patient_Lesion_Third[j] == 5:
        A46 += 1
    elif Patient_Lesion_Second[i] == 4 and Patient_Lesion_Third[j] == 0:
        A51 += 1
    elif Patient_Lesion_Second[i] == 4 and Patient_Lesion_Third[j] == 1:
        A52 += 1
    elif Patient_Lesion_Second[i] == 4 and Patient_Lesion_Third[j] == 2:
        A53 += 1
    elif Patient_Lesion_Second[i] == 4 and Patient_Lesion_Third[j] == 3:
        A54 += 1
    elif Patient_Lesion_Second[i] == 4 and Patient_Lesion_Third[j] == 4:
        A55 += 1
    elif Patient_Lesion_Second[i] == 4 and Patient_Lesion_Third[j] == 5:
        A56 += 1
    elif Patient_Lesion_Second[i] == 5 and Patient_Lesion_Third[j] == 0:
        A61 += 1
    elif Patient_Lesion_Second[i] == 5 and Patient_Lesion_Third[j] == 1:
        A62 += 1
    elif Patient_Lesion_Second[i] == 5 and Patient_Lesion_Third[j] == 2:
        A63 += 1
    elif Patient_Lesion_Second[i] == 5 and Patient_Lesion_Third[j] == 3:
        A64 += 1
    elif Patient_Lesion_Second[i] == 5 and Patient_Lesion_Third[j] == 4:
        A65 += 1
    elif Patient_Lesion_Second[i] == 5 and Patient_Lesion_Third[j] == 5:
        A66 += 1

Confusion_Matrix_First_Second = np.array([[A11, A12, A13, A14, A15, A16],
                                          [A21, A22, A23, A24, A25, A26],
                                          [A31, A32, A33, A34, A35, A36],
                                          [A41, A42, A43, A44, A45, A46],
                                          [A51, A52, A53, A54, A55, A56],
                                          [A61, A62, A63, A64, A65, A66]])

print(Confusion_Matrix_First_Second)
print(cohen_kappa_score(Patient_Lesion_First, Patient_Lesion_Second), "Line-261")
print(cohen_kappa_score(Patient_Lesion_First, Patient_Lesion_Third), "Line-262")
print(cohen_kappa_score(Patient_Lesion_Second, Patient_Lesion_Third), "Line-263")


# Patient_Lesion_First = Patient_Label_First
# Patient_Lesion_Second = Patient_Label_Second
# Patient_Lesion_Third = Patient_Label_Third

Patient_Label_Radiologist = np.zeros((len(Patient_Label_First), 6))
Patient_Lesion_Radiologist = np.zeros((len(Patient_Lesion_First), 6), dtype=int)
for i in range(len(Patient_Lesion_First)):
    patient_lesion_first = Patient_Lesion_First[i]
    patient_lesion_second = Patient_Lesion_Second[i]
    patient_lesion_third = Patient_Lesion_Third[i]
    # print(patient_lesion_first, patient_lesion_second, patient_lesion_third)
    if patient_lesion_first == 0:
        if patient_lesion_second == 0:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 3
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][0] = 2
                Patient_Lesion_Radiologist[i][1] = 1
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][0] = 2
                Patient_Lesion_Radiologist[i][2] = 1
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][0] = 2
                Patient_Lesion_Radiologist[i][3] = 1
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][0] = 2
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][0] = 2
                Patient_Lesion_Radiologist[i][5] = 1
        elif patient_lesion_second == 1:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 2
                Patient_Lesion_Radiologist[i][1] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][1] = 2
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][2] = 1
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][3] = 1
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][5] = 1
        elif patient_lesion_second == 2:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 2
                Patient_Lesion_Radiologist[i][1] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][2] = 1
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][2] = 2
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][3] = 1
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][5] = 1
        # 0-3-(0, 1, 2, 3, 4, 5)
        elif patient_lesion_second == 3:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 2
                Patient_Lesion_Radiologist[i][3] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][3] = 1
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][3] = 1
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][3] = 2
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][5] = 1
        # 0 - 4 - (0, 1, 2, 3, 4, 5)
        elif patient_lesion_second == 4:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 2
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][4] = 2
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][4] = 1
                Patient_Lesion_Radiologist[i][5] = 1
        # 0 - 5 - (0, 1, 2, 3, 4, 5)
        elif patient_lesion_second == 5:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 2
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][4] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][5] = 2
    # the first lesion of radiologist = 1
    elif patient_lesion_first == 1:
        # 1 - 0 - (0, 1, 2, 3, 4, 5)
        if patient_lesion_second == 0:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 2
                Patient_Lesion_Radiologist[i][1] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][1] = 2
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][2] = 1
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][3] = 1
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][5] = 1
        # 1 - 1 - (0, 1, 2, 3, 4, 5)
        elif patient_lesion_second == 1:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][1] = 2
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][1] = 3
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][1] = 2
                Patient_Lesion_Radiologist[i][2] = 1
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][1] = 2
                Patient_Lesion_Radiologist[i][3] = 1
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][1] = 2
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][5] = 2
        # 1 - 2 - (0, 1, 2, 3, 4, 5)
        elif patient_lesion_second == 2:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][2] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][1] = 2
                Patient_Lesion_Radiologist[i][2] = 1
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][2] = 2
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][3] = 1
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][5] = 1
        # 1 - 3 - (0, 1, 2, 3, 4, 5)
        elif patient_lesion_second == 3:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][3] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][1] = 2
                Patient_Lesion_Radiologist[i][3] = 1
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][3] = 1
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][3] = 2
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][5] = 1
        # 1 - 4 - (0, 1, 2, 3, 4, 5)
        elif patient_lesion_second == 4:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][1] = 2
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][4] = 2
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][4] = 1
                Patient_Lesion_Radiologist[i][5] = 1
        # 1 - 5 - (0, 1, 2, 3, 4, 5)
        elif patient_lesion_second == 5:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][1] = 2
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][4] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][5] = 2
    elif patient_lesion_first == 2:
        # 2 - 0 - (0, 1, 2, 3, 4, 5)
        if patient_lesion_second == 0:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 2
                Patient_Lesion_Radiologist[i][2] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][2] = 1
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][2] = 2
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][3] = 1
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][5] = 1
        # 2 - 1 - (0, 1, 2, 3, 4, 5)
        elif patient_lesion_second == 1:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][2] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][1] = 2
                Patient_Lesion_Radiologist[i][2] = 1
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][2] = 2
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][3] = 1
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][5] = 1
        # 2 - 2 - (0, 1, 2, 3, 4, 5)
        elif patient_lesion_second == 2:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][2] = 2
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][2] = 2
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][2] = 3
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][2] = 2
                Patient_Lesion_Radiologist[i][3] = 1
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][2] = 2
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][2] = 2
                Patient_Lesion_Radiologist[i][5] = 1
        # 2 - 3 - (0, 1, 2, 3, 4, 5)
        elif patient_lesion_second == 3:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][3] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][3] = 1
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][2] = 2
                Patient_Lesion_Radiologist[i][3] = 1
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][3] = 2
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][5] = 1
        # 2 - 4 - (0, 1, 2, 3, 4, 5)
        elif patient_lesion_second == 4:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][2] = 2
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][4] = 2
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][4] = 1
                Patient_Lesion_Radiologist[i][5] = 1
        # 2 - 5 - (0, 1, 2, 3, 4, 5)
        elif patient_lesion_second == 5:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][2] = 2
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][4] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][5] = 2
    elif patient_lesion_first == 3:
        # 3 - 0 - (0, 1, 2, 3, 4, 5)
        if patient_lesion_second == 0:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 2
                Patient_Lesion_Radiologist[i][3] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][3] = 1
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][3] = 1
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][3] = 2
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][5] = 1
        # 3 - 1 - (0, 1, 2, 3, 4, 5)
        elif patient_lesion_second == 1:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][3] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][1] = 2
                Patient_Lesion_Radiologist[i][3] = 1
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][3] = 1
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][3] = 2
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][5] = 1
        # 3 - 2 - (0, 1, 2, 3, 4, 5)
        elif patient_lesion_second == 2:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][3] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][3] = 1
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][2] = 2
                Patient_Lesion_Radiologist[i][3] = 1
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][3] = 2
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][5] = 1
        # 3 - 3 -(0, 1, 2, 3, 4, 5)
        elif patient_lesion_second == 3:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][3] = 2
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][3] = 2
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][3] = 2
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][3] = 3
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][3] = 2
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][3] = 2
                Patient_Lesion_Radiologist[i][5] = 1
        # 3 - 4 - (0, 1, 2, 3, 4, 5)
        elif patient_lesion_second == 4:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][3] = 2
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][4] = 2
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][4] = 1
                Patient_Lesion_Radiologist[i][5] = 1
        # 3 - 5 - (0, 1, 2, 3, 4, 5)
        elif patient_lesion_second == 5:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][3] = 2
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][4] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][5] = 2

    elif patient_lesion_first == 4:
        # 4 - 0- (0, 1, 2, 3, 4, 5)
        if patient_lesion_second == 0:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 2
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][4] = 2
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][4] = 1
                Patient_Lesion_Radiologist[i][5] = 1
        # 4 - 1 - (0, 1, 2, 3, 4, 5)
        elif patient_lesion_second == 1:
                if patient_lesion_third == 0:
                    Patient_Lesion_Radiologist[i][0] = 1
                    Patient_Lesion_Radiologist[i][1] = 1
                    Patient_Lesion_Radiologist[i][4] = 1
                elif patient_lesion_third == 1:
                    Patient_Lesion_Radiologist[i][1] = 2
                    Patient_Lesion_Radiologist[i][4] = 1
                elif patient_lesion_third == 2:
                    Patient_Lesion_Radiologist[i][1] = 1
                    Patient_Lesion_Radiologist[i][2] = 1
                    Patient_Lesion_Radiologist[i][4] = 1
                elif patient_lesion_third == 3:
                    Patient_Lesion_Radiologist[i][1] = 1
                    Patient_Lesion_Radiologist[i][3] = 1
                    Patient_Lesion_Radiologist[i][4] = 1
                elif patient_lesion_third == 4:
                    Patient_Lesion_Radiologist[i][1] = 1
                    Patient_Lesion_Radiologist[i][4] = 2
                elif patient_lesion_third == 5:
                    Patient_Lesion_Radiologist[i][1] = 1
                    Patient_Lesion_Radiologist[i][4] = 1
                    Patient_Lesion_Radiologist[i][5] = 1
        # 4 - 2 - (0, 1, 2, 3, 4, 5)
        elif patient_lesion_second == 2:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][2] = 2
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][4] = 2
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][4] = 1
                Patient_Lesion_Radiologist[i][5] = 1
        # 4 - 3 - (0, 1, 2, 3, 4, 5)
        elif patient_lesion_second == 3:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][3] = 2
                Patient_Lesion_Radiologist[i][4] = 1
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][4] = 2
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][4] = 1
                Patient_Lesion_Radiologist[i][5] = 1
        # 4 - 4 -(0, 1, 2, 3, 4, 5)
        elif patient_lesion_second == 4:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][4] = 2
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][4] = 2
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][4] = 2
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][4] = 2
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][4] = 3
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][4] = 2
                Patient_Lesion_Radiologist[i][5] = 1
        # 4 - 5 - (0, 1, 2, 3, 4, 5)
        elif patient_lesion_second == 5:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][4] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][4] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][4] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][4] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][4] = 2
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][4] = 1
                Patient_Lesion_Radiologist[i][5] = 2
    elif patient_lesion_first == 5:
        # 5 - 0- (0, 1, 2, 3, 4, 5)
        if patient_lesion_second == 0:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 2
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][4] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][5] = 2
        # 5 - 1 - (0, 1, 2, 3, 4, 5)
        elif patient_lesion_second == 1:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][1] = 2
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][4] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][5] = 2
        # 5 - 2 - (0, 1, 2, 3, 4, 5)
        elif patient_lesion_second == 2:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][2] = 2
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][4] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][5] = 2
        # 5 - 3 - (0, 1, 2, 3, 4, 5)
        elif patient_lesion_second == 3:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][3] = 2
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][4] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][5] = 2
        # 5 - 4 - (0, 1, 2, 3, 4, 5)
        elif patient_lesion_second == 4:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][4] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][4] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][4] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][4] = 1
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][4] = 2
                Patient_Lesion_Radiologist[i][5] = 1
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][4] = 1
                Patient_Lesion_Radiologist[i][5] = 2
        # 5 - 5 - (0, 1, 2, 3, 4, 5)
        elif patient_lesion_second == 5:
            if patient_lesion_third == 0:
                Patient_Lesion_Radiologist[i][0] = 1
                Patient_Lesion_Radiologist[i][5] = 2
            elif patient_lesion_third == 1:
                Patient_Lesion_Radiologist[i][1] = 1
                Patient_Lesion_Radiologist[i][5] = 2
            elif patient_lesion_third == 2:
                Patient_Lesion_Radiologist[i][2] = 1
                Patient_Lesion_Radiologist[i][5] = 2
            elif patient_lesion_third == 3:
                Patient_Lesion_Radiologist[i][3] = 1
                Patient_Lesion_Radiologist[i][5] = 2
            elif patient_lesion_third == 4:
                Patient_Lesion_Radiologist[i][4] = 1
                Patient_Lesion_Radiologist[i][5] = 2
            elif patient_lesion_third == 5:
                Patient_Lesion_Radiologist[i][5] = 3

rate = \
[
    [0,0,0,6,0],
    [0,3,0,0,3],
    [0,1,4,0,1],
    [0,0,0,0,6],
    [0,3,0,3,0],
    [2,0,4,0,0],
    [0,0,4,0,2],
    [2,0,3,1,0],
    [2,0,0,4,0],
    [0,0,0,0,6],
    [1,0,0,5,0],
    [1,1,0,4,0],
    [0,3,3,0,0],
    [1,0,0,5,0],
    [0,2,0,3,1],
    [0,0,5,0,1],
    [3,0,0,1,2],
    [5,1,0,0,0],
    [0,2,0,4,0],
    [1,0,2,0,3],
    [0,0,0,0,6],
    [0,1,0,5,0],
    [0,2,0,1,3],
    [2,0,0,4,0],
    [1,0,0,4,1],
    [0,5,0,1,0],
    [4,0,0,0,2],
    [0,2,0,4,0],
    [1,0,5,0,0],
    [0,0,0,0,6]
]
kappa = fleissKappa(rate,6)
print(kappa)
print(type(rate), type(rate[0]))

Patient_Lesion_Radiologist_Int_Array = []
for i in range(len(Patient_Lesion_Radiologist)):
    Row_List = []
    row = Patient_Lesion_Radiologist[i]
    for j in range(len(row)):
        Row_List.append(int(row[j]))
    Patient_Lesion_Radiologist_Int_Array.append(Row_List)

kappa_radiologist = fleissKappa((Patient_Lesion_Radiologist_Int_Array), 3)

CNT_ = 0
for i in range(len(Patient_Lesion_Radiologist_Int_Array)):
    case_list = Patient_Lesion_Radiologist_Int_Array[i]
    if 3 in case_list:
        continue
    else:
        CNT_ += 1

print(kappa_radiologist)
print(CNT_)