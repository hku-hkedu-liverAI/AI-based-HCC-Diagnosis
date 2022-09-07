import os
import numpy as np
import SimpleITK as sitk
Window_Level = (400, 40)


def read_mask(file_dir):
    seg_img = sitk.ReadImage(file_dir)
    vox_data = sitk.GetArrayFromImage(seg_img)
    vox_dir = seg_img.GetDirection()
    if vox_dir[0] < 0:
        vox_data = np.flip(vox_data, axis=0)
    if vox_dir[4] < 0:
        vox_data = np.flip(vox_data, axis=1)

    return vox_data


def read_dicom(file_dir):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(file_dir)
    reader.SetFileNames(dicom_names)
    dicom_img = reader.Execute()
    vol_data = sitk.GetArrayFromImage(img_trans(sitk.Cast(dicom_img, sitk.sitkFloat32)))
    vol_dir = dicom_img.GetDirection()

    return vol_data, vol_dir


def img_trans(img):
    new_img = sitk.Cast(sitk.IntensityWindowing(img,
                                                windowMinimum=Window_Level[1] - Window_Level[0]/2.0,
                                                windowMaximum=Window_Level[1] + Window_Level[0]/2),
                        sitk.sitkUInt8)
    return new_img


def compute_lesion_slice(vol_mask):
    slice_length = vol_mask.shape[0]
    slice_nonzero_list = []
    for i in range(slice_length):
        tmp = vol_mask[i, :, :]
        if np.sum(tmp) > 0:
            slice_nonzero_list.append(i)
    return slice_nonzero_list


record_split = open('/home/ra1/Documents/2D_3D_Classification_Segment_Split_Finalized_31_Oct_Check_3_Nov.txt', 'r')
readlines = record_split.readlines()
print(len(readlines))

train_id_record, train_id_label_record = [], []
test_id_record, test_id_label_record = [], []
train_image_path_list, train_masks_path_list = [], []
test_image_path_list, test_masks_path_list = [], []

CNT = 0
for line in readlines:
    CNT += 1
    if 1 < CNT < 1598:
        line_split = line.split('ata/')[1].split(' ')
        case_id = line_split[0]
        case_label = line_split[1][0]
        train_id_record.append(case_id)
        train_id_label_record.append([case_id, case_label])
        image_masks_paths = line.split(' /home')
        masks_fullpath = image_masks_paths[0]
        image_fullpath = '/home' + image_masks_paths[1].split(' ')[0]
        train_image_path_list.append(image_fullpath)
        train_masks_path_list.append(masks_fullpath)
    elif CNT > 1598:
        line_split = line.split('ata/')[1].split(' ')
        case_id = line_split[0]
        case_label = line_split[1][0]
        test_id_record.append(case_id)
        test_id_label_record.append([case_id, case_label])
        image_masks_paths = line.split(' /home')
        masks_fullpath = image_masks_paths[0]
        image_fullpath = '/home' + image_masks_paths[1].split(' ')[0]
        test_image_path_list.append(image_fullpath)
        test_masks_path_list.append(masks_fullpath)
    else:
        continue

print(len(train_image_path_list), len(train_masks_path_list), len(train_id_record))
print(len(test_image_path_list), len(test_masks_path_list), len(test_id_record))

finalized_save_rootpath = '/home/ra1/Documents/Finalized_Clas_Segm_Save'
if not os.path.exists(finalized_save_rootpath):
    os.makedirs(finalized_save_rootpath)

train_save_prefix_path = os.path.join(finalized_save_rootpath, 'Train')
if not os.path.exists(train_save_prefix_path):
    os.makedirs(train_save_prefix_path)
test_save_prefix_path = os.path.join(finalized_save_rootpath, 'Test')
if not os.path.exists(test_save_prefix_path):
    os.makedirs(test_save_prefix_path)
train_image_3d_save_prefix = os.path.join(train_save_prefix_path, 'Image_3D')
train_masks_3d_save_prefix = os.path.join(train_save_prefix_path, 'Masks_3D')
test_image_3d_save_prefix = os.path.join(test_save_prefix_path, 'Image_3D')
test_masks_3d_save_prefix = os.path.join(test_save_prefix_path, 'Masks_3D')
if not os.path.exists(train_image_3d_save_prefix):
    os.makedirs(train_image_3d_save_prefix)
if not os.path.exists(train_masks_3d_save_prefix):
    os.makedirs(train_masks_3d_save_prefix)
if not os.path.exists(test_image_3d_save_prefix):
    os.makedirs(test_image_3d_save_prefix)
if not os.path.exists(test_masks_3d_save_prefix):
    os.makedirs(test_masks_3d_save_prefix)


for idx in range(len(train_image_path_list)):
    patient_id = os.path.basename(train_image_path_list[idx])
    Volumetric_Data, _ = read_dicom(train_image_path_list[idx])  # the shape of output is [num_slices, 512, 512]
    if 'gz' in train_masks_path_list[idx]:
        Volumetric_Mask = read_mask(train_masks_path_list[idx])  # the shape of output is [512, 512, num_slices]
    else:
        sikt_t1 = sitk.ReadImage(train_masks_path_list[idx])
        Volumetric_Mask = sitk.GetArrayFromImage(sikt_t1)       # the shape of output is [N, 512, 512]
    print(Volumetric_Data.shape, Volumetric_Mask.shape, np.unique(Volumetric_Mask), patient_id,
          os.path.basename(train_masks_path_list[idx]), train_id_label_record[idx])

    nonzero_slice_list = compute_lesion_slice(Volumetric_Mask)
    min_lesion_slice_index, max_lesion_slice_index = min(nonzero_slice_list), max(nonzero_slice_list)
    print(min_lesion_slice_index, max_lesion_slice_index)

    if max_lesion_slice_index - min_lesion_slice_index <= 128:    # Slice_Number < 128
        lesion_slices_num = max_lesion_slice_index - min_lesion_slice_index
        if min_lesion_slice_index - (max_lesion_slice_index - min_lesion_slice_index)//2 >= 0:
            start_index = min_lesion_slice_index - (max_lesion_slice_index - min_lesion_slice_index) // 2
            end_index = start_index + 128
            if end_index <= Volumetric_Mask.shape[0]:
                vol_data = Volumetric_Data[start_index:end_index, :, :]
                vol_mask = Volumetric_Mask[start_index:end_index, :, :]
            else:
                vol_data = Volumetric_Data[start_index:Volumetric_Data.shape[0], :, :]
                vol_mask = Volumetric_Mask[start_index:Volumetric_Data.shape[0], :, :]
        else:
            start_index = max(0, min_lesion_slice_index - (max_lesion_slice_index-min_lesion_slice_index)//2)
            end_index = start_index + 128
            if end_index > Volumetric_Mask.shape[0]:
                vol_data = Volumetric_Data[start_index:Volumetric_Mask.shape[0], :, :]
                vol_mask = Volumetric_Mask[start_index:Volumetric_Mask.shape[0], :, :]
            else:
                vol_data = Volumetric_Data[start_index:end_index, :, :]
                vol_mask = Volumetric_Mask[start_index:end_index, :, :]

        np.save(os.path.join(train_image_3d_save_prefix, patient_id+'.npy'), vol_data)
        np.save(os.path.join(train_masks_3d_save_prefix, 'mask_' + patient_id + '.npy'), vol_mask)
    else:
        case_num = (max_lesion_slice_index - min_lesion_slice_index)//128 + 1
        if case_num == 2:
            middle_end_index = (max_lesion_slice_index + min_lesion_slice_index) // 2      # first part end_slice
            first_start_index = max(middle_end_index - 128, 0)                             # first part_start slice
            middle_start_index = middle_end_index + 1                                      # second part start slice
            second_end_index = min(middle_start_index + 128, Volumetric_Mask.shape[0])     # second part end slice
            vol_data = Volumetric_Data[first_start_index:middle_end_index, :, :]
            vol_mask = Volumetric_Mask[first_start_index:middle_end_index, :, :]
            np.save(os.path.join(train_image_3d_save_prefix, patient_id + '_1' + '.npy'), vol_data)
            np.save(os.path.join(train_masks_3d_save_prefix, 'mask_' + patient_id + '_1' + '.npy'), vol_mask)

            vol_data = Volumetric_Data[middle_start_index:second_end_index, :, :]
            vol_mask = Volumetric_Mask[middle_start_index:second_end_index, :, :]
            np.save(os.path.join(train_image_3d_save_prefix, patient_id + '_2' + '.npy'), vol_data)
            np.save(os.path.join(train_masks_3d_save_prefix, 'mask_' + patient_id + '_2' + '.npy'), vol_mask)
        elif case_num == 3:
            middle_point_index = (max_lesion_slice_index + min_lesion_slice_index) // 2
            middle_start_index = middle_point_index - 64
            middle_end_index = middle_point_index + 64
            vol_data = Volumetric_Data[middle_start_index: middle_end_index, :, :]
            vol_mask = Volumetric_Mask[middle_start_index: middle_end_index, :, :]
            np.save(os.path.join(train_image_3d_save_prefix, patient_id + '_2' + '.npy'), vol_data)
            np.save(os.path.join(train_masks_3d_save_prefix, 'mask_' + patient_id + '_2' + '.npy'), vol_mask)

            first_end_slice = middle_start_index - 1
            first_start_index = max(0, first_end_slice - 128)
            vol_data = Volumetric_Data[first_start_index: first_end_slice, :, :]
            vol_mask = Volumetric_Mask[first_start_index: first_end_slice, :, :]
            np.save(os.path.join(train_image_3d_save_prefix, patient_id + '_1' + '.npy'), vol_data)
            np.save(os.path.join(train_masks_3d_save_prefix, 'mask_' + patient_id + '_1' + '.npy'), vol_mask)

            third_start_slice = middle_end_index + 1
            third_end_slice = min(third_start_slice + 128, Volumetric_Mask.shape[0])
            vol_data = Volumetric_Data[third_start_slice: third_end_slice, :, :]
            vol_mask = Volumetric_Mask[third_start_slice: third_end_slice, :, :]
            np.save(os.path.join(train_image_3d_save_prefix, patient_id + '_3' + '.npy'), vol_data)
            np.save(os.path.join(train_masks_3d_save_prefix, 'mask_' + patient_id + '_3' + '.npy'), vol_mask)
        else:
            print(patient_id)


for idx in range(len(test_image_path_list)):
    patient_id = os.path.basename(test_image_path_list[idx])
    Volumetric_Data, _ = read_dicom(test_image_path_list[idx])  # the shape of output is [num_slices, 512, 512]
    if 'gz' in test_masks_path_list[idx]:
        Volumetric_Mask = read_mask(test_masks_path_list[idx])  # the shape of output is [512, 512, num_slices]
    else:
        sikt_t1 = sitk.ReadImage(test_masks_path_list[idx])
        Volumetric_Mask = sitk.GetArrayFromImage(sikt_t1)  # the shape of output is [N, 512, 512]
    print(Volumetric_Data.shape, Volumetric_Mask.shape, np.unique(Volumetric_Mask), patient_id,
          os.path.basename(test_masks_path_list[idx]), test_id_label_record[idx])

    nonzero_slice_list = compute_lesion_slice(Volumetric_Mask)
    min_lesion_slice_index, max_lesion_slice_index = min(nonzero_slice_list), max(nonzero_slice_list)
    print(min_lesion_slice_index, max_lesion_slice_index)

    if max_lesion_slice_index - min_lesion_slice_index <= 128:  # Slice_Number < 128
        lesion_slices_num = max_lesion_slice_index - min_lesion_slice_index
        if min_lesion_slice_index - (max_lesion_slice_index - min_lesion_slice_index) // 2 >= 0:
            start_index = min_lesion_slice_index - (max_lesion_slice_index - min_lesion_slice_index) // 2
            end_index = start_index + 128
            if end_index <= Volumetric_Mask.shape[0]:
                vol_data = Volumetric_Data[start_index:end_index, :, :]
                vol_mask = Volumetric_Mask[start_index:end_index, :, :]
            else:
                vol_data = Volumetric_Data[start_index:Volumetric_Data.shape[0], :, :]
                vol_mask = Volumetric_Mask[start_index:Volumetric_Data.shape[0], :, :]
        else:
            start_index = max(0, min_lesion_slice_index - (max_lesion_slice_index - min_lesion_slice_index) // 2)
            end_index = start_index + 128
            if end_index > Volumetric_Mask.shape[0]:
                vol_data = Volumetric_Data[start_index:Volumetric_Mask.shape[0], :, :]
                vol_mask = Volumetric_Mask[start_index:Volumetric_Mask.shape[0], :, :]
            else:
                vol_data = Volumetric_Data[start_index:end_index, :, :]
                vol_mask = Volumetric_Mask[start_index:end_index, :, :]

        np.save(os.path.join(test_image_3d_save_prefix, patient_id + '.npy'), vol_data)
        np.save(os.path.join(test_masks_3d_save_prefix, 'mask_' + patient_id + '.npy'), vol_mask)
    else:
        case_num = (max_lesion_slice_index - min_lesion_slice_index) // 128 + 1
        if case_num == 2:
            middle_end_index = (max_lesion_slice_index + min_lesion_slice_index) // 2  # first part end_slice
            first_start_index = max(middle_end_index - 128, 0)  # first part_start slice
            middle_start_index = middle_end_index + 1  # second part start slice
            second_end_index = min(middle_start_index + 128, Volumetric_Mask.shape[0])  # second part end slice
            vol_data = Volumetric_Data[first_start_index:middle_end_index, :, :]
            vol_mask = Volumetric_Mask[first_start_index:middle_end_index, :, :]
            np.save(os.path.join(test_image_3d_save_prefix, patient_id + '_1' + '.npy'), vol_data)
            np.save(os.path.join(test_masks_3d_save_prefix, 'mask_' + patient_id + '_1' + '.npy'), vol_mask)

            vol_data = Volumetric_Data[middle_start_index:second_end_index, :, :]
            vol_mask = Volumetric_Mask[middle_start_index:second_end_index, :, :]
            np.save(os.path.join(test_image_3d_save_prefix, patient_id + '_2' + '.npy'), vol_data)
            np.save(os.path.join(test_masks_3d_save_prefix, 'mask_' + patient_id + '_2' + '.npy'), vol_mask)
        elif case_num == 3:
            middle_point_index = (max_lesion_slice_index + min_lesion_slice_index) // 2
            middle_start_index = middle_point_index - 64
            middle_end_index = middle_point_index + 64
            vol_data = Volumetric_Data[middle_start_index: middle_end_index, :, :]
            vol_mask = Volumetric_Mask[middle_start_index: middle_end_index, :, :]
            np.save(os.path.join(test_image_3d_save_prefix, patient_id + '_2' + '.npy'), vol_data)
            np.save(os.path.join(test_masks_3d_save_prefix, 'mask_' + patient_id + '_2' + '.npy'), vol_mask)

            first_end_slice = middle_start_index - 1
            first_start_index = max(0, first_end_slice - 128)
            vol_data = Volumetric_Data[first_start_index: first_end_slice, :, :]
            vol_mask = Volumetric_Mask[first_start_index: first_end_slice, :, :]
            np.save(os.path.join(test_image_3d_save_prefix, patient_id + '_1' + '.npy'), vol_data)
            np.save(os.path.join(test_masks_3d_save_prefix, 'mask_' + patient_id + '_1' + '.npy'), vol_mask)

            third_start_slice = middle_end_index + 1
            third_end_slice = min(third_start_slice + 128, Volumetric_Mask.shape[0])
            vol_data = Volumetric_Data[third_start_slice: third_end_slice, :, :]
            vol_mask = Volumetric_Mask[third_start_slice: third_end_slice, :, :]
            np.save(os.path.join(test_image_3d_save_prefix, patient_id + '_3' + '.npy'), vol_data)
            np.save(os.path.join(test_masks_3d_save_prefix, 'mask_' + patient_id + '_3' + '.npy'), vol_mask)
        else:
            print(patient_id)