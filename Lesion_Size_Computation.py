import warnings
warnings.filterwarnings("ignore")
import os
import pandas as pd
import numpy as np
np.set_printoptions(precision=4, suppress=True)
import nibabel as nib
import SimpleITK as sitk
import cv2
import cc3d


def resize2d(voxels, length, width):
    resized_voxels = np.zeros((len(voxels), length, width))
    if voxels.dtype != 'int16':
        voxels = voxels.astype('int16')
    else:
        pass
    for i in range(len(voxels)):
        resized_voxels[i] = cv2.resize(voxels[i], (length, width))
    return resized_voxels


# Read dicom volume
def read_dicom(file_dir):
    reader = sitk.ImageSeriesReader()
    dicom_names = reader.GetGDCMSeriesFileNames(file_dir)
    reader.SetFileNames(dicom_names)
    dicom_img = reader.Execute()
    vol_data = sitk.GetArrayFromImage(img_trans(sitk.Cast(dicom_img, sitk.sitkFloat32)))
    vol_dir = dicom_img.GetDirection()
    return vol_data


def img_trans(img):
    new_img = sitk.Cast(sitk.IntensityWindowing(img,
                                                windowMinimum=WINDOW_LEVEL[1] - WINDOW_LEVEL[0] / 2.0,
                                                windowMaximum=WINDOW_LEVEL[1] + WINDOW_LEVEL[0] / 2.0),
                        sitk.sitkUInt8)
    return new_img


# Read segmentation file
def read_nifti(file_dir):
    file = nib.load(file_dir)
    file_voxels = file.get_data()
    file_voxels = np.transpose(file_voxels, (2, 0, 1))
    file_hdr = file.header
    file_affine = file._affine
    return file_voxels, file_hdr, file_affine


# Read segmentation file
def read_mask(file_dir):
    seg_img = sitk.ReadImage(file_dir)
    # vox_data = sitk.GetArrayFromImage(sitk.Cast(seg_img, sitk.sitkFloat32))
    vox_data = sitk.GetArrayFromImage(seg_img)
    vox_dir = seg_img.GetDirection()
    if vox_dir[0] < 0:
        vox_data = np.flip(vox_data, axis=0)
    if vox_dir[4] < 0:
        vox_data = np.flip(vox_data, axis=1)
    return vox_data


# Read the header only of segmentation file
def read_head(file_dir):
    seg_img = nib.load(file_dir)
    seg_header = seg_img.header
    voxel_x, voxel_y, voxel_z = seg_header.get_zooms()
    return voxel_x, voxel_y, voxel_z


# Given a slice, find the bounding boxes and the labels
def _bbox(img):
    a = np.where(img != 0)
    bbox = np.min(a[0]), np.max(a[0]), np.min(a[1]), np.max(a[1])
    # return row1, row2, col1, col2
    return bbox


def _bbox2(img):
    rows = np.any(img, axis=1)
    cols = np.any(img, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]
    return rmin, rmax, cmin, cmax


def _bbox_3D(img):
    r = np.any(img, axis=(1, 2))
    c = np.any(img, axis=(0, 2))
    z = np.any(img, axis=(0, 1))
    rmin, rmax = np.where(r)[0][[0, -1]]
    cmin, cmax = np.where(c)[0][[0, -1]]
    zmin, zmax = np.where(z)[0][[0, -1]]
    return rmin, rmax, cmin, cmax, zmin, zmax


# Global parameters
Length = 512
Width = 512
WINDOW_LEVEL = (400, 40)
ranseed = 1234

rootpath_mask_files = '/home/ra1/original/Finalized_Mask/Finalised Mask Update 2021_04_23'
subfolder_list = os.listdir(rootpath_mask_files)
print(subfolder_list)

"""
patient_mask_fullpath_lists = []
patient_name_list = []
for i in subfolder_list:
    subfolder_path = os.path.join(rootpath_mask_files, i)
    subfolder_patient_list = os.listdir(subfolder_path)
    for j in subfolder_patient_list:
        fullpath = os.path.join(subfolder_path, j)
        filename = os.path.basename(fullpath).split('.')[0]
        if 'P3' in filename:
            patient_mask_fullpath_lists.append(fullpath)
"""
patient_mask_fullpath_lists = ['/home/ra1/original/GZH_Output/GZH Mask Lesion Lirad/CholangCA/19-10541156.nii.gz',
                               '/home/ra1/original/GZH_Output/GZH Mask Lesion Lirad/GZH HCC/355187.nii.gz',
                               '/home/ra1/original/GZH_Output/GZH Mask Lesion Lirad/GZH HCC/345085.nii.gz',
                               '/home/ra1/original/GZH_Output/GZH Mask Lesion Lirad/GZH HCC/345041.nii.gz',
                               '/home/ra1/original/GZH_Output/GZH Mask Lesion Lirad/GZH HCC/359950.nii.gz',
                               ]


print(len(patient_mask_fullpath_lists), 'Line-119')

lesion_df = pd.DataFrame(columns=['Patient ID', 'Phase', 'Maximum Distance', 'Longest Diameter', 'Lesion_Class'])

file = open('/home/ra1/Desktop/Lesion_Size_Computation_External_Update_Five_Cases.txt', "w")
CNT = 0

for i in patient_mask_fullpath_lists:
    seg_vox, seg_hdr, _ = read_nifti(i)
    sx, sy, sz = seg_hdr.get_zooms()

    print(sx, sy, sz)
    print(III)
    labels_in = seg_vox
    connectivity = 6
    labels_out = cc3d.connected_components(labels_in, connectivity=connectivity)
    N = np.max(labels_out)
    # print(sx, sy, sz, labels_in.shape, labels_out.shape, N)
    if N > 0:
        for seg_value in range(1, N+1):
            extracted_image = labels_in * (labels_out == seg_value)
            min_y, max_y, min_x, max_x, min_z, max_z = _bbox_3D(extracted_image)
            size = max(max_y - min_y, max_x - min_x, max_z - min_z) + 1
            size_mm = max((max_y - min_y + 1) * sy, (max_x - min_x + 1) * sx, (max_z - min_z + 1) * sz)
            size_xx = (max_x - min_x + 1) * sx
            size_yy = (max_y - min_y + 1) * sy
            size_zz = (max_z - min_z + 1) * sz
            long_diag = np.sqrt(((max_y - min_y + 1) * sy)**2 + ((max_x - min_x + 1) * sx)**2 + ((max_z - min_z + 1)*sz)**2)
            lesion_class = np.max(extracted_image)
            print(os.path.basename(i).split('.')[0], lesion_class, long_diag, size_xx, size_yy, size_zz, size_mm)
            file.write(os.path.basename(i).split('.')[0] + ' ' + str(lesion_class) + ' ' + str(long_diag) + ' '
                       + str(size_xx) + ' ' + str(size_yy) + ' ' + str(size_zz) + ' ' + str(size_mm) + '\n')

file.close()








