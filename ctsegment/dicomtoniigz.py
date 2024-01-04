import SimpleITK as sitk

# 指定输入文件夹和输出文件路径
folderPath = '../data/huibikou'

# 创建 DICOM 文件读取器
reader = sitk.ImageSeriesReader()

# 加载 DICOM 文件序列
dicom_files = reader.GetGDCMSeriesFileNames(folderPath)
reader.SetFileNames(dicom_files)

# 读取 DICOM 文件序列
image = reader.Execute()

sitk.WriteImage(image, folderPath + '.nii.gz')