import SimpleITK as sitk
import vtk
# 读取NIfTI文件
nii_img = sitk.ReadImage("D:\codes\CBCTseg-1.0.2\data\huibikou.nii.gz")
img_array = sitk.GetArrayFromImage(nii_img)
pixel_array = sitk.GetArrayFromImage(nii_img)
max_hu = pixel_array.max()
min_hu = int(pixel_array.min())
print(max_hu, min_hu)
reader = vtk.vtkNIFTIImageReader()
reader.SetFileName("D:\codes\CBCTseg-1.0.2\data\huibikou.nii.gz")

# 获取数据、spacing、origin等信息
img_array = sitk.GetArrayFromImage(nii_img)
spacing = nii_img.GetSpacing()
origin = nii_img.GetOrigin()

# 设置HU值
marchingCubes = vtk.vtkMarchingCubes()
marchingCubes.SetInputConnection(reader.GetOutputPort())
marchingCubes.SetValue(-1000, -600)

# 创建一个变换对象，并设置平移
transform = vtk.vtkTransform()
transform.Translate(origin)

# 应用变换到Marching Cubes的输出
transformFilter = vtk.vtkTransformFilter()
transformFilter.SetInputConnection(marchingCubes.GetOutputPort())
transformFilter.SetTransform(transform)
transformFilter.Update()

# 平滑滤波
smooth = vtk.vtkSmoothPolyDataFilter()
smooth.SetInputConnection(transformFilter.GetOutputPort())
smooth.SetNumberOfIterations(50)  # 设置迭代次数
smooth.SetRelaxationFactor(0.1)  # 设置松弛因子
smooth.FeatureEdgeSmoothingOff()
smooth.BoundarySmoothingOn()
smooth.Update()

# 创建网格简化模块
simplifier = vtk.vtkQuadricDecimation()
simplifier.SetTargetReduction(0.8)
simplifier.SetInputConnection(smooth.GetOutputPort())

# 创建PLY写入器
plyWriter = vtk.vtkPLYWriter()
plyWriter.SetInputConnection(simplifier.GetOutputPort())
plyWriter.SetFileName("skin.ply")
# 写入PLY文件
plyWriter.Write()