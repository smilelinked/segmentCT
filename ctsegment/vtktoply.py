import os
import vtk

# 指定vtk文件文件夹
vtk_dir = '..\data\VTKfiles'
# 遍历文件夹下的所有vtk文件
for vtk_filename in os.listdir(vtk_dir):
    if not vtk_filename.endswith('.vtk'):
        continue
    # 拼接全路径
    vtk_path = os.path.join(vtk_dir, vtk_filename)
    # 读入vtk文件
    reader = vtk.vtkPolyDataReader()
    reader.SetFileName(vtk_path)
    reader.Update()
    # 添加减面过滤
    simplifier = vtk.vtkQuadricDecimation()
    simplifier.SetInputConnection(reader.GetOutputPort())
    simplifier.SetTargetReduction(0.8)  # 设置目标减面比例
    simplifier.Update()
    polydata = simplifier.GetOutput()
    # 输出的ply文件名
    ply_filename = vtk_filename.replace('.vtk', '.ply')
    ply_path = os.path.join(vtk_dir, ply_filename)
    # 写入ply文件
    writer = vtk.vtkPLYWriter()
    writer.SetFileName(ply_path)
    writer.SetInputData(polydata)
    writer.Write()
print('VTK to PLY conversion complete')
