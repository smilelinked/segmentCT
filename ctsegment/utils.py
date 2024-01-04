import numpy as np  
import SimpleITK as sitk
import itk
import vtk
import os
import torch
import cc3d
from monai.networks.nets import UNETR
# ----- MONAI ------
from monai.transforms import (
    AddChanneld,
    Compose,
    LoadImaged,
    ScaleIntensityd,
    Spacingd,
    ToTensord,
)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
DATA_TYPE = torch.float32
TRANSLATE ={
  "Maxilla" : "MAX",
  "Cranial-base" : "CB",
  "Mandible" : "MAND",
  "Cervical-vertebra" : "CV",
  "Root-canal" : "RC",
  "Mandibular-canal" : "MCAN",
  "Upper-airway" : "UAW",
  "Skin" : "SKIN",
  "Teeth" : "TEETH"
}
INV_TRANSLATE = {}
for k,v in TRANSLATE.items():
    INV_TRANSLATE[v] = k
LABELS = {
    "LARGE":{
        "MAND" : 1,
        "CB" : 2,
        "UAW" : 3,
        "MAX" : 4,
        "CV" : 5,
        "SKIN" : 6,
    },
    "SMALL":{
        "MAND" : 1,
        "RC" : 2,
        "MAX" : 4,
    }
}
LABEL_COLORS = {
    1: [216, 101, 79],
    2: [128, 174, 128],
    3: [0, 0, 0],
    4: [230, 220, 70],
    5: [111, 184, 210],
    6: [172, 122, 101],
}
NAMES_FROM_LABELS = {"LARGE":{}, "SMALL":{}}
for group,data in LABELS.items():
    for k,v in data.items():
        NAMES_FROM_LABELS[group][v] = INV_TRANSLATE[k]
MODELS_GROUP = {
    "LARGE": {
        "FF":{"MAND" : 1,"CB" : 2,"UAW" : 3,"MAX" : 4,"CV" : 5,},
        "SKIN":{"SKIN" : 1,}
    },
    "SMALL": {
        "HD-MAND":{"MAND" : 1},
        "HD-MAX":{"MAX" : 1},
        "RC":{"RC" : 1},
    },
}
def CreatePredTransform(spacing):
    pred_transforms = Compose(
        [
            LoadImaged(keys=["scan"]),
            AddChanneld(keys=["scan"]),
            ScaleIntensityd(
                keys=["scan"],minv = 0.0, maxv = 1.0, factor = None
            ),
            Spacingd(keys=["scan"],pixdim=spacing),
            ToTensord(keys=["scan"]),
        ]
    )
    return pred_transforms
def CorrectHisto(filepath,outpath,min_porcent=0.01,max_porcent = 0.95,i_min=-1500, i_max=4000):

    print("Correcting scan contrast :", filepath)
    input_img = sitk.ReadImage(filepath) 
    input_img = sitk.Cast(input_img, sitk.sitkFloat32)
    img = sitk.GetArrayFromImage(input_img)


    img_min = np.min(img)
    img_max = np.max(img)
    img_range = img_max - img_min
    # print(img_min,img_max,img_range)

    definition = 1000
    histo = np.histogram(img,definition)
    cum = np.cumsum(histo[0])
    cum = cum - np.min(cum)
    cum = cum / np.max(cum)

    res_high = list(map(lambda i: i> max_porcent, cum)).index(True)
    res_max = (res_high * img_range)/definition + img_min

    res_low = list(map(lambda i: i> min_porcent, cum)).index(True)
    res_min = (res_low * img_range)/definition + img_min

    res_min = max(res_min,i_min)
    res_max = min(res_max,i_max)


    # print(res_min,res_min)

    img = np.where(img > res_max, res_max,img)
    img = np.where(img < res_min, res_min,img)

    output = sitk.GetImageFromArray(img)
    output.SetSpacing(input_img.GetSpacing())
    output.SetDirection(input_img.GetDirection())
    output.SetOrigin(input_img.GetOrigin())
    output = sitk.Cast(output, sitk.sitkInt16)


    writer = sitk.ImageFileWriter()
    writer.SetFileName(outpath)
    writer.Execute(output)
    return output
def ItkToSitk(itk_img):
    new_sitk_img = sitk.GetImageFromArray(itk.GetArrayFromImage(itk_img), isVector=itk_img.GetNumberOfComponentsPerPixel()>1)
    new_sitk_img.SetOrigin(tuple(itk_img.GetOrigin()))
    new_sitk_img.SetSpacing(tuple(itk_img.GetSpacing()))
    new_sitk_img.SetDirection(itk.GetArrayFromMatrix(itk_img.GetDirection()).flatten())
    return new_sitk_img
def Rescale(filepath,output_spacing=[0.5, 0.5, 0.5]):
    print("Resample :", filepath, ", with spacing :", output_spacing)
    img = itk.imread(filepath)

    spacing = np.array(img.GetSpacing())
    output_spacing = np.array(output_spacing)

    if not np.array_equal(spacing,output_spacing):

        size = itk.size(img)
        scale = spacing/output_spacing

        output_size = (np.array(size)*scale).astype(int).tolist()
        output_origin = img.GetOrigin()

        #Find new origin
        output_physical_size = np.array(output_size)*np.array(output_spacing)
        input_physical_size = np.array(size)*spacing
        output_origin = np.array(output_origin) - (output_physical_size - input_physical_size)/2.0

        img_info = itk.template(img)[1]
        pixel_type = img_info[0]
        pixel_dimension = img_info[1]

        VectorImageType = itk.Image[pixel_type, pixel_dimension]
        InterpolatorType = itk.LinearInterpolateImageFunction[VectorImageType, itk.D]

        interpolator = InterpolatorType.New()
        resampled_img = ResampleImage(img,output_size,output_spacing,output_origin,img.GetDirection(),interpolator,VectorImageType)
        return resampled_img

    else:
        return img
def ResampleImage(input,size,spacing,origin,direction,interpolator,IVectorImageType,OVectorImageType):
        ResampleType = itk.ResampleImageFilter[IVectorImageType, OVectorImageType]

        # print(input)

        resampleImageFilter = ResampleType.New()
        resampleImageFilter.SetInput(input)
        resampleImageFilter.SetOutputSpacing(spacing.tolist())
        resampleImageFilter.SetOutputOrigin(origin)
        resampleImageFilter.SetOutputDirection(direction)
        resampleImageFilter.SetInterpolator(interpolator)
        resampleImageFilter.SetSize(size)
        resampleImageFilter.Update()

        resampled_img = resampleImageFilter.GetOutput()
        return resampled_img
def SetSpacing(filepath,output_spacing=[0.5, 0.5, 0.5],interpolator="Linear",outpath=-1):
    print("Reading:", filepath)
    img = itk.imread(filepath)
    spacing = np.array(img.GetSpacing())
    output_spacing = np.array(output_spacing)

    if not np.array_equal(spacing,output_spacing):
        size = itk.size(img)
        scale = spacing/output_spacing
        output_size = (np.array(size)*scale).astype(int).tolist()
        output_origin = img.GetOrigin()
        img_info = itk.template(img)[1]
        pixel_type = img_info[0]
        pixel_dimension = img_info[1]

        print(pixel_type)

        VectorImageType = itk.Image[pixel_type, pixel_dimension]

        if interpolator == "NearestNeighbor":
            InterpolatorType = itk.NearestNeighborInterpolateImageFunction[VectorImageType, itk.D]
        elif interpolator == "Linear":
            InterpolatorType = itk.LinearInterpolateImageFunction[VectorImageType, itk.D]

        interpolator = InterpolatorType.New()
        resampled_img = ResampleImage(img,output_size,output_spacing,output_origin,img.GetDirection(),interpolator,VectorImageType,VectorImageType)

        if outpath != -1:
            itk.imwrite(resampled_img, outpath)
        return resampled_img

    else:
        if outpath != -1:
            itk.imwrite(img, outpath)
        return img
def SavePrediction(img,ref_filepath, outpath, output_spacing):
    ref_img = sitk.ReadImage(ref_filepath)
    output = sitk.GetImageFromArray(img)
    output.SetSpacing(output_spacing)
    output.SetDirection(ref_img.GetDirection())
    output.SetOrigin(ref_img.GetOrigin())
    output = sitk.Cast(output, sitk.sitkInt16)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(outpath)
    writer.Execute(output)
def CleanScan(file_path):
    input_img = sitk.ReadImage(file_path)
    closing_radius = 2
    output = sitk.BinaryDilate(input_img, [closing_radius] * input_img.GetDimension())
    output = sitk.BinaryFillhole(output)
    output = sitk.BinaryErode(output, [closing_radius] * output.GetDimension())
    labels_in = sitk.GetArrayFromImage(input_img)
    out, N = cc3d.largest_k(
        labels_in, k=1, 
        connectivity=26, delta=0,
        return_N=True,
    )
    output = sitk.GetImageFromArray(out)
    output.SetSpacing(input_img.GetSpacing())
    output.SetDirection(input_img.GetDirection())
    output.SetOrigin(input_img.GetOrigin())
    output = sitk.Cast(output, sitk.sitkInt16)
    writer = sitk.ImageFileWriter()
    writer.SetFileName(file_path)
    writer.Execute(output)
def SetSpacingFromRef(filepath,refFile,interpolator = "NearestNeighbor",outpath=-1):
    img = itk.imread(filepath)
    ref = itk.imread(refFile)
    img_sp = np.array(img.GetSpacing()) 
    img_size = np.array(itk.size(img))
    ref_sp = np.array(ref.GetSpacing())
    ref_size = np.array(itk.size(ref))
    ref_origin = ref.GetOrigin()
    ref_direction = ref.GetDirection()
    Dimension = 3
    InputPixelType = itk.D
    InputImageType = itk.Image[InputPixelType, Dimension]
    reader = itk.ImageFileReader[InputImageType].New()
    reader.SetFileName(filepath)
    img = reader.GetOutput()

    if not (np.array_equal(img_sp,ref_sp) and np.array_equal(img_size,ref_size)):
        img_info = itk.template(img)[1]
        Ipixel_type = img_info[0]
        Ipixel_dimension = img_info[1]

        ref_info = itk.template(ref)[1]
        Opixel_type = ref_info[0]
        Opixel_dimension = ref_info[1]

        OVectorImageType = itk.Image[Opixel_type, Opixel_dimension]
        IVectorImageType = itk.Image[Ipixel_type, Ipixel_dimension]

        if interpolator == "NearestNeighbor":
            InterpolatorType = itk.NearestNeighborInterpolateImageFunction[InputImageType, itk.D]
            # print("Rescale Seg with spacing :", output_spacing)
        elif interpolator == "Linear":
            InterpolatorType = itk.LinearInterpolateImageFunction[InputImageType, itk.D]
            # print("Rescale Scan with spacing :", output_spacing)

        interpolator = InterpolatorType.New()
        resampled_img = ResampleImage(img,ref_size.tolist(),ref_sp,ref_origin,ref_direction,interpolator,InputImageType,InputImageType)

        output = ItkToSitk(resampled_img)
        output = sitk.Cast(output, sitk.sitkInt16)

        # if img_sp[0] > ref_sp[0]:
        closing_radius = 2
        MedianFilter = sitk.MedianImageFilter()
        MedianFilter.SetRadius(closing_radius)
        output = MedianFilter.Execute(output)


        if outpath != -1:
            writer = sitk.ImageFileWriter()
            writer.SetFileName(outpath)
            writer.Execute(output)
                # itk.imwrite(resampled_img, outpath)
        return output

    else:
        output = ItkToSitk(img)
        output = sitk.Cast(output, sitk.sitkInt16)
        if outpath != -1:
            writer = sitk.ImageFileWriter()
            writer.SetFileName(outpath)
            writer.Execute(output)
        return output
def Write(vtkdata, output_name):
	outfilename = output_name
	print("Writting:", outfilename)
	polydatawriter = vtk.vtkPolyDataWriter()
	polydatawriter.SetFileName(outfilename)
	polydatawriter.SetInputData(vtkdata)
	polydatawriter.Write()
def SavePredToVTK(file_path,temp_folder,smoothing, out_folder, model_size):
    print("Generating VTK for ", file_path)

    img = sitk.ReadImage(file_path) 
    img_arr = sitk.GetArrayFromImage(img)
    present_labels = []
    for label in range(np.max(img_arr)):
        if label+1 in img_arr:
            present_labels.append(label+1)

    for i in present_labels:
        label = i
        seg = np.where(img_arr == label, 1,0)
        output = sitk.GetImageFromArray(seg)
        output.SetOrigin(img.GetOrigin())
        output.SetSpacing(img.GetSpacing())
        output.SetDirection(img.GetDirection())
        output = sitk.Cast(output, sitk.sitkInt16)
        temp_path = temp_folder +f"/tempVTK_{label}.nrrd"
        # print(temp_path)
        writer = sitk.ImageFileWriter()
        writer.SetFileName(temp_path)
        writer.Execute(output)
        surf = vtk.vtkNrrdReader()
        surf.SetFileName(temp_path)
        surf.Update()
        # print(surf)
        dmc = vtk.vtkDiscreteMarchingCubes()
        dmc.SetInputConnection(surf.GetOutputPort())
        dmc.GenerateValues(100, 1, 100)
        # LAPLACIAN smooth
        SmoothPolyDataFilter = vtk.vtkSmoothPolyDataFilter()
        SmoothPolyDataFilter.SetInputConnection(dmc.GetOutputPort())
        SmoothPolyDataFilter.SetNumberOfIterations(smoothing)
        SmoothPolyDataFilter.SetFeatureAngle(120.0)
        SmoothPolyDataFilter.SetRelaxationFactor(0.6)
        SmoothPolyDataFilter.Update()
        model = SmoothPolyDataFilter.GetOutput()

        color = vtk.vtkUnsignedCharArray() 
        color.SetName("Colors") 
        color.SetNumberOfComponents(3) 
        color.SetNumberOfTuples( model.GetNumberOfCells() )
            
        for i in range(model.GetNumberOfCells()):
            color_tup=LABEL_COLORS[label]
            color.SetTuple(i, color_tup)
        model.GetCellData().SetScalars(color)
        outpath = out_folder + "/VTKfiles/" + os.path.basename(file_path).split('.')[0] + f"_{NAMES_FROM_LABELS[model_size][label]}_model.vtk"
        if not os.path.exists(os.path.dirname(outpath)):
            os.makedirs(os.path.dirname(outpath))
        Write(model, outpath)
def SaveSeg(file_path, spacing ,seg_arr, input_path,temp_path, outputdir,temp_folder, save_vtk, smoothing = 5, model_size= "LARGE"):
    print("Saving segmentation for ", file_path)
    SavePrediction(seg_arr,input_path,temp_path,output_spacing = spacing)
    # if clean_seg:
    #     CleanScan(temp_path)
    SetSpacingFromRef(
        temp_path,
        input_path,
        # "Linear",
        outpath=file_path
        )
    if save_vtk:
        SavePredToVTK(file_path,temp_folder, smoothing, out_folder=outputdir,model_size=model_size)
def CropSkin(skin_seg_arr, thickness):
    skin_img = sitk.GetImageFromArray(skin_seg_arr)
    skin_img = sitk.BinaryFillhole(skin_img)
    eroded_img = sitk.BinaryErode(skin_img, [thickness] * skin_img.GetDimension())
    skin_arr = sitk.GetArrayFromImage(skin_img)
    eroded_arr = sitk.GetArrayFromImage(eroded_img)
    croped_skin = np.where(eroded_arr==1, 0, skin_arr)
    out, N = cc3d.largest_k(
        croped_skin, k=1,
        connectivity=26, delta=0,
        return_N=True,
    )
    return out
def Create_UNETR(input_channel, label_nbr,cropSize):
    model = UNETR(
        in_channels=input_channel,
        out_channels=label_nbr,
        img_size=cropSize,
        feature_size=16,
        hidden_size=768,
        mlp_dim=3072,
        num_heads=12,
        pos_embed="perceptron",
        norm_name="instance",
        res_block=True,
        dropout_rate=0.05,
    )
    return model
def CleanArray(seg_arr,radius):
    input_img = sitk.GetImageFromArray(seg_arr)
    output = sitk.BinaryDilate(input_img, [radius] * input_img.GetDimension())
    output = sitk.BinaryFillhole(output)
    output = sitk.BinaryErode(output, [radius] * output.GetDimension())
    labels_in = sitk.GetArrayFromImage(output)
    out, N = cc3d.largest_k(
        labels_in, k=1,
        connectivity=26, delta=0,
        return_N=True,
    )
    return out