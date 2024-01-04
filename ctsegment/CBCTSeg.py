import random
import string
import time
import glob
import shutil
import argparse
from monai.inferers import sliding_window_inference
from monai.data import (DataLoader, Dataset)
from utils import *

def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))


def segment_ct(args):
    cropSize = args.crop_size
    temp_fold = os.path.join(args.temp_fold, "temp_" + id_generator())
    if not os.path.exists(temp_fold):
        os.makedirs(temp_fold)
    # Find available models in folder
    available_models = {}
    # print("Loading models from", args.dir_models)
    normpath = os.path.normpath("/".join([args.dir_models, '**', '']))
    for img_fn in glob.iglob(normpath, recursive=True):
        #  print(img_fn)
        basename = os.path.basename(img_fn)
        if basename.endswith(".pth"):
                model_id = basename.split("_")[1]
                if model_id == "Mask":
                    model_id = basename.split("_")[2] + "MASK"
                available_models[model_id] = img_fn
    print("Available models:", available_models)
    # Choose models to use
    models_to_use = {}
    if args.high_def:
        model_size = "SMALL"
        MODELS_DICT = MODELS_GROUP["SMALL"]
        spacing = [0.16,0.16,0.32]
    else:
        model_size = "LARGE"
        MODELS_DICT = MODELS_GROUP["LARGE"]
        spacing = [0.4,0.4,0.4]
    for model_id in MODELS_DICT.keys():
        if model_id in available_models.keys():
            for struct in args.skul_structure:
                if struct in MODELS_DICT[model_id].keys():
                    if model_id not in models_to_use.keys():
                        models_to_use[model_id] = available_models[model_id]
            # if True in [ for struct in args.skul_structure]:
    print('MODELS_DICT:', MODELS_DICT)
    print('models-to-use:',models_to_use)
    # load data
    data_list = []
    if args.output_dir != None:
        outputdir = args.output_dir

    number_of_scans = 0
    if os.path.isfile(args.input):  
        print("Loading scan :", args.input)
        img_fn = args.input
        basename = os.path.basename(img_fn)
        new_path = os.path.join(temp_fold,basename)
        temp_pred_path = os.path.join(temp_fold,"temp_Pred.nii.gz")
        if not os.path.exists(new_path):
            CorrectHisto(img_fn, new_path,0.01, 0.99)
        # new_path = img_fn
        data_list.append({"scan":new_path, "name":img_fn, "temp_path":temp_pred_path})
        number_of_scans += 1
        if args.output_dir == None:
            outputdir = os.path.dirname(args.input)
    else:
        if args.output_dir == None:
            outputdir = args.input
        scan_dir = args.input
        print("Loading data from",scan_dir )
        normpath = os.path.normpath("/".join([scan_dir, '**', '']))
        for img_fn in sorted(glob.iglob(normpath, recursive=True)):
            #  print(img_fn)
            basename = os.path.basename(img_fn)
            if True in [ext in basename for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
                if not True in [txt in basename for txt in ["_Pred","seg","Seg"]]:
                    number_of_scans += 1
        counter = 0
        for img_fn in sorted(glob.iglob(normpath, recursive=True)):
            #  print(img_fn)
            basename = os.path.basename(img_fn)
            if True in [ext in basename for ext in [".nrrd", ".nrrd.gz", ".nii", ".nii.gz", ".gipl", ".gipl.gz"]]:
                if not True in [txt in basename for txt in ["_Pred","seg","Seg"]]:
                    new_path = os.path.join(temp_fold,basename)
                    temp_pred_path = os.path.join(temp_fold,"temp_Pred.nii.gz")
                    if not os.path.exists(new_path):
                        CorrectHisto(img_fn, new_path,0.01, 0.99)
                    data_list.append({"scan":new_path, "name":img_fn, "temp_path":temp_pred_path})
                    counter += 1
    # endregion
    # region prepare data
    pred_transform = CreatePredTransform(spacing)
    pred_ds = Dataset(
        data=data_list, 
        transform=pred_transform, 
    )
    pred_loader = DataLoader(
        dataset=pred_ds,
        batch_size=1, 
        shuffle=False, 
        num_workers=args.nbr_CPU_worker, 
        pin_memory=True
    )
    # endregion
    startTime = time.time()
    seg_not_to_clean = ["CV","RC"]
    with torch.no_grad():
        for step, batch in enumerate(pred_loader):
            #region PREDICTION
            input_img, input_path,temp_path = (batch["scan"].to(DEVICE), batch["name"],batch["temp_path"])
            image = input_path[0]
            print("Working on :",image)
            baseName = os.path.basename(image)
            scan_name= baseName.split(".")
            # print(baseName)
            pred_id = "_XXXX-Seg_"+ args.prediction_ID
            if "_scan" in baseName:
                pred_name = baseName.replace("_scan",pred_id)
            elif "_Scan" in baseName:
                pred_name = baseName.replace("_Scan",pred_id)
            else:
                pred_name = ""
                for i,element in enumerate(scan_name):
                    if i == 0:
                        pred_name += element + pred_id
                    else:
                        pred_name += "." + element
            if args.save_in_folder:
                outputdir = args.output_dir
                outputdir += "/" + scan_name[0] + "_" + "SegOut"
                print("Output dir :",outputdir)
                if not os.path.exists(outputdir):
                    os.makedirs(outputdir)
            prediction_segmentation = {}

            for model_id,model_path in models_to_use.items():
                net = Create_UNETR(
                    input_channel = 1,
                    label_nbr= len(MODELS_DICT[model_id].keys()) + 1,
                    cropSize=cropSize
                ).to(DEVICE)
                print("Loading model", model_path)
                net.load_state_dict(torch.load(model_path,map_location=DEVICE))
                print("Model loaded")
                net.eval()
                val_outputs = sliding_window_inference(input_img, cropSize, args.nbr_GPU_worker, net,overlap=args.precision)
                pred_data = torch.argmax(val_outputs, dim=1).detach().cpu().type(torch.int16)
                segmentations = pred_data.permute(0, 3, 2, 1)
                print("Segmentations shape :",segmentations.shape)
                seg = segmentations.squeeze(0)
                seg_arr = seg.numpy()[:]

                for struct, label in MODELS_DICT[model_id].items():
                    sep_arr = np.where(seg_arr == label, 1, 0)
                    if (struct == "SKIN"):
                        sep_arr = CropSkin(sep_arr, 5)
                        # sep_arr = GenerateMask(sep_arr,20)
                    elif not True in [struct == id for id in seg_not_to_clean]:
                        sep_arr = CleanArray(sep_arr, 2)
                    prediction_segmentation[struct] = sep_arr

            seg_to_save = {}
            for struct in args.skul_structure:
                seg_to_save[struct] = prediction_segmentation[struct]

            save_vtk = args.gen_vtk

            if "SEPARATE" in args.merge or len(args.skul_structure) == 1:
                for struct,segmentation in seg_to_save.items():
                    file_path = os.path.join(outputdir, pred_name.replace('XXXX', struct))
                    SaveSeg(
                        file_path = file_path,
                        spacing = spacing,
                        seg_arr=segmentation,
                        input_path=input_path[0],
                        outputdir=outputdir,
                        temp_path=temp_path[0],
                        temp_folder=temp_fold,
                        save_vtk=args.gen_vtk,
                        smoothing=args.vtk_smooth,
                        model_size=model_size
                    )
                    save_vtk = False

            if "MERGE" in args.merge and len(args.skul_structure) > 1:
                print("Merging")
                file_path = os.path.join(outputdir,pred_name.replace('XXXX', "MERGED"))
                merged_seg = np.zeros(seg_arr.shape)
                for struct in args.merging_order:
                    if struct in seg_to_save.keys():
                        merged_seg = np.where(seg_to_save[struct] == 1, LABELS[model_size][struct], merged_seg)
                SaveSeg(
                    file_path=file_path,
                    spacing=spacing,
                    seg_arr=merged_seg,
                    input_path=input_path[0],
                    outputdir=outputdir,
                    temp_path=temp_path[0],
                    temp_folder=temp_fold,
                    save_vtk=save_vtk,
                    model_size=model_size
                )
            #endregion
    try:
        shutil.rmtree(temp_fold)
    except OSError as e:
        print("Error: %s : %s" % (temp_fold, e.strerror))
    print("Done in %.2f seconds" % (time.time() - startTime))
#endregion