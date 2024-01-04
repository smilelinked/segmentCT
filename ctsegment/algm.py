import argparse
import asyncio
import logging

import errno
import os
import shutil
import tempfile

from ctsegment.CBCTSeg import segment_ct
from sharement.sharedata import ct_task_queue_dict
from obs import ObsClient


class CTSegment:
    obsClient = ObsClient(
        access_key_id='NCZPQASHJNW2URNGB9SI',
        secret_access_key='lXLZ9J1yUJYMrUBYZX2oAmzc3uvbSEIOSckpEsvN',
        server='obs.cn-east-3.myhuaweicloud.com'
    )

    available = True

    def __init__(self):
        pass

    @classmethod
    def get_input_prefix(cls, user_id, case_id, typ):
        return f"doctor/{user_id}/ct/{case_id}/{typ}/ct.nii.gz"

    @classmethod
    def get_output_prefix(cls, patient_id, case_id, typ):
        return f"patient/{patient_id}/ct/{case_id}_{typ}_"

    @classmethod
    async def segment(cls):
        while True:
            # 接收新病例
            logging.info("prepare to get a new ill case")
            if "normal" not in ct_task_queue_dict:
                ct_task_queue_dict["normal"] = asyncio.PriorityQueue()
            _, information = await ct_task_queue_dict["normal"].get()
            logging.info(f"now get a new ill case: {information}")
            cls.available = False

            bucket = information.get("bucket")
            user_id = information.get("user_id")
            patient_id = information.get("patient_id")
            case_id = information.get("case_id")
            typ = information.get("type")

            try:
                tmp_dir = tempfile.mkdtemp()  # create dir
                logging.info("now we work in a temp dir " + tmp_dir)

                # 读取CT
                resp = cls.obsClient.getObject(bucket, cls.get_input_prefix(user_id, case_id, typ),
                                              downloadPath=f"{tmp_dir}/ct.nii.gz")
                if resp.status > 300:
                    logging.error(f"read file form {user_id}/{case_id}/{typ} failed")
                    continue

                # 指定图片路径
                path = tmp_dir + "/"
                os.mkdir(path + "model", 0o755)

                parser = argparse.ArgumentParser(description='Perform CBCT segmentation',
                                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
                input_group = parser.add_argument_group('directory')
                input_group.add_argument('-i', '--input', type=str, help='Path to the scans folder',
                                         default=f"{tmp_dir}/ct.nii.gz")  # 输入路径
                input_group.add_argument('-o', '--output_dir', type=str, help='Folder to save output',
                                         default=path + "model")  # 输出路径
                input_group.add_argument('-dm', '--dir_models', type=str, help='Folder with the models',
                                         default='/etc/data/segmodel')  # 模型路径
                input_group.add_argument('-temp', '--temp_fold', type=str, help='temporary folder',
                                         default='/etc/data/segmodel')  # 缓存路径
                input_group.add_argument('-ss', '--skul_structure', nargs="+", type=str,
                                         help='Skul structure to segment',
                                         default=["MAND", "MAX", "CB"])  # "MAND","MAX","CB","SKIN"
                input_group.add_argument('-hd', '--high_def', type=bool, help='Use high def models', default=False)
                input_group.add_argument('-m', '--merge', nargs="+", type=str, help='merge the segmentations',
                                         default=["MERGE"])
                input_group.add_argument('-sf', '--save_in_folder', type=bool, help='Save the output in one folder',
                                         default=False)
                input_group.add_argument('-id', '--prediction_ID', type=str, help='Generate vtk files', default="Pred")
                input_group.add_argument('-vtk', '--gen_vtk', type=bool, help='Generate vtk file', default=True)
                input_group.add_argument('-vtks', '--vtk_smooth', type=int, help='Smoothness of the vtk', default=3)
                input_group.add_argument('-sp', '--spacing', nargs="+", type=float, help='Wanted output x spacing',
                                         default=[0.2, 0.2, 0.2])
                input_group.add_argument('-cs', '--crop_size', nargs="+", type=float, help='Wanted crop size',
                                         default=[128, 128, 128])
                input_group.add_argument('-pr', '--precision', type=float, help='precision of the prediction',
                                         default=0.3)
                input_group.add_argument('-mo', '--merging_order', nargs="+", type=str, help='order of the merging',
                                         default=["SKIN", "CV", "UAW", "CB", "MAX", "MAND", "CAN", "RC"])
                input_group.add_argument('-ncw', '--nbr_CPU_worker', type=int, help='Number of worker', default=1)
                input_group.add_argument('-ngw', '--nbr_GPU_worker', type=int, help='Number of worker', default=5)

                args = parser.parse_args()
                segment_ct(args)

                shutil.make_archive(path + "model", "zip", path + 'model/')
                resp = cls.obsClient.putFile(bucket, cls.get_output_prefix(patient_id, case_id, typ) + "model.zip",
                                             path + 'model.zip')
                if resp.status >= 300:
                    logging.error(f"upload zip file failed with resp: {resp}")

                logging.info("Finished!")

            finally:
                cls.available = True
                try:
                    shutil.rmtree(tmp_dir)  # delete directory
                except OSError as exc:
                    if exc.errno != errno.ENOENT:  # ENOENT - no such file or directory
                        raise
