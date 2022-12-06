import os
import json
from traceback import print_stack
import numpy as np
import cv2
import pandas as pd

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class NIADataset(CustomDataset):

    CLASSES = ('rip',)
    # CLASSES = ('rip',)  # 1개 class 면 (class1,) 형식으로 저장

    def load_annotations(self, ann_dir):
        obs_interest = [ # 'rip_current_duration', 'rip_current_phase',
            'significant_wave_height', 'significant_wave_period',
            'wind_velocity', 'wind_direction', 'wave_direction_sprading_factor', 'spectrum_spreading_factor', 
            'peak_period', 'peak_direction', 'angle_of_incidence_of_the_beach', 'tide_level'
            ]

        ann_list = os.listdir(ann_dir)
        
        data_infos = []
        base_path = '/'.join(ann_dir.split('/')[:-1])
        for i in range(len(ann_list)):
            with open(f'{ann_dir}/{ann_list[i]}') as json_file:
                annot = json.load(json_file)
            image_info = annot['image_info']
            annots = annot['annotations']

            #FIXME intentionally choose label existing image only.
            # if 'norip' in img_name:
            #     continue

            img = cv2.imread(f'{base_path}/img/{image_info["file_name"]}')
            img_shape = img.shape
            height = int(img_shape[0])
            width = int(img_shape[1])  #1920
            
            rois = annots['bounding_count']
            label = annots['class']-1  # 이안류과제에선 한 이미지 당 여러 라벨이 나오지 않을 것이므로 for 밖으로 뺌
            obsevations = [annots[item] for item in obs_interest]
            
            bboxes = []
            labels = []
            masks_poly = []
            obs = []

            for idx in range(rois):
                polys = annots['drawing'][str(idx)]
                polys = [[item[0], height-item[1]] for item in polys]  # width height 좌표계 다른 것 보정
                mask_poly = np.array(polys).astype(np.double)
                poly1, poly2, poly3, poly4 = polys
                bbox = [poly4[0], poly4[1], poly2[0], poly2[1]]  # x1, y1, x2, y2
    
                # mask_poly = [np.array(bbox).astype(np.double).reshape(-1,4)]
                # https://mmdetection.readthedocs.io/en/latest/2_new_data_model.html
                # https://velog.io/@dust_potato/MM-Detection-Config-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0-2
                # poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
                # poly = [p for x in poly for p in x]
                labels.append(label)
                bboxes.append(bbox)
                masks_poly.append(mask_poly)
                obs.append(obsevations)

            data_infos.append(
                dict(
                    filename=image_info["file_name"],
                    width=width,
                    height=height,
                    ann=dict(
                        bboxes=np.array(bboxes).astype(np.float32).reshape(-1,4),  # 1+1 차원 배열이어야 함
                        labels=np.array(labels).astype(np.int64).reshape(-1),  # 0+1차원 배열이어야 함
                        masks= masks_poly,  # mask 좌표처럼 polygon으로 주면 알아서 mask 생성하는 방식인듯
                        obs = obs
                        )
                    )
            )

        return data_infos

    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']
