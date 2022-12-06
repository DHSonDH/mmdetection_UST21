import json
import os
import pickle

import cv2
import numpy as np

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class NIADataset(CustomDataset):

    # CLASSES = ('rip',)
    CLASSES = ('norip', 'rip')
    # CLASSES = ('rip',)  # 1개 class 면 (class1,) 형식으로 저장

    def load_annotations(self, ann_dir):
        meta_name = '_'.join(ann_dir.split('/')[1:-1])+'.pkl'
        meta_base = 'data/meta'

        if os.path.exists(f'{meta_base}/{meta_name}'):
            with open(f'{meta_base}/{meta_name}', 'rb') as f:
                var = pickle.load(f)
                print(f'{meta_base}/{meta_name}', 'found! skip meta processing')
            return var
        print(f'{meta_base}/{meta_name}', 'not found! start meta processing')

        
        obs_interest = [ # 'rip_current_duration', 'rip_current_phase',
            'significant_wave_height', 'significant_wave_period',
            'wind_velocity', 'wind_direction', 'wave_direction_sprading_factor', 'spectrum_spreading_factor', 
            'peak_period', 'peak_direction', 'angle_of_incidence_of_the_beach', 'tide_level'
        ]
        
        # when trainig or test with annotations
        if os.path.exists(ann_dir):
            ann_list = os.listdir(ann_dir)
            data_infos = []
            base_path = '/'.join(ann_dir.split('/')[:-1])

            for i in range(len(ann_list)):
                with open(f'{ann_dir}/{ann_list[i]}') as json_file:
                    annot = json.load(json_file)
                image_info = annot['image_info']
                annots = annot['annotations']
                
                # print(f'{base_path}/img/{image_info["file_name"]}')
                img = cv2.imread(f'{base_path}/img/{image_info["file_name"]}')
                img_shape = img.shape
                height = int(img_shape[0])  # 1080
                width = int(img_shape[1])  # 1920
                bboxes = []
                labels = []
                masks_poly = []
                obs = []

                rois = annots['bounding_count'] + 1 # all image 에 background 1개짜리 줄거니까
                
                observations = [annots[item] for item in obs_interest]
                if 'Null' in observations:
                    print(f"null in observation of {image_info['file_name']}")
                    observations = [0]*len(obs_interest)
                else:
                    observations[3] -= annots['angle_of_incidence_of_the_beach']  # wave direction - angle of incidence
                    
                for idx in range(rois):
                    if idx ==0:
                        label = 0
                        mask_poly = np.array([[0,  0],
                                            [width,  0],
                                            [width,  height],
                                            [0,  height]], dtype=float)
                        bbox = [0, 0, width, height]  
                        # x1(upper left), y1 , x2(lower right), y2
                    else:
                        # rip current recording
                        label = annots['class']  # 0: norip, 1:rip
                        polys = annots['drawing'][idx-1]  #실제기록은 0부터 되있으므로
                        # 이미지 좌표계를 따라서 시계방향 order
                        polys = [[item[0], item[1]] for item in polys]  # width height 좌표계 다른 것 보정
                        mask_poly = np.array(polys).astype(np.double)
                        poly1, poly2, poly3, poly4 = polys
                        bbox = [poly1[0], poly1[1], poly3[0], poly3[1]]  # x1, y1, x2, y2

                    labels.append(label)  # 한 이미지가 가지는 모든 bbox 라벨들 + background labeling
                    bboxes.append(bbox)
                    masks_poly.append(mask_poly)
                    obs.append(observations)

                data_infos.append(
                    dict(
                        filename=image_info["file_name"],
                        width=width,
                        height=height,
                        annots=dict(
                            bboxes=np.array(bboxes).astype(np.float32).reshape(-1,4),  # 1+1 차원 배열이어야 함
                            labels=np.array(labels).astype(np.int64).reshape(-1),  # 0+1차원 배열이어야 함
                            masks= masks_poly,  # mask 좌표처럼 polygon으로 주면 알아서 mask 생성하는 방식인듯
                            obs = obs
                        )
                    )
                )
            
            # save meta file
            with open(f'{meta_base}/{meta_name}', 'wb') as fp:
                pickle.dump(data_infos, fp)
        

        else: # When test image with no annotations 
            ''' just add background annotations, label == 0 '''
            base_path = '/'.join(ann_dir.split('/')[:-1])
            print(f'=======  Loading image from {base_path} ==========')
            img_dir = base_path+'/img'
            test_img_list = os.listdir(img_dir)

            data_infos = []
            for i, img_name in enumerate(test_img_list):

                # if i>100:
                #     break

                img = cv2.imread(f'{base_path}/img/{img_name}')
                img_shape = img.shape
                height = int(img_shape[0])  # 1080
                width = int(img_shape[1])  # 1920

                bbox = [0, 0, width, height]  
                label = 0
                mask_poly = np.array([[0,  0],
                                    [width,  0],
                                    [width,  height],
                                    [0,  height]], dtype=float)
                observations = [0]*10
                
                bboxes = [bbox]
                labels = [label]  # 한 이미지가 가지는 모든 bbox 라벨들 + background labeling
                masks_poly = [mask_poly]
                obs = [observations]

                data_infos.append(
                    dict(
                        filename=img_name,
                        width=width,
                        height=height,
                        annots=dict(
                            bboxes=np.array(bboxes).astype(np.float32).reshape(-1,4),  # 1+1 차원 배열이어야 함
                            labels=np.array(labels).astype(np.int64).reshape(-1),  # 0+1차원 배열이어야 함
                            masks= masks_poly,  # mask 좌표처럼 polygon으로 주면 알아서 mask 생성하는 방식인듯
                            obs = obs
                        )
                    )
                )
        
        # save meta file
        with open(f'{meta_base}/{meta_name}', 'wb') as fp:
            pickle.dump(data_infos, fp)
                
        return data_infos  #25128 # autolabeltest: 5132
        # from start to return, time took 2279 instances: 58-sec
        # from start to return, time took 25128 instances: 160-sec
        # norip 추가 코드 : 390-sec

    def get_ann_info(self, idx):
        return self.data_infos[idx]['annots']
