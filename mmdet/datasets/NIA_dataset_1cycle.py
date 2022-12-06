import mmcv
import numpy as np
import cv2
import pandas as pd

from .builder import DATASETS
from .custom import CustomDataset


@DATASETS.register_module()
class NIADataset_1cycle(CustomDataset):

    CLASSES = ('rip',)
    # CLASSES = ('rip',)  # 1개 class 면 (class1,) 형식으로 저장

    def load_annotations(self, ann_file):
        
        df = pd.read_csv(ann_file)
        # ann_list = mmcv.list_from_file(ann_file)
        ann_list = df['fname'].to_list()

        data_infos = []
        path = '/'.join(ann_file.split('/')[:-1])
        for i in range(len(df)):
            # label = df['fname'][i].split('_')[0]
            filename = f'{path}/img/{df["fname"][i]}'
            
            #FIXME intentionally choose label existing image only.
            if 'norip' in filename:
                continue
            
            img = cv2.imread(filename)

            img_shape = img.shape
            width = int(img_shape[0])
            height = int(img_shape[1])
            
            # anns = ann_line.split(' ')
            bbox = df.iloc[i,1:5].tolist()
            label = df.iloc[i,-1].tolist()-1
            
            w1, h1, w2, h2 = bbox[0], bbox[1], bbox[2], bbox[3]

            # bbox_number = 1  # FIXME
            # bboxes = []
            # labels = []
            # for anns in ann_list[i + 4:i + 4 + bbox_number]:
            #     bboxes.append([float(ann) for ann in anns[:4]])
            #     labels.append(int(anns[4]))

            mask_poly = [np.array([[w1,h1],[w1, h2],[w2, h2], [w2, h1]]).astype(np.double)]
            # mask_poly = [np.array(bbox).astype(np.double).reshape(-1,4)]

            # https://mmdetection.readthedocs.io/en/latest/2_new_data_model.html
            # https://velog.io/@dust_potato/MM-Detection-Config-%EC%9D%B4%ED%95%B4%ED%95%98%EA%B8%B0-2
            # poly = [(x + 0.5, y + 0.5) for x, y in zip(px, py)]
            # poly = [p for x in poly for p in x]
            # add iscrowd

            data_infos.append(
                dict(
                    filename=ann_list[i],
                    width=width,
                    height=height,
                    ann=dict(
                        bboxes=np.array(bbox).astype(np.float32).reshape(-1,4),  # 1+1 차원 배열이어야 함
                        labels=np.array(label).astype(np.int64).reshape(-1),  # 0+1차원 배열이어야 함
                        masks= mask_poly,  # mask 좌표처럼 polygon으로 주면 알아서 mask 생성하는 방식인듯
                        
                        )
                    )
            )

        return data_infos

    def get_ann_info(self, idx):
        return self.data_infos[idx]['ann']
