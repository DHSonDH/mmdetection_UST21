# Copyright (c) OpenMMLab. All rights reserved.
import collections

from mmcv.utils import build_from_cfg

from ..builder import PIPELINES


@PIPELINES.register_module()
class Compose:
    """Compose multiple transforms sequentially.

    Args:
        transforms (Sequence[dict | callable]): Sequence of transform object or
            config dict to be composed.
    """

    def __init__(self, transforms):
        assert isinstance(transforms, collections.abc.Sequence)
        self.transforms = []
        for transform in transforms:
            if isinstance(transform, dict):
                transform = build_from_cfg(transform, PIPELINES)
                self.transforms.append(transform)
            elif callable(transform):
                self.transforms.append(transform)
            else:
                raise TypeError('transform must be callable or a dict')

    def __call__(self, data):
        """Call function to apply transforms sequentially.

        Args:
            data (dict): A result dict contains the data to transform.

        Returns:
           dict: Transformed data.
        """
        for t in self.transforms:
            data = t(data)
            # if type(t).__name__ == 'Resize':
            # if type(t).__name__ != 'Collect':
            #     try:
            #         print(f"img_shape : {data['img_shape']}, pad_shape : {data['pad_shape']}")  
            #         # image shape과 pad shape의 discrepancy. 특히 pad shape이 img_shape보다 작으면 error남. img_shape이 직사각형이라 짧은 부분이 문제가 됨. 
            #         # reshape이던 crop이던 이미지 원본 가로세로 비율을 유지하면서 바뀌어서 직사각형이 되는 듯 함
            #         # 따라서, 언제 이미지 reshape이 일어나는지, 이를 어떻게 바꾸면 될 지 고민하면 됨. 무조건 정사각형으로 바꾸면 되는것인지 ?
            #     except:
            #         print(f"img_shape : {data['img_shape']}, org_shape : {data['ori_shape']}")  

            if data is None:
                return None
        return data

    def __repr__(self):
        format_string = self.__class__.__name__ + '('
        for t in self.transforms:
            str_ = t.__repr__()
            if 'Compose(' in str_:
                str_ = str_.replace('\n', '\n    ')
            format_string += '\n'
            format_string += f'    {str_}'
        format_string += '\n)'
        return format_string
