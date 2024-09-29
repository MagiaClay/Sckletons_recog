import os.path as osp
from mmengine.fileio import list_from_file
from mmengine.dataset import BaseDataset

from mmengine.registry import DATASETS  # 此处要从mmengine 导入而不是mmaction
import data_pipeline


# OpenMMLab中的所有 Dataset 类都必须继承自 mmengine 中的 BaseDataset 类。我们可以通过覆盖 load_data_list 方法来定制注释加载过程。

@DATASETS.register_module()
class DatasetZelda(BaseDataset):
    def __init__(self, ann_file, pipeline, data_root, data_prefix=dict(video=''),
                 test_mode=False, modality='RGB', **kwargs):
        self.modality = modality
        super(DatasetZelda, self).__init__(ann_file=ann_file, pipeline=pipeline, data_root=data_root,
                                           data_prefix=data_prefix, test_mode=test_mode,
                                           **kwargs)

    def load_data_list(self):
        data_list = []
        fin = list_from_file(self.ann_file)
        for line in fin:
            line_split = line.strip().split()
            filename, label = line_split
            label = int(label)
            filename = osp.join(self.data_prefix['video'], filename)
            data_list.append(dict(filename=filename, label=label))  # 将每条数据和对应的label进行打包
        return data_list

    def get_data_info(self, idx: int) -> dict:
        data_info = super().get_data_info(idx)
        data_info['modality'] = self.modality  # 利用重写该函数来增加resul的信息
        return data_info


def data_loader_temp():
    train_pipeline_cfg = [
        dict(type='VideoInit'),
        dict(type='VideoSample', clip_len=16, num_clips=1, test_mode=False),
        dict(type='VideoDecode'),
        dict(type='VideoResize', r_size=256),
        dict(type='VideoCrop', c_size=224),
        dict(type='VideoFormat'),
        dict(type='VideoPack')
    ]

    val_pipeline_cfg = [
        dict(type='VideoInit'),
        dict(type='VideoSample', clip_len=16, num_clips=5, test_mode=True),
        dict(type='VideoDecode'),
        dict(type='VideoResize', r_size=256),
        dict(type='VideoCrop', c_size=224),
        dict(type='VideoFormat'),
        dict(type='VideoPack')
    ]
    # 配置文件
    train_dataset_cfg = dict(
        type='DatasetZelda',
        ann_file='kinetics_tiny_train_video.txt',
        pipeline=train_pipeline_cfg,
        data_root='../../mmaction2/data/kinetics400_tiny/',
        data_prefix=dict(video='train'))

    val_dataset_cfg = dict(
        type='DatasetZelda',
        ann_file='kinetics_tiny_val_video.txt',
        pipeline=val_pipeline_cfg,
        data_root='../../mmaction2/data/kinetics400_tiny/',
        data_prefix=dict(video='val'))

    train_dataset = DATASETS.build(train_dataset_cfg)  # 利用已经注册的Dataset进行build

    packed_results = train_dataset[0]  # 暂时用第一个样本测试

    inputs = packed_results['inputs']
    data_sample = packed_results['data_samples']
    # 测试导入的dataset的内容
    # print('shape of the inputs: ', inputs.shape)
    #
    # # 获取输入的信息
    # print('image_shape: ', data_sample.img_shape)
    # print('num_clips: ', data_sample.num_clips)
    # print('clip_len: ', data_sample.clip_len)
    #
    # # 获取输入的标签
    # print('label: ', data_sample.gt_label)

    from mmengine.runner import Runner

    BATCH_SIZE = 2

    train_dataloader_cfg = dict(
        batch_size=BATCH_SIZE,
        num_workers=0,
        persistent_workers=False,
        sampler=dict(type='DefaultSampler', shuffle=True),
        dataset=train_dataset_cfg)

    val_dataloader_cfg = dict(
        batch_size=BATCH_SIZE,
        num_workers=0,
        persistent_workers=False,
        sampler=dict(type='DefaultSampler', shuffle=False),
        dataset=val_dataset_cfg)

    train_data_loader = Runner.build_dataloader(dataloader=train_dataloader_cfg)
    val_data_loader = Runner.build_dataloader(dataloader=val_dataloader_cfg)
    # 构建dataset的迭代器：
    batched_packed_results = next(iter(train_data_loader))  # 迭代器，迭代一次获得结果

    batched_inputs = batched_packed_results['inputs']
    batched_data_sample = batched_packed_results['data_samples']

    assert len(batched_inputs) == BATCH_SIZE
    assert len(batched_data_sample) == BATCH_SIZE

    return batched_packed_results # 如果需要一次次返回则再经行一次额next()
