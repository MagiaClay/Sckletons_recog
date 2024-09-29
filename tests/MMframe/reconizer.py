# 构建 recognizer，它主要由三部分组成：用于批处理和规范化数据的 data preprocessor，用于特征提取的 backbone 和用于分类的 cls_head 。
import torch
from mmengine.model import BaseDataPreprocessor, stack_batch
from mmengine.registry import MODELS
import dataset_dataloader
import torch
import copy


# 滴哦数据进行预处理，归一化
@MODELS.register_module()
class DataPreprocessorZelda(BaseDataPreprocessor):
    def __init__(self, mean, std):
        super().__init__()

        self.register_buffer(
            'mean',
            torch.tensor(mean, dtype=torch.float32).view(-1, 1, 1, 1),
            False)
        self.register_buffer(
            'std',
            torch.tensor(std, dtype=torch.float32).view(-1, 1, 1, 1),
            False)

    def forward(self, data, training=False):
        data = self.cast_data(data)  # 复制打他
        inputs = data['inputs']
        batch_inputs = stack_batch(inputs)  # 批处理
        batch_inputs = (batch_inputs - self.mean) / self.std  # 归一化
        data['inputs'] = batch_inputs
        return data  # 以张量的形式返回


# backbone、cls——head和recognizer的实现如下
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmengine.model import BaseModel, BaseModule, Sequential
from mmengine.structures import LabelData


# 用于特征提取的backbone
@MODELS.register_module()
class BackBoneZelda(BaseModule):
    # Kaiming一种初始化方法，为了防止初始化导致梯度消失或者梯度爆炸
    def __init__(self, init_cfg=None):
        if init_cfg is None:
            #
            init_cfg = [dict(type='Kaiming', layer='Conv3d', mode='fan_out', nonlinearity="relu"),
                        dict(type='Constant', layer='BatchNorm3d', val=1, bias=0)]

        super(BackBoneZelda, self).__init__(init_cfg=init_cfg)
        # 后续了解卷基层的作用
        self.conv1 = Sequential(nn.Conv3d(3, 64, kernel_size=(3, 7, 7),
                                          stride=(1, 2, 2), padding=(1, 3, 3)),
                                nn.BatchNorm3d(64), nn.ReLU())
        self.maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2),
                                    padding=(0, 1, 1))

        self.conv = Sequential(nn.Conv3d(64, 128, kernel_size=3, stride=2, padding=1),
                               nn.BatchNorm3d(128), nn.ReLU())

    # 前向传播：两层卷基层一层池化层
    def forward(self, imgs):
        # imgs: [batch_size*num_views, 3, T, H, W]
        # features: [batch_size*num_views, 128, T/2, H//8, W//8]
        features = self.conv(self.maxpool(self.conv1(imgs)))
        return features


# head分类器，用于classes的分类，同样再model中注册
@MODELS.register_module()
class ClsHeadZelda(BaseModule):
    # 线性层分类，输入【分类数】【通道数】【随机dopout用于降低拟合】
    def __init__(self, num_classes, in_channels, dropout=0.5, average_clips='prob', init_cfg=None):
        if init_cfg is None:
            init_cfg = dict(type='Normal', layer='Linear', std=0.01)

        super(ClsHeadZelda, self).__init__(init_cfg=init_cfg)

        self.num_classes = num_classes
        self.in_channels = in_channels
        self.average_clips = average_clips

        if dropout != 0:
            self.dropout = nn.Dropout(dropout)  # Dropout初始化
        else:
            self.dropout = None

        self.fc = nn.Linear(self.in_channels, self.num_classes)
        self.pool = nn.AdaptiveAvgPool3d(1)
        self.loss_fn = nn.CrossEntropyLoss()  # 用交叉熵作为loss

    def forward(self, x):
        N, C, T, H, W = x.shape
        x = self.pool(x)
        x = x.view(N, C)
        assert x.shape[1] == self.in_channels

        if self.dropout is not None:
            x = self.dropout(x)

        cls_scores = self.fc(x)
        return cls_scores  # 输出为分类的的风

    def loss(self, feats, data_samples):
        cls_scores = self(feats)
        labels = torch.stack([x.gt_label for x in data_samples])
        labels = labels.squeeze()

        if labels.shape == torch.Size([]):
            labels = labels.unsqueeze(0)

        loss_cls = self.loss_fn(cls_scores, labels)
        return dict(loss_cls=loss_cls)

    def predict(self, feats, data_samples):
        cls_scores = self(feats)
        num_views = cls_scores.shape[0] // len(data_samples)
        # assert num_views == data_samples[0].num_clips
        cls_scores = self.average_clip(cls_scores, num_views)

        for ds, sc in zip(data_samples, cls_scores):
            pred = LabelData(item=sc)
            ds.pred_scores = pred
        return data_samples

    # 再许多clips上平均分类scores，使用不同的平均方法（‘score’, or 'prob' or None 取决于test_cfg）,从而获得最终类别的评分，只在测试mode中使用
    def average_clip(self, cls_scores, num_views):
        if self.average_clips not in ['score', 'prob', None]:
            raise ValueError(f'{self.average_clips} is not supported. '
                             f'Currently supported ones are '
                             f'["score", "prob", None]')

        total_views = cls_scores.shape[0]
        cls_scores = cls_scores.view(total_views // num_views, num_views, -1)

        if self.average_clips is None:
            return cls_scores
        elif self.average_clips == 'prob':  # 如果是基于概率，则使用softmax函数，常用于解决多分类问题，把数据转换为(0, 1)之间的值
            cls_scores = F.softmax(cls_scores, dim=2).mean(dim=1)
        elif self.average_clips == 'score':
            cls_scores = cls_scores.mean(dim=1)

        return cls_scores


# 前两者的整合的识别器
@MODELS.register_module()
class RecognizerZelda(BaseModel):
    def __init__(self, backbone, cls_head, data_preprocessor):
        super().__init__(data_preprocessor=data_preprocessor)

        self.backbone = MODELS.build(backbone)
        self.cls_head = MODELS.build(cls_head)

    # 接触特征
    def extract_feat(self, inputs):
        inputs = inputs.view((-1,) + inputs.shape[2:])
        return self.backbone(inputs)

    # 计算loss
    def loss(self, inputs, data_samples):
        feats = self.extract_feat(inputs)
        loss = self.cls_head.loss(feats, data_samples)
        return loss

    # 计算预测值
    def predict(self, inputs, data_samples):
        feats = self.extract_feat(inputs)
        predictions = self.cls_head.predict(feats, data_samples)
        return predictions

    # 分类器功能
    def forward(self, inputs, data_samples=None, mode='tensor'):
        if mode == 'tensor':
            return self.extract_feat(inputs)
        elif mode == 'loss':
            return self.loss(inputs, data_samples)
        elif mode == 'predict':
            return self.predict(inputs, data_samples)
        else:
            raise RuntimeError(f'Invalid mode: {mode}')


def recoginizer_temp():
    model_cfg = dict(
        type='RecognizerZelda',
        backbone=dict(type='BackBoneZelda'),
        cls_head=dict(
            type='ClsHeadZelda',
            num_classes=2,
            in_channels=128,
            average_clips='prob'),
        data_preprocessor=dict(
            type='DataPreprocessorZelda',
            mean=[123.675, 116.28, 103.53],
            std=[58.395, 57.12, 57.375]))

    model = MODELS.build(model_cfg)

    # 训练
    model.train()
    model.init_weights()  # 初始化权重
    data_batch_train = copy.deepcopy(dataset_dataloader.data_loader_temp())  # 传入一个打他loader的实例
    data = model.data_preprocessor(data_batch_train, training=True)
    loss = model(**data, mode='loss')  # 此处存在解引用计算模型loss
    # print('loss dict: ', loss)

    # 验证，with是使得torch.no_grad()对象实现了上下文管理协议，可以
    with torch.no_grad():
        model.eval()
        data_batch_test = copy.deepcopy(dataset_dataloader.data_loader_temp())
        data = model.data_preprocessor(data_batch_test, training=False)
        predictions = model(**data, mode='predict')
    print('Label of Sample[0]', predictions[0].gt_label)
    print('Scores of Sample[0]', predictions[0].pred_scores.item)
    return predictions
