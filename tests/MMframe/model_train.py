from mmengine.runner import Runner
import dataset_dataloader
import reconizer
import AccuracyMetric
# 训练配置
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=10, val_interval=1)
val_cfg = dict(type='ValLoop')
optim_wrapper = dict(optimizer=dict(type='Adam', lr=0.01))

# 以下是模型配置、加载器配置、以及识别器配置
BATCH_SIZE = 2
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
metric_cfg = dict(type='AccuracyMetric', topk=(1, 5))

#
runner = Runner(model=model_cfg, work_dir='./work_dirs/guide',
                train_dataloader=train_dataloader_cfg,
                train_cfg=train_cfg,
                val_dataloader=val_dataloader_cfg,
                val_cfg=val_cfg,
                optim_wrapper=optim_wrapper,
                val_evaluator=[metric_cfg],
                default_scope='mmaction')
runner.train()