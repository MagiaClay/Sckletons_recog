import os

from mmaction.apis import inference_recognizer, init_recognizer

os.chdir('../mmaction2')

config_path = 'configs/recognition/tsn/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb.py'
checkpoint_path = 'work_dirs/tsn_imagenet-pretrained-r50_8xb32-1x1x3-100e_kinetics400-rgb/best_acc_top1_epoch_2.pth' # 可以是本地路径
img_path = 'demo/demo.mp4'   # 您可以指定自己的图片路径

# 从配置文件和权重文件中构建模型
model = init_recognizer(config_path, checkpoint_path, device="cuda:0")  # device 可以是 'cuda:0'
# 对单个视频进行测试
result = inference_recognizer(model, img_path)
print(result)