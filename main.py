from mmaction.apis import inference_recognizer, init_recognizer
import os
os.chdir('mmaction2')
config_path = 'data/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint.py'
checkpoint_path = 'data/slowonly_r50_8xb16-u48-240e_ntu60-xsub-keypoint_20220815-38db104b.pth' # 可以是本地路径
img_path = 'data/kinetics400_tiny/train/27_CSXByd3s.mp4'   # 您可以指定自己的图片路径

# 从配置文件和权重文件中构建模型
model = init_recognizer(config_path, checkpoint_path, device="cuda:0")  # device 可以是 'cuda:0'
# 对单个视频进行测试
result = inference_recognizer(model, img_path)