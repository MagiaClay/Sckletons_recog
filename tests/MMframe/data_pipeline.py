import mmcv
import decord
import numpy as np
from mmcv.transforms import TRANSFORMS, BaseTransform, to_tensor
from mmaction.structures import ActionDataSample


# 实现 解码、采样、调整大小、裁剪、格式化 和 打包 视频数据和相应的标签
@TRANSFORMS.register_module()
class VideoInit(BaseTransform):
    # 视屏参数初始化
    def transform(self, results):
        container = decord.VideoReader(results['filename'])
        results['total_frames'] = len(container)
        results['video_reader'] = container
        return results


@TRANSFORMS.register_module()
class VideoSample(BaseTransform):
    # 时评采样
    def __init__(self, clip_len, num_clips, test_mode=False):
        self.clip_len = clip_len
        self.num_clips = num_clips
        self.test_mode = test_mode

    def transform(self, results):
        total_frames = results['total_frames']
        interval = total_frames // self.clip_len

        if self.test_mode:
            # 使测试期间的采样具有确定性
            np.random.seed(42)

        inds_of_all_clips = []
        for i in range(self.num_clips):
            bids = np.arange(self.clip_len) * interval
            offset = np.random.randint(interval, size=bids.shape)
            inds = bids + offset
            inds_of_all_clips.append(inds)

        results['frame_inds'] = np.concatenate(inds_of_all_clips)
        results['clip_len'] = self.clip_len
        results['num_clips'] = self.num_clips
        return results


@TRANSFORMS.register_module()
class VideoDecode(BaseTransform):
    # 视频解码
    def transform(self, results):
        frame_inds = results['frame_inds']
        container = results['video_reader']

        imgs = container.get_batch(frame_inds).asnumpy()
        imgs = list(imgs)

        results['video_reader'] = None
        del container

        results['imgs'] = imgs
        results['img_shape'] = imgs[0].shape[:2]
        return results


@TRANSFORMS.register_module()
class VideoResize(BaseTransform):
    # 视频缩放
    def __init__(self, r_size):
        self.r_size = (np.inf, r_size)

    def transform(self, results):
        img_h, img_w = results['img_shape']
        new_w, new_h = mmcv.rescale_size((img_w, img_h), self.r_size)

        imgs = [mmcv.imresize(img, (new_w, new_h))
                for img in results['imgs']]
        results['imgs'] = imgs
        results['img_shape'] = imgs[0].shape[:2]
        return results


@TRANSFORMS.register_module()
class VideoCrop(BaseTransform):
    # 视频裁剪
    def __init__(self, c_size):
        self.c_size = c_size

    def transform(self, results):
        img_h, img_w = results['img_shape']
        center_x, center_y = img_w // 2, img_h // 2
        x1, x2 = center_x - self.c_size // 2, center_x + self.c_size // 2
        y1, y2 = center_y - self.c_size // 2, center_y + self.c_size // 2
        imgs = [img[y1:y2, x1:x2] for img in results['imgs']]
        results['imgs'] = imgs
        results['img_shape'] = imgs[0].shape[:2]
        return results


@TRANSFORMS.register_module()
class VideoFormat(BaseTransform):
    # 视频格式化与标准化
    def transform(self, results):
        num_clips = results['num_clips']
        clip_len = results['clip_len']
        imgs = results['imgs']

        # [num_clips*clip_len, H, W, C]
        imgs = np.array(imgs)
        # [num_clips, clip_len, H, W, C]
        imgs = imgs.reshape((num_clips, clip_len) + imgs.shape[1:])
        # [num_clips, C, clip_len, H, W]
        imgs = imgs.transpose(0, 4, 1, 2, 3)

        results['imgs'] = imgs
        return results


@TRANSFORMS.register_module()
class VideoPack(BaseTransform):
    # 视频信息打包
    def __init__(self, meta_keys=('img_shape', 'num_clips', 'clip_len')):
        self.meta_keys = meta_keys

    def transform(self, results):
        packed_results = dict()
        inputs = to_tensor(results['imgs'])
        data_sample = ActionDataSample().set_gt_label(results['label'])
        metainfo = {k: results[k] for k in self.meta_keys if k in results}
        data_sample.set_metainfo(metainfo)
        packed_results['inputs'] = inputs
        packed_results['data_samples'] = data_sample
        return packed_results


if __name__ == '__main__':
    import os.path as osp
    from mmengine.dataset import Compose

    pipeline_cfg = [
        dict(type='VideoInit'),
        dict(type='VideoSample', clip_len=16, num_clips=1, test_mode=False),
        dict(type='VideoDecode'),
        dict(type='VideoResize', r_size=256),
        dict(type='VideoCrop', c_size=224),
        dict(type='VideoFormat'),
        dict(type='VideoPack')
    ]  # 用字典进行测试

    pipeline = Compose(pipeline_cfg)  # mmengine自带的信息整合，因为继承的BaseTransform已在注册表中，因此一次构建pipeline
    data_prefix = '../../mmaction2/data/kinetics400_tiny/train'
    # 对某一条单独信息经行分析
    results = dict(filename=osp.join(data_prefix, 'D32_1gwq35E.mp4'), label=0)
    packed_results = pipeline(results)

    inputs = packed_results['inputs']
    data_sample = packed_results['data_samples']

    # print('shape of the inputs: ', inputs.shape)
    #
    # # 获取输入的信息
    # print('image_shape: ', data_sample.img_shape)
    # print('num_clips: ', data_sample.num_clips)
    # print('clip_len: ', data_sample.clip_len)
    #
    # # 获取输入的标签
    # print('label: ', data_sample.gt_label)
