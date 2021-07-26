from scipy.io.matlab.mio import loadmat
import torch.utils.data as data
import scipy.io as sio
import torch
import torchvision.transforms as trans
import numpy as np
from pathlib import Path
from PIL import Image
import cv2

prepared_dataset_path = '/home/data/wjc_data/xgaze_224_prepare_two'
mean_file = '/home/data/wjc_data/src/data/'

def loadMeta(path):
    try:
        meta = sio.loadmat(str(path), squeeze_me=True, struct_as_record=False)
        print('success to load {}'.format(path.stem))
    except:
        print('fail to load the {}'.format(path.stem))
        return None
    return meta


class SubtractMean(object):
    """Normalize an tensor image with mean.
    """

    def __init__(self, meanImg):
        self.meanImg = trans.ToTensor()(meanImg / 255)

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        return tensor.sub(self.meanImg)


class ITrackDataXgaze(data.Dataset):
    def __init__(self, dataPath, split='train', faceSize=(224, 224), eyeSize=(50, 50), is_equalization=False):
        self.dataPath = dataPath
        self.faceSize = faceSize
        self.eyeSize = eyeSize
        self.is_equalization = is_equalization

        print('load {} Dataset'.format(split))

        self.faceMean = loadMeta(Path(mean_file).joinpath('mean_face_224.mat'))['image_mean']
        self.eyeLeftMean = loadMeta(Path(mean_file).joinpath('mean_left_224.mat'))['image_mean']
        self.eyeRightMean = loadMeta(Path(mean_file).joinpath('mean_right_224.mat'))['image_mean']

        self.transFace = trans.Compose([
            trans.Resize(self.faceSize),
            trans.ToTensor(),
            SubtractMean(meanImg=self.faceMean),
            #trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transLeftEye = trans.Compose([
            trans.Resize(self.faceSize),
            trans.ToTensor(),
            SubtractMean(meanImg=self.eyeLeftMean),
            # trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        self.transRightEye = trans.Compose([
            trans.Resize(self.faceSize),
            trans.ToTensor(),
            SubtractMean(meanImg=self.eyeRightMean),
            # trans.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])

        if split == 'test':
            self.dataset = Path(self.dataPath) / 'test1'
            self.meta = loadMeta(self.dataset.joinpath('meta_test.mat'))
        elif split == 'train':
            self.dataset = Path(self.dataPath) / 'train'
            self.meta = loadMeta(self.dataset.joinpath('meta_train.mat'))
        else:
            self.dataset = Path(self.dataPath) / 'val'
            self.meta = loadMeta(self.dataset.joinpath('meta_validate.mat'))

    def loadImage(self, path):
        try:
            image = Image.open(str(path)).convert('RGB')
        except:
            raise RuntimeError('Could load the image')
        return image

    def __getitem__(self, index):
        face_path = self.dataset.joinpath(
            'subject{:0>4d}/face/{:0>6d}.jpg'.format(self.meta['subject'][index], self.meta['frameIndex'][index]))
        left_eye_path = self.dataset.joinpath(
            'subject{:0>4d}/left_eye/{:0>6d}.jpg'.format(self.meta['subject'][index], self.meta['frameIndex'][index]))
        right_eye_path = self.dataset.joinpath(
            'subject{:0>4d}/right_eye/{:0>6d}.jpg'.format(self.meta['subject'][index], self.meta['frameIndex'][index]))

        if self.is_equalization:
            if self.meta['is_equalization']>512:
                face = self.loadImage(face_path)
                img_yuv = cv2.cvtColor(face, cv2.COLOR_BGR2YUV)
                img_yuv[:, :, 0] = cv2.equalizeHist(img_yuv[:, :, 0])
                face = cv2.cvtColor(img_yuv, cv2.COLOR_YUV2RGB)
                if self.meta['left_eye_pos'][index] is not None and self.meta['right_eye_pos'][index] is not None:
                    left_eye_pos = self.meta['left_eye_pos'][index]
                    right_eye_pos = self.meta['right_eye_pos'][index]
                    left_eye = face[left_eye_pos[0], left_eye_pos[1],left_eye_pos[2],left_eye_pos[3]]
                    right_eye = face[right_eye_pos[0], right_eye_pos[1],right_eye_pos[2],right_eye_pos[3]]
                else:
                    left_eye = self.loadImage(left_eye_path)
                    right_eye = self.loadImage(right_eye_path)
        else:
            face = self.loadImage(face_path)
            left_eye = self.loadImage(left_eye_path)
            right_eye = self.loadImage(right_eye_path)

        face = self.transFace(face)
        left_eye = self.transLeftEye(left_eye)
        right_eye = self.transRightEye(right_eye)
        #print(left_eye.shape)

        if self.dataset.stem != 'test1':
            eye_direction = self.meta['face_gaze_direction'][index]
            eye_direction = np.array(eye_direction)
            eye_direction = torch.FloatTensor(eye_direction)
        else:
            eye_direction = []
        return face, left_eye, right_eye, eye_direction

    def __len__(self):
        return self.meta['subject'].shape[0]
        #return 100
