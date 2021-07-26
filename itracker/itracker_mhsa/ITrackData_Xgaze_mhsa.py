import torch.utils.data as data
import scipy.io as sio
import torch
import torchvision.transforms as trans
import numpy as np
from pathlib import Path
from PIL import Image

prepared_dataset_path = '/home/data/wjc_data/xgaze_224_prepare'
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


SPILT = 600000


class ITrackDataXgaze(data.Dataset):
    def __init__(self, dataPath, split='train', faceSize=(224, 224), eyeSize=(50, 50)):
        self.dataPath = dataPath
        self.faceSize = faceSize
        self.eyeSize = eyeSize

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
            self.dataset = Path(self.dataPath) / 'test'
            self.meta = loadMeta(self.dataset.joinpath('meta_test.mat'))
        else:
            self.dataset = Path(self.dataPath) / 'train'
            self.meta = loadMeta(self.dataset.joinpath('meta_train.mat'))
            if split == 'train':
                self.meta = {
                    'subject': self.meta['subject'][0:SPILT],
                    'frameIndex': self.meta['frameIndex'][0:SPILT],
                    'face_gaze_direction': self.meta['face_gaze_direction'][0:SPILT]
                }
            else:
                self.meta = {
                    'subject': self.meta['subject'][SPILT:],
                    'frameIndex': self.meta['frameIndex'][SPILT:],
                    'face_gaze_direction': self.meta['face_gaze_direction'][SPILT:]
                }

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

        face = self.loadImage(face_path)
        left_eye = self.loadImage(left_eye_path)
        right_eye = self.loadImage(right_eye_path)

        face = self.transFace(face)
        left_eye = self.transLeftEye(left_eye)
        right_eye = self.transRightEye(right_eye)
        #print(left_eye.shape)

        if self.dataset.stem == 'train':
            eye_direction = self.meta['face_gaze_direction'][index]
            eye_direction = np.array(eye_direction)
            eye_direction = torch.FloatTensor(eye_direction)
        else:
            eye_direction = []
        return face, left_eye, right_eye, eye_direction

    def __len__(self):
        return self.meta['subject'].shape[0]
        # return 2000
