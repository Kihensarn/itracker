import mediapipe as mp
import cv2
import temp
from PIL import Image
# from imutils import face_utils
import os
import h5py
import numpy as np
import temp
import os


# from eyes_features import get_landmarks


eye_detect_file = '/home/data/wjc_data/src/data/'
eye_cascade = cv2.CascadeClassifier(os.path.join(eye_detect_file, 'haarcascade_eye.xml'))
face_cascade = cv2.CascadeClassifier(os.path.join(eye_detect_file, 'haarcascade_frontalface_default.xml'))
face_landmarks_path = '/home/data/wjc_data/xgaze_224_prepare/shape_predictor_68_face_landmarks.dat'


#mediapipe solutions
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)
# face_mesh = mp_face_mesh.FaceMesh(
#     static_image_mode=True,
#     max_num_faces=1,
#     min_detection_confidence=1
# )
draw_spec = mp_drawing.DrawingSpec(thickness=1, circle_radius=1)
draw_spec2 = mp_drawing.DrawingSpec(thickness=2, circle_radius=1, color=(255, 255, 255))

#dlib solutions
# detector = dlib.get_frontal_face_detector()
# predictor = dlib.shape_predictor(face_landmarks_path)


def test_mediapipe(picture_path, is_cv2=True):
    if is_cv2:
        image = cv2.imread(picture_path)
        image_out = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    else:
        image_out = Image.open(picture_path)
        image = image_out.convert('BGR')

    results = face_mesh.process(image_out)
    if results.multi_face_landmarks is not None:
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                image=image,
                landmark_list=face_landmarks,
                connections=mp_face_mesh.FACE_CONNECTIONS,
                landmark_drawing_spec=draw_spec,
                connection_drawing_spec=draw_spec2
            )

    face_mesh.close()
    return image


def test_mediapipe_image(image_out, is_cv2=True):

    image = cv2.cvtColor(image_out, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image)
    face_landmarks_list = {}
    x = []
    y = []
    x1 = []
    y1 = []

    # if results.multi_face_landmarks is not None:
    #     print('success to detect the eyes')
    #     for face_landmarks in results.multi_face_landmarks:
    #         # mp_drawing.draw_landmarks(
    #         #     image=image_out,
    #         #     landmark_list=face_landmarks,
    #         #     connections=mp_face_mesh.FACE_CONNECTIONS,
    #         #     landmark_drawing_spec=draw_spec,
    #         #     connection_drawing_spec=draw_spec2
    #         # )
    #         face_landmarks_list = get_landmarks(face_landmarks)
    # temp.face_features['leftEyeUpper0'].extend(temp.face_features['leftEyeLower0'])
    # temp.face_features['rightEyeUpper0'].extend(temp.face_features['rightEyeLower0'])
    # for position in temp.face_features['leftEyeUpper0']:
    #     if position in face_landmarks_list:
    #         x.append(face_landmarks_list[position][0])
    #         y.append(face_landmarks_list[position][1])
    # print(max(x)-min(x), max(y)-min(y))
    # for position in temp.face_features['rightEyeUpper0']:
    #     if position in face_landmarks_list:
    #         x1.append(face_landmarks_list[position][0])
    #         y1.append(face_landmarks_list[position][1])
    # print(max(x1)-min(x1), max(y1)-min(y1))
    face_mesh.close()
    return image_out


def cv2_detect_eyes(image, is_cv2=True):
    # if is_cv2:
    #     image = cv2.imread(picture_path)
    # else:
    #     image = Image.open(picture_path)
    #     image = image.convert('BGR')

    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # eyes = eye_cascade.detectMultiScale(gray_image)
    eyes = detect_eyes([0, 0, 224, 224], gray_image)
    for (ex, ey, ew, eh) in eyes:
        cv2.rectangle(image, (ex, ey), (ex + ew, ey + eh), (0, 255, 255), 1)
        print(ex,  ey, ew, eh)
    return image
    # cv2.imshow('image', image)
    # cv2.waitKey(0)
    # cv2.destroyAllWindows()


def detect_eyes(face, gray):
    [x, y, w, h] = face
    roi_gray = gray[y:y + h, x:x + w]

    eyes = eye_cascade.detectMultiScale(roi_gray)
    eyes_sorted_by_size = sorted(eyes, key=lambda x: -x[2])
    largest_eyes = eyes_sorted_by_size[:2]
    # sort by x position
    largest_eyes.sort(key=lambda x: x[0])
    # offset by face start
    return list(map(lambda eye: [face[0] + eye[0], face[1] + eye[1], eye[2], eye[3]], largest_eyes))


def get_rect(points, ratio=1.0, scale=1):  # ratio = w:h
    x = points[:, 0]
    y = points[:, 1]

    x_expand = 0.1 * (max(x) - min(x))
    y_expand = 0.1 * (max(y) - min(y))

    x_max, x_min = max(x) + x_expand, min(x) - x_expand
    y_max, y_min = max(y) + y_expand, min(y) - y_expand

    # h:w=1:2
    if (y_max - y_min) * ratio < (x_max - x_min):
        h = (x_max - x_min) / ratio
        pad = (h - (y_max - y_min)) / 2
        y_max += pad
        y_min -= pad
    else:
        h = (y_max - y_min)
        pad = (h * ratio - (x_max - x_min)) / 2
        x_max += pad
        x_min -= pad

    int(x_min), int(x_max), int(y_min), int(y_max)
    bbox = [int(x_min), int(y_min), int(x_max - x_min), int(y_max - y_min)]
    bbox = np.array(bbox)

    aSrc = np.maximum(bbox[:2], 0)
    bSrc = np.minimum(bbox[:2] + bbox[2:], (224*scale, 224*scale))
    rect = np.concatenate([aSrc, bSrc])

    return rect


# def test_dlib(image):
#     face_landmarks = detector(image, 10)
#     print(len(face_landmarks))
#     for index, face_landmark in enumerate(face_landmarks):
#         shape = predictor(image, face_landmarks[index])
#         # shape = face_utils.shape_to_np(shape)
#         print(len(shape))


class ImageShowing(object):
    def __init__(self, row, col):
        self.num_i = 0
        self.img_size = 250
        self.img_show = np.zeros((self.img_size * row, self.img_size * col, 3), dtype=np.uint8)  # initial a empty image
        self.col = col
        self.row = row
        self.is_show = True

    def show_image(self, input_image):
        if self.is_show:
            cv2.namedWindow("image")

            num_r = self.num_i // self.col
            num_c = self.num_i - num_r * self.col

            input_image = cv2.resize(input_image, (self.img_size, self.img_size))

            if self.num_i < self.row * self.col:
                self.img_show[self.img_size * num_r:self.img_size * (num_r + 1),
                              self.img_size * num_c:self.img_size * (num_c + 1)] = input_image

            self.num_i = self.num_i + 1
            if self.num_i >= self.row * self.col:
                while True:
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    cv2.putText(self.img_show, 'Please press L to the next sample, and ESC to exit', (10, 30),
                                font, 1.0, (0, 255, 0), 2, cv2.LINE_AA)

                    cv2.imshow('image', self.img_show)
                    input_key = cv2.waitKey(0)
                    if input_key == 27:  # ESC key to exit
                        cv2.destroyAllWindows()
                        self.is_show = False
                        break
                    elif input_key == 108:  # l key to the next
                        self.num_i = 0
                        break
                    else:
                        continue


if __name__ == '__main__':
    # envpath = '/home/wjc/anaconda3/envs/pytorch/lib/python3.8/site-packages/cv2/qt/plugins/platform'
    # os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath
    # img_show = ImageShowing(1, 1)
    # with h5py.File('/home/data/wjc_data/xgaze_224/train/subject0038.h5') as train:
    #     for i in range(0, 1000):
    #         image = train['face_patch'][i, :]
    #         face_patch = test_mediapipe_image(image)
    #         img_show.show_image(face_patch)
    path = '/home/data/wjc_data/xgaze_224_prepare/train/subject0038/face/000003.jpg'
    image = Image.open(path)
    image.show()

