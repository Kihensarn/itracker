import shutil
import numpy as np
import temp
import mediapipe as mp
import cv2
from test_mediapipe import detect_eyes
import h5py


#size
image_size = [224, 224]
eye_size = 50


#mediapipe solutions
mp_face_mesh = mp.solutions.face_mesh
mp_drawing = mp.solutions.drawing_utils
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=True,
    max_num_faces=1,
    min_detection_confidence=0.5
)

draw_spec = mp_drawing.DrawingSpec(thickness=0, circle_radius=0)
draw_spec2 = mp_drawing.DrawingSpec(thickness=2, circle_radius=1, color=(255, 255, 255))

#get the landmarks of two eyes
left_eye_landmarks = temp.face_features['leftEyeUpper0'] + temp.face_features['leftEyeLower0']
right_eye_landmarks = temp.face_features['rightEyeUpper0'] + temp.face_features['rightEyeLower0']


# crop two eyes in one picture
def crop_two_eyes(another_image, split):
    image_out = cv2.cvtColor(another_image, cv2.COLOR_BGR2RGB)
    results = face_mesh.process(image_out)
    face_landmarks_list = {}
    is_available = 4

    # use mediapipe to get face features
    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            face_landmarks_list = get_landmarks(face_landmarks)
    # use cv2 to get face features
    else:
        if split == 'train':
            is_available = 3
            left_eye_image_cv2 = np.zeros((75, 75, 3))
            right_eye_image_cv2 = np.zeros((75, 75, 3))
            left_eye_pos = []
            right_eye_pos = []
            return left_eye_image_cv2, right_eye_image_cv2, left_eye_pos, right_eye_pos, is_available
        else:
            gray_image = cv2.cvtColor(another_image, cv2.COLOR_BGR2GRAY)
            eyes = detect_eyes([0, 0, 224, 224], gray_image)
            if len(eyes) == 2:
                rec_1 = eyes[0]
                rec_2 = eyes[1]
                if rec_1[0] < rec_2[0]:
                    left_eye_image_cv2 = another_image[rec_2[1]:rec_2[1] + rec_2[3], rec_2[0]:rec_2[0] + rec_2[2], :]
                    right_eye_image_cv2 = another_image[rec_1[1]:rec_1[1] + rec_1[3], rec_1[0]:rec_1[0] + rec_1[2], :]
                else:
                    right_eye_image_cv2 = another_image[rec_2[1]:rec_2[1] + rec_2[3], rec_2[0]:rec_2[0] + rec_2[2], :]
                    left_eye_image_cv2 = another_image[rec_1[1]:rec_1[1] + rec_1[3], rec_1[0]:rec_1[0] + rec_1[2], :]
            elif len(eyes) == 1:
                rec = eyes[0]
                if (rec[0] + rec[2] / 2) < 112:
                    right_eye_image_cv2 = another_image[rec[1]:rec[1] + rec[3], rec[0]:rec[0] + rec[2], :]
                    left_eye_image_cv2 = np.zeros((eye_size, eye_size, 3))
                else:
                    left_eye_image_cv2 = another_image[rec[1]:rec[1] + rec[3], rec[0]:rec[0] + rec[2], :]
                    right_eye_image_cv2 = np.zeros((eye_size, eye_size, 3))
            else:
                left_eye_image_cv2 = np.zeros((eye_size, eye_size, 3))
                right_eye_image_cv2 = np.zeros((eye_size, eye_size, 3))
            left_eye_pos = []
            right_eye_pos = []
            return left_eye_image_cv2, right_eye_image_cv2, left_eye_pos, right_eye_pos, len(eyes)

    image_trans = image_out.transpose(2, 0, 1)
    # left_eye_image, left_eye_pos = crop_eye(image_trans, left_eye_landmarks, face_landmarks_list)
    # right_eye_image, right_eye_pos = crop_eye(image_trans, right_eye_landmarks, face_landmarks_list)
    left_eye_image, left_eye_pos = crop_eye2(image_trans, temp.left_eye_connection, face_landmarks_list)
    right_eye_image, right_eye_pos = crop_eye2(image_trans, temp.right_eye_connection, face_landmarks_list)
    left_eye_image_cv2 = cv2.cvtColor(left_eye_image.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
    right_eye_image_cv2 = cv2.cvtColor(right_eye_image.transpose(1, 2, 0), cv2.COLOR_RGB2BGR)
    return left_eye_image_cv2, right_eye_image_cv2, left_eye_pos, right_eye_pos, is_available


# crop eyes image which size is 75*75*3
def crop_eye(image, landmarks, all_face_landmarks):

    # index = cvt2index(connections)
    xpos = []
    ypos = []
    for position in landmarks:
        if position in all_face_landmarks:
            xpos.append(all_face_landmarks[position][0])
            ypos.append(all_face_landmarks[position][1])

    # maximum and minimum pos of x,y
    xpos_max = max(xpos)
    xpos_min = min(xpos)
    ypos_max = max(ypos)
    ypos_min = min(ypos)
    eye_pos = [ypos_min, ypos_max, xpos_min, xpos_max]

    # # eye center pos
    # center_x = int((xpos_min + xpos_max) / 2)
    # center_y = int((ypos_min + ypos_max) / 2)

    # # crop range
    # crop_xpos_min = max(center_x - int(eye_size / 2), 0)
    # crop_xpos_max = min(center_x + int(eye_size / 2), 224)
    # crop_ypos_min = max(center_y - int(eye_size / 2), 0)
    # crop_ypos_max = min(center_y + int(eye_size / 2), 224)

    #crop image
    image_cropped = image[:, ypos_min:ypos_max, xpos_min:xpos_max]
    return image_cropped, eye_pos


def crop_eye2(image, connections, all_face_landmarks):
    index = cvt2index(connections)
    xpos = []
    ypos = []
    for position in index:
        if position in all_face_landmarks:
            xpos.append(all_face_landmarks[position][0])
            ypos.append(all_face_landmarks[position][1])

    # maximum and minimum pos of x,y
    xpos_max = max(xpos)
    xpos_min = min(xpos)
    ypos_max = max(ypos)
    ypos_min = min(ypos)
    # eye_pos = [ypos_min, ypos_max, xpos_min, xpos_max]

    # eye center pos
    center_x = int((xpos_min + xpos_max) / 2)
    center_y = int((ypos_min + ypos_max) / 2)

    # crop range
    crop_xpos_min = max(center_x - int(eye_size / 2), 0)
    crop_xpos_max = min(center_x + int(eye_size / 2), 224)
    crop_ypos_min = max(center_y - int(eye_size / 2), 0)
    crop_ypos_max = min(center_y + int(eye_size / 2), 224)
    eye_pos = [crop_ypos_min, crop_ypos_max, crop_xpos_min, crop_xpos_max]

    #crop image
    image_cropped = image[:, crop_ypos_min:crop_ypos_max, crop_xpos_min:crop_xpos_max]
    return image_cropped, eye_pos

# convert connection tuple to unique index list
def cvt2index(connections):
    index = []
    for connection in connections:
        index.append(connection[0])
        index.append(connection[1])
    return list(set(index))


VISIBILITY_THRESHOLD = 0.5
PRESENCE_THRESHOLD = 0.5


# get a dictionary which store the face_landmarks position
def get_landmarks(landmark_list):
    image_cols, image_rows = image_size
    idx_to_coordinates = {}
    for idx, landmark in enumerate(landmark_list.landmark):
        if ((landmark.HasField('visibility') and
             landmark.visibility < VISIBILITY_THRESHOLD) or
                (landmark.HasField('presence') and
                 landmark.presence < PRESENCE_THRESHOLD)):
            continue
        landmark_px = mp_drawing._normalized_to_pixel_coordinates(landmark.x, landmark.y,
                                                                  image_cols, image_rows)
        if landmark_px:
            idx_to_coordinates[idx] = landmark_px
    return idx_to_coordinates


if __name__ == '__main__':
    with h5py.File('/home/data/wjc_data/xgaze_224/test/subject0001.h5') as train:
            image = train['face_patch'][46, :]
            a = crop_two_eyes(image, split='test')
            print(a[0].shape)
    



