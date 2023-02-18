import numpy as np
import matplotlib.pyplot as plt
import nibabel as nib
import os
import pandas as pd
import torch
from torch import nn
import torch.nn.functional as F
import glob

from skimage.color import rgb2gray
import cv2
from skimage import measure
import mediapipe as mp


def add_face_detection(mp_drawing,
                       results, 
                       image):
    if results.detections:
        counter = 0
        annotated_image = image.copy()
        for detection in results.detections:
            mp_drawing.draw_detection(annotated_image, detection)
            counter += 1
        return annotated_image, counter
    else:
        return image, 0
        
def add_face_box_to_img(image, convert=True):
  '''
  Face bounding box obtained with open cv haar cascades.
  Image modified inplace!
  '''
  face_cascade_name = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
  haar_cascade = cv2.CascadeClassifier(face_cascade_name)
  faces_rect = haar_cascade.detectMultiScale(image, scaleFactor = 1.1,
                                             minNeighbors = 3, minSize=(100, 100))
  color_img = cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
  for (x,y,w,h) in faces_rect:
     cv2.rectangle(color_img, (x, y), (x+w, y+h), (220, 20, 60), 2)
  return color_img, faces_rect

def add_facial_landmarks(mp_drawing,
                         mp_face_mesh,
                         mp_drawing_styles,
                         results, image):
    """
    Add facial landmarks to img for plotting
    """
    if results.multi_face_landmarks:
        counter = 0
        for face_landmarks in results.multi_face_landmarks:
            mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_tesselation_style())
            mp_drawing.draw_landmarks(
                    image=image,
                    landmark_list=face_landmarks,
                    connections=mp_face_mesh.FACEMESH_CONTOURS,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=mp_drawing_styles
                    .get_default_face_mesh_contours_style())
        counter += 1
        return image, counter
    else:
        return image, 0

def normalization_eyes(results, mp_face_mesh):
    """
    Get the interocular distance in the ct img with landmarks
    """
    mesh = np.array([(p.x, p.y) for p in results.multi_face_landmarks[0].landmark])

    # find landmark indices for eyes
    left_eye_idx = np.unique(list(mp_face_mesh.FACEMESH_LEFT_EYE))
    right_eye_idx = np.unique(list(mp_face_mesh.FACEMESH_RIGHT_EYE))

    # get midpoint for left eye
    left_eye = sorted(mesh[left_eye_idx], key=lambda x: x[0])
    left_eye_pt = (.5*(left_eye[-1][0] + left_eye[0][0]), .5*(left_eye[-1][1] + left_eye[0][1]))

    # get midpoint for right eye
    right_eye = sorted(mesh[right_eye_idx], key=lambda x: x[0])
    right_eye_pt = (.5*(right_eye[-1][0] + right_eye[0][0]), .5*(right_eye[-1][1] + right_eye[0][1]))

    # get normalized distance between both midpoints
    return right_eye_pt, left_eye_pt, np.linalg.norm(np.array(right_eye_pt) - np.array(left_eye_pt))

def get_distance_landmarks(res_ct, res_pet, mp_face_mesh, iod=None):
    """
    Implement the mean absolute distance betwwen the vertex locations,
    normalized by the distance between the eye centers (Interocular distance)
    """
    if res_ct.multi_face_landmarks is None or res_pet.multi_face_landmarks is None:
        print("Cannot compute the distance")
        return 

    # don't use normalization
    if iod is None:
        iod = 1

    mesh_ct = np.array([(p.x, p.y) for p in res_ct.multi_face_landmarks[0].landmark])
    mesh_pet = np.array([(p.x, p.y) for p in res_pet.multi_face_landmarks[0].landmark])
    return 100*np.sum(np.abs(mesh_ct - mesh_pet))/(iod*mp_face_mesh.FACEMESH_NUM_LANDMARKS)

def plot(img1, img2, title1, title2):
    fig, axs = plt.subplots(1, 2, figsize=(12, 6))
    axs[0].imshow(img1, cmap='gray')
    axs[0].axis('off')
    axs[0].set(title=title1)

    axs[1].imshow(img2, cmap='gray')
    axs[1].axis('off')
    axs[1].set(title=title2)
    plt.show()

