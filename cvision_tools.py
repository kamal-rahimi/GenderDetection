"""
Methods to read and preprocess image data
"""

import numpy as np
import os
import cv2
import parse

FERET_IMAGE_PATH = "data/colorferet/dvd1/data/smaller/"
FERET_INFO_PATH = "data/colorferet/dvd1/data/ground_truths/name_value/"

INFO_FROMAT_STRING = "id={}\ngender={}\nyob={}\nrace={}"


def read_feret_subject(subject_dir, file_, subject_image_path, subject_info_path):
    """ Reads an image and the image info from feret data set
    Args:

    Returns:
        image: a nummpy array of the image data
        label: a python dictionary of the image info
    """ 
    subject_image_full_path = os.path.abspath(os.path.join(subject_image_path, file_))
    subject_info_full_path1 = os.path.abspath(os.path.join(subject_info_path, "{}.txt".format(subject_dir)))
    subject_info_full_path2 = os.path.abspath(os.path.join(subject_info_path, file_))
    
    image = cv2.imread(subject_image_full_path)
    
    with open(subject_info_full_path1, "r") as f:
        string = f.read()
        parsed = parse.parse(INFO_FROMAT_STRING, string)
        label = {}
        label["id"] = parsed[0]
        label["gender"] = parsed[1]
        label["yob"] = parsed[2]
        label["race"] = parsed[3]

    return image, label


def read_feret_data():
    """ Reads the images and info from feret data set
    Args:

    Returns:
        images: a python array of the images data
        labels: a python array of the image info
    """ 
    images = []
    labels = []
    
    for subject_dir in os.listdir(FERET_IMAGE_PATH):
        subject_image_path = os.path.join(FERET_IMAGE_PATH, subject_dir)
        subject_info_path  = os.path.join(FERET_INFO_PATH, subject_dir)
        
        for file_ in os.listdir(subject_image_path):
            if file_.endswith("_fa.ppm"):
                image, label = read_feret_subject(subject_dir, file_, subject_image_path, subject_info_path)
                images.append(image)
                labels.append(label)

    return images, labels


face_cascade = cv2.CascadeClassifier('/home/kamal/.local/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
def detect_face(image):
    """ Detecs face in an image
    Args:
        image: a numpy array containing image data
    Return:
       face[0]: A box indicating  the location of one face in the image (x, y, w, h)
    """
    faces = face_cascade.detectMultiScale(images[0], 1.1, 5)
    return faces[0]


def crop_face(image):
    """ Crops the face part of input image
    Args:
        image: a numpy array containing image data
    Return:
        cropped_face: the cropped face image
    """
    x, y, w, h = detect_face(image)
    cropped_face = image[y:y+h, x:x+w]
    return cropped_face


def convert_to_gray(image):
    """ Converts to grayscale
    Args:
        image: a numpy array containing image data
    Return:
        gray_image: the image in grayscale format
    """
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    return gray_image


def resize_with_pad(image, height, width):
    """ Resizes the image without changing scale (pads are added if necessary)
    Args:
        image: a numpy array containing image data
        height: ouput image height
        width: output image width
    Return:
        resized_image: resized image
    """
    h, w, _ = image.shape
    top_pad, bottom_pad, left_pad, right_pad = (0, 0, 0, 0)
    if ( h / w < height / width):
       pad = height / width * w - h
       left_pad = int(pad / 2)
       right_pad = int(pad / 2)
    else:
        pad = width / height * h - w
        top_pad = int(pad / 2)
        bottom_pad = int(pad / 2)

    padded_image = cv2.copyMakeBorder(image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=[0 ,0, 0])

    resized_image = cv2.resize(padded_image, (height, width))

    return resized_image


images, labels = read_feret_data()

faces = [crop_face(image) for image in images]

faces = [resize_with_pad(face, 100, 100) for face in faces]

faces = [convert_to_gray(face) for face in faces]

#print(images[0])
#print(labels[0])
"""
for image, face in zip(images, faces):
    cv2.imshow("image[40]",image)
    cv2.imshow("img",face)
    cv2.waitKey(0)
#cv2.destroyAllWindows()
"""