"""
Methods to read and preprocess image data
"""

import numpy as np
import os
import cv2
import parse

from sklearn.model_selection import train_test_split

image_height = 100
image_width = 100
image_n_channels = 1 

FERET_IMAGE_PATH = "data/colorferet/dvd1/data/smaller/"
FERET_INFO_PATH  = "data/colorferet/dvd1/data/ground_truths/name_value/"


INFO_FROMAT_STRING = "id={}\ngender={}\nyob={}\nrace={}"

class ColorfretDataset():
    def __init__(self):
        self.X_train = None
        self.X_valid = None
        self.X_test = None
        self.y_train = None
        self.y_valid = None
        self.y_test = None
    
    def read(self, test_size=0.2, valid_size=0, gray=True):
        images, labels = read_feret_data()
        images = np.array(images)
        faces = [crop_face(image) for image in images]
        faces = [resize_with_pad(face, image_height, image_width) for face in faces]
        if gray==True:
            faces = [convert_to_gray(face) for face in faces]
            faces = np.array(faces)
            faces=faces.reshape(-1, image_height, image_width, image_n_channels)

        X_train, X_test, y_train, y_test = train_test_split(faces, labels, test_size=test_size, random_state=42)
        X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=valid_size, random_state=42)

        self.X_train = X_train
        self.X_valid = X_valid
        self.X_test = X_test
        self.y_train = y_train
        self.y_valid = y_valid
        self.y_test = y_test


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

def read_image(image_path):
    """ Reads an image from an image path
    Args:
        image_path: a string indicating the image path
    Returns:
        image: a nummpy array of the image data
    """ 
    image = cv2.imread(image_path)

    return image


face_cascade = cv2.CascadeClassifier('/home/kamal/.local/lib/python3.6/site-packages/cv2/data/haarcascade_frontalface_default.xml')
def detect_face(image):
    """ Detecs face in an image
    Args:
        image: a numpy array containing image data
    Return:
       face[0]: A box indicating  the location of one face in the image (x, y, w, h)
    """
    faces = face_cascade.detectMultiScale(image, 1.1, 5)
    #print(faces)
    if faces != ():
        return faces[0]
    else:
        return 0, 0, image.shape[0], image.shape[1]
    #return faces[0] if faces != () else image 


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

def main():
    images, labels = read_feret_data()
    faces = [crop_face(image) for image in images]
    faces = [resize_with_pad(face, 100, 100) for face in faces]
    faces = [convert_to_gray(face) for face in faces]
    #print(images[0])
    #print(labels[0])
    for image, face in zip(images, faces):
        cv2.imshow("image[40]",image)
        cv2.imshow("img",face)
        cv2.waitKey(0)
    #cv2.destroyAllWindows()


def __init__():
    pass