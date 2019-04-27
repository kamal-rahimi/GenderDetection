import numpy as np
import os
import cv2
import parse

from cvision_tools import detect_face, crop_face, convert_to_gray, resize_with_pad

from sklearn.model_selection import train_test_split

image_height = 100
image_width = 100
image_n_channels = 1 

FERET_IMAGE_PATH = "data/colorferet/dvd1/data/smaller/"
FERET_INFO_PATH  = "data/colorferet/dvd1/data/ground_truths/name_value/"


FERET_INFO_FROMAT_STRING = "id={}\ngender={}\nyob={}\nrace={}"

class ColorfretDataset():
    """ This calss is to read the images and label info from feret data set
    """
    def __init__(self):
        self.X_train = None
        self.X_valid = None
        self.X_test = None
        self.y_train = None
        self.y_valid = None
        self.y_test = None
    
    def read(self, test_size=0.2, valid_size=0, gray=True):
        images, labels = self.read_feret_data()
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


    def read_feret_data(self):
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
                    image, label = self.read_feret_subject(subject_dir, file_, subject_image_path, subject_info_path)
                    images.append(image)
                    labels.append(label)

        return images, labels

    def read_feret_subject(self, subject_dir, file_, subject_image_path, subject_info_path):
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
            parsed = parse.parse(FERET_INFO_FROMAT_STRING, string)
            label = {}
            label["id"] = parsed[0]
            label["gender"] = parsed[1]
            label["yob"] = parsed[2]
            label["race"] = parsed[3]

        return image, label