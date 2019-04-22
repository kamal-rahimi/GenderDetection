"""
"""

import tensorflow as tf 
import numpy as np
import pickle
import argparse

from cvision_tools import read_image, detect_face, crop_face, convert_to_gray, resize_with_pad

import cv2 as cv2

image_height = 100
image_width = 100
image_n_channels = 1

GENDER_ENCODER_PATH = "./model/gender_encoder.pickle"
RACE_ENCODER_PATH = "./model/race_encoder.pickle"

GENDER_PREDICTION_MODEL_PATH = "./model/gender_model"
RACE_PREDICTION_MODEL_PATH = "./model/race_model"


def prepare_image(image_path):
    image = read_image(image_path)
    image = np.array(image)
    face  = crop_face(image)
    face  = resize_with_pad(face, image_height, image_width)
    face  = convert_to_gray(face)
    face  = np.array(face)
    face  = face.reshape(-1, image_height, image_width, image_n_channels)
    face  = face.astype('float32')
    face /= 255
    return image, face


def indetify_gender(face):
    gender_encoder = pickle.load(open(GENDER_ENCODER_PATH, 'rb'))

    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(GENDER_PREDICTION_MODEL_PATH + '.meta')
    graph = tf.get_default_graph()

    X = graph.get_tensor_by_name("cnn_graph_gender/X:0")
    Y_proba_gender = graph.get_tensor_by_name("cnn_graph_gender/Y_proba_gender:0")

    with tf.Session() as predit_sess:
        saver.restore(predit_sess, GENDER_PREDICTION_MODEL_PATH)
        probs = Y_proba_gender.eval(feed_dict={X: face})
        gender_index = np.argmax(probs)
        predicted_gender = gender_encoder.inverse_transform([gender_index])
        prob_gender = probs[0, gender_index]

        print( predicted_gender )
        print("Confidennce={}".format(prob_gender))
        #print( logits_gender.eval(feed_dict={X: face}) )

    return predicted_gender, prob_gender


def indetify_race(face):
    race_encoder = pickle.load(open(RACE_ENCODER_PATH, 'rb'))

    tf.reset_default_graph()
    saver = tf.train.import_meta_graph(RACE_PREDICTION_MODEL_PATH + '.meta')
    graph = tf.get_default_graph()

    X = graph.get_tensor_by_name("cnn_graph_race/X:0")
    Y_proba_race = graph.get_tensor_by_name("cnn_graph_race/Y_proba_race:0")

    with tf.Session() as predit_sess:
        saver.restore(predit_sess, RACE_PREDICTION_MODEL_PATH)
        probs = Y_proba_race.eval(feed_dict={X: face})
        race_index = np.argmax(probs)
        predicted_race = race_encoder.inverse_transform([race_index])
        prob_race = probs[0, race_index]
        print( predicted_race )
        print("Confidennce={}".format(prob_race))
    
    return predicted_race, prob_race
        #print( logits_gender.eval(feed_dict={X: face}) )

def display_image(image, predicted_gender, prob_gender):
    image = resize_with_pad(image, 600, 600)
    x, y, w, h = detect_face(image)
    cv2.rectangle(image, (x,y), (x+w,y+h), (255,0,0), 2)
    cv2.putText(image, "{}".format(predicted_gender[0]), (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
    cv2.putText(image, "Prob: {:.2f}".format(prob_gender), (10, 130), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 0, 0), 2)
    cv2.imshow("Image", image)
    cv2.waitKey(0)

def main():

    ap = argparse.ArgumentParser()
    ap.add_argument("-p", "--path", type=str, default="./data/test/obama.jpg", help="Specify the path to the input image")
    args = vars(ap.parse_args())
    image_path = args["path"]

    image, face = prepare_image(image_path)
    predicted_gender, prob_gender = indetify_gender(face)
    #predicted_race, prob_race = indetify_race(face)
    display_image(image, predicted_gender, prob_gender)


if __name__ == "__main__":
    main()