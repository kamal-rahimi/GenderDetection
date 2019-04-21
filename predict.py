"""
"""

import tensorflow as tf 
import numpy as np
import pickle

from cvision_tools import read_image, detect_face, crop_face, convert_to_gray, resize_with_pad

image_height = 100
image_width = 100
image_n_channels = 1

GENDER_ENCODER_PATH = "./model/gender_encoder.pickle"
RACE_ENCODER_PATH = "./model/race_encoder.pickle"

GENDER_PREDICTION_MODEL_PATH = "./model/gender_model"
RACE_PREDICTION_MODEL_PATH = "./model/race_model"


#logits_race = graph.get_tensor_by_name("cnn_graph/logits_race:0")

#for i in range( len(race_encoder.classes_) ):
#    print(race_encoder.inverse_transform([i]))


image_path = "./data/test/ross.jpg"

image = read_image(image_path)
image = np.array(image)
face  = crop_face(image)
face  = resize_with_pad(face, image_height, image_width)
face  = convert_to_gray(face)
face  = np.array(face)
face  = face.reshape(-1, image_height, image_width, image_n_channels)
face  = face.astype('float32')
face /= 255


gender_encoder = pickle.load(open(GENDER_ENCODER_PATH, 'rb'))

tf.reset_default_graph()
saver = tf.train.import_meta_graph(GENDER_PREDICTION_MODEL_PATH + '.meta')
graph = tf.get_default_graph()

X = graph.get_tensor_by_name("cnn_gender_graph/X:0")
Y_proba_gender = graph.get_tensor_by_name("cnn_gender_graph/Y_proba_gender:0")
Y_proba_race = graph.get_tensor_by_name("cnn_gender_graph/Y_proba_race:0")

with tf.Session() as predit_sess:
    saver.restore(predit_sess, GENDER_PREDICTION_MODEL_PATH)
    probs = Y_proba_gender.eval(feed_dict={X: face})
    gender_index = np.argmax(probs)
    print( gender_encoder.inverse_transform([gender_index]) )
    print("Confidennce={}".format(probs[0, gender_index]))
    #print( logits_gender.eval(feed_dict={X: face}) )


race_encoder = pickle.load(open(RACE_ENCODER_PATH, 'rb'))

tf.reset_default_graph()
saver = tf.train.import_meta_graph(RACE_PREDICTION_MODEL_PATH + '.meta')
graph = tf.get_default_graph()

X = graph.get_tensor_by_name("cnn_race_graph/X:0")
Y_proba_gender = graph.get_tensor_by_name("cnn_race_graph/Y_proba_gender:0")
Y_proba_race = graph.get_tensor_by_name("cnn_race_graph/Y_proba_race:0")

with tf.Session() as predit_sess:
    saver.restore(predit_sess, RACE_PREDICTION_MODEL_PATH)
    probs = Y_proba_race.eval(feed_dict={X: face})
    race_index = np.argmax(probs)
    print( race_encoder.inverse_transform([race_index]) )
    print("Confidennce={}".format(probs[0, race_index]))
    #print( logits_gender.eval(feed_dict={X: face}) )