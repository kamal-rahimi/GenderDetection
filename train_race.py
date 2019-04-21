"""
Creates a convolutionaly neural network (CNN) in Tensorflow and trains the network
to identify the race of the person in an input image.
"""

import cv2
import numpy as np
import tensorflow as tf

import tensorflow as tf

from imblearn.over_sampling import RandomOverSampler
from collections import Counter
#import keras
#from keras import backend as K

#tf_sess = tf.Session()
#K.set_session(tf_sess)

from cvision_tools import ColorfretDataset, read_feret_data, detect_face, crop_face, convert_to_gray, resize_with_pad

from sklearn.preprocessing import LabelEncoder

import pickle

image_height = 100
image_width = 100
image_n_channels = 1

n_epochs = 300
batch_size = 10

initial_learning_rate = 0.001
decay_steps = 20000
decay_rate = 1/2

RACE_ENCODER_PATH = "./model/race_encoder.pickle"
RACE_PREDICTION_MODEL_PATH = "./model/race_model"


def main():
    dataset = ColorfretDataset()
    dataset.read()

    X_train         = np.array(dataset.X_train)
    X_test          = np.array(dataset.X_test)

    y_race_train    = np.array([y['race'] for y in dataset.y_train])
    y_race_test     = np.array([y['race'] for y in dataset.y_test])

    y_race_train    = [y if (y != 'Native-American' and y != 'Pacific-Islander' and y != 'Asian-Southern') else 'Other' for y in  y_race_train ]
    y_race_test     = [y if (y != 'Native-American' and y != 'Pacific-Islander' and y != 'Asian-Southern') else 'Other' for y in  y_race_test ] 

    X_train         = X_train.astype('float32')
    X_test          = X_test.astype('float32')
    X_train        /= 255
    X_test         /= 255

    race_enc   = LabelEncoder()
    y_race_train  = race_enc.fit_transform(y_race_train)
    y_race_test   = race_enc.transform(y_race_test)
    pickle.dump(race_enc, open(RACE_ENCODER_PATH, 'wb'))

    num_race_outputs = len(race_enc.classes_)

    race_uniques, race_counts = np.unique(y_race_train, return_counts=True)
    print(race_enc.inverse_transform(race_uniques))
    print (race_counts)

    random_over_sampler = RandomOverSampler(random_state=42)
    nsamples, nx, ny, nz = X_train.shape
    d2_X_train = X_train.reshape((nsamples,nx*ny*nz))
    d2_X_train, y_race_train = random_over_sampler.fit_resample(np.array(d2_X_train), y_race_train)
    X_train = d2_X_train.reshape(-1, nx, ny, nz)

    print(sorted(Counter(y_race_train).items()))


    with tf.name_scope("cnn_race_graph"):
        X = tf.placeholder(tf.float32, shape = [None, image_height, image_width, image_n_channels], name="X")
        #y = tf.placeholder(tf.float32, shape = [None, image_height, image_weight, image_n_channels])
        y = tf.placeholder(tf.int32, shape = [None])
        loss_weight = tf.placeholder(tf.float32, shape = [None])
        training = tf.placeholder_with_default(False, shape=[], name='training')

        conv1 = tf.layers.conv2d(X, filters=8, kernel_size=4,
                                strides=1, padding='SAME',
                                activation=tf.nn.relu, name="conv1")

        pool2 = tf.nn.max_pool(conv1, ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding="SAME")

        pool2_drop = tf.layers.dropout(pool2, 0.05, training=training)

        conv3 = tf.layers.conv2d(pool2_drop, filters=32, kernel_size=4,
                                strides=1, padding='SAME',
                                activation=tf.nn.relu, name="pool2")

        pool4 = tf.nn.max_pool(conv3, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")

        shape = tf.shape(pool4)

        pool4_flat = tf.reshape(pool4, shape=[-1, 32 * 5 * 5])
        pool4_flat_drop = tf.layers.dropout(pool4_flat, 0.1, training=training)

        fc1 = tf.layers.dense(pool4_flat_drop, 32, name="fc1")
        bn1 = tf.layers.batch_normalization(fc1, training=training, momentum=0.9)
        bn1_act = tf.nn.relu(bn1)

        logits_race = tf.layers.dense(fc1, num_race_outputs, name="logits_race")
        Y_proba_race = tf.nn.softmax(logits_race, name="Y_proba_race")

        global_step = tf.Variable(0, trainable=False, name="global_step")
        learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate)

    with tf.name_scope("train_race"):
        xentropy_race = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_race, labels=y)
        loss_race = tf.reduce_mean(xentropy_race)
        weighted_loss_race = tf.math.multiply(loss_race, loss_weight)
        optimizer_race = tf.train.AdamOptimizer(learning_rate=learning_rate) # beta1=0.8
        #optimizer = tf.train.MomentumOptimizer(learning_rate=0.00005, momentum=0.9, use_nesterov=True)
        training_op_race = optimizer_race.minimize(loss_race, global_step=global_step)

    with tf.name_scope("eval_race"):
        correct_race = tf.nn.in_top_k(logits_race, y, 1)
        accuracy_race = tf.reduce_mean(tf.cast(correct_race, tf.float32))

    with tf.name_scope("init_and_save"):
        init = tf.global_variables_initializer()
        saver = tf.train.Saver()

    with tf.Session() as train_race_sess:
        init.run()
        print("Training for race identification")
        for epoch in range(n_epochs):
            n_it = X_train.shape[0]//batch_size
            for it in range(n_it):
                X_batch = X_train[it*batch_size:(it+1)*batch_size,:,:,:]
                y_batch = y_race_train[it*batch_size:(it+1)*batch_size]
                race_weight_batch =  [1/race_counts[i] for i in y_batch] 
                #print(race_weight_batch)
                train_race_sess.run(training_op_race, feed_dict={X: X_batch, y: y_batch, training: True, loss_weight:race_weight_batch})
            acc_batch = accuracy_race.eval(feed_dict={X: X_batch, y: y_batch, loss_weight:np.ones(len(y_batch))})
            acc_train = accuracy_race.eval(feed_dict={X: X_train[1:100], y: y_race_train[1:100], loss_weight:np.ones(100)})
            acc_test = accuracy_race.eval(feed_dict={X: X_test, y: y_race_test, loss_weight:np.ones(len(y_race_test))})
            print(epoch, "Last batch accuracy:", acc_batch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

            save_path_race = saver.save(train_race_sess, RACE_PREDICTION_MODEL_PATH)


if __name__ == "__main__":
    main()