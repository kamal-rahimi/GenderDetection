"""
Creates a convolutionaly neural network (CNN) in Tensorflow and trains the network
to identify the race of the person in an input image.
"""

import tensorflow as tf

from cvision_tools import ColorfretDataset, read_feret_data, detect_face, crop_face, convert_to_gray, resize_with_pad

import numpy as np
from sklearn.preprocessing import LabelEncoder
from imblearn.over_sampling import RandomOverSampler
import pickle


image_height = 100
image_width = 100
image_n_channels = 1

n_epochs = 300
batch_size = 10

initial_learning_rate = 0.001
decay_steps = 4000
decay_rate = 1/2

RACE_ENCODER_PATH = "./model/race_encoder.pickle"
RACE_PREDICTION_MODEL_PATH = "./model/race_model"


def prepare_race_data():
    """ Prepares training and test data for race identification model from Colofret image dataset 
    (Face area in each image is detected and cropped)  
    Args:
    Returns:
        X_train: a numpy array of the train face image data
        X_test: a numpy array of the test face image data
        y_race_train: a numpy array of the train race lables
        y_race_test: a numpy array of the test race lables
    """
    dataset = ColorfretDataset()
    dataset.read()

    X_train         = np.array(dataset.X_train).astype('float32')
    X_test          = np.array(dataset.X_test).astype('float32')
    X_train        /= 255
    X_test         /= 255

    y_race_train    = np.array([y['race'] for y in dataset.y_train])
    y_race_test     = np.array([y['race'] for y in dataset.y_test])

    y_race_train    = [y if (y != 'Native-American' and y != 'Pacific-Islander' and y != 'Asian-Southern') else 'Other' for y in  y_race_train ]
    y_race_test     = [y if (y != 'Native-American' and y != 'Pacific-Islander' and y != 'Asian-Southern') else 'Other' for y in  y_race_test ] 

    return X_train, X_test, y_race_train, y_race_test


def create_race_encoder(y_race):
    """ Creates a race label encoder model
    Args:
        y_race: a numpy array of race labels
    Returns:
        race_enc: a race label encoder model
    """
    race_enc = LabelEncoder()
    race_enc.fit(y_race)
    pickle.dump(race_enc, open(RACE_ENCODER_PATH, 'wb'))
    return race_enc

def over_sample(X, y):
    random_over_sampler = RandomOverSampler(random_state=42)
    nsamples, nx, ny, nz = X.shape
    d2_X = X.reshape((nsamples,nx*ny*nz))
    d2_X_os, y_os = random_over_sampler.fit_resample(np.array(d2_X), y)
    X_os = d2_X.reshape(-1, nx, ny, nz)
    #print(sorted(Counter(y_os).items()))
    return X_os, y_os


def train_race_model(X_train, X_test, y_train, y_test, race_encoder):
    """ Creates a convoluational neural network (CNN) and trains the model to identify race
    of an input face image
    Args:
        X_train: a numpy array of the train image data
        X_test: a numpy array of the test image data
        y_train: a numpy array of the train race lables
        y_test: a numpy array of the test race lables
    Returns:
    """

    num_race_outputs = len(race_encoder.classes_)
    race_uniques, race_counts = np.unique(y_train, return_counts=True)

    with tf.name_scope("cnn_graph_race"):
        X = tf.placeholder(tf.float32, shape = [None, image_height, image_width, image_n_channels], name="X")
        #y = tf.placeholder(tf.float32, shape = [None, image_height, image_weight, image_n_channels])
        y = tf.placeholder(tf.int32, shape = [None])
        loss_weight = tf.placeholder(tf.float32, shape = [None])
        training = tf.placeholder_with_default(False, shape=[], name='training')

        conv1 = tf.layers.conv2d(X, filters=8, kernel_size=4,
                                strides=1, padding='SAME',
                                activation=tf.nn.relu, name="conv1")

        pool2 = tf.nn.max_pool(conv1, ksize=[1, 5, 5, 1], strides=[1, 5, 5, 1], padding="SAME")

        pool2_drop = tf.layers.dropout(pool2, 0.1, training=training)

        conv3 = tf.layers.conv2d(pool2_drop, filters=32, kernel_size=4,
                                strides=1, padding='SAME',
                                activation=tf.nn.relu, name="pool2")

        pool4 = tf.nn.max_pool(conv3, ksize=[1, 4, 4, 1], strides=[1, 4, 4, 1], padding="SAME")

        shape = tf.shape(pool4)

        pool4_flat = tf.reshape(pool4, shape=[-1, 32 * 5 * 5])
        pool4_flat_drop = tf.layers.dropout(pool4_flat, 0.5, training=training)

        fc1 = tf.layers.dense(pool4_flat_drop, 32, name="fc1")
        bn1 = tf.layers.batch_normalization(fc1, training=training, momentum=0.9)
        bn1_act = tf.nn.relu(bn1)

        logits_race = tf.layers.dense(bn1_act, num_race_outputs, name="logits_race")
        Y_proba_race = tf.nn.softmax(logits_race, name="Y_proba_race")

        global_step = tf.Variable(0, trainable=False, name="global_step")
        learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate)

    with tf.name_scope("train_race"):
        xentropy_race = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_race, labels=y)
        loss_race = tf.reduce_mean(xentropy_race)
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
                y_batch = y_train[it*batch_size:(it+1)*batch_size]
                train_race_sess.run(training_op_race, feed_dict={X: X_batch, y: y_batch, training: True})
            
            acc_batch = accuracy_race.eval(feed_dict={X: X_batch, y: y_batch})
            acc_train = accuracy_race.eval(feed_dict={X: X_train[1:100], y: y_train[1:100]})
            acc_test = accuracy_race.eval(feed_dict={X: X_test, y: y_test})
            print(epoch, "Last batch accuracy:", acc_batch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

            save_path_race = saver.save(train_race_sess, RACE_PREDICTION_MODEL_PATH)


def main():
    X_train, X_test, y_race_train, y_race_test = prepare_race_data()
    
    race_encoder = create_race_encoder(y_race_train)
    
    y_train = race_encoder.transform(y_race_train)
    y_test  = race_encoder.transform(y_race_test)
    
    X_train, y_train = over_sample(X_train, y_train)
    
    train_race_model(X_train, X_test, y_train, y_test, race_encoder)


if __name__ == "__main__":
    main()