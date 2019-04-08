"""

"""
import cv2
import numpy as np
import tensorflow as tf

import tensorflow as tf
#import keras
#from keras import backend as K

tf_sess = tf.Session()
#K.set_session(tf_sess)

from cvision_tools import ColorfretDataset, read_feret_data, detect_face, crop_face, convert_to_gray, resize_with_pad

from sklearn.preprocessing import LabelEncoder


"""
#print(images[0])
#print(labels[0])
for image, face in zip(images, faces):
    cv2.imshow("image[40]",image)
    cv2.imshow("img",face)
    cv2.waitKey(0)
#cv2.destroyAllWindows()
"""

dataset = ColorfretDataset()
dataset.read()

X_train     = np.array(dataset.X_train)
X_test      = np.array(dataset.X_test)

y_gender_train   = np.array([y['gender'] for y in dataset.y_train])
y_gender_test    = np.array([y['gender'] for y in dataset.y_test])

y_race_train   = np.array([y['race'] for y in dataset.y_train])
y_race_test    = np.array([y['race'] for y in dataset.y_test])

X_train     = X_train.astype('float32')
X_test      = X_test.astype('float32')
X_train    /= 255
X_test     /= 255

gender_enc = LabelEncoder()
y_gender_train  = gender_enc.fit_transform(y_gender_train)
y_gender_test   = gender_enc.transform(y_gender_test)

race_enc   = LabelEncoder()
y_race_train  = race_enc.fit_transform(y_race_train)
y_race_test   = race_enc.transform(y_race_test)

# print(y_gender_train[1:10])

num_gender_outputs = len(gender_enc.classes_)
num_race_outputs = len(race_enc.classes_)

gender_uniques, gender_counts = np.unique(y_gender_train, return_counts=True)
#print(gender_enc.inverse_transform(g_uniques))
#print (g_counts)

race_uniques, race_counts = np.unique(y_race_train, return_counts=True)
#print(race_enc.inverse_transform(r_uniques))
#print (r_counts)

height = 100
width = 100
n_channels = 1  

X = tf.placeholder(tf.float32, shape = [None, height, width, n_channels])
#y = tf.placeholder(tf.float32, shape = [None, height, weight, n_channels])
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

fc1 = tf.layers.dense(pool4_flat, 32, activation=tf.nn.relu, name="fc1")

logits_gender = tf.layers.dense(fc1, num_gender_outputs, name="output_gender")
logits_race = tf.layers.dense(fc1, num_race_outputs, name="output_race")
Y_proba_gender = tf.nn.softmax(logits_gender, name="Y_proba_gender")
Y_proba_race = tf.nn.softmax(logits_race, name="Y_proba_race")

initial_learning_rate =0.001
decay_steps = 2000
decay_rate = 1/2
global_step = tf.Variable(0, trainable=False, name="global_step")
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate)

with tf.name_scope("train_gender"):
    xentropy_gender = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_gender, labels=y)
    loss_gender = tf.reduce_mean(xentropy_gender)
    weighted_loss_gender = tf.math.multiply(loss_gender, loss_weight)
    optimizer_gender = tf.train.AdamOptimizer(learning_rate=learning_rate) # beta1=0.8
    #optimizer = tf.train.MomentumOptimizer(learning_rate=0.00005, momentum=0.9, use_nesterov=True)
    training_op_gender = optimizer_gender.minimize(loss_gender, global_step=global_step)

with tf.name_scope("eval_gender"):
    correct_gender = tf.nn.in_top_k(logits_gender, y, 1)
    accuracy_gender = tf.reduce_mean(tf.cast(correct_gender, tf.float32))

with tf.name_scope("train_race"):
    xentropy_race = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits_race, labels=y)
    loss_race = tf.reduce_mean(xentropy_race)
    weighted_loss_race = tf.math.multiply(loss_gender, loss_weight)
    optimizer_race = tf.train.AdamOptimizer(learning_rate=learning_rate) # beta1=0.8
    #optimizer = tf.train.MomentumOptimizer(learning_rate=0.00005, momentum=0.9, use_nesterov=True)
    training_op_race = optimizer_race.minimize(loss_race, global_step=global_step)

with tf.name_scope("eval_race"):
    correct_race = tf.nn.in_top_k(logits_race, y, 1)
    accuracy_race = tf.reduce_mean(tf.cast(correct_race, tf.float32))


with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()

n_epochs = 200
batch_size = 10
"""
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./model/'))
    print(sess.run('training:0'))
"""

with tf.Session() as train_gender_sess:
    init.run()
    print("Training for gender identification")
    for epoch in range(n_epochs):
        n_it = X_train.shape[0]//batch_size
        for it in range(n_it):
            X_batch = X_train[it*batch_size:(it+1)*batch_size,:,:,:]
            y_batch = y_gender_train[it*batch_size:(it+1)*batch_size]
            gender_weight_batch =  [1/gender_counts[i] for i in y_batch] 
            train_gender_sess.run(training_op_gender, feed_dict={X: X_batch, y: y_batch, training: True, loss_weight:gender_weight_batch})
        acc_batch = accuracy_gender.eval(feed_dict={X: X_batch, y: y_batch, loss_weight:np.ones(len(y_batch))})
        acc_train = accuracy_gender.eval(feed_dict={X: X_train[1:100], y: y_gender_train[1:100], loss_weight:np.ones(100)})
        acc_test = accuracy_gender.eval(feed_dict={X: X_test, y: y_gender_test, loss_weight:np.ones(len(y_gender_test))})
        print(epoch, "Last batch accuracy:", acc_batch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

        save_path_gender = saver.save(train_gender_sess, "./model/model_gender")

with tf.Session() as train_race_sess:
    init.run()
    print("Training for race identification")
    for epoch in range(n_epochs):
        n_it = X_train.shape[0]//batch_size
        for it in range(n_it):
            X_batch = X_train[it*batch_size:(it+1)*batch_size,:,:,:]
            y_batch = y_race_train[it*batch_size:(it+1)*batch_size]
            race_weight_batch =  [1/race_counts[i] for i in y_batch] 
            train_race_sess.run(training_op_race, feed_dict={X: X_batch, y: y_batch, training: True, loss_weight:race_weight_batch})
        acc_batch = accuracy_race.eval(feed_dict={X: X_batch, y: y_batch, loss_weight:np.ones(len(y_batch))})
        acc_train = accuracy_race.eval(feed_dict={X: X_train[1:100], y: y_race_train[1:100], loss_weight:np.ones(100)})
        acc_test = accuracy_race.eval(feed_dict={X: X_test, y: y_race_test, loss_weight:np.ones(len(y_race_test))})
        print(epoch, "Last batch accuracy:", acc_batch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

        save_path_race = saver.save(train_race_sess, "./model/model_race")

