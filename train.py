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

y_g_train   = np.array([y['gender'] for y in dataset.y_train])
y_g_test    = np.array([y['gender'] for y in dataset.y_test])

y_r_train   = np.array([y['race'] for y in dataset.y_train])
y_r_test    = np.array([y['race'] for y in dataset.y_test])

X_train     = X_train.astype('float32')
X_test      = X_test.astype('float32')
X_train    /= 255
X_test     /= 255

gender_enc = LabelEncoder()
y_g_train  = gender_enc.fit_transform(y_g_train)
y_g_test   = gender_enc.transform(y_g_test)

race_enc   = LabelEncoder()
y_r_train  = race_enc.fit_transform(y_r_train)
y_r_test   = race_enc.transform(y_r_test)

print(y_g_train[1:10])

num_outputs = len(race_enc.classes_)

uniques, counts = np.unique(y_g_train, return_counts=True)
print(gender_enc.inverse_transform(uniques))
print (counts)

uniques, counts = np.unique(y_r_train, return_counts=True)
print(race_enc.inverse_transform(uniques))
print (counts)

height = 100
width = 100
n_channels = 1  

X = tf.placeholder(tf.float32, shape = [None, height, width, n_channels])
#y = tf.placeholder(tf.float32, shape = [None, height, weight, n_channels])
y = tf.placeholder(tf.int32, shape = [None])
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

logits = tf.layers.dense(fc1, num_outputs, name="output")
Y_proba = tf.nn.softmax(logits, name="Y_proba")

initial_learning_rate =0.001
decay_steps = 2000
decay_rate = 1/2
global_step = tf.Variable(0, trainable=False, name="global_step")
learning_rate = tf.train.exponential_decay(initial_learning_rate, global_step, decay_steps, decay_rate)

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate) # beta1=0.8
    #optimizer = tf.train.MomentumOptimizer(learning_rate=0.00005, momentum=0.9, use_nesterov=True)
    training_op = optimizer.minimize(loss, global_step=global_step)


with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


n_epochs = 100
batch_size = 10
"""
with tf.Session() as sess:
    saver.restore(sess, tf.train.latest_checkpoint('./model/'))
    print(sess.run('training:0'))
"""
with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_it = X_train.shape[0]//batch_size
        for it in range(n_it):
            X_batch = X_train[it*batch_size:(it+1)*batch_size,:,:,:]
            y_batch = y_r_train[it*batch_size:(it+1)*batch_size]
            #print(shape.eval(feed_dict={X: X_batch, y: y_batch}))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: X_train[1:100], y: y_r_train[1:100]})
        acc_test = accuracy.eval(feed_dict={X: X_test, y: y_r_test})
        print(epoch, "Last batch accuracy:", acc_batch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

        save_path = saver.save(sess, "./model/model2")