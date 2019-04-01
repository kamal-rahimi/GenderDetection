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

from cvision_tools import read_feret_data, detect_face, crop_face, convert_to_gray, resize_with_pad

from sklearn.model_selection import train_test_split

class Dataset():
    def __init__(self):
        self.X_train = None
        self.X_valid = None
        self.X_test = None
        self.y_train = None
        self.y_valid = None
        self.y_test = None
    
    def read(self):
        images, labels = read_feret_data()
        images = np.array(images)
        faces = [crop_face(image) for image in images]
        faces = [resize_with_pad(face, 100, 100) for face in faces]
        faces = [convert_to_gray(face) for face in faces]
        faces = np.array(faces)
        faces=faces.reshape(-1, 100, 100, 1)
        genders = [1 if label["gender"]=="Female" else 0 for label in labels]
        genders = np.array(genders)
        print(np.unique(genders))
        
        X_train, X_test, y_train, y_test = train_test_split(faces, genders, test_size=0.2, random_state=42)
        #X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

        self.X_train = X_train
        #self.X_valid = X_valid
        self.X_test = X_test
        self.y_train = y_train
        #self.y_valid = y_valid
        self.y_test = y_test

data_set = Dataset()
data_set.read()
print(data_set.X_train.shape)
print(data_set.y_train.shape)
print(len(data_set.X_train))
print(len(data_set.y_train))
#print(len(data_set.X_valid))
#print(len(data_set.y_valid))
print(len(data_set.X_test))
print(len(data_set.y_test))
"""
#print(images[0])
#print(labels[0])
for image, face in zip(images, faces):
    cv2.imshow("image[40]",image)
    cv2.imshow("img",face)
    cv2.waitKey(0)
#cv2.destroyAllWindows()
"""

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

conv3 = tf.layers.conv2d(pool2, filters=16, kernel_size=4,
                         strides=1, padding='SAME',
                         activation=tf.nn.relu, name="pool2")


pool4 = tf.nn.max_pool(conv3, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding="SAME")

shape = tf.shape(pool4)

pool4_flat = tf.reshape(pool4, shape=[-1, 16 * 10 * 10])
pool4_flat_drop = tf.layers.dropout(pool4_flat, 0.8, training=training)

fc1 = tf.layers.dense(pool4_flat, 32, activation=tf.nn.relu, name="fc1")
fc1_drop = tf.layers.dropout(fc1, 0.2, training=training)
logits = tf.layers.dense(fc1, 2, name="output")
Y_proba = tf.nn.softmax(logits, name="Y_proba")

with tf.name_scope("train"):
    xentropy = tf.nn.sparse_softmax_cross_entropy_with_logits(logits=logits, labels=y)
    loss = tf.reduce_mean(xentropy)
    optimizer = tf.train.AdamOptimizer()
    #optimizer = tf.train.MomentumOptimizer(learning_rate=0.00005, momentum=0.9, use_nesterov=True)
    training_op = optimizer.minimize(loss)


with tf.name_scope("eval"):
    correct = tf.nn.in_top_k(logits, y, 1)
    accuracy = tf.reduce_mean(tf.cast(correct, tf.float32))

with tf.name_scope("init_and_save"):
    init = tf.global_variables_initializer()
    saver = tf.train.Saver()


n_epochs = 200
batch_size = 10

with tf.Session() as sess:
    init.run()
    for epoch in range(n_epochs):
        n_it = data_set.X_train.shape[0]//batch_size
        for it in range(n_it):
            X_batch = data_set.X_train[it*batch_size:(it+1)*batch_size,:,:,:]
            y_batch = data_set.y_train[it*batch_size:(it+1)*batch_size]
            #print(shape.eval(feed_dict={X: X_batch, y: y_batch}))
            sess.run(training_op, feed_dict={X: X_batch, y: y_batch, training: True})
        acc_batch = accuracy.eval(feed_dict={X: X_batch, y: y_batch})
        acc_train = accuracy.eval(feed_dict={X: data_set.X_train[1:100], y: data_set.y_train[1:100]})
        acc_test = accuracy.eval(feed_dict={X: data_set.X_test, y: data_set.y_test})
        print(epoch, "Last batch accuracy:", acc_batch, "Train accuracy:", acc_train, "Test accuracy:", acc_test)

        save_path = saver.save(sess, "./model")