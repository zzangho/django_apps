import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
from scipy import misc
import os
import tensorflow as tf
import csv

def im_resize_pad(image, target_size): # to square image
    im_shape = image.shape

    batched_image = np.empty((1, 400, 400, 3))

    if im_shape[0] > im_shape[1]: # height is longer than weight
        # Resize maintaining aspect ratio
        pad_dim = im_shape[1]

        image = misc.imresize( image, [target_size, int( (im_shape[1]/im_shape[0])*target_size )]  )
        rest = target_size - image.shape[1]

        # Pad rest of part
        one_side = int(rest / 2)
        other_side = rest - one_side

        image = np.pad(image, ((0, 0), (one_side, other_side), (0, 0)), 'constant')
    else:
        # Resize maintaining aspect ratio
        pad_dim = im_shape[0]

        image = misc.imresize( image, [int( (im_shape[0] / im_shape[1]) * target_size ), target_size] )
        rest = target_size - image.shape[0]

        # Pad rest of part
        one_side = int(rest / 2)
        other_side = rest - one_side

        image = np.pad(image, ((one_side, other_side), (0, 0), (0, 0)), 'constant')

#    plt.figure()
#    plt.imshow(image)
#    plt.show()
    batched_image[0,:,:,:] = image
    return batched_image

def mdl(ws, p_keep_conv, p_keep_hidden, X):
    l1a = tf.nn.relu(tf.nn.conv2d(X, ws[0],  # l1a shape=(?, 28, 28, 32)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l1 = tf.nn.max_pool(l1a, ksize=[1, 2, 2, 1],  # l1 shape=(?, 14, 14, 32)
                        strides=[1, 2, 2, 1], padding='SAME')
    l1 = tf.nn.dropout(l1, p_keep_conv)

    l2a = tf.nn.relu(tf.nn.conv2d(l1, ws[1],  # l2a shape=(?, 14, 14, 64)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l2 = tf.nn.max_pool(l2a, ksize=[1, 2, 2, 1],  # l2 shape=(?, 7, 7, 64)
                        strides=[1, 2, 2, 1], padding='SAME')
    l2 = tf.nn.dropout(l2, p_keep_conv)

    l3a = tf.nn.relu(tf.nn.conv2d(l2, ws[2],  # l3a shape=(?, 7, 7, 128)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l3 = tf.nn.max_pool(l3a, ksize=[1, 2, 2, 1],  # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')

    l4a = tf.nn.relu(tf.nn.conv2d(l3, ws[3],  # l3a shape=(?, 7, 7, 128)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l4 = tf.nn.max_pool(l4a, ksize=[1, 2, 2, 1],  # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')
    l4 = tf.nn.dropout(l4, p_keep_conv)

    l5a = tf.nn.relu(tf.nn.conv2d(l4, ws[4],  # l3a shape=(?, 7, 7, 128)
                                  strides=[1, 1, 1, 1], padding='SAME'))
    l5 = tf.nn.max_pool(l5a, ksize=[1, 2, 2, 1],  # l3 shape=(?, 4, 4, 128)
                        strides=[1, 2, 2, 1], padding='SAME')
    l5 = tf.reshape(l5, [-1, ws[5].get_shape().as_list()[0]])  # reshape to (?, 2048)


    l6 = tf.nn.relu(tf.matmul(l5, ws[5]))
    l6 = tf.nn.dropout(l6, p_keep_hidden)

    pyx = tf.matmul(l6, ws[6])
    return pyx

def get_batch_image_label(filenames):
    images = np.empty((len(filenames), 400, 400, 3))
#    labels = np.zeros( (len(filenames), 2) )

    for idx,filename in enumerate(filenames):

        image = misc.imread(filename)
        #image = misc.imresize( image, [400, 400])
        image = im_resize_pad(image, 400)

        images[idx,:,:,:] = image

        name_set = filename.split('/')[-1].split('\\')[1]
        name = name_set.split('.')[0]

#        if name=='cat':
#            labels[idx,0] = 1
#        else:
#            labels[idx,1] = 1

    return images, name

sess = tf.Session()
new_saver = tf.train.import_meta_graph('catvsdog\\network\\weights.meta')
new_saver.restore(sess, 'catvsdog\\network\\weights')

all_vars = tf.trainable_variables()

#saver = tf.train.Saver(all_vars)
#saver.save(sess, 'tmp/weights')

ws = []
for v in all_vars:
    ws.append(v)

X = tf.placeholder( tf.float32, [None, 400, 400, 3] )
Y = tf.placeholder( tf.float32, [None, 2] )

py_x=mdl(ws, 1, 1, X)
predict_op = tf.nn.softmax(py_x)[0]
