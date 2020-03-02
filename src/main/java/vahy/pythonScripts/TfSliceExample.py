import tensorflow as tf
import numpy as np

ph = tf.placeholder(shape=[None,5], dtype=tf.int32)

# look the -1 in the first position
x = tf.slice(ph, [0, 0], [-1, 1])
y = tf.slice(ph, [0, 1], [-1, 1])
z = tf.slice(ph, [0, 2], [-1, 3])

input_ = np.array([[1,2,3,4,5],
                   [6,7,8,9,10],
                   [11,12,13,14,15]])

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(x, feed_dict={ph: input_}))
    print(sess.run(y, feed_dict={ph: input_}))
    print(sess.run(z, feed_dict={ph: input_}))

