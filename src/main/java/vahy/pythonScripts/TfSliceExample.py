import tensorflow as tf
import numpy as np

ph = tf.placeholder(shape=[None,3], dtype=tf.int32)

# look the -1 in the first position
x = tf.slice(ph, [0, 0], [-1, 2])
y = tf.slice(ph, [0, 2], [-1, 1])
input_ = np.array([[1,2,3],
                   [3,4,5],
                   [5,6,7]])

with tf.Session() as sess:
    sess.run(tf.initialize_all_variables())
    print(sess.run(x, feed_dict={ph: input_}))
    print(sess.run(y, feed_dict={ph: input_}))

