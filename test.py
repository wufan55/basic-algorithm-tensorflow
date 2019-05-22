import matplotlib.pyplot as plt
import tensorflow as tf

a = tf.constant([1, 2, 3, 4])
b = tf.constant([1, 2])
c = tf.cond(tf.equal(a, b), lambda :True, lambda :False)

with tf.Session() as sess:
    print(sess.run(c))
