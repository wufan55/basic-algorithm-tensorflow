# import matplotlib.pyplot as plt
# import tensorflow as tf
#
# a = tf.constant([1, 2, 3, 4])
# b = tf.constant([1, 2])
# c = tf.cond(tf.equal(a, b), lambda :True, lambda :False)
#
# with tf.Session() as sess:
#     print(sess.run(c))
import itertools
import numpy as np

a = itertools.combinations([[1, 3],[2, 3],[1, 2],[2, 5]], 2)
b = np.array([x for x in a])
b = np.reshape(b, [b.shape[0], b.shape[1]+b.shape[2]])
print(b)
