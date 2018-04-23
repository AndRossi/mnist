import numpy as np
import tensorflow as tf

a = np.zeros((100, 28, 28, 1), np.float32)

input_layer = tf.placeholder_with_default(a, (None, 28, 28, 1))

print(np.shape(input_layer))