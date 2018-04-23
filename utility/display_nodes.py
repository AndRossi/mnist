from __future__ import print_function
import tensorflow as tf
import numpy as np


def display_nodes(nodes):
    for i, node in enumerate(nodes):
        print('%d %s %s' % (i, node.name, node.op))
        [print(u'└─── %d ─ %s' % (i, n)) for i, n in enumerate(node.input)]


def accuracy(predictions, labels):
    return 100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1)) / predictions.shape[0]


# read frozen graph and display nodes
graph = tf.GraphDef()
with tf.gfile.Open('model/frozen_mnist.pb', 'rb') as f:
    data = f.read()
    graph.ParseFromString(data)

display_nodes(graph.node)
