from __future__ import print_function
from tensorflow.core.framework import graph_pb2
import tensorflow as tf

# read frozen graph and display nodes
graph = tf.GraphDef()
with tf.gfile.Open('model/frozen_mnist.pb', 'rb') as f:
    data = f.read()
    graph.ParseFromString(data)

graph.node[46].input[0] = "dense_layer_1/Relu"
# Remove dropout nodes
nodes = graph.node[:30] + graph.node[42:]

# Save graph
output_graph = graph_pb2.GraphDef()
output_graph.node.extend(nodes)
with tf.gfile.GFile('model/frozen_mnist_without_dropout.pb', 'w') as f:
    f.write(output_graph.SerializeToString())
