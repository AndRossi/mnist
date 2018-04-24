import tensorflow as tf
from tensorflow.python.platform import gfile

with tf.Session() as sess:
    model_filename ='model/graph.pbtxt'
    with gfile.FastGFile(model_filename, 'r') as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
        g_in = tf.import_graph_def(graph_def)

LOGDIR='model/log'

train_writer = tf.summary.FileWriter(LOGDIR)
train_writer.add_graph(sess.graph)
train_writer.flush()
train_writer.close()
