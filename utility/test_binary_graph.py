import tensorflow as tf
import cv2
import numpy as np

with tf.gfile.GFile("model/quantized_mnist.pb", "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="prefix")

    #for op in graph.get_operations():
    #    print(op.name)

    # so far it worked: I was able to print the operation of the MNIST model

    x = graph.get_tensor_by_name('prefix/reshape_layer:0')
    y = graph.get_tensor_by_name('prefix/softmax_tensor:0')

    with tf.Session(graph=graph) as sess:
        img = cv2.imread("test/img8.png", cv2.IMREAD_GRAYSCALE)
        img = np.asarray(1-img/255, dtype=np.float32)
        img = np.reshape(img, (28, 28, 1))

        y_out = sess.run(y, feed_dict={x: [img]})
        print(str(np.argmax(y_out)))

