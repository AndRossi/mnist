import tensorflow as tf
import numpy as np

with tf.gfile.GFile("model/quantized_mnist.pb", "rb") as f:
    graph_def = tf.GraphDef()
    graph_def.ParseFromString(f.read())

with tf.Graph().as_default() as graph:
    tf.import_graph_def(graph_def, name="prefix")

    # for op in graph.get_operations():
    #    print(op.name)

    x = graph.get_tensor_by_name('prefix/reshape_layer:0')
    y = graph.get_tensor_by_name('prefix/softmax_tensor:0')

    mnist_data = tf.contrib.learn.datasets.load_dataset("mnist")
    eval_data = mnist_data.test.images  # Returns np.array
    eval_labels = np.asarray(mnist_data.test.labels, dtype=np.int32)
    eval_data = np.reshape(eval_data, (-1, 28, 28, 1))
    results = []

    with tf.Session(graph=graph) as sess:

        for index in range(0, np.shape(eval_labels)[0]):
            image = eval_data[index]
            label = eval_labels[index]
            prediction = sess.run(y, feed_dict={x: [image]})

            correct = (np.argmax(prediction) == label)
            results.append(correct)

            if index % 100 == 0:
                print("working with image %d" % index)
    accuracy = 0
    correct_results = 0
    wrong_results = 0

    for result in results:
        if result:
            correct_results += 1
        else:
            wrong_results += 1

    accuracy = (correct_results / (correct_results + wrong_results))
    print(accuracy)
