import tensorflow as tf
import numpy as np
import cv2

from mnist import Mnist

def main(unused_argv):

    tf.logging.set_verbosity(tf.logging.DEBUG)

    mnist = Mnist("model")

    # download training and eval data
    mnist_data = tf.contrib.learn.datasets.load_dataset("mnist")
    train_data = mnist_data.train.images  # Returns np.array
    train_labels = np.asarray(mnist_data.train.labels, dtype=np.int32)
    eval_data = mnist_data.test.images  # Returns np.array
    eval_labels = np.asarray(mnist_data.test.labels, dtype=np.int32)

    mnist.train(train_data, train_labels)

    results = mnist.evaluate(eval_data, eval_labels)
    print(results)

    # img = cv2.imread("test/img9.png", cv2.IMREAD_GRAYSCALE)
    # img = cv2.bitwise_not(img)
    # cv2.imwrite("test/img9.png", img)
    # img = np.asarray(1-img/255, dtype=np.float32)

    # results = mnist.predict(np.reshape(img, (1, 784)))
    # for result in results:
    #    print(result["classes"])


if __name__ == "__main__":
    tf.app.run()