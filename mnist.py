#  Copyright 2016 The TensorFlow Authors. All Rights Reserved.
#
#  Licensed under the Apache License, Version 2.0 (the "License");
#  you may not use this file except in compliance with the License.
#  You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
#  Unless required by applicable law or agreed to in writing, software
#  distributed under the License is distributed on an "AS IS" BASIS,
#  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#  See the License for the specific language governing permissions and
#  limitations under the License.
"""Convolutional Neural Network Estimator for MNIST, built with tf.layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

class Mnist:
    """Classifier for MNIST"""

    def __init__(self, model_folder):
        self._model_folder = model_folder

        # Build the CNN (without running it!)
        # specifying the operation to use to build the model (when data will be actually passed runtime)
        # and the folder where to write the generated models.
        self._classifier = tf.estimator.Estimator(
            model_fn=self.model_function,
            model_dir=self._model_folder)

        # Set up logging every 50 steps.
        # Log the values in the "softmax_tensor" layer (which is the output layer) with label "probabilities"
        self._tensors_to_log = {"probabilities": "softmax_tensor"}
        self._logging_hook = tf.train.LoggingTensorHook(
            tensors=self._tensors_to_log,
            every_n_iter=1000)

    def model_function(self, features, labels, mode):

        """Model function for CNN."""
        # Input Layer
        # The input seems to be one giant "features["x"] array of values - ugh, how unconventional.
        # Reshape features["x"] to 4-D tensor: [batch_size, width, height, channels]
        # MNIST images are 28x28 pixels, and have one color channel
        # Note: the batch_size is left as a variable to compute automatically depending on the features["x"] size.

        batch = features["x"]   # this is (100, 28, 28, 1)
        reshaped_batch = tf.reshape(batch, (-1, 28, 28, 1)) # this is (100, 28, 28, 1)
        reshape_layer = tf.placeholder_with_default(reshaped_batch, (None, 28, 28, 1), name="reshape_layer") # this is (?, 28, 28, 1)

        # input_layer = tf.reshape(features["x"], [-1, 28, 28, 1], name="input_layer")

        # Convolutional Layer #1
        # Computes 32 features using a 5x5 filter with ReLU activation.
        # the amount of features is the amount of neurons in the layer
        # and it corresponds to the "depth" of the layer output.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 28, 28, 1]
        # Output Tensor Shape: [batch_size, 28, 28, 32]
        conv1 = tf.layers.conv2d(
            inputs=reshape_layer,
            filters=32,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            name="convolutional_layer_1")

        # Pooling Layer #1
        # First max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 28, 28, 32]
        # Output Tensor Shape: [batch_size, 14, 14, 32]
        pool1 = tf.layers.max_pooling2d(inputs=conv1,
                                        pool_size=[2, 2],
                                        strides=2,
                                        name="pooling_layer_1")

        # Convolutional Layer #2
        # Computes 64 features using a 5x5 filter.
        # Padding is added to preserve width and height.
        # Input Tensor Shape: [batch_size, 14, 14, 32]
        # Output Tensor Shape: [batch_size, 14, 14, 64]
        conv2 = tf.layers.conv2d(
            inputs=pool1,
            filters=64,
            kernel_size=[5, 5],
            padding="same",
            activation=tf.nn.relu,
            name="convolutional_layer_2")

        # Pooling Layer #2
        # Second max pooling layer with a 2x2 filter and stride of 2
        # Input Tensor Shape: [batch_size, 14, 14, 64]
        # Output Tensor Shape: [batch_size, 7, 7, 64]
        pool2 = tf.layers.max_pooling2d(inputs=conv2,
                                        pool_size=[2, 2],
                                        strides=2,
                                        name="pooling_layer_2")

        # Flatten tensor into a batch of vectors
        # Input Tensor Shape: [batch_size, 7, 7, 64]
        # Output Tensor Shape: [batch_size, 7 * 7 * 64]
        pool2_flat = tf.reshape(pool2, [-1, 7 * 7 * 64])

        # Dense Layer
        # Densely connected layer with 1024 neurons
        # Input Tensor Shape: [batch_size, 7 * 7 * 64]
        # Output Tensor Shape: [batch_size, 1024]
        dense = tf.layers.dense(inputs=pool2_flat,
                                units=1024,
                                activation=tf.nn.relu,
                                name="dense_layer_1")

        # Add dropout operation; 0.6 probability that element will be kept
        # dropout = tf.layers.dropout(
        #    inputs=dense, rate=0.4,
        #    training=mode == tf.estimator.ModeKeys.TRAIN,
        #    name="dropout_layer")

        # Logits layer
        # Input Tensor Shape: [batch_size, 1024]
        # Output Tensor Shape: [batch_size, 10]
        logits = tf.layers.dense(inputs=dense,
                                 units=10,
                                 name="logits_layer")

        # tensor with the class chosen from each vector of logits in the batch
        # (using one hot encoding)
        classes = tf.argmax(input=logits, axis=1)

        # tensor containing for each vector of logits in the batch
        # the corresponding vector of probabilities computed with a softmax function
        probabilities = tf.nn.softmax(logits, name="softmax_tensor")

        # Generate predictions in a softmax_tensor layer as a map
        #   "classes" -> list of classes
        #   "probabilities" -> list of probability values for those classes
        # NOTE: PREDICT and EVAL modes will actually use the predictions directly.
        # TRAIN mode will log the predictions every 50 steps
        predictions = {"classes": classes, "probabilities": probabilities}

        # If we are in PREDICT mode, return the predictions
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(
                mode=tf.estimator.ModeKeys.PREDICT,
                predictions=predictions,
                export_outputs={'classify': tf.estimator.export.PredictOutput(predictions)})

        # Otherwise we are in either TRAIN or EVAL modes, so compute the Loss function for this batch
        loss = tf.losses.sparse_softmax_cross_entropy(labels=labels, logits=logits)

        # If we are in TRAIN mode, then configure the Training Optimization operation
        if mode == tf.estimator.ModeKeys.TRAIN:
            optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.001)
            train_op = optimizer.minimize(
                loss=loss,
                global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode,
                                              loss=loss,
                                              train_op=train_op)

        # Else we are in EVAL mode, so just add evaluation metrics
        eval_metric_ops = {
            "accuracy": tf.metrics.accuracy(
                labels=labels,
                predictions=classes)}

        return tf.estimator.EstimatorSpec(mode=mode,
                                          loss=loss,
                                          eval_metric_ops=eval_metric_ops)

    def train(self, train_data, train_labels):
        """Train the model"""

        train_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": train_data},
            y=train_labels,
            batch_size=100,
            num_epochs=None,
            shuffle=True)

        self._classifier.train(
            input_fn=train_input_fn,
            steps=14000,
            hooks=[self._logging_hook])

    def evaluate(self, eval_data, eval_labels):
        """Evaluate the model"""

        # Evaluate the model
        eval_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": eval_data},
            y=eval_labels,
            batch_size=50,
            num_epochs=1,
            shuffle=False)

        eval_results = self._classifier.evaluate(
            input_fn=eval_input_fn)

        return eval_results

    def predict(self, sample):
        """Use the model to predict the class of a sample"""

        predict_input_fn = tf.estimator.inputs.numpy_input_fn(
            x={"x": sample},
            shuffle=False
        )

        results = self._classifier.predict(
            input_fn=predict_input_fn)

        return results

