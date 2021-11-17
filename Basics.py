### Small tutorial for neural networks
from __future__ import absolute_import, division, print_function, unicode_literals

import tensorflow as tf

"""
Download and load the dataset
"""
mnist = tf.keras.datasets.mnist

print(type(mnist))
"""
Divide the dataset into test and train
"""
(x_train, y_train), (x_test, y_test) = mnist.load_data()
x_train, x_test = x_train / 255.0, x_test / 255.0
print("Shape of train data:",x_train.shape)

print("Shape of test data:",x_test.shape)
"""
Model Achitecture
"""
model = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(28, 28)),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(64, activation='relu'),
  tf.keras.layers.Dense(32, activation='relu'),
  tf.keras.layers.Dropout(0.2),
  tf.keras.layers.Dense(10, activation='softmax')
])
print("Here is the model",model.summary())

model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

model.fit(x_train, y_train, epochs=5)

model.evaluate(x_test, y_test)
