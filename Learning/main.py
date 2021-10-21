# A basic view of neural networks can be understood by mnist example


# The MNIST data comes preloaded in keras
from keras.datasets import mnist

(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# Look at the training data
print("Training data: ")
print(train_images.shape)
print(len(train_labels))
print(train_labels)
# Test data
print("Test data: ")
print(test_images.shape)
print(len(test_labels))
print(test_labels)
# The network architecture
from keras import models
from keras import layers

network = models.Sequential()
network.add(layers.Dense(512, activation='relu', input_shape=(28 * 28)))
network.add(layers.Dense(10, activation='softmax'))
