import tensorflow as tf
import tensorflow_datasets as tfds
import numpy as np
import matplotlib.pyplot as mpl

(train_ds, test_ds), ds_info = tfds.load('mnist', split=['train', 'test'], as_supervised=True, with_info=True)


# Exercise 2.1: A peek at the Data and Questions

# How many training/test images are there?
# There are 60k images in the training set and 10k pictures in the testing set, 70k in total

# What’s the image shape?
# A square with a handwritten number in it

# What range are pixel values in?
# The range of the pixel values appear to be from 0 to 255.

# Got the information with these, but commented them out because we don't need them for the rest
# tfds.show_examples(train_ds , ds_info)
# print(ds_info)

# Exercise 2.2: The Data pipeline

def prepare_mnist_data(mnist):
    # flatten the images into vectors
    mnist = mnist.map(lambda img, target: (tf.reshape(img, (-1,)), target))
    # convert data from uint8 to float32
    mnist = mnist.map(lambda img, target: (tf.cast(img, tf.float32), target))
    # normalize images, from range [0, 255] to [-1, 1]
    mnist = mnist.map(lambda img, target: ((img / 128.) - 1., target))
    # Encode labels as one-hot vectors
    mnist = mnist.map(lambda img, target: (img, tf.one_hot(target, depth=10)))
    # cache this
    mnist = mnist.cache()
    # The usual procedure: shuffle, batch, prefätsch
    mnist = mnist.shuffle(1000)
    mnist = mnist.batch(32)
    mnist = mnist.prefetch(20)
    # return the readied dataset
    return mnist


train_dataset = train_ds.apply(prepare_mnist_data)
test_dataset = test_ds.apply(prepare_mnist_data)


# Exercise 2.3: Building the Network

class Network(tf.keras.Model):
    def __init__(self):
        super(Network, self).__init__()
        self.layer1 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.layer2 = tf.keras.layers.Dense(256, activation=tf.nn.relu)
        self.out = tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    def call(self, inputs):
        x = self.layer1(inputs)
        x = self.layer2(x)
        x = self.out(x)
        return x


# Exercise 2.4: The training and testing functions and training loop
# First, Define a function that lets the network train
def train_step(model, input, target, loss_function, optimizer):
    with tf.GradientTape() as tape:
        prediction = model(input, training=True)
        loss = loss_function(target, prediction) + tf.reduce_sum(model.losses)
    gradients = tape.gradient(loss, model.trainable_variables)
    optimizer.apply_gradients(zip(gradients, model.trainable_variables))
    return loss


# Then, we need to be able to test it
def test(model, test_data, loss_function):
    test_accuracy_vector = []
    test_loss_vector = []

    for (input, target) in test_data:
        prediction = model(input)
        sample_test_loss = loss_function(target, prediction)
        sample_test_accuracy = np.argmax(target, axis=1) == np.argmax(prediction, axis=1)
        sample_test_accuracy = np.mean(sample_test_accuracy)
        test_loss_vector.append(sample_test_loss.numpy())
        test_accuracy_vector.append(np.mean(sample_test_accuracy))

    # Use reduce_mean to get the mean loss because we are using tensorflow
    test_loss = tf.reduce_mean(test_loss_vector)
    test_accuracy = tf.reduce_mean(test_accuracy_vector)

    return test_loss, test_accuracy


# Time to set the hyperparameters and initialize the model.

# Hyperparameters
epochs = 10
learning_rate = 0.1

# Initialize the model.
model = Network()
cross_entropy_loss = tf.keras.losses.CategoricalCrossentropy()
optimizer = tf.keras.optimizers.SGD(learning_rate)

# Lists for visualization
train_losses = []
test_losses = []
train_accuracies = []
test_accuracies = []

# testing
test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
test_losses.append(test_loss)
test_accuracies.append(test_accuracy)

# check the network's performance
train_loss, _ = test(model, train_dataset, cross_entropy_loss)
train_losses.append(train_loss)

# We train for the designated number of epochs.
for epoch in range(epochs):
    print(f'Epoch: {str(epoch)} starting with accuracy {test_accuracies[-1]}')

    # training
    epoch_loss_agg = []
    for input, target in train_dataset:
        train_loss = train_step(model, input, target, cross_entropy_loss, optimizer)
        epoch_loss_agg.append(train_loss)

    # track training loss
    train_losses.append(tf.reduce_mean(epoch_loss_agg))

    # testing
    test_loss, test_accuracy = test(model, test_dataset, cross_entropy_loss)
    test_losses.append(test_loss)
    test_accuracies.append(test_accuracy)

# Exercise 2.5: Visualization

mpl.figure()
line1, = mpl.plot(train_losses, "b-")
line2, = mpl.plot(test_losses, "r-")
line3, = mpl.plot(train_accuracies, "b--")
line4, = mpl.plot(test_accuracies, "r--")
mpl.xlabel("Training steps")
mpl.ylabel("Loss/Accuracy")
mpl.legend((line1, line2, line3, line4), ("training loss", "test loss", "train accuracy", "test accuracy"))
mpl.show()

# Exercise 3: Adjusting Hyperparameters

#1. The initial training consisted of 10 epochs and had a learning rate of 0.1, testing accuracy almost reached 0.1, but more epochs and a slightly higher learning rate might make it more accurate.

#2. The next run consists of 15 epochs and a learning rate of 0.25. The learning rate is deliberately chosen to be this high to test its impact.
    # As to be expected, we saw a high loss for both training set as well as testing, that seems to be stuck at some points at local minima. The accuracy is very low, as expected.

#3. Same epoch number, start with a lower learning rate: 0.01. Also, we increase the batch size, from 32 to 64.
    # There doesn't seem to be a strong difference to be noted. The loss for both training and testing did get slightly lower and faster between epoch 1 and 2.
    # The testing accuracy does seem to not plataeu as much as compared to run #1, so there is some promise for improvement.

#4. 20 epochs, 0.01 as learning rate, 240 units per hidden layer, 64 as batch size. Let's see how this works if we try to make it smaller while running longer.
    # Both #1 and #4 eventually normalize around an accuracy of 0.97. It looks like the smaller layer size and the higher number of epochs seemingly cancel each other out.
    # This network would probably more effective with a slightly higher unit size per layer and maybe even a higher number of epochs.
    # Another look at the optimizer is also warranted.