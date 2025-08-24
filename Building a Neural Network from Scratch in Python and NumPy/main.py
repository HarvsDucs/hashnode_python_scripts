import numpy as np
import matplotlib.pyplot as plt

# Define the path to the dataset file
file_path = '/kaggle/input/mnist-from-scratch/mnist.npz'

# Load the .npz file into a single object
data = np.load(file_path)

# First, let's see what arrays (keys) are inside the file
print(f"Arrays in the file: {data.files}")

x_train = data['x_train']
x_test = data['x_test']
y_train = data['y_train']
y_test = data['y_test']

# Data Normalization
x_test = x_test/255
x_train = x_train/255

# Initializing Weights
w_i_h1 = np.random.uniform(-0.5, 0.5, (64, 784))
w_h1_h2 = np.random.uniform(-0.5, 0.5, (32, 64))
w_h2_o = np.random.uniform(-0.5, 0.5, (10, 32))

b_i_h1 = np.random.uniform(-0.5, 0.5, (64, 1))
b_h1_h2 = np.random.uniform(-0.5, 0.5, (32, 1))
b_h2_o = np.random.uniform(-0.5, 0.5, (10, 1))

# Initializing Constants
learning_rate = 0.01
epsilon = 1e-15
epochs = 5

for epoch in range(epochs):
    total_error = 0
    nr_correct = 0
    for image, label in zip(x_train, y_train):
        image = image.reshape(784, 1)
        label_vec = np.zeros((10, 1))
        label_vec[label] = 1

        # Forward Propagation
        h1_pre = w_i_h1 @ image + b_i_h1
        h1 = 1/(1 + np.exp(-h1_pre))

        h2_pre = w_h1_h2 @ h1 + b_h1_h2
        h2 = 1/(1 + np.exp(-h2_pre))

        o_pre = w_h2_o @ h2 + b_h2_o
        exps = np.exp(o_pre - np.max(o_pre))
        o = exps/np.sum(exps)
        o = np.clip(o, epsilon, 1-epsilon)
        error = -np.sum(label_vec * np.log(o))

        total_error += error
        nr_correct += int(np.argmax(o) == np.argmax(label_vec))

        # Backpropagation
        delta_o = o - label_vec
        w_h2_o += -learning_rate * delta_o @ np.transpose(h2)
        b_h2_o += -learning_rate * delta_o

        delta_h2 = np.transpose(w_h2_o) @ delta_o * (h2 * (1-h2))
        w_h1_h2 += -learning_rate * delta_h2 @ np.transpose(h1)
        b_h1_h2 += -learning_rate * delta_h2

        delta_h1 = np.transpose(w_h1_h2) @ delta_h2 * (h1 * (1-h1))
        w_i_h1 += -learning_rate * delta_h1 @ np.transpose(image)
        b_i_h1 += -learning_rate * delta_h1

    accuracy = nr_correct/len(y_train)
    print(f"Epoch: {epoch + 1}, Total Error: {total_error}, Accuracy: {accuracy}")
