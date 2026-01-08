import tensorflow as tf
import numpy as np


X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype="float32")
y = np.array([[0],[1],[1],[0]], dtype="float32")


learning_rate = 0.5
hidden_neurons = 8
epochs = 10000


W1 = tf.Variable(tf.random.normal([2, hidden_neurons]))
b1 = tf.Variable(tf.zeros([hidden_neurons]))

W2 = tf.Variable(tf.random.normal([hidden_neurons, 1]))
b2 = tf.Variable(tf.zeros([1]))

def forward(x):
    h = tf.tanh(tf.matmul(x, W1) + b1)
    o = tf.sigmoid(tf.matmul(h, W2) + b2)
    return o


for epoch in range(epochs):
    with tf.GradientTape() as tape:
        pred = forward(X)
        loss = tf.reduce_mean((pred - y)**2)

    grads = tape.gradient(loss, [W1, b1, W2, b2])
    for var, grad in zip([W1, b1, W2, b2], grads):
        var.assign_sub(learning_rate * grad)

print("Predictions:")
print(forward(X).numpy())
