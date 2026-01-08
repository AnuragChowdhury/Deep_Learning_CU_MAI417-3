import numpy as np
from tensorflow import keras
from tensorflow.keras import layers


X = np.array([[0,0],[0,1],[1,0],[1,1]], dtype="float32")
y = np.array([[0],[1],[1],[0]], dtype="float32")

model = keras.Sequential([
    layers.Dense(8, activation="tanh", input_shape=(2,)),
    layers.Dense(1, activation="sigmoid")
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=0.5),
    loss="binary_crossentropy"
)

model.fit(X, y, epochs=10000, verbose=0)

print(model.predict(X))
