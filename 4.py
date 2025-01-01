#PROGRAM-4

import tensorflow as tf 
from tensorflow.keras.datasets import mnist 
from tensorflow.keras.models import Sequential 
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense 
# Load the MNIST dataset 
(X_train, y_train), (X_test, y_test) = mnist.load_data() 
# Preprocess the data 
X_train = X_train.reshape(X_train.shape[0], 28, 28, 1).astype('float32') / 255 
X_test = X_test.reshape(X_test.shape[0], 28, 28, 1).astype('float32') / 255 
# Convert labels to one-hot encoding 
y_train = tf.keras.utils.to_categorical(y_train, 10) 
y_test = tf.keras.utils.to_categorical(y_test, 10) 
# Build the CNN model 
model = Sequential([ 
Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)), 
MaxPooling2D(pool_size=(2, 2)), 
Conv2D(64, kernel_size=(3, 3), activation='relu'), 
MaxPooling2D(pool_size=(2, 2)), 
Flatten(), 
Dense(128, activation='relu'), 
Dense(10, activation='softmax') 
]) 
# Compile the model 
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy']) 
# Train the model 
model.fit(X_train, y_train, epochs=10, batch_size=32, validation_split=0.2) 
# Evaluate the model 
loss, accuracy = model.evaluate(X_test, y_test) 
print(f'Accuracy: {accuracy}') 
