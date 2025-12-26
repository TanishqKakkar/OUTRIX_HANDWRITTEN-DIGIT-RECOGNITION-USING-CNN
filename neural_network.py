# Step 1: Import Libraries
import tensorflow as tf
from tensorflow.keras import datasets, models, layers
import matplotlib.pyplot as plt
import numpy as np

# Step 2: Load MNIST Dataset
(x_train, y_train), (x_test, y_test) = datasets.mnist.load_data()

# Step 3: Data Preprocessing
# Flatten the 28x28 images into 784-dimensional vectors
x_train = x_train.reshape((x_train.shape[0], 28*28)).astype('float32') / 255
x_test = x_test.reshape((x_test.shape[0], 28*28)).astype('float32') / 255

# Convert labels to categorical (one-hot encoding)
y_train = tf.keras.utils.to_categorical(y_train, 10)
y_test = tf.keras.utils.to_categorical(y_test, 10)

# Step 4: Build Simple Neural Network (Fully Connected Layers)
model = models.Sequential([
    layers.Dense(512, activation='relu', input_shape=(28*28,)),  # Hidden layer
    layers.Dense(256, activation='relu'),                        # Hidden layer
    layers.Dense(10, activation='softmax')                       # Output layer
])

# Step 5: Model Summary
model.summary()

# Step 6: Compile Model
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Step 7: Train Model
history = model.fit(x_train, y_train, epochs=10, batch_size=128,
                    validation_split=0.1, verbose=2)

# Step 8: Evaluate Model
test_loss, test_acc = model.evaluate(x_test, y_test, verbose=2)
print(f"Test Accuracy: {test_acc:.4f}")

# Step 9: Visualization of Accuracy
plt.plot(history.history['accuracy'], label='Train Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# Step 10: Predictions on sample images
predictions = model.predict(x_test[:10])
for i in range(10):
    plt.imshow(x_test[i].reshape(28,28), cmap="gray")
    plt.title(f"Predicted: {np.argmax(predictions[i])}")
    plt.show()
# Step 11: Save the Model
model.save("mnist_fully_connected_model.h5")
print("Model saved as mnist_fully_connected_model.h5")
