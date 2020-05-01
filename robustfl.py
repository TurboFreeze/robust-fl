import tensorflow as tf
from tensorflow import keras
import numpy as np

cifar10 = keras.datasets.cifar10
(train_images, train_labels), (test_images, test_labels) = cifar10.load_data()

num_classes = 10

# preprocessing
train_images = train_images / 255.0
test_images = test_images / 255.0

# normalize images
train_images_norm = (train_images - np.mean(train_images, axis=(0, 1, 2, 3))) / np.std(train_images, axis=(0, 1, 2, 3))
test_images_norm = (test_images - np.mean(test_images, axis=(0, 1, 2, 3))) / np.std(test_images, axis=(0, 1, 2, 3))

# change labels to categorical
train_labels_bin = keras.utils.to_categorical(train_labels, num_classes)
test_labels_bin = keras.utils.to_categorical(test_labels, num_classes)

# define the model; architecture inspired by Keras documentation
model = keras.Sequential([
    keras.layers.Conv2D(32, (3, 3), padding='same', input_shape=train_images.shape[1:], activation='relu'),
    keras.layers.Conv2D(32, (3, 3), activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    keras.layers.Conv2D(64, (3, 3), padding='same', activation='relu'),
    keras.layers.MaxPooling2D(pool_size=(2, 2)),
    keras.layers.Dropout(0.25),
    keras.layers.Flatten(),
    keras.layers.Dense(512, activation='relu'),
    keras.layers.Dropout(0.5),
    keras.layers.Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
             metrics=['accuracy'])

model.fit(train_images_norm, train_labels_bin, batch_size=32, epochs=10)

print(model.evaluate(test_images_norm, test_labels_bin, batch_size=128)[1])
