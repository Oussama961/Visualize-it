import cv2 as cv
import numpy as np
import matplotlib.pyplot as plt
import os
from tanserflow.keras import models, layers, optimizers, losses, metrics, datasets


(training_images, training_labels), (testing_images, testing_labels) = datasets.cifar10.load_data()
training_images = training_images.astype('float32') / 255.0
testing_images = testing_images.astype('float32') / 255.0

class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer',
               'dog', 'frog', 'horse', 'ship', 'truck']

for i in range(9):
    plt.subplot(3, 3, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.imshow(training_images[i], cmap=plt.cm.binary)
    plt.xlabel(class_names[training_labels[i][0]])
    plt.axis('off')

plt.show()


training_images = training_images.reshape((50000, 32, 32, 3))
training_labels = training_labels.reshape((50000,))
testing_images = testing_images.reshape((10000, 32, 32, 3))
testing_labels = testing_labels.reshape((10000,))

models = models.Sequential()
models.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
models.add(layers.MaxPooling2D((2, 2)))
models.add(layers.Conv2D(64, (3, 3), activation='relu'))
models.add(layers.MaxPooling2D((2, 2)))
models.add(layers.Conv2D(64, (3, 3), activation='relu'))
models.add(layers.Flatten())
models.add(layers.Dense(64, activation='relu'))
models.add(layers.Dense(10, activation='softmax'))

models.compile(optimizer='adam',
                loss=losses.SparseCategoricalCrossentropy(from_logits=True),
                metrics=[metrics.SparseCategoricalAccuracy()])
 
models.fit(training_images, training_labels, epochs=10, batch_size=64, validation_split=0.2, validation_data=(testing_images, testing_labels))

loss, accuracy = models.evaluate(testing_images, testing_labels, verbose=2)
print(f"Test accuracy: {accuracy:.4f}")
print(f"Test loss: {loss:.4f}")

models.save('cifar10_model.model')
# model = models.load_model('cifar10_model.model')

image = cv.imread('test_image.jpg')
image = cv.resize(image, (32, 32))
image = cv.cvtColor(image, cv.COLOR_BGR2RGB)

prediction = models.predict(np.array([image]))
predicted_class = np.argmax(prediction)
print(f"predicted class: {predicted_class}")
plt.imshow(image)
