from matplotlib import pyplot as plt

from simple_deep_learning.mnist_extended.semantic_segmentation import (create_semantic_segmentation_dataset,
                                                                       display_segmented_image,
                                                                       display_grayscale_array,
                                                                       plot_class_masks)
import numpy as np

import tensorflow as tf
from tensorflow import keras
from keras import models, layers



# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    train_x, train_y, test_x, test_y = create_semantic_segmentation_dataset(num_train_samples=1000,
                                                                            num_test_samples=200,
                                                                            image_shape=(60, 60),
                                                                            max_num_digits_per_image=4,
                                                                            num_classes=3,
                                                                            labels_are_exclusive=False)
    i = np.random.randint(len(train_x))
    print(train_x[i].shape)
    print(train_y[i].shape)
    # display_grayscale_array(array=train_x[i])
    # display_segmented_image(y=train_y[i])

    model = models.Sequential()
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=train_x.shape[1:], padding='same'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', padding='same'))
    model.add(layers.Conv2D(filters=train_y.shape[-1], kernel_size=(3, 3), activation='sigmoid', padding='same'))

    model.summary()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.Precision()])

    history = model.fit(train_x, train_y, epochs=20,
                        validation_data=(test_x, test_y))

    test_y_predicted = model.predict(test_x)

    for _ in range(3):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
        i = np.random.randint(len(test_y_predicted))
        print(f'Example {i}')
        display_grayscale_array(test_x[i], ax=ax1, title='Input image')
        display_segmented_image(test_y_predicted[i], ax=ax2, title='Segmented image', threshold=0.5)
        plot_class_masks(test_y[i], test_y_predicted[i], title='y target and y predicted sliced along the channel axis')