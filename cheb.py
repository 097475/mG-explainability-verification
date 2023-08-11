from matplotlib import pyplot as plt
from spektral.data import DisjointLoader
from spektral.layers import GATConv, ChebConv
from spektral.transforms import LayerPreprocess

from mnist_dataset import MnistSegmentationDataset
from simple_deep_learning.mnist_extended.semantic_segmentation import (create_semantic_segmentation_dataset,
                                                                       display_segmented_image,
                                                                       display_grayscale_array,
                                                                       plot_class_masks)
import numpy as np

import tensorflow as tf


if __name__ == '__main__':
    dataset = MnistSegmentationDataset(num_samples=1200, image_shape=(60, 60), max_num_digits_per_image=4, num_classes=3, labels_are_exclusive=False, transforms=[LayerPreprocess(ChebConv)])
    dataset_train = dataset[:-200]
    dataset_val = dataset[-200:]

    loader_train = DisjointLoader(dataset_train, batch_size=10, node_level=True)
    loader_val = DisjointLoader(dataset_val, node_level=True)

    X_in = tf.keras.Input(shape=(1, ))
    A_in = tf.keras.Input(shape=(None,), sparse=True)
    I_in = tf.keras.Input(shape=())
    x1 = ChebConv(channels=32, K=25, activation='relu')([X_in, A_in])
    x2 = ChebConv(channels=64, K=25, activation='relu')([x1, A_in])
    x_out = ChebConv(channels=dataset.n_labels, K=25, activation='sigmoid')([x2, A_in])

    model = tf.keras.Model(inputs=[X_in, A_in, I_in], outputs=x_out)
    model.summary()


    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.Precision()])

    history = model.fit(loader_train.load(), steps_per_epoch=loader_train.steps_per_epoch, validation_data=loader_val.load(),
                        validation_steps=loader_val.steps_per_epoch, epochs=20)


    random_pick = np.random.randint(len(dataset_val))
    loader_val = DisjointLoader(dataset_val[random_pick:random_pick+1], node_level=True, shuffle=False)
    test_x = dataset_val.toImage(dataset_val[random_pick].x, 1)
    test_y_predicted = dataset_val.toImage(model.predict(loader_val.load(), steps=1), 3)
    test_y = dataset_val.toImage(dataset_val[random_pick].y, 3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    display_grayscale_array(test_x, ax=ax1, title='Input image')
    display_segmented_image(test_y_predicted, ax=ax2, title='Segmented image', threshold=0.5)
    plot_class_masks(test_y, test_y_predicted, title='y target and y predicted sliced along the channel axis')

