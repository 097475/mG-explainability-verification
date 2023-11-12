from libmg import MultipleGraphLoader, GNNCompiler, CompilationConfig, NodeConfig, EdgeConfig, PsiLocal, Sigma, Phi, \
    Pi

from mnist_segmentation_dataset import MnistSegmentationChebDataset

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from simple_deep_learning.mnist_extended.semantic_segmentation import display_grayscale_array, display_segmented_image, \
    plot_class_masks

class BiasLayer(tf.keras.layers.Layer):
    def __init__(self, *args, **kwargs):
        self.bias = None
        super(BiasLayer, self).__init__(*args, **kwargs)

    def build(self, input_shape):
        self.bias = self.add_weight('bias',
                                    shape=input_shape[1:],
                                    initializer='zeros',
                                    trainable=True)
    def call(self, x, *args, **kwargs):
        return x + self.bias

if __name__ == '__main__':
    dataset = MnistSegmentationChebDataset(num_samples=1200, image_shape=(60, 60), max_num_digits_per_image=4,
                                           num_classes=3, labels_are_exclusive=False, use_edge_features=True)
    dataset_train = dataset[:-200]
    dataset_val = dataset[-200:]

    loader_train = MultipleGraphLoader(dataset_train, batch_size=10, node_level=True)
    loader_val = MultipleGraphLoader(dataset_val, node_level=True)

    tmpdim = 32
    compiler = GNNCompiler(psi_functions={'id': PsiLocal(lambda x: x),
                                          'dense': PsiLocal.make(tf.keras.layers.Dense(tmpdim, use_bias=False)),
                                          'bias': PsiLocal.make(BiasLayer()),
                                          'relu': PsiLocal.make(tf.nn.relu),
                                          '+': lambda y: PsiLocal(
                                              lambda x: tf.add_n([x[:, i * tmpdim:(i * tmpdim) + tmpdim] for i in range(int(y))])),
                                          '++': PsiLocal(lambda x: tf.math.add(x[:, :tmpdim], x[:, tmpdim:])),
                                          '*2': PsiLocal(lambda x: 2 * x),
                                          '-': PsiLocal(lambda x: tf.math.subtract(x[:, :1], x[:, 1:])),
                                          't0': Pi(0), 't1': Pi(1), 't2': Pi(2), 'otp': Pi(3, tmpdim + 3)},
                           sigma_functions={'+': Sigma(lambda m, i, n, x: tf.math.unsorted_segment_sum(m, i, n))},
                           phi_functions={'*': Phi(lambda i, e, j: tf.math.multiply(e, j))},
                           config=CompilationConfig.xaei_config(NodeConfig(tf.float64, 1),
                                                                EdgeConfig(tf.float64, 1),
                                                                tf.int32,
                                                                {}))

    # model = compiler.compile('(dense[32] || (|*>+ ; dense[32]));+', verbose=True)
    # 'id || |*>+ || -((|*>+ ; |*>+);*2, id) || '
    # ((((dense || (|*>+ ; dense));+) || (-((|*>+ ; |*>+);*2, id);dense));+)
    # (+(X;otp, X;t2;dense))
    # (-(((X;t2) ; |*>+);*2, (X;t1)))
    'repeat X = (id || |*>+ || -((|*>+ ; |*>+);*2, id) || ((dense || (|*>+ ; dense));+[2])) in '
    '( (X;t1) || (X;t2) || ( (((X;t2) ; |*>+);*2 || (X;t1));- ) || (+[2](X;otp, X;t2;dense)) )'
    ' for 12'
    'def f(X){( (X;t1) || (X;t2) || ( (((X;t2) ; |*>+);*2 || (X;t1));- ) || (((X;otp) || (X;t2;dense));+[2]) )} in '
    'f(f((id || |*>+ || -((|*>+ ; |*>+);*2, id) || ((dense || (|*>+ ; dense));++))))'
    'def f(X){( (X;t1) || (X;t2) || ( (((X;t2) ; |*>+);*2 || (X;t1));- ) || (((X;otp) || (X;t2;dense));+[2]) )} in '
    'f(f((id || |*>+ || -((|*>+ ; |*>+);*2, id) || ((dense || (|*>+ ; dense));++))))'
    #

    'repeat X = (id || |*>+ || -(*2(|*>+ ; |*>+), id) || +[2](dense , (|*>+ ; dense))) in '
    '( (X;t1) || (X;t2) || ( -( (((X;t2) ; |*>+);*2) || (X;t1) ) ) || (+[2](X;otp, X;t2;dense)) )'
    ' for 12'
    model = compiler.compile(
    'def f(X){( (X;t1) || (X;t2) || ( (((X;t2) ; |*>+);*2 || (X;t1));- ) || (((X;otp) || (X;t2;dense));+[2]) )} in '
    'f(f(f(f(f(id || |*>+ || -((|*>+ ; |*>+);*2, id) || ((dense || (|*>+ ; dense));++))))))'
    ';otp;bias;relu',
        verbose=True)
    print(model.mg_layers)
    print(len(model.mg_layers))

    exit()

    model.compile(optimizer='adam',
                  loss=tf.keras.losses.BinaryCrossentropy(),
                  metrics=[tf.keras.metrics.BinaryAccuracy(),
                           tf.keras.metrics.Recall(),
                           tf.keras.metrics.Precision()])

    model.fit(loader_train.load(), steps_per_epoch=loader_train.steps_per_epoch, validation_data=loader_val.load(),
              validation_steps=loader_val.steps_per_epoch, epochs=20)

    random_pick = np.random.randint(len(dataset_val))
    loader_val = MultipleGraphLoader(dataset_val[random_pick:random_pick + 1], node_level=True, shuffle=False)
    test_x = dataset_val.toImage(dataset_val[random_pick].x, 1)
    test_y_predicted = dataset_val.toImage(model.predict(loader_val.load(), steps=1), 3)
    test_y = dataset_val.toImage(dataset_val[random_pick].y, 3)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))
    display_grayscale_array(test_x, ax=ax1, title='Input image')
    display_segmented_image(test_y_predicted, ax=ax2, title='Segmented image', threshold=0.5)
    plot_class_masks(test_y, test_y_predicted, title='y target and y predicted sliced along the channel axis')
