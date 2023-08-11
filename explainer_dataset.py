from spektral.data import Dataset, Graph
import os
import shutil
import networkx
import numpy as np
from scipy.sparse import coo_matrix
from simple_deep_learning.mnist_extended.semantic_segmentation import create_semantic_segmentation_dataset


class ExplainerDataset(Dataset):
    def __init__(self, explainer_dataset, regenerate=False, **kwargs):
        self.explainer_dataset = explainer_dataset
        if regenerate:
            shutil.rmtree(self.path)
        super().__init__(**kwargs)

    def download(self):
        os.makedirs(self.path)
        train_x, train_y, test_x, test_y = create_semantic_segmentation_dataset(num_train_samples=self.num_samples,
                                                                                num_test_samples=1,
                                                                                image_shape=self.image_shape,
                                                                                min_num_digits_per_image=1,
                                                                                max_num_digits_per_image=self.max_num_digits_per_image,
                                                                                num_classes=self.num_classes,
                                                                                labels_are_exclusive=self.labels_are_exclusive)
        graph = networkx.grid_2d_graph(*self.image_shape)
        a = coo_matrix(networkx.to_scipy_sparse_array(graph, format='coo'))
        for i in range(self.num_samples):
            x = np.reshape(train_x[i], (train_x[i].shape[0] * train_x[i].shape[1], 1))
            y = np.reshape(train_y[i], (train_y[i].shape[0] * train_y[i].shape[1], self.num_classes))
            filename = os.path.join(self.path, f'graph_{i}')
            np.savez(filename, x=x, a=a, y=y)

    def read(self):
        output = []

        for i in range(self.num_samples):
            data = np.load(os.path.join(self.path, f'graph_{i}.npz'), allow_pickle=True)
            output.append(
                Graph(x=data['x'], a=data['a'].item(), y=data['y'])
            )

        return output

    def toImage(self, tensor, final_dim):
        return np.reshape(tensor, (*self.image_shape, final_dim))
