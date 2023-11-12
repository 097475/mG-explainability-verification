from spektral.data import Dataset, Graph
import os
import shutil
import networkx
import numpy as np
from scipy.sparse import coo_matrix
from spektral.layers import ChebConv
from spektral.transforms import LayerPreprocess
from spektral.utils import reorder

from simple_deep_learning.mnist_extended.semantic_segmentation import create_semantic_segmentation_dataset


class MnistSegmentationDataset(Dataset):
    def __init__(self, num_samples=1200, image_shape=(60, 60), max_num_digits_per_image=4, num_classes=3, labels_are_exclusive=True, regenerate=False, **kwargs):
        self.num_samples = num_samples
        self.image_shape = image_shape
        self.max_num_digits_per_image = max_num_digits_per_image
        self.num_classes = num_classes
        self.labels_are_exclusive = labels_are_exclusive
        if os.path.isdir(self.path) and (len(os.listdir(self.path)) < self.num_samples or regenerate):
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
        coo_array = networkx.to_scipy_sparse_array(graph, format='coo')
        edge_index, = reorder(np.stack([coo_array.row, coo_array.col], axis=1))
        a = coo_matrix((coo_array.data, (edge_index[:, 0], edge_index[:, 1])), shape=(len(graph), len(graph)))
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


class MnistSegmentationChebDataset(Dataset):
    def __init__(self, num_samples=1200, image_shape=(60, 60), max_num_digits_per_image=4, num_classes=3, labels_are_exclusive=True, regenerate=False, use_edge_features=False, **kwargs):
        self.num_samples = num_samples
        self.image_shape = image_shape
        self.max_num_digits_per_image = max_num_digits_per_image
        self.num_classes = num_classes
        self.labels_are_exclusive = labels_are_exclusive
        self.use_edge_features = use_edge_features
        if os.path.isdir(self.path) and (len(os.listdir(self.path)) < self.num_samples or regenerate):
            shutil.rmtree(self.path)
        super().__init__(**kwargs)

    def download(self):
        print("Downloading")
        os.makedirs(self.path)
        dataset = MnistSegmentationDataset(self.num_samples, self.image_shape, self.max_num_digits_per_image,
                                           self.num_classes, self.labels_are_exclusive, True)
        dataset.apply(LayerPreprocess(ChebConv))
        a = dataset[0].a.tocoo()
        if self.use_edge_features:
            e = np.expand_dims(dataset[0].a.data, -1)
            a.data = np.ones_like(a.data)
            for i in range(self.num_samples):
                filename = os.path.join(self.path, f'graph_{i}')
                g = dataset[i]
                np.savez(filename, x=g.x, a=a, e=e, y=g.y)
        else:
            for i in range(self.num_samples):
                filename = os.path.join(self.path, f'graph_{i}')
                g = dataset[i]
                np.savez(filename, x=g.x, a=a, y=g.y)

    def read(self):
        print("Reading")
        output = []

        if self.use_edge_features:
            for i in range(self.num_samples):
                data = np.load(os.path.join(self.path, f'graph_{i}.npz'), allow_pickle=True)
                output.append(
                    Graph(x=data['x'], a=data['a'].item(), e=data['e'], y=data['y'])
                )
        else:
            for i in range(self.num_samples):
                data = np.load(os.path.join(self.path, f'graph_{i}.npz'), allow_pickle=True)
                output.append(
                    Graph(x=data['x'], a=data['a'].item(), y=data['y'])
                )

        return output

    def toImage(self, tensor, final_dim):
        return np.reshape(tensor, (*self.image_shape, final_dim))

    @property
    def path(self):
        return os.path.join(super().path, 'edges' if self.use_edge_features else 'noedges')


if __name__ == '__main__':
    dataset = MnistSegmentationChebDataset(use_edge_features=False, regenerate=False)
    print(dataset)
    print(dataset.n_edge_features)