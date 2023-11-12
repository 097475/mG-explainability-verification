import networkx as nx
from libmg import print_graph
from spektral.data import Dataset, Graph
import os
import shutil
import numpy as np
from scipy.sparse import coo_matrix
from torch_geometric.datasets import ExplainerDataset
from torch_geometric.datasets.graph_generator import BAGraph
from torch_geometric.utils import to_networkx


class ExplainerSyntheticDataset(Dataset):
    def __init__(self, explainer_dataset, regenerate=False, **kwargs):
        self.explainer_dataset = explainer_dataset
        self.num_graphs = len(self.explainer_dataset)
        if regenerate:
            shutil.rmtree(self.path)
        super().__init__(**kwargs)

    def download(self):
        os.makedirs(self.path)
        for i, g in enumerate(self.explainer_dataset):
            a = coo_matrix((np.ones(g.num_edges, dtype=int), (g.edge_index[0], g.edge_index[1])),
                           shape=(g.num_nodes, g.num_nodes))
            x = np.ones((g.num_nodes, 1))
            y = np.expand_dims(g.y, -1)
            filename = os.path.join(self.path, f'graph_{i}')
            np.savez(filename, x=x, a=a, y=y)

    def read(self):
        output = []

        for i in range(self.num_graphs):
            data = np.load(os.path.join(self.path, f'graph_{i}.npz'), allow_pickle=True)
            output.append(
                Graph(x=data['x'], a=data['a'].item(), y=data['y'])
            )

        return output


if __name__ == '__main__':
    dataset = ExplainerSyntheticDataset(
        ExplainerDataset(graph_generator=BAGraph(num_nodes=300, num_edges=5), motif_generator='house', num_motifs=80, num_graphs=1),
        False)
    print_graph(dataset[0], show_labels=True)

