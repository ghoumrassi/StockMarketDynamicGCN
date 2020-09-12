from torch_geometric.data import Dataset, Data, Batch, DataLoader
import torch
from pathlib import Path
import numpy as np
import random


class TestDataset(Dataset):
    def __init__(self, root):
        super().__init__(root=root)

    def process(self):
        data_dir = Path(self.processed_dir)
        data_list = []

        for date in range(5):

            X = torch.tensor([[0,  1,  2,  3],
                              [4,  5,  6,  7],
                              [8,  9,  10, 11],
                              [12, 13, 14, 15]]
            )
            y = torch.tensor([0, 1, 2, 3])
            e = torch.tensor([[0, 3, 2],
                              [1, 2, 0]])
            w = torch.tensor([4, 2, 3])

            X *= date
            data = Data(x=X, y=y, edge_index=e, edge_attr=w)
            data_list.append(data)
            if len(data_list) == 3:
                batch = Batch.from_data_list(data_list)
                torch.save(batch, (data_dir / self.processed_file_names[date-2]))
                data_list.pop(0)

    @property
    def processed_file_names(self):
        return ['test', 'test2', 'test3']

    def len(self):
        return len(self.processed_file_names)

    def get(self, i):
        data_dir = Path(self.processed_dir)
        data = torch.load((data_dir / self.processed_file_names[i]))
        data.seq = data.batch
        del data.batch
        return data


def make_X_array():
    X = np.random.randn(10000, 4)
    flag = np.random.rand(10000) > 0.9

    for i in range(X.shape[1] - 1):
        X[:, i][flag] += 10
    return X



if __name__ == '__main__':
    # ds = TestDataset(root='E:/test')
    # dl = DataLoader(ds, batch_size=2)
    # for d in dl:
    #     print(d.x)
    #     d.x.reshape(2, 3, -1, 4)

    make_X_array()