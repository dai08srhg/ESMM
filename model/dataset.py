import torch
import pandas as pd


class EsmmDataset(torch.utils.data.Dataset):
    def __init__(self, train_X: pd.DataFrame, train_yz: pd.DataFrame):
        self.train_X = torch.from_numpy(train_X.values).long()
        self.train_y = torch.from_numpy(train_yz['click_supervised'].values).float()  # label of click
        self.train_z = torch.from_numpy(train_yz['kpi_supervised'].values).float()  # label of post-click-conversion
        self.data_num = len(train_X)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_x = self.train_X[idx]
        out_y = self.train_y[idx]
        out_z = self.train_z[idx]
        return out_x, out_y, out_z