import torch
import pandas as pd


class EsmmDataset(torch.utils.data.Dataset):
    """
    Dataset of ESMM
    """

    def __init__(self, df: pd.DataFrame):
        df_feature = df.drop(columns=['click', 'conversion'])

        self.X = torch.from_numpy(df_feature.values).long()
        self.click = torch.from_numpy(df['click'].values).float()  # label of click
        self.conversion = torch.from_numpy(df['conversion'].values).float()  # label of conversion

        self.data_num = len(self.X)

    def __len__(self):
        return self.data_num

    def __getitem__(self, idx):
        out_X = self.X[idx]
        out_click = self.click[idx]
        out_conversion = self.conversion[idx]
        return out_X, out_click, out_conversion
