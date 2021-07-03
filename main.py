from typing import List, Tuple
import torch
from torch import nn, optim
import pandas as pd
import hydra
from pathlib import Path
from category_encoders import OrdinalEncoder
from model.models import FeatureExtractor, CtrNetwork, CvrNetwork, ESMM
from model.dataset import EsmmDataset
import tqdm


WORK_DIR = Path().resolve()


def load_data() -> pd.DataFrame: 
    # TODO implement data_load
    return df_train
    
    
def get_embedding_dims(train_X: pd.DataFrame, embedding_dim) -> Tuple(List((int, int)), int):
    """Get embedding layer size and concat_feature_vec_dim"""
    field_dims = list(train_X.max())
    field_dims = list(map(lambda x: x+1, field_dims))  # 各特徴量の最大値＋１

    embedding_sizes = []
    concat_dim = 0
    for field_dim in field_dims:
        embedding_sizes.append((field_dim, embedding_dim))
        concat_dim += embedding_dim
    return embedding_sizes, concat_dim


def train(df_train_encoded: pd.DataFrame, device, cfg):
    category_columns = list(cfg.columns.feature_columns)
    supervised = [cfg.columns.click_supervised, cfg.columns.kpi_supervised]

    # Split feature label
    train_X = df_train_encoded[category_columns]
    train_yz = df_train_encoded[supervised]

    # Build model
    embedding_sizes, concat_dim = get_embedding_dims(train_X, cfg.embedding_dim)
    feature_extractor = FeatureExtractor(embedding_sizes)
    ctr_network = CtrNetwork(concat_dim)
    cvr_network = CvrNetwork(concat_dim)
    model = ESMM(feature_extractor, ctr_network, cvr_network)
    model = model.to(device)

    # Settings
    batch_size = 64
    loss_fn = loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    epochs = 30

    # Build dataloader
    dataset = EsmmDataset(train_X, train_yz)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Start fitting
    model.train()
    for epoch in range(epochs):
        running_total_loss = 0.0
        running_ctr_loss = 0.0
        running_ctcvr_loss = 0.0
        for i, (inputs, y_labels, z_labels) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs = inputs.to(device)
            y_labels = torch.unsqueeze(y_labels.to(device), 1)
            z_labels = torch.unsqueeze(z_labels.to(device), 1)

            # Initialize gradient
            optimizer.zero_grad()  
            # caluculate losses
            p_ctr, p_ctcvr = model(inputs)
            ctr_loss = loss_fn(p_ctr, y_labels)
            ctcvr_loss = loss_fn(p_ctcvr, z_labels)
            total_loss = ctr_loss + ctcvr_loss
            # Backpropagation
            total_loss.backward()
            # Update parameters
            optimizer.step()

            running_total_loss += total_loss.item()
            running_ctr_loss += ctr_loss.item()
            running_ctcvr_loss += ctcvr_loss.item()

        running_total_loss = running_total_loss / (i+1)
        running_ctr_loss = running_ctr_loss / (i+1)
        running_ctcvr_loss = running_ctcvr_loss / (i+1)
        print(f'epoch: {epoch+1}, total_loss: {running_total_loss}, ctr_loss: {running_ctr_loss}, ctcvr_loss: {running_ctcvr_loss}')


@hydra.main(config_path=f'{WORK_DIR}/conf/conf.yaml')
def main(cfg):
    # Load data
    df_train = load_data()

    # Encode dataset
    category_columns = list(cfg.columns.feature_columns)
    encoder = OrdinalEncoder(cols=category_columns, handle_unknown='impute').fit(df_train)
    df_train_encoded = encoder.transform(df_train)

    # Start train
    device = 'cpu'
    train(df_train_encoded, device, cfg)


if __name__ == '__main__':
    main()