from typing import List, Tuple
import torch
from torch import nn, optim
import pandas as pd
from pathlib import Path
from category_encoders import OrdinalEncoder
from model.esmm import ESMM
from model.dataset import EsmmDataset
import tqdm

WORK_DIR = Path().resolve()


def load_data() -> pd.DataFrame:
    # TODO implement data_load
    return df


def get_embedding_size(df: pd.DataFrame, embedding_dim: int) -> List[Tuple[int, int]]:
    """
    Get embedding size
    Args:
        df (pd.DataFrame): Train dataset
        embedding_dim (int): Number of embedded dimensions
    Returns:
        List[Tuple[int, int]]: List of (Unique number of categories, embedding_dim)
    """
    df_feature = df.drop(columns=['click', 'conversion'])

    # Get embedding layer size
    max_idxs = list(df_feature.max())
    embedding_sizes = []
    for i in max_idxs:
        embedding_sizes.append((int(i + 1), embedding_dim))

    return embedding_sizes


def train(df: pd.DataFrame):
    if torch.cuda.is_available():
        device = 'cuda'
    else:
        device = 'cpu'

    # Build model
    embedding_sizes = get_embedding_size(df, 5)
    model = ESMM(embedding_sizes)
    model = model.to(device)

    # Settings
    batch_size = 64
    loss_fn = loss_fn = nn.BCELoss()
    optimizer = optim.SGD(model.parameters(), lr=0.01)
    epochs = 30

    # Build dataloader
    dataset = EsmmDataset(df)
    train_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=2)

    # Start fitting
    model.train()
    for epoch in range(epochs):
        running_total_loss = 0.0
        running_ctr_loss = 0.0
        running_ctcvr_loss = 0.0
        for i, (inputs, click, conversion) in tqdm.tqdm(enumerate(train_loader), total=len(train_loader)):
            inputs = inputs.to(device)
            click = torch.unsqueeze(click.to(device), 1)
            conversion = torch.unsqueeze(conversion.to(device), 1)

            # Initialize gradient
            optimizer.zero_grad()
            # caluculate losses
            p_ctr, p_ctcvr = model(inputs)
            ctr_loss = loss_fn(p_ctr, click)
            ctcvr_loss = loss_fn(p_ctcvr, conversion)
            total_loss = ctr_loss + ctcvr_loss
            # Backpropagation
            total_loss.backward()
            # Update parameters
            optimizer.step()

            running_total_loss += total_loss.item()
            running_ctr_loss += ctr_loss.item()
            running_ctcvr_loss += ctcvr_loss.item()

        running_total_loss = running_total_loss / len(train_loader)
        running_ctr_loss = running_ctr_loss / len(train_loader)
        running_ctcvr_loss = running_ctcvr_loss / len(train_loader)
        print(
            f'epoch: {epoch+1}, total_loss: {running_total_loss}, ctr_loss: {running_ctr_loss}, ctcvr_loss: {running_ctcvr_loss}'
        )


def main():
    # Load data
    df = load_data()

    # Encode dataset
    category_columns = ['feature1', 'feature2', 'feature3']
    encoder = OrdinalEncoder(cols=category_columns, handle_unknown='impute').fit(df)
    df = encoder.transform(df)

    # Start train
    train(df)


if __name__ == '__main__':
    main()
