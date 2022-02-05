import torch
import torch.nn as nn
from typing import List, Tuple


class FeatureExtractor(nn.Module):
    """
    Embedding layer for encoding categorical variables.
    """

    def __init__(self, embedding_sizes: List[Tuple[int, int]]):
        """
        Args:
            embedding_sizes (List[Tuple[int, int]]): List of (Unique categorical variables + 1, embedding dim)
        """
        super(FeatureExtractor, self).__init__()
        self.embedding_layers = nn.ModuleList(
            [nn.Embedding(unique_size, embedding_dim) for unique_size, embedding_dim in embedding_sizes])

    def forward(self, category_inputs):
        # Embedding each variable
        h = [embedding_layer(category_inputs[:, i]) for i, embedding_layer in enumerate(self.embedding_layers)]
        # Concat each vector
        h = torch.cat(h, dim=1)  # size = (minibath, embeding_dim * Number of categorical variables)
        return h


class CtrNetwork(nn.Module):
    """NN for CTR prediction"""

    def __init__(self, input_dim):
        super(CtrNetwork, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        p = self.mlp(inputs)
        return self.sigmoid(p)


class CvrNetwork(nn.Module):
    """NN for CVR prediction"""

    def __init__(self, input_dim):
        super(CvrNetwork, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features=input_dim, out_features=128),
            nn.ReLU(),
            nn.Linear(in_features=128, out_features=1),
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, inputs):
        p = self.mlp(inputs)
        return self.sigmoid(p)


class ESMM(nn.Module):
    """ESMM"""

    def __init__(self, embedding_sizes: List[Tuple[int, int]]):
        super(ESMM, self).__init__()
        self.feature_extractor = FeatureExtractor(embedding_sizes)

        input_dim = 0
        for _, embedding_dim in embedding_sizes:
            input_dim += embedding_dim
        self.ctr_network = CtrNetwork(input_dim)
        self.cvr_network = CvrNetwork(input_dim)

    def forward(self, inputs):
        # embedding
        h = self.feature_extractor(inputs)
        # Predict pCTR
        p_ctr = self.ctr_network(h)
        # Predict pCVR
        p_cvr = self.cvr_network(h)
        # Predict pCTCVR
        p_ctcvr = torch.mul(p_ctr, p_cvr)
        return p_ctr, p_ctcvr
