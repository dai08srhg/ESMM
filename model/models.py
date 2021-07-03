import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    """
    カテゴリ変数をembedding layerでencodeする
    CTRnetworkとCVRnetworkでパラメータを共有する
    """
    def __init__(self, embedding_sizes):
        super(FeatureExtractor, self).__init__()
        # カテゴリ変数のembedding_layer
        self.embedding_layers = nn.ModuleList([nn.Embedding(categories, size) for categories, size in embedding_sizes])

    def forward(self, category_inputs):
        h = [embedding_layer(category_inputs[:, i]) for i, embedding_layer in enumerate(self.embedding_layers)]
        # カテゴリ変数の特徴量ベクトルをconcat
        h = torch.cat(h, dim=1)  # size = (インスタンス数, embeding_dim*カテゴリ変数の数)
        return h


class CtrNetwork(nn.Module):
    """CTR予測を行うnetwork"""
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
    """CVR予測を行うnetwork"""
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
    def __init__(self, feature_extractor: FeatureExtractor, ctr_network: CtrNetwork, cvr_network: CvrNetwork):
        super(ESMM, self).__init__()
        self.feature_extractor = feature_extractor
        self.ctr_network = ctr_network
        self.cvr_network = cvr_network
    
    def forward(self, inputs):
        h = self.feature_extractor(inputs)  # encode
        # Predict pCTR
        p_ctr = self.ctr_network(h)
        # Predict pCVR
        p_cvr = self.cvr_network(h)
        # Predict pCTCVR
        p_ctcvr = torch.mul(p_ctr, p_cvr)
        return p_ctr, p_ctcvr
