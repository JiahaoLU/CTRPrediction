import torch
import numpy as np
import torch.nn as nn


class FieldAwareFactorizationMachineModel(nn.Module):
    """
    A pytorch implementation of Field-aware Factorization Machine.

    Reference:
        Y Juan, et al. Field-aware Factorization Machines for CTR Prediction, 2015.
    """

    def __init__(self, field_dims, embed_dim):
        """
        Initialisation of the FFM model
        :param field_dims: list of dimensions of each field
        :param embed_dim: dimension of latent vector
        """
        super().__init__()
        self.linear = FeaturesLinear(field_dims)
        self.ffm = FieldAwareFactorizationMachine(field_dims, embed_dim)

    def forward(self, x):
        """
        :param x: Integer tensor of size ``(batch_size, num_fields)``
        """
        ffm_term = torch.sum(torch.sum(self.ffm(x), dim=1), dim=1, keepdim=True)
        x = self.linear(x) + ffm_term
        return torch.sigmoid(x.squeeze(1))


class FeaturesLinear(nn.Module):

    def __init__(self, field_dims, output_dim=1):
        """
        Initialisation of linear first-degree terms. Use nn.Embedding to realize
        inner product of input one-hot encoding vector and weight vector
        :param field_dims: list of dimensions of each field
        :param output_dim: the sum of linear terms
        """
        super().__init__()
        self.fc = nn.Embedding(sum(field_dims), output_dim)
        self.bias = nn.Parameter(torch.zeros((output_dim,)))
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)

    def forward(self, x):
        """
        :param x: Integer tensor of size ``(batch_size, num_fields)``
        """
        x = x.to(torch.int64)  # field-relative one hot encoding
        x = x + x.new_tensor(self.offsets).unsqueeze(0)  # absolute-position one hot encoding
        return torch.sum(self.fc(x), dim=1) + self.bias  # element wise linear poly sum


class FieldAwareFactorizationMachine(nn.Module):

    def __init__(self, field_dims, embed_dim):
        """
        Use latent vector product to replace simple scalar weight factor.
        More adaptive to situations where dependency exists among fields.
        :param field_dims: list of dimensions of each field
        :param embed_dim: dimension of latent vector
        """
        super().__init__()
        self.num_fields = len(field_dims)
        self.embeddings = nn.ModuleList([
            nn.Embedding(sum(field_dims), embed_dim) for _ in range(self.num_fields)
        ])
        self.offsets = np.array((0, *np.cumsum(field_dims)[:-1]), dtype=np.long)
        for embedding in self.embeddings:
            nn.init.xavier_uniform_(embedding.weight.data)

    def forward(self, x):
        """
        :param x: Integer tensor of size ``(batch_size, num_fields)``
        """
        x = x.to(torch.int64)
        x = x + x.new_tensor(self.offsets).unsqueeze(0)
        xs = [self.embeddings[i](x) for i in range(self.num_fields)]  # generation of latent mat
        ix = list()
        for i in range(self.num_fields - 1):
            for j in range(i + 1, self.num_fields):
                ix.append(xs[j][:, i] * xs[i][:, j])  # product of latent vector
        ix = torch.stack(ix, dim=1)
        return ix
