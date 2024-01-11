from typing import List
from torch import nn
import torch
from scvi.nn import one_hot

from enum import Enum


class TRAIN_MODE(int, Enum):
    RECONST = 0
    RECONST_CF = 1
    KL_Z = 2
    CLASSIFICATION = 3
    ADVERSARIAL = 4


class LOSS_KEYS(str, Enum):
    LOSS = "loss"
    RECONST_LOSS_X = "rec_x"
    RECONST_LOSS_X_CF = "rec_x_cf"
    KL_Z = "kl_z"
    CLASSIFICATION_LOSS = "ce"
    ACCURACY = "acc"
    F1 = "f1"


LOSS_KEYS_LIST = [
    LOSS_KEYS.LOSS,
    LOSS_KEYS.RECONST_LOSS_X,
    LOSS_KEYS.RECONST_LOSS_X_CF,
    LOSS_KEYS.KL_Z,
    LOSS_KEYS.CLASSIFICATION_LOSS,
    LOSS_KEYS.ACCURACY,
    LOSS_KEYS.F1
]


def one_hot_cat(n_cat_list: List[int], cat_covs: torch.Tensor):
    cat_list = list()
    if cat_covs is not None:
        cat_list = list(torch.split(cat_covs, 1, dim=1))
    one_hot_cat_list = []
    if len(n_cat_list) > len(cat_list):
        raise ValueError("nb. categorical args provided doesn't match init. params.")
    for n_cat, cat in zip(n_cat_list, cat_list):
        if n_cat and cat is None:
            raise ValueError("cat not provided while n_cat != 0 in init. params.")
        if n_cat > 1:  # n_cat = 1 will be ignored - no additional information
            if cat.size(1) != n_cat:
                onehot_cat = one_hot(cat, n_cat)
            else:
                onehot_cat = cat  # cat has already been one_hot encoded
            one_hot_cat_list += [onehot_cat]
    u_cat = torch.cat(*one_hot_cat_list) if len(one_hot_cat_list) > 1 else one_hot_cat_list[0]
    return u_cat

class PerturbationNetwork(nn.Module): # from CPA code
    def __init__(self,
                 n_perts,
                 n_latent,
                 doser_type='logsigm',
                 n_hidden=None,
                 n_layers=None,
                 dropout_rate: float = 0.0,
                 drug_embeddings=None,):
        super().__init__()
        self.n_latent = n_latent
        
        if drug_embeddings is not None:
            self.pert_embedding = drug_embeddings
            self.pert_transformation = nn.Linear(drug_embeddings.embedding_dim, n_latent)
            self.use_rdkit = True
        else:
            self.use_rdkit = False
            self.pert_embedding = nn.Embedding(n_perts, n_latent, padding_idx=CPA_REGISTRY_KEYS.PADDING_IDX)
            
        self.doser_type = doser_type
        if self.doser_type == 'mlp':
            self.dosers = nn.ModuleList()
            for _ in range(n_perts):
                self.dosers.append(
                    FCLayers(
                        n_in=1,
                        n_out=1,
                        n_hidden=n_hidden,
                        n_layers=n_layers,
                        use_batch_norm=False,
                        use_layer_norm=True,
                        dropout_rate=dropout_rate
                    )
                )
        else:
            self.dosers = GeneralizedSigmoid(n_perts, non_linearity=self.doser_type)

    def forward(self, perts, dosages):
        """
            perts: (batch_size, max_comb_len)
            dosages: (batch_size, max_comb_len)
        """
        bs, max_comb_len = perts.shape
        perts = perts.long()
        scaled_dosages = self.dosers(dosages, perts)  # (batch_size, max_comb_len)

        drug_embeddings = self.pert_embedding(perts)  # (batch_size, max_comb_len, n_drug_emb_dim)

        if self.use_rdkit:
            drug_embeddings = self.pert_transformation(drug_embeddings.view(bs * max_comb_len, -1)).view(bs, max_comb_len, -1)

        z_drugs = torch.einsum('bm,bme->bme', [scaled_dosages, drug_embeddings])  # (batch_size, n_latent)

        z_drugs = torch.einsum('bmn,bm->bmn', z_drugs, (perts != 0).int()).sum(dim=1)  # mask single perts

        return z_drugs # (batch_size, n_latent)