from typing import List
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

    LOSS_ACC = "loss_atac"
    RECONST_LOSS_X_ACC = "rec_x_atac"
    RECONST_LOSS_X_CF_ACC = "rec_x_cf_atac"
    KL_Z_ACC = "kl_z_atac"
    CLASSIFICATION_LOSS_ACC = "ce_atac"
    ACCURACY_ACC = "acc_atac"
    F1_ACC = "f1_atac"

LOSS_KEYS_LIST = [
    LOSS_KEYS.LOSS,
    LOSS_KEYS.RECONST_LOSS_X,
    LOSS_KEYS.RECONST_LOSS_X_CF,
    LOSS_KEYS.KL_Z,
    LOSS_KEYS.CLASSIFICATION_LOSS,
    LOSS_KEYS.ACCURACY,
    LOSS_KEYS.F1,
    
    LOSS_KEYS.LOSS_ACC,
    LOSS_KEYS.RECONST_LOSS_X_ACC,
    LOSS_KEYS.RECONST_LOSS_X_CF_ACC,
    LOSS_KEYS.KL_Z_ACC,
    LOSS_KEYS.CLASSIFICATION_LOSS_ACC,
    LOSS_KEYS.ACCURACY_ACC,
    LOSS_KEYS.F1_ACC
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
