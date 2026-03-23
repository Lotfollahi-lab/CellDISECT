from typing import Dict, List, Optional

import logging
import numpy as np
from torch import nn
import torch
from scvi.nn import one_hot

from enum import Enum

logger = logging.getLogger(__name__)


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

##############################################################################
# Perturbation prediction utilities
##############################################################################


def parse_perturbation(name: str, delimiter: str = "+") -> List[str]:
    """Split a (possibly combinatorial) perturbation name into atomic components.

    Parameters
    ----------
    name
        Perturbation label, e.g. ``"GeneA+GeneB"`` or ``"ctrl"``.
    delimiter
        Separator used for combinatorial perturbations.

    Returns
    -------
    List of atomic perturbation names.
    """
    return [comp.strip() for comp in name.split(delimiter)]


def validate_perturbation_embeddings(
    adata,
    perturbation_key: str,
    embedding_key: str,
    delimiter: str = "+",
) -> None:
    """Check that every atomic perturbation in *adata.obs* has an entry in *adata.uns*.

    Parameters
    ----------
    adata
        Annotated data object.
    perturbation_key
        Column in ``adata.obs`` containing perturbation labels.
    embedding_key
        Key in ``adata.uns`` whose value is a ``dict[str, array]`` mapping
        atomic perturbation names to their vector representations.
    delimiter
        Separator used for combinatorial perturbations.

    Raises
    ------
    KeyError
        If ``embedding_key`` is not found in ``adata.uns``.
    ValueError
        If any atomic perturbation is missing from the embeddings dictionary.
    """
    if embedding_key not in adata.uns:
        raise KeyError(
            f"Perturbation embedding key '{embedding_key}' not found in adata.uns. "
            f"Available keys: {list(adata.uns.keys())}"
        )

    emb_dict = adata.uns[embedding_key]
    if not isinstance(emb_dict, dict):
        raise TypeError(
            f"adata.uns['{embedding_key}'] must be a dict mapping perturbation names "
            f"to embedding vectors, got {type(emb_dict)}."
        )

    unique_labels = adata.obs[perturbation_key].unique()
    all_atomic = set()
    for label in unique_labels:
        for comp in parse_perturbation(str(label), delimiter):
            all_atomic.add(comp)

    missing = all_atomic - set(emb_dict.keys())
    if missing:
        raise ValueError(
            f"The following atomic perturbations in adata.obs['{perturbation_key}'] "
            f"do not have embeddings in adata.uns['{embedding_key}']: {sorted(missing)}"
        )

    dims = {k: np.asarray(v).shape for k, v in emb_dict.items() if k in all_atomic}
    unique_shapes = set(dims.values())
    if len(unique_shapes) > 1:
        raise ValueError(
            f"All predefined perturbation embeddings must have the same dimensionality. "
            f"Found shapes: {unique_shapes}"
        )

    logger.info(
        f"Validated {len(all_atomic)} atomic perturbations across "
        f"{len(unique_labels)} unique labels."
    )


def build_perturbation_embedding_matrix(
    category_names: List[str],
    predefined_embeddings: Dict[str, np.ndarray],
    delimiter: str = "+",
) -> torch.Tensor:
    """Build an embedding matrix for all perturbation categories.

    For combinatorial perturbations the component embeddings are summed.

    Parameters
    ----------
    category_names
        Ordered list of perturbation category names (from AnnData mapping).
    predefined_embeddings
        Dictionary mapping atomic perturbation names to vectors.
    delimiter
        Separator for combinatorial perturbations.

    Returns
    -------
    Tensor of shape ``(n_categories, emb_dim)``.
    """
    rows = []
    for cat_name in category_names:
        components = parse_perturbation(str(cat_name), delimiter)
        component_vecs = [
            torch.as_tensor(
                np.asarray(predefined_embeddings[comp], dtype=np.float32)
            )
            for comp in components
        ]
        combined = torch.stack(component_vecs).sum(dim=0)
        rows.append(combined)
    return torch.stack(rows)


def perturbation_metrics(
    pred: np.ndarray,
    true: np.ndarray,
    ctrl: np.ndarray,
    top_n_de: int = 20,
) -> dict:
    """Compute standard perturbation prediction evaluation metrics.

    Parameters
    ----------
    pred
        Predicted mean gene expression, shape ``(n_genes,)`` or ``(n_cells, n_genes)``.
        If 2-D the mean across cells is taken.
    true
        Ground-truth mean gene expression (same shape convention).
    ctrl
        Control (source) mean gene expression (same shape convention).
    top_n_de
        Number of top differentially expressed genes to evaluate.

    Returns
    -------
    Dictionary with the following keys:

    - ``pearson_mean`` -- Pearson *r* between predicted and true mean expression
    - ``pearson_delta`` -- Pearson *r* between predicted and true *delta* (vs ctrl)
    - ``mse`` -- Mean squared error of mean expression
    - ``top_de_pearson`` -- Pearson *r* on top-N DE genes (ranked by |true - ctrl|)
    - ``top_de_cosine`` -- Cosine similarity on top-N DE genes
    """
    from scipy.stats import pearsonr

    def _to_1d(arr):
        arr = np.asarray(arr, dtype=np.float64)
        return arr.mean(axis=0) if arr.ndim == 2 else arr

    pred_mean = _to_1d(pred)
    true_mean = _to_1d(true)
    ctrl_mean = _to_1d(ctrl)

    delta_pred = pred_mean - ctrl_mean
    delta_true = true_mean - ctrl_mean

    r_mean, _ = pearsonr(pred_mean, true_mean)
    r_delta, _ = pearsonr(delta_pred, delta_true)
    mse = float(np.mean((pred_mean - true_mean) ** 2))

    de_ranking = np.argsort(-np.abs(delta_true))
    top_idx = de_ranking[:top_n_de]

    r_de, _ = pearsonr(pred_mean[top_idx], true_mean[top_idx])

    cos_num = np.dot(delta_pred[top_idx], delta_true[top_idx])
    cos_den = (
        np.linalg.norm(delta_pred[top_idx]) * np.linalg.norm(delta_true[top_idx])
        + 1e-8
    )
    cos_de = float(cos_num / cos_den)

    return {
        "pearson_mean": float(r_mean),
        "pearson_delta": float(r_delta),
        "mse": mse,
        f"top{top_n_de}_de_pearson": float(r_de),
        f"top{top_n_de}_de_cosine": cos_de,
    }


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