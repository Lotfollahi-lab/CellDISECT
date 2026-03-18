"""Minimal example: perturbation prediction with CellDISECT.

This script demonstrates how to:
1. Prepare predefined perturbation embeddings in adata.uns
2. Set up and train a CellDISECT model with perturbation support
3. Predict expression for seen, unseen, and combinatorial perturbations
4. Evaluate predictions with perturbation_metrics
"""

import numpy as np
import scanpy as sc
import scvi
import torch

torch.set_float32_matmul_precision("medium")

import warnings

warnings.simplefilter("ignore", UserWarning)

from celldisect import CellDISECT, perturbation_metrics

scvi.settings.seed = 42

# ============================================================
# 1. Load data
# ============================================================
adata = sc.read_h5ad("PATH/TO/PERTURBATION_DATA.h5ad")
adata = adata[adata.X.sum(1) != 0].copy()

# ============================================================
# 2. Prepare predefined perturbation embeddings
# ============================================================
# Load pre-computed gene embeddings (e.g. GenePT, ESM, scGPT).
# This should be a dict: {gene_name: np.ndarray of shape (emb_dim,)}
gene_embeddings = np.load("PATH/TO/GENE_EMBEDDINGS.npy", allow_pickle=True).item()

# Store in adata.uns under a chosen key
adata.uns["pert_embeddings"] = gene_embeddings

# ============================================================
# 3. Define covariate keys
# ============================================================
perturbation_key = "perturbation"  # column in adata.obs with pert labels
cats = ["cell_type", perturbation_key]

# ============================================================
# 4. Setup AnnData with perturbation support
# ============================================================
CellDISECT.setup_anndata(
    adata,
    layer="counts",
    categorical_covariate_keys=cats,
    continuous_covariate_keys=[],
    perturbation_key=perturbation_key,
    perturbation_embedding_key="pert_embeddings",
    perturbation_combination_delimiter="+",
)

# ============================================================
# 5. Train the model
# ============================================================
arch_dict = {
    "n_layers": 2,
    "n_hidden": 128,
    "n_latent_shared": 32,
    "n_latent_attribute": 32,
    "dropout_rate": 0.1,
}

model = CellDISECT(adata, **arch_dict)

model.train(
    max_epochs=200,
    batch_size=256,
    recon_weight=20,
    cf_weight=0.8,
    beta=0.003,
    clf_weight=0.05,
    adv_clf_weight=0.014,
    adv_period=5,
    n_cf=1,
    early_stopping=True,
    save_best=True,
    plan_kwargs={"lr": 0.003, "weight_decay": 5e-5},
)

# model.save("path/to/save/model", overwrite=True)

# ============================================================
# 6. Predict a SEEN perturbation
# ============================================================
seen_pert = "GeneA"  # a perturbation present in training data
x_ctrl, x_true, x_pred = model.predict_perturbation(
    adata,
    perturbation=seen_pert,
    source_perturbation="ctrl",
    cats=cats,
    perturbation_key=perturbation_key,
)

metrics = perturbation_metrics(
    x_pred.numpy(), x_true.numpy(), x_ctrl.numpy()
)
print(f"Seen perturbation '{seen_pert}' metrics: {metrics}")

# ============================================================
# 7. Predict an UNSEEN perturbation
# ============================================================
unseen_pert = "GeneX"
unseen_emb = gene_embeddings[unseen_pert]  # embedding for unseen gene

x_ctrl, x_true, x_pred = model.predict_perturbation(
    adata,
    perturbation=unseen_pert,
    source_perturbation="ctrl",
    cats=cats,
    perturbation_key=perturbation_key,
    new_embeddings={unseen_pert: unseen_emb},
)
print(f"Unseen perturbation '{unseen_pert}' prediction shape: {x_pred.shape}")

# ============================================================
# 8. Predict a COMBINATORIAL perturbation
# ============================================================
combo_pert = "GeneA+GeneB"

x_ctrl, x_true, x_pred = model.predict_perturbation(
    adata,
    perturbation=combo_pert,
    source_perturbation="ctrl",
    cats=cats,
    perturbation_key=perturbation_key,
)

if x_true is not None:
    metrics = perturbation_metrics(
        x_pred.numpy(), x_true.numpy(), x_ctrl.numpy()
    )
    print(f"Combinatorial perturbation '{combo_pert}' metrics: {metrics}")
else:
    print(f"Combinatorial perturbation '{combo_pert}' prediction shape: {x_pred.shape}")
