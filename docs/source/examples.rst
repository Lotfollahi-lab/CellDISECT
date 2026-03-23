=================
Example Scripts
=================

CellDISECT comes with example scripts that demonstrate the complete workflow for training, inference, and perturbation prediction. These scripts are located in the ``examples/`` directory of the repository:

* `training_example.py <https://github.com/Lotfollahi-lab/CellDISECT/blob/main/examples/training_example.py>`_
* `inference_example.py <https://github.com/Lotfollahi-lab/CellDISECT/blob/main/examples/inference_example.py>`_
* `perturbation_example.py <https://github.com/Lotfollahi-lab/CellDISECT/blob/main/examples/perturbation_example.py>`_

Training Example
-----------------

The ``training_example.py`` script demonstrates how to train a CellDISECT model with customizable architecture and training parameters. Here's a breakdown of the key components:

.. code-block:: python

    import scvi
    import scanpy as sc
    import torch
    from lightning.pytorch.loggers import WandbLogger
    from celldisect import CellDISECT

    # Load and prepare data
    adata = sc.read_h5ad('PATH/TO/DATA.h5ad')
    adata = adata[adata.X.sum(1) != 0].copy()

    # Define covariates
    cats = [
        'cat1',
        'cat2',
        'cat3',
    ]
    cell_type_included = False  # Set to True if cell type annotation is in cats

Key Configuration Options:

1. **Architecture Parameters**:

.. code-block:: python

    arch_dict = {
        'n_layers': 2,
        'n_hidden': 128,
        'n_latent_shared': 32,
        'n_latent_attribute': 32,
        'dropout_rate': 0.1,
        'weighted_classifier': False,
    }

2. **Training Parameters**:

.. code-block:: python

    train_dict = {
        'max_epochs': 1000,
        'batch_size': 256,
        'recon_weight': 20,
        'cf_weight': 0.8,
        'beta': 0.003,
        'clf_weight': 0.05,
        'adv_clf_weight': 0.014,
        'adv_period': 5,
        'n_cf': 1,
        'early_stopping_patience': 6,
        'early_stopping': True,
        'save_best': True,
    }

3. **Training Plan Parameters**:

.. code-block:: python

    plan_kwargs = {
        'lr': 0.003,
        'weight_decay': 0.00005,
        'ensemble_method_cf': True,
        'lr_patience': 5,
        'lr_factor': 0.5,
        'lr_scheduler_metric': 'loss_validation',
        'n_epochs_kl_warmup': 10,
    }

The script includes optional Weights & Biases (wandb) integration for training monitoring.

Inference Example
------------------

The ``inference_example.py`` script shows how to load a trained model and perform various types of inference. Key features include:

1. **Loading a trained model**:

.. code-block:: python

    model = CellDISECT.load(f"{pre_path}/{model_name}", adata=adata)

2. **Extracting different latent representations**:

- Z_0 (shared latent space)
- Z_i (covariate-specific latent spaces)
- Z_{-i} (complement latent spaces)
- Z_{0+Z_i} (combined latent spaces)

3. **Computing neighbors and UMAP visualizations** for all latent representations:

.. code-block:: python

    # Compute neighbors and UMAPs for each latent space
    for i in range(len(cats) + 1):
        if i == 0:
            latent_name = f"CellDISECT_Z_{i}"
        else:
            label = cats[i - 1]
            latent_name = f"CellDISECT_Z_{label}"
        
        latent = ad.AnnData(X=adata.obsm[f"{latent_name}"], obs=adata.obs)
        sc.pp.neighbors(adata=latent, use_rep="X")
        sc.tl.umap(adata=latent)

The script also includes commented-out plotting code that you can use in your analysis notebooks.

Using the Examples
-------------------

1. Copy the relevant example script to your working directory
2. Modify the paths and parameters according to your needs:

   - ``PATH/TO/DATA.h5ad``: Path to your input data
   - ``PATH/TO/SAVE/YOUR/MODEL``: Where to save the trained model
   - ``cats``: List of categorical covariates from your data
   - Architecture and training parameters as needed

3. For inference, make sure to:

   - Use the same covariate list as during training
   - Specify the correct path to your trained model
   - Adjust the output path for saving results

Perturbation Example
---------------------

The ``perturbation_example.py`` script demonstrates how to use CellDISECT for perturbation prediction. It covers:

1. **Preparing predefined embeddings**:

.. code-block:: python

    # Load gene embeddings (e.g. GenePT, ESM)
    gene_embeddings = np.load('PATH/TO/GENE_EMBEDDINGS.npy', allow_pickle=True).item()
    adata.uns['pert_embeddings'] = gene_embeddings

2. **Setting up with perturbation support**:

.. code-block:: python

    CellDISECT.setup_anndata(
        adata,
        layer='counts',
        categorical_covariate_keys=['cell_type', 'perturbation'],
        perturbation_key='perturbation',
        perturbation_embedding_key='pert_embeddings',
        perturbation_combination_delimiter='+',
    )

3. **Predicting perturbations** (seen, unseen, or combinatorial):

.. code-block:: python

    x_ctrl, x_true, x_pred = model.predict_perturbation(
        adata,
        perturbation='GeneA+GeneB',
        source_perturbation='ctrl',
        cats=['cell_type', 'perturbation'],
        perturbation_key='perturbation',
    )

4. **Evaluating predictions**:

.. code-block:: python

    from celldisect import perturbation_metrics
    metrics = perturbation_metrics(x_pred.numpy(), x_true.numpy(), x_ctrl.numpy())

These scripts serve as comprehensive templates for working with CellDISECT and can be adapted to your specific use case. 