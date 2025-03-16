import logging
from typing import List, Literal, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch
from anndata import AnnData
import anndata as ad
import scanpy as sc
import random
from scipy import sparse

from sklearn.utils.class_weight import compute_class_weight

from scvi import REGISTRY_KEYS
from scvi.data import AnnDataManager
from scvi.data.fields import (
    CategoricalJointObsField,
    CategoricalObsField,
    LayerField,
    NumericalJointObsField,
    NumericalObsField,
)
from scvi.model.base import UnsupervisedTrainingMixin
from scvi.utils import setup_anndata_dsp
from scvi.dataloaders._data_splitting import DataSplitter
from scvi.dataloaders._ann_dataloader import AnnDataLoader
from scvi.train import TrainRunner
from scvi.model.base import RNASeqMixin, VAEMixin, BaseModelClass
logger = logging.getLogger(__name__)

from ._module import CellDISECTModule
from .data import AnnDataSplitter
from .trainingplan import CellDISECTTrainingPlan

from scvi.train._callbacks import SaveBestState

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def set_seed(seed):
    """Set seed for all random number generators for reproducibility."""
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)

class CellDISECT(
    RNASeqMixin,
    VAEMixin,
    UnsupervisedTrainingMixin,
    BaseModelClass
):
    """CellDISECT model for single-cell RNA sequencing data analysis.

    Parameters
    ----------
    adata
        AnnData object that has been registered via :meth:`~scvi.model.SCVI.setup_anndata`.
    n_hidden
        Number of nodes per hidden layer.
    n_latent_shared
        Dimensionality of the shared latent space.
    n_latent_attribute
        Dimensionality of the latent space for each sensitive attribute.
    n_layers
        Number of hidden layers used for encoder and decoder neural networks.
    dropout_rate
        Dropout rate for neural networks.
    gene_likelihood
        One of:

        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of:

        * ``'normal'`` - Normal distribution
        * ``'ln'`` - Logistic normal distribution (Normal(0, I) transformed by softmax)
    split_key
        Key in `adata.obs` to split the data into training, validation, and test sets.
    train_split
        Values in `split_key` to be used for training.
    valid_split
        Values in `split_key` to be used for validation.
    test_split
        Values in `split_key` to be used for testing.
    weighted_classifier
        Whether to use weighted classifiers for categorical covariates.
    **model_kwargs
        Additional keyword arguments for the model.
    """

    _module_cls = CellDISECTModule
    _data_splitter_cls = AnnDataSplitter
    _training_plan_cls = CellDISECTTrainingPlan
    _train_runner_cls = TrainRunner

    def __init__(
            self,
            adata: AnnData,
            n_hidden: int = 128,
            n_latent_shared: int = 10,
            n_latent_attribute: int = 10,
            n_layers: int = 1,
            dropout_rate: float = 0.1,
            gene_likelihood: Literal["zinb", "nb", "poisson"] = "zinb",
            latent_distribution: Literal["normal", "ln"] = "normal",
            split_key: str = None,
            train_split: Union[str, List[str]] = ["train"],
            valid_split: Union[str, List[str]] = ["valid"],
            test_split: Union[str, List[str]] = ["ood"],
            weighted_classifier=False,
            **model_kwargs,
    ):
        """
        Initialize the CellDISECT model.

        Parameters
        ----------
        adata : AnnData
            AnnData object that has been registered via :meth:`~scvi.model.SCVI.setup_anndata`.
        n_hidden : int, optional
            Number of nodes per hidden layer, by default 128.
        n_latent_shared : int, optional
            Dimensionality of the shared latent space, by default 10.
        n_latent_attribute : int, optional
            Dimensionality of the latent space for each sensitive attribute, by default 10.
        n_layers : int, optional
            Number of hidden layers used for encoder and decoder neural networks, by default 1.
        dropout_rate : float, optional
            Dropout rate for neural networks, by default 0.1.
        gene_likelihood : Literal["zinb", "nb", "poisson"], optional
            Gene likelihood distribution, by default "zinb".
        latent_distribution : Literal["normal", "ln"], optional
            Latent distribution, by default "normal".
        split_key : str, optional
            Key in `adata.obs` to split the data into training, validation, and test sets, by default None.
        train_split : Union[str, List[str]], optional
            Values in `split_key` to be used for training, by default ["train"].
        valid_split : Union[str, List[str]], optional
            Values in `split_key` to be used for validation, by default ["valid"].
        test_split : Union[str, List[str]], optional
            Values in `split_key` to be used for testing, by default ["ood"].
        weighted_classifier : bool, optional
            Whether to use weighted classifiers for categorical covariates, by default False.
        **model_kwargs : dict
            Additional keyword arguments for the model.
        """
        super().__init__(adata)

        self._data_loader_cls = AnnDataLoader
        self.split_key = split_key
        n_cats_per_cov = (
            self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).n_cats_per_key
            if REGISTRY_KEYS.CAT_COVS_KEY in self.adata_manager.data_registry
            else None
        )
        self.classifier_weights = None
        if weighted_classifier:
            if REGISTRY_KEYS.CAT_COVS_KEY not in self.adata_manager.data_registry:
                raise ValueError(
                    "Cannot use weighted classifier without categorical covariates."
                )
            self.classifier_weights = []
            for covar in self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            ).field_keys:
                y = self.adata.obs[covar].values
                classes = np.unique(y)
                class_weight = compute_class_weight(class_weight="balanced", classes=classes, y=y)
                self.classifier_weights.append(class_weight)
        self.module = self._module_cls(
            n_input=self.summary_stats.n_vars,
            n_cats_per_cov=n_cats_per_cov,
            n_hidden=n_hidden,
            n_latent_shared=n_latent_shared,
            n_latent_attribute=n_latent_attribute,
            n_layers=n_layers,
            dropout_rate=dropout_rate,
            gene_likelihood=gene_likelihood,
            latent_distribution=latent_distribution,
            classifier_weights=self.classifier_weights,
            **model_kwargs,
        )
        if split_key is not None:
            train_indices = np.where(adata.obs.loc[:, split_key].isin(train_split))[0]
            valid_indices = np.where(adata.obs.loc[:, split_key].isin(valid_split))[0]
            test_indices = np.where(adata.obs.loc[:, split_key].isin(test_split))[0]

            self.train_indices = train_indices
            self.valid_indices = valid_indices
            self.test_indices = test_indices

        self._model_summary_string = (
            "CellDISECT Model with the following params: \nn_hidden: {}, n_latent_shared: {}, n_latent_attribute: {}"
            ", n_layers: {}, dropout_rate: {}, gene_likelihood: {}, latent_distribution: {}"
        ).format(
            n_hidden,
            n_latent_shared,
            n_latent_attribute,
            n_layers,
            dropout_rate,
            gene_likelihood,
            latent_distribution,
        )
        self.init_params_ = self._get_init_params(locals())

    @classmethod
    @setup_anndata_dsp.dedent
    def setup_anndata(
            cls,
            adata: AnnData,
            layer: Optional[str] = None,
            batch_key: Optional[str] = None,
            labels_key: Optional[str] = None,
            size_factor_key: Optional[str] = None,
            categorical_covariate_keys: Optional[List[str]] = None,
            continuous_covariate_keys: Optional[List[str]] = None,
            add_cluster_covariate: bool = False,
            clustering_normalize_counts: bool = True,
            **kwargs,
    ):
        """
        Set up the AnnData object for the CellDISECT model.

        This method configures the AnnData object by registering the necessary fields and optionally adding a cluster covariate.

        Parameters
        ----------
        adata : AnnData
            AnnData object to be set up.
        layer : Optional[str], optional
            Layer in `adata` to use as the count data, by default None.
        batch_key : Optional[str], optional
            Key in `adata.obs` for batch information, by default None.
        labels_key : Optional[str], optional
            Key in `adata.obs` for labels, by default None.
        size_factor_key : Optional[str], optional
            Key in `adata.obs` for size factors, by default None.
        categorical_covariate_keys : Optional[List[str]], optional
            List of keys in `adata.obs` for categorical covariates, by default None.
        continuous_covariate_keys : Optional[List[str]], optional
            List of keys in `adata.obs` for continuous covariates, by default None.
        add_cluster_covariate : bool, optional
            Whether to add a cluster covariate to `adata.obs`, by default False.
        clustering_normalize_counts : bool, optional
            Whether to normalize counts before clustering, by default True.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        None
        """
        setup_method_args = cls._get_setup_method_args(**locals())
        if add_cluster_covariate:
            cls.add_cluster_covariate(
                adata,
                normalize_counts=clustering_normalize_counts
                )

        anndata_fields = [
            LayerField(REGISTRY_KEYS.X_KEY, layer, is_count_data=True),
            CategoricalObsField(REGISTRY_KEYS.BATCH_KEY, batch_key),
            CategoricalObsField(REGISTRY_KEYS.LABELS_KEY, labels_key),
            NumericalObsField(
                REGISTRY_KEYS.SIZE_FACTOR_KEY, size_factor_key, required=False
            ),
            CategoricalJointObsField(
                REGISTRY_KEYS.CAT_COVS_KEY, categorical_covariate_keys
            ),
            NumericalJointObsField(
                REGISTRY_KEYS.CONT_COVS_KEY, continuous_covariate_keys
            ),
        ]
        if add_cluster_covariate:
            anndata_fields.append(
                CategoricalObsField('cluster', '_cluster')
                )

        adata_manager = AnnDataManager(
            fields=anndata_fields, setup_method_args=setup_method_args
        )
        adata_manager.register_fields(adata, **kwargs)
        cls.register_manager(adata_manager)

    @classmethod
    def add_cluster_covariate(
        cls,
        adata: AnnData,
        normalize_counts: bool = True):
        """
        Run PCA on the gene expression matrix and run Leiden clustering on the PCA components
        to create a cluster covariate to be added to the `adata.obs`.

        Parameters
        ----------
        adata : AnnData
            AnnData object containing the single-cell RNA sequencing data.
        normalize_counts : bool, optional
            If True, takes the counts from the `adata.layers['counts']` and log normalizes them, by default True.

        Returns
        -------
        None
        """
        logger.info("Adding cluster covariate to adata.obs")
        if '_cluster' in adata.obs.keys():
            logger.warning(
                "Cluster covariate already present in adata.obs, remove in case you want to re-run, skipping!")
            return

        if normalize_counts:
            logger.info("Normalizing counts")
            adata.X = adata.layers['counts'].copy()
            # Normalizing to median total counts
            sc.pp.normalize_total(adata)
            # Logarithmize the data
            sc.pp.log1p(adata)

        logger.info("Running PCA and Leiden clustering")
        sc.tl.pca(adata, random_state=0)
        sc.pp.neighbors(adata, use_rep='X_pca', random_state=0)
        sc.tl.leiden(adata, key_added='_cluster', flavor='igraph', n_iterations=2, random_state=0)

        return
        

    # call this method after training the model with this held-out:
    # covs[cov_idx] = cov_value_cf, covs[others_idx] = adata.obs[others_idx]
    @torch.no_grad()
    def predict_given_covs_depricated(
            self,
            adata: AnnData,  # source anndata with fixed cov values
            cats: List[str],
            cov_idx: int,  # index in cats starting from 0
            cov_value_cf,
            batch_size: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        self._check_if_trained(warn=False)

        adata_cf = adata.copy()
        cov_name = cats[cov_idx]
        adata_cf.obs[cov_name] = pd.Categorical([cov_value_cf for _ in adata_cf.obs[cov_name]])

        CellDISECT.setup_anndata(
            adata_cf,
            layer='counts',
            categorical_covariate_keys=cats,
            continuous_covariate_keys=[]
        )

        adata_cf = self._validate_anndata(adata_cf)

        scdl = self._make_data_loader(
            adata=adata_cf, batch_size=batch_size
        )

        px_cf_mean_list = []

        for tensors in scdl:
            px_cf = self.module.sub_forward(idx=cov_idx + 1, x=tensors[REGISTRY_KEYS.X_KEY].to(device),
                                            cat_covs=tensors[REGISTRY_KEYS.CAT_COVS_KEY].to(device))

            px_cf_mean_list.append(px_cf.mean)

        px_cf_mean_tensor = torch.cat(px_cf_mean_list, dim=0)
        px_cf_mean_pred = torch.mean(px_cf_mean_tensor, dim=0)

        px_cf_variance = torch.sub(px_cf_mean_tensor, px_cf_mean_pred)
        px_cf_variance = torch.pow(px_cf_variance, 2)
        px_cf_variance_pred = torch.mean(px_cf_variance, dim=0)

        return px_cf_mean_pred, px_cf_variance_pred

    @torch.no_grad()
    def predict_counterfactuals(
            self,
            adata: AnnData,
            cov_names: list[str],
            cov_values: list[str],
            cov_values_cf: list[str],
            cats: list[str],
            n_samples_from_source: Optional[int] = None,
            seed: Optional[int] = 0
    ):
        """Predicts counterfactuals for a given subset of data.

        This function estimates the counterfactual outcomes for a subset of data based on specified changes in covariate values.

        Parameters
        ----------
        adata : AnnData
            The subset of the data for which the counterfactuals are to be predicted.
        cov_names : list[str]
            Names of the covariates that are to be changed.
        cov_values : list[str]
            Original values for the covariates that are to be changed.
        cov_values_cf : list[str]
            Counterfactual values for the covariates that are to be changed.
        cats : list[str]
            Names of the categorical covariates.
        n_samples_from_source : Optional[int], optional
            Number of samples to take from the source data to predict the counterfactuals.
            If None, all samples from the source data are used. Defaults to None.
        seed : Optional[int], optional
            Random seed for reproducibility. Defaults to 0.
            Only used if `n_samples_from_source` is not None.

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
            Control (source), True Counterfactuals, and Predicted Counterfactual COUNTS (not log-transformed).

        Examples
        --------
        Single covariate change::

            cats = ['cell_type', 'condition']
            cell_type_to_check = ['CD4 T',]
            cov_names = ['condition']
            cov_values = ['ctrl']
            cov_values_cf = ['stimulated']
            n_samples_from_source = 500
            x_ctrl, x_true, x_pred = model.predict_counterfactuals(
                adata[(adata.obs['cell_type'].isin(cell_type_to_check))].copy(),
                cov_names=cov_names,
                cov_values=cov_values,
                cov_values_cf=cov_values_cf,
                cats=cats,
                n_samples_from_source=n_samples_from_source,
            )

        Multiple covariate change::

            cats = ['tissue', 'Sample ID', 'sex', 'Age_bin', 'CoarseCellType']
            cell_type_to_check = 'Epithelial cell (luminal)'
            cov_names = ['sex', 'tissue']
            cov_values = ['female', 'breast']
            cov_values_cf = ['male', 'prostate gland']
            n_samples_from_source = None
            x_ctrl, x_true, x_pred = model.predict_counterfactuals(
                adata[adata.obs['Broad cell type'] == cell_type_to_check].copy(),
                cov_names=cov_names,
                cov_values=cov_values,
                cov_values_cf=cov_values_cf,
                cats=cats,
                n_samples_from_source=n_samples_from_source,
            )
        """
        # Copy the counts layer to the main matrix
        adata.X = adata.layers['counts'].copy()
        adata.obs['idx'] = [i for i in range(len(adata))]

        # Identify true and source indices based on covariate values
        true_indices = pd.DataFrame(
            [adata.obs[cov_name] == cov_values_cf[i] for i, cov_name in enumerate(cov_names)]
            ).all(0).values
        true_idx = list(adata[true_indices].obs['idx'])

        source_indices = pd.DataFrame(
            [adata.obs[cov_name] == cov_values[i] for i, cov_name in enumerate(cov_names)]
            ).all(0).values
        source_idx = list(adata[source_indices].obs['idx'])

        # Create true and source AnnData objects
        true_adata = adata[adata.obs['idx'].isin(true_idx)].copy()
        source_adata = adata[adata.obs['idx'].isin(source_idx)].copy()

        # Sample from source data if specified
        if n_samples_from_source is not None:
            random.seed(seed)
            chosen_ids = random.sample(range(len(source_adata)), n_samples_from_source)
            source_adata = source_adata[chosen_ids].copy()
        adata_cf = source_adata.copy()

        # Update covariate values in the counterfactual data
        for i, cov_name in enumerate(cov_names):
            adata_cf.obs.loc[:, cov_name] = pd.Categorical(
                [cov_values_cf[i] for _ in adata_cf.obs[cov_name]]
            )
        batch_size = len(adata_cf)
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Setup AnnData for the counterfactual data
        self.setup_anndata(
            adata_cf,
            layer='counts',
            categorical_covariate_keys=cats,
            continuous_covariate_keys=[]
        )
        adata_cf = self._validate_anndata(adata_cf)
        source_adata = self._validate_anndata(source_adata)

        # Create data loaders for source and counterfactual data
        scdl_cf = self._make_data_loader(
            adata=adata_cf, batch_size=batch_size
        )
        scdl = self._make_data_loader(
            adata=source_adata, batch_size=batch_size
        )

        # Predict counterfactuals
        px_cf_mean_list = []
        for tensors, tensors_cf in zip(scdl, scdl_cf):
            _, pxs_cf = self.module.sub_forward_cf_avg(
                x=tensors[REGISTRY_KEYS.X_KEY].to(device),
                cat_covs=tensors[REGISTRY_KEYS.CAT_COVS_KEY].to(device),
                cat_covs_cf=tensors_cf[REGISTRY_KEYS.CAT_COVS_KEY].to(device)
            )

            for px_cf in pxs_cf:
                if px_cf is None:
                    continue
                x_cf = px_cf.mu
                px_cf_mean_list.append(x_cf)

        # Compute mean predictions
        px_cf_mean_tensor = torch.stack(px_cf_mean_list, dim=0)
        px_cf_mean_pred = torch.mean(px_cf_mean_tensor, dim=0)  # (n_cells, n_genes)

        # Convert predictions to numpy arrays
        px_cf_mean_pred = px_cf_mean_pred.to('cpu').detach().numpy()
        px_cf_mean_tensor = px_cf_mean_tensor.to('cpu').detach().numpy()

        # Create AnnData object for predictions
        px_cf_mean_tensor = ad.AnnData(px_cf_mean_pred)
        px_cf_mean_tensor = torch.tensor(px_cf_mean_tensor.X)

        # Get true and control counts
        if sparse.issparse(true_adata.X):
            true_x_count = torch.tensor(true_adata.X.toarray())
        else:
            true_x_count = torch.tensor(true_adata.X)
        if sparse.issparse(source_adata.X):
            cf_x_count = torch.tensor(source_adata.X.toarray())
        else:
            cf_x_count = torch.tensor(source_adata.X)

        x_true = true_x_count
        x_pred = px_cf_mean_tensor
        x_ctrl = cf_x_count

        return x_ctrl, x_true, x_pred


    @torch.no_grad()
    def get_latent_representation(
            self,
            adata: Optional[AnnData] = None,
            indices: Optional[Sequence[int]] = None,
            batch_size: Optional[int] = None,
            nullify_cat_covs_indices: Optional[List[int]] = None,
            nullify_shared: Optional[bool] = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
        """
        Get the latent representation of the data.

        This function computes the latent representation of the data using the trained model.
        It allows for optional nullification of specific categorical covariates or the shared latent space.

        Parameters
        ----------
        adata : Optional[AnnData]
            Annotated data object. If None, uses the data registered with the model.
        indices : Optional[Sequence[int]]
            Optional indices to subset the data.
        batch_size : Optional[int]
            Batch size to use for data loading. If None, uses the default batch size.
        nullify_cat_covs_indices : Optional[List[int]]
            List of indices of categorical covariates to nullify in the latent space. If None, no covariates are nullified.
        nullify_shared : Optional[bool]
            If True, nullifies the shared latent space. Defaults to False.

        Returns
        -------
        Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
            The latent representation of the data. If `nullify_cat_covs_indices` or `nullify_shared` is specified,
            returns a tuple of the latent representation and the nullified latent representation.
        """
        self._check_if_trained(warn=False)

        adata = self._validate_anndata(adata)
        scdl = self._make_data_loader(
            adata=adata, indices=indices, batch_size=batch_size
        )
        latent = []
        for tensors in scdl:
            inference_inputs = self.module._get_inference_input(tensors)
            outputs = self.module.inference(**inference_inputs,
                                            nullify_cat_covs_indices=nullify_cat_covs_indices,
                                            nullify_shared=nullify_shared)

            latent += [outputs["z_concat"].cpu()]

        return torch.cat(latent).numpy()

    @torch.no_grad()
    def get_cat_covariate_latents(
            self,
    ):
        """
        Returns the embeddings of the categorical covariates.

        This function retrieves the embeddings of the categorical covariates from the trained model.
        It also returns the mappings of the categorical covariates.

        Parameters
        ----------
        self : object
            The instance of the class.

        Returns
        -------
        Tuple[dict, dict]
            A tuple containing two dictionaries:
            - covar_embeddings: Dictionary where keys are covariate names and values are the embeddings as numpy arrays.
            - covar_mappings: Dictionary where keys are covariate names and values are the mappings of the covariates.
        """
        self._check_if_trained(warn=False)
        covar_names = self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY).values()[0]

        covar_embeddings = {}
        covar_mappings = {}
        for name, emb in zip(covar_names, self.module.covars_embeddings.values()):
            mappings = self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)['mappings'][name]
            covar_embeddings[name] = emb.weight.cpu().detach().numpy()
            covar_mappings[name] = mappings

        return covar_embeddings, covar_mappings

    def train(
            self,
            max_epochs: Optional[int] = None,
            accelerator: str = "auto",
            devices: Union[int, list[int], str] = "auto",
            train_size: float = 0.8,
            validation_size: Optional[float] = None,
            batch_size: int = 256,
            early_stopping: bool = True,
            save_best: bool = False,
            plan_kwargs: Optional[dict] = None,
            recon_weight: float = 10, # RECONST_LOSS_X weight
            cf_weight: float = 1,  # RECONST_LOSS_X_CF weight
            beta: float = 1,  # KL Zi weight
            clf_weight: float = 50,  # Si classifier weight
            adv_clf_weight: float = 10,  # adversarial classifier weight
            adv_period: int = 1,  # adversarial training period
            n_cf: int = 10,  # number of X_cf recons (a random permutation of n VAEs and a random half-batch subset for each trial)
            kappa_optimizer2: bool = True,
            n_epochs_pretrain_ae: int = 0,
            use_gpu: Optional[Union[str, int, bool]] = None,  # Kept for backwards compatibility
            **trainer_kwargs,
    ):
        """
        Train the model.

        Parameters
        ----------
        max_epochs : Optional[int]
            Number of passes through the dataset. If `None`, defaults to `np.min([round((20000 / n_cells) * 400), 400])`.
        accelerator : str
            Accelerator to use for training. One of 'cpu', 'gpu', 'tpu', 'ipu', 'auto'.
            'auto' will use GPU if available, otherwise CPU.
        devices : Union[int, list[int], str]
            Devices to use for training. Can be an int (use that many devices), a list of ints
            (use these specific devices), or a string ('auto' will use all available devices).
        train_size : float
            Size of the training set in the range [0.0, 1.0].
        validation_size : Optional[float]
            Size of the validation set. If `None`, defaults to 1 - `train_size`. If `train_size + validation_size < 1`, the remaining cells belong to a test set.
        batch_size : int
            Minibatch size to use during training.
        early_stopping : bool
            Perform early stopping. Additional arguments can be passed in `**kwargs`. See :class:`~scvi.train.Trainer` for further options.
        save_best : bool
            Save the best model state with respect to the validation loss (default), or use the final state in the training procedure.
        plan_kwargs : Optional[dict]
            Keyword arguments for :class:`~scvi.train.TrainingPlan`. Keyword arguments passed to `train()` will overwrite values present in `plan_kwargs`, when appropriate.
        recon_weight : float
            Weight for the reconstruction loss of X.
        cf_weight : float
            Weight for the reconstruction loss of X_cf.
        beta : float
            Weight for the KL divergence of Zi.
        clf_weight : float
            Weight for the Si classifier loss.
        adv_clf_weight : float
            Weight for the adversarial classifier loss.
        adv_period : int
            Adversarial training period.
        n_cf : int
            Number of X_cf reconstructions (a random permutation of n VAEs and a random half-batch subset for each trial).
        kappa_optimizer2 : bool
            Whether to use the second kappa optimizer.
        n_epochs_pretrain_ae : int
            Number of epochs to pretrain the autoencoder.
        use_gpu : Optional[Union[str, int, bool]]
            Deprecated. Use accelerator='gpu' and devices instead.
        **trainer_kwargs
            Other keyword arguments for :class:`~scvi.train.Trainer`.

        Returns
        -------
        None
        """
        # Ensure reproducibility by setting all seeds
        from scvi import settings
        if hasattr(settings, 'seed') and settings.seed is not None:
            set_seed(settings.seed)
        
        n_cells = self.adata.n_obs
        if max_epochs is None:
            max_epochs = int(np.min([round((20000 / n_cells) * 400), 400]))

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else {}
        
        # Handle deprecated use_gpu parameter for backward compatibility
        if use_gpu is not None:
            import warnings
            warnings.warn(
                "The use_gpu parameter is deprecated. Please use accelerator='gpu' instead.",
                DeprecationWarning,
                stacklevel=2,
            )
            if use_gpu:
                accelerator = "gpu"
                devices = 1 if isinstance(use_gpu, bool) else use_gpu

        if self.split_key is not None:
            data_splitter = AnnDataSplitter(
                self.adata_manager,
                train_indices=self.train_indices,
                valid_indices=self.valid_indices,
                test_indices=self.test_indices,
                batch_size=batch_size,
                use_gpu=accelerator == "gpu",  # Convert back for backward compatibility
                drop_last=True,
            )
        else:
            data_splitter = DataSplitter(
                adata_manager=self.adata_manager,
                train_size=train_size,
                validation_size=validation_size,
                batch_size=batch_size,
            )

        training_plan = self._training_plan_cls(self.module,
                                                recon_weight=recon_weight,
                                                cf_weight=cf_weight,
                                                beta=beta,
                                                clf_weight=clf_weight,
                                                adv_clf_weight=adv_clf_weight,
                                                adv_period=adv_period,
                                                n_cf=n_cf,
                                                kappa_optimizer2=kappa_optimizer2,
                                                n_epochs_pretrain_ae=n_epochs_pretrain_ae,
                                                **plan_kwargs)

        es = "early_stopping"
        trainer_kwargs[es] = (
            early_stopping if es not in trainer_kwargs.keys() else trainer_kwargs[es]
        )

        if save_best:
            checkpoint = SaveBestState(
                monitor="loss_validation", mode="min", period=1, verbose=True
            )
            trainer_kwargs["callbacks"] = [] if "callbacks" not in trainer_kwargs else trainer_kwargs["callbacks"]
            trainer_kwargs["callbacks"].append(checkpoint)

        trainer_kwargs['enable_checkpointing'] = True

        trainer_kwargs['early_stopping_monitor'] = "loss_validation"
        runner = self._train_runner_cls(
            self,
            training_plan=training_plan,
            data_splitter=data_splitter,
            max_epochs=max_epochs,
            accelerator=accelerator,
            devices=devices,
            **trainer_kwargs,
        )
        return runner()
