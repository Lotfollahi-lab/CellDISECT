import logging
from typing import List, Literal, Optional, Sequence, Tuple, Union
import random
import numpy as np
import pandas as pd
import torch
from anndata import AnnData
import anndata as ad
import scanpy as sc
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
from scvi.autotune._types import Tunable, TunableMixin
logger = logging.getLogger(__name__)

from ._module import CellDISECTModule, PerturbationEmbedding
from .data import AnnDataSplitter
from .trainingplan import CellDISECTTrainingPlan
from .utils import validate_perturbation_embeddings, parse_perturbation

from scvi.train._callbacks import SaveBestState
import torch.nn as nn


class CellDISECT(
    RNASeqMixin,
    VAEMixin,
    UnsupervisedTrainingMixin,
    BaseModelClass,
    TunableMixin
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
            use_bias: bool = True,
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
        use_bias : bool, optional
            Whether to use bias terms in the neural networks, by default True.
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
        # Extract perturbation configuration if it was set during setup_anndata
        pert_config = adata.uns.get('_celldisect_perturbation_config', None)
        self._pert_config = pert_config
        if pert_config is not None:
            pert_cov_idx = pert_config['perturbation_cov_idx']
            pert_emb_key = pert_config['perturbation_embedding_key']
            predefined_embs = adata.uns[pert_emb_key]
            cat_covs_registry = self.adata_manager.get_state_registry(
                REGISTRY_KEYS.CAT_COVS_KEY
            )
            pert_category_names = list(
                cat_covs_registry.mappings[pert_config['perturbation_key']]
            )
            delimiter = pert_config['perturbation_combination_delimiter']

            model_kwargs['perturbation_cov_idx'] = pert_cov_idx
            model_kwargs['predefined_pert_embeddings'] = predefined_embs
            model_kwargs['perturbation_category_names'] = pert_category_names
            model_kwargs['perturbation_combination_delimiter'] = delimiter

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
            bias=use_bias,
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
            perturbation_key: Optional[str] = None,
            perturbation_embedding_key: Optional[str] = None,
            perturbation_combination_delimiter: str = "+",
            **kwargs,
    ):
        """
        Set up the AnnData object for the CellDISECT model.

        This method configures the AnnData object by registering the necessary
        fields and optionally adding a cluster covariate.  When
        ``perturbation_key`` is provided, the corresponding column in
        ``adata.obs`` is treated as a perturbation covariate whose embeddings
        come from ``adata.uns[perturbation_embedding_key]`` rather than being
        learned during training.

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
        perturbation_key : Optional[str], optional
            Column in ``adata.obs`` that contains perturbation labels (e.g.
            ``"GeneA"``, ``"GeneA+GeneB"``).  When set, the perturbation
            covariate uses predefined embeddings instead of learned ones.
        perturbation_embedding_key : Optional[str], optional
            Key in ``adata.uns`` whose value is a ``dict[str, np.ndarray]``
            mapping atomic perturbation names to their vector
            representations (e.g. ESM or GenePT embeddings).  Required when
            ``perturbation_key`` is set.
        perturbation_combination_delimiter : str, optional
            Delimiter for combinatorial perturbation labels, by default ``"+"``.
        **kwargs
            Additional keyword arguments.

        Returns
        -------
        None
        """
        # Handle perturbation covariate
        if perturbation_key is not None:
            if perturbation_embedding_key is None:
                raise ValueError(
                    "When `perturbation_key` is set, `perturbation_embedding_key` "
                    "must also be provided."
                )
            validate_perturbation_embeddings(
                adata, perturbation_key, perturbation_embedding_key,
                perturbation_combination_delimiter,
            )
            if categorical_covariate_keys is None:
                categorical_covariate_keys = []
            if perturbation_key not in categorical_covariate_keys:
                categorical_covariate_keys = list(categorical_covariate_keys)
                categorical_covariate_keys.append(perturbation_key)
                logger.info(
                    f"Auto-added perturbation_key '{perturbation_key}' to "
                    f"categorical_covariate_keys."
                )

            pert_cov_idx = categorical_covariate_keys.index(perturbation_key)
            adata.uns['_celldisect_perturbation_config'] = {
                'perturbation_key': perturbation_key,
                'perturbation_embedding_key': perturbation_embedding_key,
                'perturbation_combination_delimiter': perturbation_combination_delimiter,
                'perturbation_cov_idx': pert_cov_idx,
            }

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
            px_cf = self.module.sub_forward(idx=cov_idx + 1, x=tensors[REGISTRY_KEYS.X_KEY].to(self.device),
                                            cat_covs=tensors[REGISTRY_KEYS.CAT_COVS_KEY].to(self.device))

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

    @torch.no_grad()
    def get_gene_importance(self) -> pd.DataFrame:
        """
        Computes gene importance for each encoder.

        This method calculates a gene importance score for each gene based on the weights
        of the encoders. The importance is calculated by propagating the weights through
        the network layers to obtain an effective weight matrix from input genes to the
        latent space. This method is most accurate when the model is trained without biases
        and with a non-negative activation function like ReLU. If biases are present,
        they are ignored in this calculation.

        Returns
        -------
        pd.DataFrame
            A pandas DataFrame containing gene importance scores. The rows correspond to
            genes, and the columns correspond to the different encoders (e.g., 'shared',
            'attribute_0', 'attribute_1', ...).
        """
        self._check_if_trained(warn=False)
        self.module.eval()

        n_genes = self.summary_stats.n_vars
        gene_names = self.adata_manager.adata.var_names.to_list()
        importances_df = pd.DataFrame(index=gene_names)

        encoder_names = ["shared"] + [f"attribute_{i}" for i in range(self.module.zs_num)]

        for i, encoder in enumerate(self.module.z_encoders_list):
            sequences = [seq for seq in encoder.encoder.fc_layers]
            linear_layers = []
            for seq in sequences:
                for layer in seq:
                    if isinstance(layer, nn.Linear):
                        linear_layers.append(layer)

            
            # Add the mean encoder layer
            linear_layers.append(encoder.mean_encoder)

            # Get weights and compute effective weight matrix
            # Note: PyTorch Linear layer weights are (out_features, in_features)
            # We transpose them to (in_features, out_features) for matrix multiplication
            w1 = linear_layers[0].weight.t()
            
            # The input to the encoder is genes + covariates, we only want the gene part
            w_eff = w1[:n_genes, :]

            for layer in linear_layers[1:]:
                w = layer.weight.t()
                w_eff = torch.relu(w_eff) @ w
            
            # Calculate importance scores
            gene_importances = torch.sum(torch.abs(w_eff), dim=1)
            normalized_importances = gene_importances / torch.sum(gene_importances)
            
            importances_df[encoder_names[i]] = normalized_importances.cpu().numpy()

        return importances_df

    def train(
            self,
            max_epochs: Optional[int] = None,
            use_gpu: Optional[Union[str, int, bool]] = True,
            train_size: float = 0.8,
            validation_size: Optional[float] = None,
            batch_size: int = 256,
            early_stopping: bool = True,
            save_best: bool = False,
            plan_kwargs: Optional[dict] = None,
            recon_weight: Tunable[Union[float, int]] = 10, # RECONST_LOSS_X weight
            cf_weight: Tunable[Union[float, int]] = 1,  # RECONST_LOSS_X_CF weight
            beta: Tunable[Union[float, int]] = 1,  # KL Zi weight
            clf_weight: Tunable[Union[float, int]] = 50,  # Si classifier weight
            adv_clf_weight: Tunable[Union[float, int]] = 10,  # adversarial classifier weight
            adv_period: Tunable[int] = 1,  # adversarial training period
            n_cf: Tunable[int] = 10,  # number of X_cf recons (a random permutation of n VAEs and a random half-batch subset for each trial)
            kappa_optimizer2: bool = True,
            n_epochs_pretrain_ae: int = 0,
            **trainer_kwargs,
    ):
        """
        Train the model.

        Parameters
        ----------
        max_epochs : Optional[int]
            Number of passes through the dataset. If `None`, defaults to `np.min([round((20000 / n_cells) * 400), 400])`.
        use_gpu : Optional[Union[str, int, bool]]
            Whether to use GPU for training. Can be a boolean, string, or integer specifying the GPU device.
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
        recon_weight : Tunable[Union[float, int]]
            Weight for the reconstruction loss of X.
        cf_weight : Tunable[Union[float, int]]
            Weight for the reconstruction loss of X_cf.
        beta : Tunable[Union[float, int]]
            Weight for the KL divergence of Zi.
        clf_weight : Tunable[Union[float, int]]
            Weight for the Si classifier loss.
        adv_clf_weight : Tunable[Union[float, int]]
            Weight for the adversarial classifier loss.
        adv_period : Tunable[int]
            Adversarial training period.
        n_cf : Tunable[int]
            Number of X_cf reconstructions (a random permutation of n VAEs and a random half-batch subset for each trial).
        kappa_optimizer2 : bool
            Whether to use the second kappa optimizer.
        n_epochs_pretrain_ae : int
            Number of epochs to pretrain the autoencoder.
        **trainer_kwargs
            Other keyword arguments for :class:`~scvi.train.Trainer`.

        Returns
        -------
        None
        """
        n_cells = self.adata.n_obs
        if max_epochs is None:
            max_epochs = int(np.min([round((20000 / n_cells) * 400), 400]))

        plan_kwargs = plan_kwargs if isinstance(plan_kwargs, dict) else {}

        if self.split_key is not None:
            data_splitter = AnnDataSplitter(
                self.adata_manager,
                train_indices=self.train_indices,
                valid_indices=self.valid_indices,
                test_indices=self.test_indices,
                batch_size=batch_size,
                use_gpu=use_gpu,
                drop_last=3,
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
            use_gpu=use_gpu,
            **trainer_kwargs,
        )
        return runner()

    @torch.no_grad()
    def generate(
        self,
        n_samples: int,
        covariates: dict[str, str],
        library_size: Optional[float] = None,
        adata_for_library_size: Optional[AnnData] = None,
        random_seed: Optional[int] = None,
        include_shared_latent: bool = False,
    ):
        """
        Generate new cells conditionally based on specified categorical covariates.

        This method generates `n_samples` new cells for a given combination of categorical covariates.
        The generation process involves sampling from the latent spaces and then decoding to
        get gene expression profiles.

        The shared latent space (`z_shared`) is sampled from a standard normal distribution.
        The attribute-specific latent spaces (`zs`) are sampled from their respective prior encoders,
        conditioned on the provided covariates.

        Parameters
        ----------
        n_samples
            Number of cells to generate.
        covariates
            A dictionary mapping categorical covariate names to the desired values for generation.
            All categorical covariates used to train the model must be specified.
        library_size
            A specific library size to use for generation.
            Mutually exclusive with `adata_for_library_size`.
        adata_for_library_size
            An AnnData object to infer the median library size from. If cells with the specified
            `covariates` combination exist in this AnnData object, their median library size is used.
            Otherwise, the median library size of the entire `adata_for_library_size` is used.
            Mutually exclusive with `library_size`.
        random_seed
            A random seed for reproducibility.
        include_shared_latent
            Whether to include the expression from the decoder associated with the shared latent space (Z_0)
            in the `average_expression`. If `False`, the average is computed over attribute-specific
            decoders only. We suggest setting this to `False` for most use cases. The shared decoder
            is assumed to be `decoder_0`.

        Returns
        -------
        A tuple containing:
            - **average_expression** (numpy.ndarray): An array of shape `(n_samples, n_vars)`
              representing the average gene expression across the selected decoders.
            - **all_expressions** (dict[str, numpy.ndarray]): A dictionary where keys are
              decoder names (`decoder_0`, `decoder_1`, etc.) and values are numpy arrays of
              shape `(n_samples, n_vars)` representing the generated gene expression counts
              from each individual decoder.

        Examples
        --------
        >>> n_generated_cells = 10
        >>> desired_covariates = {"cell_type": "T-cell", "condition": "diseased"}
        >>> avg_expr, all_expr = model.generate(
        ...     n_samples=n_generated_cells,
        ...     covariates=desired_covariates,
        ...     adata_for_library_size=adata
        ... )

        To compute a custom average using only a subset of decoders (e.g., decoder_0 and decoder_2):
        >>> import numpy as np
        >>> custom_avg = np.mean(
        ...     [all_expr["decoder_0"], all_expr["decoder_2"]], axis=0
        ... )
        """
        if library_size is None and adata_for_library_size is None:
            raise ValueError("Either library_size or adata_for_library_size must be provided.")
        if adata_for_library_size is not None:
            adata_for_library_size = adata_for_library_size.copy()
        self._check_if_trained(warn=False)
        self.module.eval()
        if random_seed is not None:
            random.seed(random_seed)
            torch.manual_seed(random_seed)
            np.random.seed(random_seed)

        cat_covs_registry = self.adata_manager.get_state_registry(
            REGISTRY_KEYS.CAT_COVS_KEY
        )
        cov_names = cat_covs_registry.field_keys
        mappings = cat_covs_registry.mappings

        if set(covariates.keys()) != set(cov_names):
            raise ValueError(
                "Please provide a value for every categorical covariate. "
                f"Required: {cov_names}, but got: {list(covariates.keys())}"
            )

        cat_indices = []
        for cov_name in cov_names:
            cov_value = covariates[cov_name]
            mapping = mappings[cov_name]
            if cov_value not in mapping:
                raise ValueError(
                    f"Covariate value '{cov_value}' not found for covariate '{cov_name}'. "
                    f"Available values are: {list(mapping)}"
                )
            value_idx = np.where(mapping == cov_value)[0][0]
            cat_indices.append(value_idx)

        cat_covs_tensor = (
            torch.tensor(cat_indices, device=self.device).unsqueeze(0).expand(n_samples, -1)
        )

        # 1. Sample z_shared from a standard Gaussian
        z_shared = torch.randn(
            n_samples, self.module.n_latent_shared, device=self.device
        )

        # 2. Sample zs from prior encoders
        prior_emb_in = []
        for i, embedding_indices in enumerate(cat_covs_tensor.t()):
            emb = self.module.covars_embeddings[str(i)](embedding_indices.long())
            emb = emb + torch.randn_like(emb) * 0.1
            prior_emb_in.append(emb)

        zs = []
        for i in range(len(self.module.z_prior_encoders_list)):
            # Encoder returns (distribution, sample)
            _, z_s_i = self.module.z_prior_encoders_list[i](prior_emb_in[i])
            zs.append(z_s_i)

        # 3. Decode
        if library_size is not None:
            library = torch.full(
                (n_samples, 1), library_size
            ).to(self.device)
        elif adata_for_library_size is not None:
            # If there are any cells for the covariate combination, use the library size of the subset, if not, use the library size of the entire dataset
            cov_subset_mask = []
            for cov_name, cov_value in covariates.items():
                cov_subset_mask.append(adata_for_library_size.obs[cov_name] == cov_value)
            cov_subset_mask = np.all(cov_subset_mask, axis=0)
            if np.any(cov_subset_mask):
                print(f"Using library size of the subset for the covariate combination {covariates}.")
                print(f"Number of cells in the subset: {np.sum(cov_subset_mask)}")
                adata_for_library_size = adata_for_library_size[cov_subset_mask]
            else:
                print(f"No cells found for the covariate combination {covariates} in the adata_for_library_size dataset.")
                print(f"Using library size of the entire dataset.")
                
            library_global = torch.log(torch.tensor(adata_for_library_size.X.sum(1)).unsqueeze(1))
            library_median = library_global.median().item()
            print(f"Library size median: {library_median}")
            library = torch.full(
                (n_samples, 1), library_median
            ).to(self.device)
        else:
            raise ValueError("Either library_size or adata_for_library_size must be provided.")
            
        generative_kwargs = {
            "z_shared": z_shared,
            "zs": zs,
            "library": library,
            "cat_covs": cat_covs_tensor,
        }
        generative_outputs = self.module.generative(**generative_kwargs)
        pxs = generative_outputs["px"]

        # 4. Return generated expressions from all decoders
        all_expressions = {}
        for i, px in enumerate(pxs):
            all_expressions[f"decoder_{i}"] = px.mu.cpu().detach().numpy()

        # 5. Compute average expression
        if include_shared_latent:
            average_expression = np.mean(list(all_expressions.values()), axis=0)
        else:
            average_expression = np.mean(list(all_expressions.values())[1:], axis=0)

        return average_expression, all_expressions

    # ------------------------------------------------------------------
    # Perturbation prediction API
    # ------------------------------------------------------------------

    def add_perturbation_embedding(
        self,
        name: str,
        embedding: np.ndarray,
    ) -> None:
        """Register an embedding vector for an unseen atomic perturbation.

        After calling this method the model can predict counterfactuals for
        the new perturbation (or combinatorial perturbations that include it).

        Parameters
        ----------
        name
            Atomic perturbation name (e.g. ``"GeneC"``).
        embedding
            Vector representation with the same dimensionality as the
            predefined embeddings used during training.
        """
        if self._pert_config is None:
            raise RuntimeError(
                "This model was not set up with perturbation support. "
                "Use perturbation_key in setup_anndata first."
            )
        pert_idx = self._pert_config['perturbation_cov_idx']
        pert_emb_module = self.module.covars_embeddings[str(pert_idx)]
        if not isinstance(pert_emb_module, PerturbationEmbedding):
            raise RuntimeError("Perturbation embedding module not found.")
        pert_emb_module.add_perturbation(name, np.asarray(embedding, dtype=np.float32))

    def get_perturbation_embeddings(self) -> Tuple[List[str], np.ndarray]:
        """Return current perturbation names and their (combined) embeddings.

        Returns
        -------
        names
            Ordered list of perturbation category names.
        embeddings
            Numpy array of shape ``(n_perturbations, emb_dim)``.
        """
        if self._pert_config is None:
            raise RuntimeError(
                "This model was not set up with perturbation support."
            )
        pert_idx = self._pert_config['perturbation_cov_idx']
        pert_emb_module = self.module.covars_embeddings[str(pert_idx)]
        names = list(pert_emb_module._category_names)
        weights = pert_emb_module.embedding.weight.data.cpu().numpy()
        return names, weights

    # ------------------------------------------------------------------
    # Perturbation prediction API
    # ------------------------------------------------------------------

    @torch.no_grad()
    def predict_perturbation(
        self,
        adata: AnnData,
        perturbation: str,
        source_perturbation: str,
        cats: List[str],
        perturbation_key: str,
        new_embeddings: Optional[dict] = None,
        n_samples_from_source: Optional[int] = None,
        seed: Optional[int] = 0,
        device: Optional[Union[str, torch.device]] = None,
        batch_size: Optional[int] = 256,
        source_adata: Optional[AnnData] = None,
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor], torch.Tensor]:
        """Predict gene expression after a perturbation.

        Memory-efficient implementation that avoids copying large adata objects.

        Parameters
        ----------
        adata
            AnnData object containing cells. Must have a ``'counts'`` layer.
        perturbation
            Target perturbation label (e.g. ``"GeneA"`` or ``"GeneA+GeneB"``).
        source_perturbation
            Source / control perturbation label (e.g. ``"ctrl"``).
        cats
            List of categorical covariate keys (same order used during training).
        perturbation_key
            Which element of ``cats`` is the perturbation covariate.
        new_embeddings
            Optional dictionary mapping atomic perturbation names to embedding vectors.
        n_samples_from_source
            If set, randomly sample this many cells from the source group.
        seed
            Random seed for reproducibility.
        device
            Device to run prediction on. If ``None``, uses the model's device.
        batch_size
            Batch size for forward passes. Default 256.
        source_adata
            Optional AnnData with control cells.

        Returns
        -------
        x_ctrl
            Source (control) counts, shape ``(n_source, n_genes)``.
        x_true
            Target (ground-truth) counts if cells exist, else ``None``.
        x_pred
            Predicted counts, shape ``(n_source, n_genes)``.
        """
        self._check_if_trained(warn=False)
        
        if self._pert_config is None:
            raise RuntimeError(
                "This model was not set up with perturbation support. "
                "Use perturbation_key in setup_anndata first."
            )

        pert_idx = self._pert_config['perturbation_cov_idx']
        pert_emb_module = self.module.covars_embeddings[str(pert_idx)]
        delimiter = self._pert_config['perturbation_combination_delimiter']

        # Register new embeddings if provided
        if new_embeddings is not None:
            for name, vec in new_embeddings.items():
                if name not in pert_emb_module._predefined_embeddings:
                    pert_emb_module._predefined_embeddings[name] = np.asarray(vec, dtype=np.float32)

        # Validate all perturbation components have embeddings
        for comp in parse_perturbation(perturbation, delimiter):
            if comp not in pert_emb_module._predefined_embeddings:
                raise ValueError(f"Atomic perturbation '{comp}' has no embedding.")
        for comp in parse_perturbation(source_perturbation, delimiter):
            if comp not in pert_emb_module._predefined_embeddings:
                raise ValueError(f"Source perturbation component '{comp}' has no embedding.")

        # Determine device
        _device = torch.device(device) if isinstance(device, str) else (device or self.device)

        # Get source cell indices (NO COPY of adata)
        data_source = source_adata if source_adata is not None else adata
        source_mask = data_source.obs[perturbation_key] == source_perturbation
        source_indices = np.where(source_mask)[0]
        
        if len(source_indices) == 0:
            raise ValueError(f"No cells with {perturbation_key}='{source_perturbation}' found.")

        # Sample if requested
        if n_samples_from_source is not None and n_samples_from_source < len(source_indices):
            random.seed(seed)
            source_indices = np.array(random.sample(list(source_indices), n_samples_from_source))

        # Extract source counts directly (minimal memory - just the subset we need)
        X_source = data_source.layers['counts'][source_indices]
        if sparse.issparse(X_source):
            X_source = X_source.toarray()
        x_ctrl = torch.tensor(X_source, dtype=torch.float32)

        # Extract source covariates directly using the TRAINED model's manager
        # We need the categorical covariate indices from the training data mapping
        source_cat_covs = []
        trained_cat_registry = self.adata_manager.get_state_registry(REGISTRY_KEYS.CAT_COVS_KEY)
        
        for i, cat_key in enumerate(cats):
            cat_values = data_source.obs[cat_key].iloc[source_indices].values
            mapping = trained_cat_registry.mappings[cat_key]
            
            # Convert category values to indices
            cat_indices = []
            for val in cat_values:
                if val in mapping:
                    cat_indices.append(np.where(mapping == val)[0][0])
                else:
                    raise ValueError(f"Category value '{val}' not found in training mapping for '{cat_key}'")
            source_cat_covs.append(torch.tensor(cat_indices, dtype=torch.long))
        
        cat_covs_tensor = torch.stack(source_cat_covs, dim=1)  # (n_cells, n_cats)

        # Build new perturbation mapping that includes the target perturbation
        old_category_names = list(pert_emb_module._category_names)
        new_mapping = list(old_category_names)
        if perturbation not in new_mapping:
            new_mapping.append(perturbation)
        if source_perturbation not in new_mapping:
            new_mapping.append(source_perturbation)
        
        # Rebuild embedding for new mapping
        pert_emb_module.rebuild_for_mapping(new_mapping)
        
        # Get indices in new mapping
        source_pert_idx = new_mapping.index(source_perturbation)
        target_pert_idx = new_mapping.index(perturbation)
        
        # Update source cat_covs to use source perturbation index
        cat_covs_tensor[:, pert_idx] = source_pert_idx

        # Move model to device if needed
        module_device = next(self.module.parameters()).device
        if module_device != _device:
            self.module.to(_device)

        # Run prediction in batches
        n_cells = len(source_indices)
        _batch_size = batch_size if batch_size is not None else n_cells
        px_cf_mean_list = []

        try:
            for start in range(0, n_cells, _batch_size):
                end = min(start + _batch_size, n_cells)
                
                x_batch = x_ctrl[start:end].to(_device)
                cat_covs_batch = cat_covs_tensor[start:end].to(_device)
                
                # Create counterfactual covariates (change perturbation)
                cat_covs_cf = cat_covs_batch.clone()
                cat_covs_cf[:, pert_idx] = target_pert_idx
                
                # Forward pass
                _, pxs_cf = self.module.sub_forward_cf_avg(
                    x=x_batch,
                    cat_covs=cat_covs_batch,
                    cat_covs_cf=cat_covs_cf,
                )
                
                # Average predictions from all decoders
                batch_preds = [px_cf.mu for px_cf in pxs_cf if px_cf is not None]
                if batch_preds:
                    batch_mean = torch.stack(batch_preds, dim=0).mean(dim=0)
                    px_cf_mean_list.append(batch_mean.cpu())
                
                # Clear GPU memory
                del x_batch, cat_covs_batch, cat_covs_cf
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

            x_pred = torch.cat(px_cf_mean_list, dim=0)

        finally:
            # Restore original mapping
            pert_emb_module.rebuild_for_mapping(old_category_names)
            if module_device != _device:
                self.module.to(module_device)

        # Get ground truth (on-demand, no copy of full adata)
        x_true = None
        target_mask = adata.obs[perturbation_key] == perturbation
        if target_mask.any():
            X_true = adata.layers['counts'][target_mask]
            if sparse.issparse(X_true):
                X_true = X_true.toarray()
            x_true = torch.tensor(X_true, dtype=torch.float32)

        return x_ctrl, x_true, x_pred

    @torch.no_grad()
    def predict_perturbations(
        self,
        adata: AnnData,
        perturbations: List[str],
        source_perturbation: str,
        cats: List[str],
        perturbation_key: str,
        new_embeddings: Optional[dict] = None,
        n_samples_from_source: Optional[int] = None,
        seed: Optional[int] = 0,
        device: Optional[Union[str, torch.device]] = None,
        batch_size: Optional[int] = 256,
        source_adata: Optional[AnnData] = None,
    ) -> dict:
        """Predict gene expression for multiple perturbations.

        Parameters
        ----------
        adata
            AnnData object containing cells. Must have a ``'counts'`` layer.
        perturbations
            List of target perturbation labels.
        source_perturbation
            Source / control perturbation label.
        cats
            List of categorical covariate keys.
        perturbation_key
            Which element of ``cats`` is the perturbation covariate.
        new_embeddings
            Optional dictionary mapping atomic perturbation names to embedding vectors.
        n_samples_from_source
            If set, randomly sample this many cells from the source group.
        seed
            Random seed for reproducibility.
        device
            Device to run prediction on.
        batch_size
            Batch size for forward passes. Default 256.
        source_adata
            Optional AnnData with control cells.

        Returns
        -------
        dict
            Dictionary mapping each perturbation to ``(x_ctrl, x_true, x_pred)``.
        """
        if not perturbations:
            return {}

        results = {}
        for pert in perturbations:
            x_ctrl, x_true, x_pred = self.predict_perturbation(
                adata=adata,
                perturbation=pert,
                source_perturbation=source_perturbation,
                cats=cats,
                perturbation_key=perturbation_key,
                new_embeddings=new_embeddings,
                n_samples_from_source=n_samples_from_source,
                seed=seed,
                device=device,
                batch_size=batch_size,
                source_adata=source_adata,
            )
            results[pert] = (x_ctrl, x_true, x_pred)
        
        return results
