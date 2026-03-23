import logging
import random
from typing import Callable, Dict, Iterable, Literal, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import kl_divergence as kl
from torchmetrics import Accuracy, F1Score

from scvi import REGISTRY_KEYS
from scvi.autotune._types import Tunable
from scvi.distributions import NegativeBinomial, Poisson, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, auto_move_data
from scvi.nn import DecoderSCVI, Encoder

torch.backends.cudnn.benchmark = True
from .utils import *
from scvi.module._classifier import Classifier

_logger = logging.getLogger(__name__)

dim_indices = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class PerturbationEmbedding(nn.Module):
    """Embedding module backed by predefined external vectors (e.g. ESM, GenePT).

    For combinatorial perturbations (e.g. ``"GeneA+GeneB"``), component
    embeddings are summed.  The weight matrix is **frozen** by default.

    Parameters
    ----------
    predefined_embeddings
        Mapping from atomic perturbation name to its vector representation.
    category_names
        Ordered list of category names that mirrors the integer-index mapping
        produced by scvi-tools' ``CategoricalJointObsField``.
    combination_delimiter
        String used to split combinatorial perturbation labels.
    """

    def __init__(
        self,
        predefined_embeddings: Dict[str, np.ndarray],
        category_names: list,
        combination_delimiter: str = "+",
    ):
        super().__init__()
        self.combination_delimiter = combination_delimiter
        self._predefined_embeddings: Dict[str, np.ndarray] = dict(predefined_embeddings)
        self._category_names: list = list(category_names)

        emb_matrix = build_perturbation_embedding_matrix(
            self._category_names, self._predefined_embeddings, combination_delimiter
        )
        self.embedding_dim = emb_matrix.shape[1]
        self.embedding = nn.Embedding(
            emb_matrix.shape[0], self.embedding_dim
        )
        self.embedding.weight.data.copy_(emb_matrix)
        self.embedding.weight.requires_grad = False

    # ------------------------------------------------------------------
    # Runtime expansion for unseen perturbations
    # ------------------------------------------------------------------
    def add_perturbation(
        self,
        name: str,
        embedding: Optional[np.ndarray] = None,
    ) -> int:
        """Register a new perturbation (possibly unseen during training).

        Parameters
        ----------
        name
            Perturbation label. Can be combinatorial (``"A+B"``).
        embedding
            Vector for an *atomic* perturbation that is not yet in the
            predefined dictionary.  Ignored for names already known.

        Returns
        -------
        Integer index of the (new or existing) perturbation.
        """
        if name in self._category_names:
            return self._category_names.index(name)

        components = parse_perturbation(name, self.combination_delimiter)
        for comp in components:
            if comp not in self._predefined_embeddings:
                if embedding is not None and len(components) == 1:
                    self._predefined_embeddings[comp] = np.asarray(embedding, dtype=np.float32)
                elif embedding is not None:
                    raise ValueError(
                        f"Cannot add atomic embedding via `embedding` arg when name "
                        f"'{name}' is combinatorial. Add each component separately first."
                    )
                else:
                    raise ValueError(
                        f"Atomic perturbation '{comp}' (from '{name}') has no predefined "
                        f"embedding. Provide it via the `embedding` argument."
                    )

        component_vecs = [
            torch.as_tensor(np.asarray(self._predefined_embeddings[c], dtype=np.float32))
            for c in components
        ]
        combined = torch.stack(component_vecs).sum(dim=0).unsqueeze(0)

        old_weight = self.embedding.weight.data
        device = old_weight.device
        new_weight = torch.cat([old_weight, combined.to(device)], dim=0)
        self.embedding = nn.Embedding(new_weight.shape[0], self.embedding_dim)
        self.embedding.weight.data.copy_(new_weight)
        self.embedding.weight.requires_grad = False
        self.embedding = self.embedding.to(device)

        self._category_names.append(name)
        return len(self._category_names) - 1

    def rebuild_for_mapping(self, new_category_names: list) -> None:
        """Rebuild the weight matrix for a different category-to-index mapping."""
        emb_matrix = build_perturbation_embedding_matrix(
            new_category_names, self._predefined_embeddings, self.combination_delimiter
        )
        device = self.embedding.weight.device
        self.embedding = nn.Embedding(emb_matrix.shape[0], self.embedding_dim)
        self.embedding.weight.data.copy_(emb_matrix.to(device))
        self.embedding.weight.requires_grad = False
        self.embedding = self.embedding.to(device)
        self._category_names = list(new_category_names)

    def forward(self, indices: torch.Tensor) -> torch.Tensor:
        return self.embedding(indices.long())

    @property
    def weight(self):
        """Alias so callers expecting ``nn.Embedding`` attributes still work."""
        return self.embedding.weight


class CellDISECTModule(BaseModuleClass):
    """
    Variational auto-encoder module.

    Parameters
    ----------
    n_input : int
        Number of input genes.
    n_hidden : Tunable[int], optional
        Number of nodes per hidden layer, by default 128.
    n_latent_shared : Tunable[int], optional
        Dimensionality of the shared latent space (Z_{-s}), by default 10.
    n_latent_attribute : Tunable[int], optional
        Dimensionality of the latent space for each sensitive attribute (Z_{s_i}), by default 10.
    n_layers : Tunable[int], optional
        Number of hidden layers used for encoder and decoder NNs, by default 1.
    n_cats_per_cov : Optional[Iterable[int]], optional
        Number of categories for each extra categorical covariate, by default None.
    dropout_rate : Tunable[float], optional
        Dropout rate for neural networks, by default 0.1.
    log_variational : bool, optional
        Log(data+1) prior to encoding for numerical stability. Not normalization, by default True.
    gene_likelihood : Tunable[Literal["zinb", "nb", "poisson"]], optional
        One of 'nb' (Negative binomial distribution), 'zinb' (Zero-inflated negative binomial distribution), or 'poisson' (Poisson distribution), by default "zinb".
    latent_distribution : Tunable[Literal["normal", "ln"]], optional
        One of 'normal' (Isotropic normal) or 'ln' (Logistic normal with normal params N(0, 1)), by default "normal".
    deeply_inject_covariates : Tunable[bool], optional
        Whether to concatenate covariates into output of hidden layers in encoder/decoder. This option only applies when `n_layers` > 1. The covariates are concatenated to the input of subsequent hidden layers, by default True.
    use_batch_norm : Tunable[Literal["encoder", "decoder", "none", "both"]], optional
        Whether to use batch norm in layers, by default "both".
    use_layer_norm : Tunable[Literal["encoder", "decoder", "none", "both"]], optional
        Whether to use layer norm in layers, by default "none".
    var_activation : Optional[Callable], optional
        Callable used to ensure positivity of the variational distributions' variance. When `None`, defaults to `torch.exp`, by default None.
    use_custom_embs : bool, optional
        Whether to use custom embeddings, by default False.
    embeddings : Union[torch.Tensor, List[torch.Tensor]], optional
        Custom embeddings to use if `use_custom_embs` is True, by default None.
    classifier_weights : Optional[list], optional
        Weights for the classifiers, by default None.
    bias : bool, optional
        Whether to use bias in the encoder and decoder layers, by default True.
    perturbation_cov_idx : Optional[int], optional
        Index (in the categorical covariates list) of the perturbation covariate.
        When set, that covariate uses predefined embeddings instead of learned ones.
    predefined_pert_embeddings : Optional[dict], optional
        Mapping from atomic perturbation name to its vector representation
        (e.g. ESM or GenePT embeddings). Required when ``perturbation_cov_idx``
        is not None.
    perturbation_category_names : Optional[list], optional
        Ordered category names for the perturbation covariate (mirrors the
        integer-index mapping from the AnnData manager).
    perturbation_combination_delimiter : str, optional
        Delimiter for combinatorial perturbation labels, by default ``"+"``.
    """

    def __init__(
            self,
            n_input: int,
            n_hidden: Tunable[int] = 128,
            n_latent_shared: Tunable[int] = 10,
            n_latent_attribute: Tunable[int] = 10,
            n_layers: Tunable[int] = 1,
            n_cats_per_cov: Optional[Iterable[int]] = None,
            dropout_rate: Tunable[float] = 0.1,
            log_variational: bool = True,
            gene_likelihood: Tunable[Literal["zinb", "nb", "poisson"]] = "zinb",
            latent_distribution: Tunable[Literal["normal", "ln"]] = "normal",
            deeply_inject_covariates: Tunable[bool] = True,
            use_batch_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "both",
            use_layer_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "none",
            var_activation: Optional[Callable] = None,
            use_custom_embs: bool = False,
            embeddings: Union[torch.Tensor, List[torch.Tensor]] = None,
            classifier_weights: Optional[list] = None,
            bias: bool = True,
            perturbation_cov_idx: Optional[int] = None,
            predefined_pert_embeddings: Optional[Dict[str, np.ndarray]] = None,
            perturbation_category_names: Optional[list] = None,
            perturbation_combination_delimiter: str = "+",
    ):
        super().__init__()
        self.dispersion = "gene"
        self.n_latent_shared = n_latent_shared
        self.n_latent_attribute = n_latent_attribute
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        self.latent_distribution = latent_distribution
        self.perturbation_cov_idx = perturbation_cov_idx

        self.px_r = torch.nn.Parameter(torch.randn(n_input))

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # Encoders

        n_input_encoder = n_input

        self.n_cat_list = list([] if n_cats_per_cov is None else n_cats_per_cov)

        # ---- per-covariate embeddings + projections ----
        _has_perturbation = (
            perturbation_cov_idx is not None
            and predefined_pert_embeddings is not None
            and perturbation_category_names is not None
        )
        self._has_perturbation = _has_perturbation

        if use_custom_embs and not _has_perturbation:
            # Legacy path: single custom embedding for ALL covariates
            self.covars_embeddings = nn.ModuleDict(
                {
                    str(key): torch.nn.Embedding(embedding.shape[0], embedding.shape[1])
                    for key, embedding in enumerate([embeddings])
                }
            )
            self.covars_embeddings['0'].weight.data.copy_(embeddings)
            self.covars_embeddings['0'].weight.requires_grad = False
            self.emb_projections = nn.ModuleDict(
                {str(k): nn.Identity() for k in range(len(self.n_cat_list))}
            )
            emb_dim_reducer = nn.Linear(
                self.covars_embeddings['0'].weight.shape[1], n_latent_shared
            )
            self.pert_encoder = emb_dim_reducer
        else:
            self.covars_embeddings = nn.ModuleDict()
            self.emb_projections = nn.ModuleDict()

            for i, n_cats in enumerate(self.n_cat_list):
                if _has_perturbation and i == perturbation_cov_idx:
                    pert_emb = PerturbationEmbedding(
                        predefined_pert_embeddings,
                        perturbation_category_names,
                        perturbation_combination_delimiter,
                    )
                    self.covars_embeddings[str(i)] = pert_emb
                    self.emb_projections[str(i)] = nn.Linear(
                        pert_emb.embedding_dim, n_latent_shared
                    )
                else:
                    self.covars_embeddings[str(i)] = nn.Embedding(n_cats, n_latent_shared)
                    self.emb_projections[str(i)] = nn.Identity()

            self.pert_encoder = nn.Identity()

        # Determine prior-encoder input size per covariate
        self._prior_input_dims = []
        for i in range(len(self.n_cat_list)):
            if _has_perturbation and i == perturbation_cov_idx:
                self._prior_input_dims.append(
                    self.covars_embeddings[str(i)].embedding_dim
                )
            elif use_custom_embs and not _has_perturbation:
                self._prior_input_dims.append(embeddings.shape[1])
            else:
                self._prior_input_dims.append(n_latent_shared)

        self.zs_num = len(self.n_cat_list)

        self.classifier_weights = classifier_weights
        if self.classifier_weights is not None:
            assert len(
                self.classifier_weights
                ) == self.zs_num, "classifier_weights should have the same length as the number of categocrical covariates."

        self.z_encoders_list = nn.ModuleList(
            [
                Encoder(
                    n_input_encoder + len(self.n_cat_list) * n_latent_shared,
                    n_latent_shared,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    distribution=latent_distribution,
                    use_batch_norm=use_batch_norm_encoder,
                    use_layer_norm=use_layer_norm_encoder,
                    var_activation=var_activation,
                    return_dist=True,
                    bias=bias,
                ).to(device)
            ]
        )

        self.z_encoders_list.extend(
            [
                Encoder(
                    n_input_encoder + len(self.n_cat_list) * n_latent_shared,
                    n_latent_attribute,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    distribution=latent_distribution,
                    use_batch_norm=use_batch_norm_encoder,
                    use_layer_norm=use_layer_norm_encoder,
                    var_activation=var_activation,
                    return_dist=True,
                    bias=bias,
                ).to(device)
                for k in range(self.zs_num)
            ]
        )

        self.z_prior_encoders_list = nn.ModuleList(
            [
                Encoder(
                    self._prior_input_dims[k],
                    n_latent_attribute,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    distribution=latent_distribution,
                    use_batch_norm=use_batch_norm_encoder,
                    use_layer_norm=use_layer_norm_encoder,
                    var_activation=var_activation,
                    return_dist=True,
                    bias=bias,
                ).to(device)
                for k in range(self.zs_num)
            ]
        )

        # Decoders

        self.x_decoders_list = nn.ModuleList(
            [
                DecoderSCVI(
                    n_latent_shared + len(self.n_cat_list) * n_latent_shared,
                    n_input,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    use_batch_norm=use_batch_norm_decoder,
                    use_layer_norm=use_layer_norm_decoder,
                    scale_activation="softmax",
                ).to(device)
            ]
        )

        self.x_decoders_list.extend(
            [
                DecoderSCVI(
                    n_latent_attribute * len(self.n_cat_list),
                    n_input,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    use_batch_norm=use_batch_norm_decoder,
                    use_layer_norm=use_layer_norm_decoder,
                    scale_activation="softmax",
                ).to(device)
                for k in range(self.zs_num)
            ]
        )

        self.n_latent = n_latent_shared + n_latent_attribute * self.zs_num

        self.s_classifiers_list = nn.ModuleList([])
        for i in range(self.zs_num):
            self.s_classifiers_list.append(
                Classifier(
                    n_input=n_latent_attribute,
                    n_labels=self.n_cat_list[i],
                    logits=True,
                ).to(device)
            )

    def _get_inference_input(
            self,
            tensors: dict[str, torch.Tensor]
        ) -> dict[str, torch.Tensor]:
        """
        Prepares the input for the inference step.

        Parameters
        ----------
        tensors : dict[str, torch.Tensor]
            Dictionary containing the input tensors.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing the processed input tensors for inference.
        """
        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key]

        x = tensors[REGISTRY_KEYS.X_KEY]

        input_dict = {
            "x": x,
            "cat_covs": cat_covs,
        }
        return input_dict

    def _get_generative_input(
            self,
            tensors: dict[str, torch.Tensor],
            inference_outputs: dict[str, torch.Tensor],
    ) -> dict[str, torch.Tensor]:
        """
        Prepares the input for the generative step.

        Parameters
        ----------
        tensors : dict[str, torch.Tensor]
            Dictionary containing the input tensors.
        inference_outputs : dict[str, torch.Tensor]
            Dictionary containing the outputs from the inference step.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing the processed input tensors for the generative step.
        """
        input_dict = {
            "z_shared": inference_outputs["z_shared"],
            "zs": inference_outputs["zs"],  # a list of all zs
            "library": inference_outputs["library"],
            "cat_covs": inference_outputs["cat_covs"],
        }
        return input_dict

    # ------------------------------------------------------------------
    # Embedding helper
    # ------------------------------------------------------------------
    def _get_covariate_embeddings(self, cat_covs: torch.Tensor):
        """Look up embeddings for every covariate and project them.

        Returns
        -------
        emb_flat
            ``(batch, n_covs * n_latent_shared)`` -- projected & flattened,
            ready to concatenate with gene expression for encoders / Dec_0.
        raw_embs
            List of length ``n_covs``, each ``(batch, raw_dim_i)`` -- the raw
            (unprojected) embeddings used as input for prior encoders.
        projected_3d
            ``(n_covs, batch, n_latent_shared)`` -- projected embeddings kept
            in 3-D so callers can easily select subsets for decoders.
        """
        raw_embs = []
        projected_embs = []
        for i, embedding_indices in enumerate(cat_covs.t()):
            raw = self.covars_embeddings[str(i)](embedding_indices.long())
            raw_embs.append(raw)
            projected = self.emb_projections[str(i)](raw)
            projected_embs.append(projected)

        # Legacy path (use_custom_embs without perturbation): apply batch pert_encoder
        if not isinstance(self.pert_encoder, nn.Identity):
            stacked = torch.stack(raw_embs, dim=1)          # (B, n_covs, raw_dim)
            projected_all = self.pert_encoder(stacked)       # (B, n_covs, n_latent_shared)
            projected_embs = list(projected_all.unbind(dim=1))

        projected_3d = torch.stack(projected_embs, dim=0)    # (n_covs, B, n_latent_shared)
        emb_flat = projected_3d.permute(1, 0, 2).reshape(
            projected_3d.shape[1], -1
        )  # (B, n_covs * n_latent_shared)
        return emb_flat, raw_embs, projected_3d

    # ------------------------------------------------------------------
    # Inference / generative / forward helpers
    # ------------------------------------------------------------------

    @auto_move_data
    def inference(self,
                  x,
                  cat_covs,
                  nullify_cat_covs_indices: Optional[List[int]] = None,
                  nullify_shared: Optional[bool] = False,
                  ) -> dict[str, torch.Tensor]:
        """
        Perform the inference step of the model.

        Parameters
        ----------
        x : torch.Tensor
            Input gene expression data.
        cat_covs : torch.Tensor
            Categorical covariates.
        nullify_cat_covs_indices : Optional[List[int]], optional
            Indices of categorical covariates to nullify, by default None.
        nullify_shared : Optional[bool], optional
            Whether to nullify the shared latent space, by default False.

        Returns
        -------
        dict[str, torch.Tensor]
            Dictionary containing the inference outputs.
        """
        nullify_cat_covs_indices = [] if nullify_cat_covs_indices is None else nullify_cat_covs_indices

        x_ = x
        library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        cat_in = torch.split(cat_covs, 1, dim=1)

        emb_flat, raw_embs, _ = self._get_covariate_embeddings(cat_covs)

        # the expression data and the embeddings are concatenated
        # and passed through the first encoder to get the shared latent space Z_0
        qz_shared, z_shared = self.z_encoders_list[0](torch.hstack((x_, emb_flat)))
        z_shared = z_shared.to(device)
    
        # zs
        encoders_outputs = []
        encoders_inputs = [torch.hstack((x_, emb_flat)) for _ in cat_in]

        for i in range(len(self.z_encoders_list) - 1):
            encoders_outputs.append(self.z_encoders_list[i + 1](encoders_inputs[i]))

        qzs = [enc_out[0] for enc_out in encoders_outputs]
        zs = [enc_out[1].to(device) for enc_out in encoders_outputs]
        
        # zs_prior (use raw embeddings as input -- prior encoders accept
        # the native embedding dimension for each covariate)
        encoders_prior_outputs = []
        for i in range(len(self.z_prior_encoders_list)):
            encoders_prior_outputs.append(self.z_prior_encoders_list[i](raw_embs[i]))

        qzs_prior = [enc_out[0] for enc_out in encoders_prior_outputs]
        zs_prior = [enc_out[1].to(device) for enc_out in encoders_prior_outputs]
        
        # nullify if required

        if nullify_shared:
            z_shared = torch.zeros_like(z_shared).to(device)

        for i in range(self.zs_num):
            if i in nullify_cat_covs_indices:
                zs[i] = torch.zeros_like(zs[i]).to(device)

        zs_concat = torch.cat(zs, dim=-1)
        z_concat = torch.cat([z_shared, zs_concat], dim=-1)

        output_dict = {
            "z_shared": z_shared,
            "zs": zs,
            "zs_prior": zs_prior,
            "qz_shared": qz_shared,
            "qzs": qzs,
            "qzs_prior": qzs_prior,
            "z_concat": z_concat,
            "library": library,
            "cat_covs": cat_covs,
        }
        return output_dict

    @auto_move_data
    def generative(
            self,
            z_shared,
            zs,
            library,
            cat_covs,
            ):
        """
        Perform the generative step of the model.

        Parameters
        ----------
        z_shared : torch.Tensor
            Shared latent space tensor.
        zs : list of torch.Tensor
            List of latent space tensors for each sensitive attribute.
        library : torch.Tensor
            Library size tensor.
        cat_covs : torch.Tensor
            Categorical covariates tensor.

        Returns
        -------
        dict
            Dictionary containing the generative outputs.
        """
        output_dict = {"px": []}

        z = [z_shared] + zs

        emb_flat, _, projected_3d = self._get_covariate_embeddings(cat_covs)

        # Create embeddings for all covariates except the ith one for the decoder
        all_cats_but_one = []
        for i in range(self.zs_num):
            cov_indices = list(set(range(self.zs_num)) - {i})
            ith_emb = projected_3d[cov_indices, :, :]         # (n-1, B, n_latent_shared)
            ith_emb = ith_emb.permute(1, 0, 2)                # (B, n-1, n_latent_shared)
            ith_emb = ith_emb.reshape(ith_emb.shape[0], -1)   # (B, (n-1)*n_latent_shared)
            all_cats_but_one.append(ith_emb)

        # Dec_0 takes all covariates; Dec_i takes all covariates except i
        dec_cats_in = [emb_flat] + all_cats_but_one

        for dec_count in range(self.zs_num + 1):
            x_decoder = self.x_decoders_list[dec_count]
            dec_covs = dec_cats_in[dec_count]
            x_decoder_input = z[dec_count]
            
            px_scale, px_r, px_rate, px_dropout = x_decoder(
                self.dispersion,
                torch.hstack((x_decoder_input, dec_covs)),
                library,
            )
            px_r = torch.exp(self.px_r)

            if self.gene_likelihood == "zinb":
                px = ZeroInflatedNegativeBinomial(
                    mu=px_rate,
                    theta=px_r,
                    zi_logits=px_dropout,
                    scale=px_scale,
                )
            elif self.gene_likelihood == "nb":
                px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
            elif self.gene_likelihood == "poisson":
                px = Poisson(px_rate, scale=px_scale)

            output_dict["px"] += [px]
        return output_dict

    def sub_forward(self, idx,
                    x, cat_covs,
                    detach_x=False,
                    detach_z=False):
        """
        Performs forward (inference + generative) only on encoder/decoder idx.

        Parameters
        ----------
        idx : int
            Index of encoder/decoder in [1, ..., self.zs_num].
        x : torch.Tensor
            Input gene expression data.
        cat_covs : torch.Tensor
            Categorical covariates.
        detach_x : bool, optional
            Whether to detach the input tensor `x`, by default False.
        detach_z : bool, optional
            Whether to detach the latent representation `z`, by default False.

        Returns
        -------
        torch.distributions.Distribution
            The reconstructed gene expression distribution.
        """
        x_ = x
        if detach_x:
            x_ = x.detach()

        library = torch.log(x_.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        emb_flat, _, projected_3d = self._get_covariate_embeddings(cat_covs)
        
        qz, z = (self.z_encoders_list[idx](torch.hstack((x_, emb_flat))))
        
        if detach_z:
            z = z.detach()

        cov_indices = list(set(range(self.zs_num)) - {idx - 1})
        ith_emb = projected_3d[cov_indices, :, :]
        ith_emb = ith_emb.permute(1, 0, 2)
        ith_emb = ith_emb.reshape(ith_emb.shape[0], -1)

        x_decoder = self.x_decoders_list[idx]

        px_scale, px_r, px_rate, px_dropout = x_decoder(
            self.dispersion,
            torch.hstack((z, ith_emb)),
            library,
        )
        px_r = torch.exp(self.px_r)

        if self.gene_likelihood == "zinb":
            px = ZeroInflatedNegativeBinomial(
                mu=px_rate,
                theta=px_r,
                zi_logits=px_dropout,
                scale=px_scale,
            )
        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
        elif self.gene_likelihood == "poisson":
            px = Poisson(px_rate, scale=px_scale)
        return px

    def classification_logits(self, inference_outputs):
        """
        Compute classification logits for each sensitive attribute.

        Parameters
        ----------
        inference_outputs : dict[str, torch.Tensor]
            Dictionary containing the outputs from the inference step.

        Returns
        -------
        list[torch.Tensor]
            List of logits for each sensitive attribute.
        """
        zs = inference_outputs["zs"]
        logits = []
        for i in range(self.zs_num):
            s_i_classifier = self.s_classifiers_list[i]
            logits_i = s_i_classifier(zs[i])
            logits += [logits_i]

        return logits


    def sub_forward_cf(
                    self, 
                    idx,
                    x, 
                    cat_covs,
                    cat_covs_cf=None,
                    detach_x=False,
                    detach_z=False):
        """
        Perform counterfactual forward pass for a specific encoder/decoder.

        Parameters
        ----------
        idx : int
            Index of the encoder/decoder to use.
        x : torch.Tensor
            Input gene expression data.
        cat_covs : torch.Tensor
            Original categorical covariates.
        cat_covs_cf : torch.Tensor, optional
            Counterfactual categorical covariates. If None, use original covariates.
        detach_x : bool, optional
            Whether to detach the input tensor `x`.
        detach_z : bool, optional
            Whether to detach the latent representation `z`.

        Returns
        -------
        torch.distributions.Distribution
            The reconstructed gene expression distribution.
        """
        x_ = x
        if detach_x:
            x_ = x.detach()

        library = torch.log(x_.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        emb_flat, _, projected_3d = self._get_covariate_embeddings(cat_covs)

        qz, z = (self.z_encoders_list[idx](torch.hstack((x_, emb_flat))))

        if detach_z:
            z = z.detach()

        if cat_covs_cf is None:
            cov_indices = list(set(range(self.zs_num)) - {idx - 1})
            ith_emb = projected_3d[cov_indices, :, :]
            ith_emb = ith_emb.permute(1, 0, 2)
            ith_emb = ith_emb.reshape(ith_emb.shape[0], -1)
        else:
            # If there is only one covariate, the attribute-specific decoder (decoder_i for i > 0)
            # cannot be used for counterfactual predictions.
            if self.zs_num == 1:
                return None
            _, _, cf_proj_3d = self._get_covariate_embeddings(cat_covs_cf)
            cov_indices = [i for i in range(self.zs_num) if i != idx - 1]
            ith_emb = cf_proj_3d[cov_indices, :, :]
            ith_emb = ith_emb.permute(1, 0, 2)
            ith_emb = ith_emb.reshape(ith_emb.shape[0], -1)

        x_decoder = self.x_decoders_list[idx]

        px_scale, px_r, px_rate, px_dropout = x_decoder(
            self.dispersion,
            torch.hstack((z, ith_emb)),
            library,
        )
        px_r = torch.exp(self.px_r)
        cf_difference = (cat_covs == cat_covs_cf).to(device)
        px_scale = px_scale[cf_difference[:, idx-1]]
        px_rate = px_rate[cf_difference[:, idx-1]]
        px_dropout = px_dropout[cf_difference[:, idx-1]]
        if px_scale.shape[0] == 0:
            return None

        if self.gene_likelihood == "zinb":
            px = ZeroInflatedNegativeBinomial(
                mu=px_rate,
                theta=px_r,
                zi_logits=px_dropout,
                scale=px_scale,
            )
        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
        elif self.gene_likelihood == "poisson":
            px = Poisson(px_rate, scale=px_scale)
        return px


    def sub_forward_cf_z0(self,
                    x,
                    cat_covs,
                    cat_covs_cf=None,
                    detach_x=False,
                    detach_z=False):
        """
        Performs counterfactual forward (inference + generative) only on encoder/decoder 0.

        Parameters
        ----------
        x : torch.Tensor
            Input gene expression data.
        cat_covs : torch.Tensor
            Original categorical covariates.
        cat_covs_cf : torch.Tensor, optional
            Counterfactual categorical covariates. If None, use original covariates.
        detach_x : bool, optional
            Whether to detach the input tensor `x`, by default False.
        detach_z : bool, optional
            Whether to detach the latent representation `z`, by default False.

        Returns
        -------
        torch.distributions.Distribution
            The reconstructed gene expression distribution.
        """
        x_ = x
        if detach_x:
            x_ = x.detach()

        library = torch.log(x_.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        emb_flat, _, projected_3d = self._get_covariate_embeddings(cat_covs)
        
        qz, z = (self.z_encoders_list[0](torch.hstack((x_, emb_flat))))
        
        if detach_z:
            z = z.detach()

        if cat_covs_cf is None:
            ith_emb = projected_3d.permute(1, 0, 2)           # (B, n_covs, n_latent_shared)
            ith_emb = ith_emb.reshape(ith_emb.shape[0], -1)   # (B, n_covs * n_latent_shared)
        else:
            cf_emb_flat, _, _ = self._get_covariate_embeddings(cat_covs_cf)
            ith_emb = cf_emb_flat

        # decoder 0 takes all covariates and is unaware of them in its latent
        x_decoder = self.x_decoders_list[0]

        px_scale, px_r, px_rate, px_dropout = x_decoder(
            self.dispersion,
            torch.hstack((z, ith_emb)),
            library,
        )
        px_r = torch.exp(self.px_r)

        if self.gene_likelihood == "zinb":
            px = ZeroInflatedNegativeBinomial(
                mu=px_rate,
                theta=px_r,
                zi_logits=px_dropout,
                scale=px_scale,
            )
        elif self.gene_likelihood == "nb":
            px = NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
        elif self.gene_likelihood == "poisson":
            px = Poisson(px_rate, scale=px_scale)
        return px


    def sub_forward_cf_avg(
            self,
            x,
            cat_covs,
            cat_covs_cf=None,
            detach_x=False,
            detach_z=False):
        """
        Perform counterfactual forward pass for all encoders/decoders and average the results.

        Parameters
        ----------
        x : torch.Tensor
            Input gene expression data.
        cat_covs : torch.Tensor
            Original categorical covariates.
        cat_covs_cf : torch.Tensor, optional
            Counterfactual categorical covariates. If None, use original covariates.
        detach_x : bool, optional
            Whether to detach the input tensor `x`, by default False.
        detach_z : bool, optional
            Whether to detach the latent representation `z`, by default False.

        Returns
        -------
        torch.Tensor
            The average counterfactual gene expression.
        list
            List of reconstructed gene expression distributions.
        """
        xs = []
        pxs = []

        for i in range(self.zs_num):
            # Doing counterfactual forward pass for each encoder/decoder 1, 2, ..., zs_num
            px = self.sub_forward_cf(i+1, x, cat_covs, cat_covs_cf, detach_x, detach_z)
            pxs.append(px)
            if px is None:
                # all cells in the batch have their covariate i changed
                # no output from decoder i+1
                continue
            xs.append(px.mean)

        # Doing counterfactual forward pass for encoder/decoder 0
        px = self.sub_forward_cf_z0(x, cat_covs, cat_covs_cf, detach_x, detach_z)
        pxs.append(px)
        xs.append(px.mean)

        # we take the average of the counterfactual gene expression
        # predictions from all the encoder/decoders to get the final counterfactual gene expression
        x_avg = torch.mean(torch.cat(xs), dim=0)
        return x_avg, pxs


    def compute_clf_metrics(self, logits, cat_covs):
        """
        Compute classification metrics: Cross-Entropy (CE) loss, Accuracy, and F1 score.

        Parameters
        ----------
        logits : list[torch.Tensor]
            List of logits for each sensitive attribute.
        cat_covs : torch.Tensor
            Tensor containing the categorical covariates.

        Returns
        -------
        tuple
            A tuple containing the mean CE loss, accuracy, and F1 score.
        """
        # CE, ACC, F1
        cats = torch.split(cat_covs, 1, dim=1)
        ce_losses = []
        accuracy_scores = []
        f1_scores = []
        if len(logits) == self.zs_num:
            adversarial = False
        else:
            adversarial = True
        for i in range(self.zs_num):
            s_i = one_hot_cat([self.n_cat_list[i]], cats[i]).to(device)
            if adversarial:
                for j in range(self.zs_num):
                    logits_index = i * self.zs_num + j
                    if self.classifier_weights is not None:
                        weight = torch.tensor(self.classifier_weights[i]).to(device)
                        ce_losses += [F.cross_entropy(logits[logits_index], s_i, weight=weight)]
                    else:
                        ce_losses += [F.cross_entropy(logits[logits_index], s_i)]
                    kwargs = {"task": "multiclass", "num_classes": self.n_cat_list[i]}
                    predicted_labels = torch.argmax(logits[logits_index], dim=-1, keepdim=True).to(device)
                    acc = Accuracy(**kwargs).to(device)
                    accuracy_scores.append(acc(predicted_labels, cats[i]).to(device))
                    F1 = F1Score(**kwargs).to(device)
                    f1_scores.append(F1(predicted_labels, cats[i]).to(device))   
            else:     
                if self.classifier_weights is not None:
                    weight = torch.tensor(self.classifier_weights[i]).to(device)
                    ce_losses += [F.cross_entropy(logits[i], s_i, weight=weight)]
                else:
                    ce_losses += [F.cross_entropy(logits[i], s_i)]
                kwargs = {"task": "multiclass", "num_classes": self.n_cat_list[i]}
                predicted_labels = torch.argmax(logits[i], dim=-1, keepdim=True).to(device)
                acc = Accuracy(**kwargs).to(device)
                accuracy_scores.append(acc(predicted_labels, cats[i]).to(device))
                F1 = F1Score(**kwargs).to(device)
                f1_scores.append(F1(predicted_labels, cats[i]).to(device))

        ce_loss_mean = sum(ce_losses) / len(ce_losses)
        accuracy = sum(accuracy_scores) / len(accuracy_scores)
        f1 = sum(f1_scores) / len(f1_scores)

        return ce_loss_mean, accuracy, f1

    def loss(
            self,
            tensors,
            inference_outputs,
            generative_outputs,
            recon_weight: Tunable[Union[float, int]], # RECONST_LOSS_X weight
            cf_weight: Tunable[Union[float, int]],  # RECONST_LOSS_X_CF weight
            beta: Tunable[Union[float, int]],  # KL Zi weight
            clf_weight: Tunable[Union[float, int]],  # Si classifier weight
            n_cf: Tunable[int],  # number of X_cf recons (X_cf = a random permutation of X)
            kl_weight: float = 1.0,
            ensemble_method_cf=True,
    ):
        """
        Compute the loss for the model.

        Parameters
        ----------
        tensors : dict
            Dictionary containing the input tensors.
        inference_outputs : dict
            Dictionary containing the outputs from the inference step.
        generative_outputs : dict
            Dictionary containing the outputs from the generative step.
        recon_weight : Tunable[Union[float, int]]
            Weight for the reconstruction loss of X.
        cf_weight : Tunable[Union[float, int]]
            Weight for the reconstruction loss of X_cf.
        beta : Tunable[Union[float, int]]
            Weight for the KL divergence of Zi.
        clf_weight : Tunable[Union[float, int]]
            Weight for the Si classifier loss.
        n_cf : Tunable[int]
            Number of X_cf reconstructions (X_cf = a random permutation of X).
        kl_weight : float, optional
            Weight for the KL divergence, by default 1.0.
        ensemble_method_cf : bool, optional
            Whether to use the new counterfactual method, by default True.

        Returns
        -------
        dict
            Dictionary containing the computed losses and metrics.
        """
        # reconstruction loss X
        x = tensors[REGISTRY_KEYS.X_KEY]

        reconst_loss_x_list = [-torch.mean(px.log_prob(x).mean(-1)) for px in generative_outputs["px"]]
        reconst_loss_x_dict = {'x_' + str(i): reconst_loss_x_list[i] for i in range(len(reconst_loss_x_list))}
        reconst_loss_x = sum(reconst_loss_x_list) / len(reconst_loss_x_list)

        # reconstruction loss X' (counterfactual)
        cat_covs = tensors[REGISTRY_KEYS.CAT_COVS_KEY]
        batch_size = x.size(dim=0)

        reconst_loss_x_cf_list = []

        for _ in range(n_cf):
            # shuffle cell covariates within batch
            idx_shuffled = list(range(batch_size))

            # choose a random permutation of X as X_cf
            if 'cluster' in tensors.keys():
                # if the data is clustered, we shuffle the data within each cluster
                # meaning each index will be replaced by another index within the same cluster
                # this is to ensure that the counterfactuals are still within the same cluster
                # in cases such as when the Cell Type is not given as a covariate
                cluster = tensors['cluster']
                cluster_unique = torch.unique(cluster)
                for c in cluster_unique:
                    idx_c = torch.where(cluster == c)[0]
                    idx_c_shuffled = idx_c[torch.randperm(idx_c.size(0))]
                    for i, idx in enumerate(idx_c):
                        idx_shuffled[idx] = idx_c_shuffled[i]
            else:
                # if the data is not clustered, we shuffle the data randomly
                random.shuffle(idx_shuffled)
            idx_shuffled = torch.tensor(idx_shuffled).to(device)

            x_ = x
            # x_cf is a random permutation of x based on idx_shuffled
            x_cf = torch.index_select(x, 0, idx_shuffled).to(device)

            cat_cov_ = cat_covs # batch_size x n_cat_covs
            # cat_cov_cf is a random permutation of cat_covs based on idx_shuffled
            cat_cov_cf = torch.index_select(cat_covs, 0, idx_shuffled).to(device)

            cat_cov_cf_split = torch.split(cat_cov_cf, 1, dim=1)
            # cat_cov_cf_split is a list of tensors, each tensor is a column of cat_cov_cf
            # i.e. covariate values for each covariate in the batch

            # a random ordering for diffusing through n VAEs
            perm = list(range(self.zs_num))
            random.shuffle(perm)

            if ensemble_method_cf:
                # This is going to tell us which covariates are different between
                # cat_covs and cat_cov_cf in each row of cat_covs and cat_cov_cf
                cf_difference = (cat_covs == cat_cov_cf).to(device) # batch_size x n_cat_covs: bool
                # Add one column of all True to the end of cf_difference: batch_size x (n_cat_covs+1)
                cf_difference = torch.cat([cf_difference, torch.ones_like(cf_difference[:, 0]).unsqueeze(1)], dim=1).type(torch.bool)
                # details in sub_forward_cf_avg, sub_forward_cf, and sub_forward_cf_z0
                # in short, pxs is a list of counterfactually predicted gene expression distributions from all encoder/decoders
                _, pxs = self.sub_forward_cf_avg(x_, cat_cov_, cat_cov_cf)

                # some dists in pxs might be None if all cells in the batch have their encoder decoder related covariate changed
                # we only compare the cells in the batch for each enc/dec where the corresponding covariate has not been changed
                # in enc/dec 0 however, since there is no problem with changing even all the covariates, we can use all the cells (that's why we added a column of all True to cf_difference)
                log_probs = [px_.log_prob(x_cf[cf_difference[:, i]])
                             for i, px_ in enumerate(pxs) if px_ is not None]
                probs = [torch.exp(log_prob) for log_prob in log_probs]
                mean_probs = torch.mean(torch.cat(probs), dim=0)
                nll = -torch.log(mean_probs)
                reconst_loss_x_cf_list.append(torch.mean(nll))
                
            else:
                for idx in perm:
                    # cat_cov_[idx] (possibly) changes to cat_cov_cf[idx]
                    cat_cov_split = list(torch.split(cat_cov_, 1, dim=1))
                    cat_cov_split[idx] = cat_cov_cf_split[idx]
                    cat_cov_ = torch.cat(cat_cov_split, dim=1)
                    # use enc/dec idx+1 to get px_ and feed px_.mean as the next x_
                    px_ = self.sub_forward(idx + 1, x_, cat_cov_)
                    x_ = px_.mean

                reconst_loss_x_cf_list.append(-torch.mean(px_.log_prob(x_cf).sum(-1)))

        reconst_loss_x_cf = sum(reconst_loss_x_cf_list) / n_cf

        # KL divergence Z

        kl_z_list = [torch.mean(kl(qzs, qzs_prior).sum(dim=1)) for qzs, qzs_prior in
                     zip(inference_outputs["qzs"], inference_outputs["qzs_prior"])]

        kl_z_dict = {'z_' + str(i+1): kl_z_list[i] for i in range(len(kl_z_list))}
        kl_loss = sum(kl_z_list) / len(kl_z_list)

        # classification metrics: CE, ACC, F1

        logits = self.classification_logits(inference_outputs)
        ce_loss_mean, accuracy, f1 = self.compute_clf_metrics(logits, cat_covs)

        # total loss
        loss = reconst_loss_x * recon_weight + \
               reconst_loss_x_cf * cf_weight + \
               kl_loss * kl_weight * beta + \
               ce_loss_mean * clf_weight

        loss_dict = {
            LOSS_KEYS.LOSS: loss,
            LOSS_KEYS.RECONST_LOSS_X: reconst_loss_x_dict,
            LOSS_KEYS.RECONST_LOSS_X_CF: reconst_loss_x_cf,
            LOSS_KEYS.KL_Z: kl_z_dict,
            LOSS_KEYS.CLASSIFICATION_LOSS: ce_loss_mean,
            LOSS_KEYS.ACCURACY: accuracy,
            LOSS_KEYS.F1: f1
        }

        return loss_dict
