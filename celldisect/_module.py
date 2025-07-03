import random
from typing import Callable, Iterable, Literal, Optional, Union

import torch
import torch.nn.functional as F
from torch import nn
from torch.distributions import kl_divergence as kl, Normal
from torchmetrics import Accuracy, F1Score

from scvi import REGISTRY_KEYS
from scvi.autotune._types import Tunable
from scvi.distributions import NegativeBinomial, Poisson, ZeroInflatedNegativeBinomial
from scvi.module.base import BaseModuleClass, auto_move_data
from scvi.nn import DecoderSCVI, Encoder

torch.backends.cudnn.benchmark = True
from .utils import *
from scvi.module._classifier import Classifier

dim_indices = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'


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
    gene_likelihood : Tunable[Literal["zinb", "nb", "poisson", "normal"]], optional
        One of 'nb' (Negative binomial distribution), 'zinb' (Zero-inflated negative binomial distribution), 'poisson' (Poisson distribution), or 'normal' (Normal distribution), by default "zinb".
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
            gene_likelihood: Tunable[Literal["zinb", "nb", "poisson", "normal"]] = "zinb",
            latent_distribution: Tunable[Literal["normal", "ln"]] = "normal",
            deeply_inject_covariates: Tunable[bool] = True,
            use_batch_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "both",
            use_layer_norm: Tunable[Literal["encoder", "decoder", "none", "both"]] = "none",
            var_activation: Optional[Callable] = None,
            use_custom_embs: bool = False,
            embeddings: Union[torch.Tensor, List[torch.Tensor]] = None,
            classifier_weights: Optional[list] = None,
    ):
        super().__init__()
        self.dispersion = "gene"
        self.n_latent_shared = n_latent_shared
        self.n_latent_attribute = n_latent_attribute
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        # Automatically deactivate if useless
        self.latent_distribution = latent_distribution

        self.px_r = torch.nn.Parameter(torch.randn(n_input))

        use_batch_norm_encoder = use_batch_norm == "encoder" or use_batch_norm == "both"
        use_batch_norm_decoder = use_batch_norm == "decoder" or use_batch_norm == "both"
        use_layer_norm_encoder = use_layer_norm == "encoder" or use_layer_norm == "both"
        use_layer_norm_decoder = use_layer_norm == "decoder" or use_layer_norm == "both"

        # Encoders

        n_input_encoder = n_input

        self.n_cat_list = list([] if n_cats_per_cov is None else n_cats_per_cov)
        if use_custom_embs:
            self.covars_embeddings = nn.ModuleDict(
                {
                    str(key): torch.nn.Embedding(embedding.shape[0], embedding.shape[1])
                    for key, embedding in enumerate([embeddings])
                }
            )
            self.covars_embeddings['0'].weight.data.copy_(embeddings)
            self.covars_embeddings['0'].weight.requires_grad = False
        else:
            self.covars_embeddings = nn.ModuleDict(
                {
                    str(key): torch.nn.Embedding(unique_covars, n_latent_shared)
                    for key, unique_covars in enumerate(self.n_cat_list)
                }
            )

        emb_dim_reducer = nn.Linear(self.covars_embeddings['0'].weight.shape[1], n_latent_shared)
        self.pert_encoder = emb_dim_reducer if use_custom_embs else nn.Identity()

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
                    # n_cat_list=self.n_cat_list,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    distribution=latent_distribution,
                    # inject_covariates=deeply_inject_covariates,
                    use_batch_norm=use_batch_norm_encoder,
                    use_layer_norm=use_layer_norm_encoder,
                    var_activation=var_activation,
                    return_dist=True,
                ).to(device)
            ]
        )

        self.z_encoders_list.extend(
            [
                Encoder(
                    n_input_encoder + len(self.n_cat_list) * n_latent_shared,
                    n_latent_attribute,
                    # n_cat_list=self.n_cat_list,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    distribution=latent_distribution,
                    # inject_covariates=deeply_inject_covariates,
                    use_batch_norm=use_batch_norm_encoder,
                    use_layer_norm=use_layer_norm_encoder,
                    var_activation=var_activation,
                    return_dist=True,
                ).to(device)
                for k in range(self.zs_num)
            ]
        )

        self.z_prior_encoders_list = nn.ModuleList(
            [
                Encoder(
                    n_latent_shared if use_custom_embs == False else embeddings.shape[1],
                    n_latent_attribute,
                    # n_cat_list=[self.n_cat_list[k]],
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    dropout_rate=dropout_rate,
                    distribution=latent_distribution,
                    # inject_covariates=deeply_inject_covariates,
                    use_batch_norm=use_batch_norm_encoder,
                    use_layer_norm=use_layer_norm_encoder,
                    var_activation=var_activation,
                    return_dist=True,
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
                    # n_cat_list=self.n_cat_list,
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    # inject_covariates=deeply_inject_covariates,
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
                    # n_cat_list=[self.n_cat_list[i] for i in range(len(self.n_cat_list)) if i != k],
                    n_layers=n_layers,
                    n_hidden=n_hidden,
                    # inject_covariates=deeply_inject_covariates,
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

    def _validate_gene_likelihood_data_compatibility(self, x: torch.Tensor):
        """
        Validate that the gene likelihood is compatible with the input data.
        
        Parameters
        ----------
        x : torch.Tensor
            Input gene expression data
            
        Raises
        ------
        ValueError
            If there's a mismatch between gene likelihood and data type
        """
        if self.gene_likelihood in ["zinb", "nb", "poisson"]:
            if torch.any(x < 0):
                raise ValueError(
                    f"Gene likelihood '{self.gene_likelihood}' requires non-negative data, "
                    f"but found negative values. Consider using 'normal' for log-normalized or "
                    f"continuous data."
                )
            if torch.any(x != torch.floor(x)) and self.gene_likelihood != "poisson":
                # Allow non-integer for poisson as it can handle continuous rate parameters
                import warnings
                warnings.warn(
                    f"Gene likelihood '{self.gene_likelihood}' typically expects count data "
                    f"(integers), but found non-integer values. This might indicate "
                    f"log-normalized data - consider using 'normal' distribution.",
                    UserWarning
                )
        elif self.gene_likelihood == "normal":
            if torch.all(x >= 0) and torch.all(x == torch.floor(x)):
                import warnings
                warnings.warn(
                    f"Gene likelihood 'normal' is used with what appears to be count data "
                    f"(non-negative integers). Consider using 'zinb', 'nb', or 'poisson' "
                    f"for count data.",
                    UserWarning
                )

    def _compute_library_size(self, x: torch.Tensor) -> torch.Tensor:
        """
        Compute library size based on gene likelihood type.
        
        Parameters
        ----------
        x : torch.Tensor
            Input gene expression data
            
        Returns
        -------
        torch.Tensor
            Library size tensor
        """
        if self.gene_likelihood in ["zinb", "nb", "poisson"]:
            # For count data, library size is log of total counts per cell
            return torch.log(x.sum(1) + 1e-6).unsqueeze(1)  # Add small epsilon to avoid log(0)
        else:
            # For continuous data (normal), library size is not meaningful
            # Return zeros with same shape
            return torch.zeros(x.shape[0], 1, device=x.device)

    def _preprocess_input(self, x: torch.Tensor) -> torch.Tensor:
        """
        Preprocess input based on gene likelihood and log_variational setting.
        
        Parameters
        ----------
        x : torch.Tensor
            Input gene expression data
            
        Returns
        -------
        torch.Tensor
            Preprocessed input data
        """
        if self.log_variational and self.gene_likelihood in ["zinb", "nb", "poisson"]:
            # Apply log(1+x) transformation only for count data
            return torch.log(1 + x)
        else:
            # For normal distribution or when log_variational is False, return as-is
            return x

    def _create_distribution(self, px_scale: torch.Tensor, px_r: torch.Tensor, 
                           px_rate: torch.Tensor, px_dropout: torch.Tensor) -> torch.distributions.Distribution:
        """
        Create the appropriate distribution based on gene_likelihood.
        
        Parameters
        ----------
        px_scale : torch.Tensor
            Scale parameter
        px_r : torch.Tensor  
            Dispersion parameter (for count distributions)
        px_rate : torch.Tensor
            Rate/mean parameter
        px_dropout : torch.Tensor
            Dropout parameter (for zero-inflated distributions)
            
        Returns
        -------
        torch.distributions.Distribution
            The appropriate distribution object
        """
        if self.gene_likelihood == "zinb":
            return ZeroInflatedNegativeBinomial(
                mu=px_rate,
                theta=px_r,
                zi_logits=px_dropout,
                scale=px_scale,
            )
        elif self.gene_likelihood == "nb":
            return NegativeBinomial(mu=px_rate, theta=px_r, scale=px_scale)
        elif self.gene_likelihood == "poisson":
            return Poisson(px_rate, scale=px_scale)
        elif self.gene_likelihood == "normal":
            # For normal distribution, use px_rate as mean and px_r as log std
            # This follows scvi-tools conventions for continuous distributions
            std = torch.exp(px_r)  # Ensure positive std
            return Normal(px_rate, std)
        else:
            raise ValueError(f"Unsupported gene likelihood: {self.gene_likelihood}")

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

        # Validate data compatibility with gene likelihood
        self._validate_gene_likelihood_data_compatibility(x)
        
        # Compute library size and preprocess input using helper methods
        library = self._compute_library_size(x)
        x_ = self._preprocess_input(x)

        # cat_covs are shaped like (batch_size, n_cat_covs)
        # we split them into a list of n_cat_covs (batch_size, 1) tensors
        # where each tensor is a column of cat_covs (each will contain one categorical covariate)
        cat_in = torch.split(cat_covs, 1, dim=1)
        # z_shared
        emb = []
        for i, embedding in enumerate(cat_covs.t()):
            # emb will be a list of embeddings for each categorical covariate
            # for the batch. Each embedding is of shape (batch_size, emb_dim)
            emb.append(self.covars_embeddings[str(i)](embedding.long()))

        prior_emb_in = emb[:] # save a copy for later
        emb = torch.stack(emb, dim=0) # unique_covs x batch_size x emb_dim
        emb = torch.permute(emb, (1, 0, 2)) # batch_size x unique_covs x emb_dim
        emb = self.pert_encoder(emb)
        emb = emb.reshape(emb.shape[0], -1) # batch_size x (unique_covs * emb_dim)
        # each row in emb now represents the embedding for all covariates in that single cell

        # the expression data and the embeddings are concatenated
        # and passed through the first encoder to get the shared latent space Z_0
        qz_shared, z_shared = self.z_encoders_list[0](torch.hstack((x_, emb)))
        z_shared = z_shared.to(device)
    
        # zs
        encoders_outputs = []
        encoders_inputs = [torch.hstack((x_, emb)) for _ in cat_in]

        for i in range(len(self.z_encoders_list) - 1):
            encoders_outputs.append(self.z_encoders_list[i + 1](encoders_inputs[i]))

        qzs = [enc_out[0] for enc_out in encoders_outputs]
        zs = [enc_out[1].to(device) for enc_out in encoders_outputs]
        
        # zs_prior
        encoders_prior_outputs = []
        for i in range(len(self.z_prior_encoders_list)):
            encoders_prior_outputs.append(self.z_prior_encoders_list[i](prior_emb_in[i]))

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

        cats_splits = torch.split(cat_covs, 1, dim=1)
        emb = []
        for i, embedding in enumerate(cat_covs.t()):
            emb.append(self.covars_embeddings[str(i)](embedding.long()))
        emb = torch.stack(emb, dim=0) # unique_covs x batch_size x emb_dim
        full_embs_ubd = emb.clone() # unique_covs x batch_size x emb_dim
        emb = torch.permute(emb, (1, 0, 2)) # batch_size x unique_covs x emb_dim
        emb = self.pert_encoder(emb)
        emb = emb.reshape(emb.shape[0], -1) # batch_size x (unique_covs * emb_dim)
        # each row in emb now represents the embedding for all covariates in that single cell

        # Create embeddings for all covariates except the ith one for the decoder
        all_cats_but_one = []
        for i in range(self.zs_num): # for each categorical covariate
            cov_indices = list(set(range(self.zs_num)) - {i}) # all indices except i
            # embeddings for all covariates except the ith one
            ith_emb = full_embs_ubd[cov_indices, :, :] # (unique_covs-1) x batch_size x emb_dim
            ith_emb = torch.permute(ith_emb, (1, 0, 2)) # batch_size x (unique_covs-1) x emb_dim
            ith_emb = ith_emb.reshape(ith_emb.shape[0], -1) # batch_size x ((unique_covs-1) * emb_dim)
            all_cats_but_one.append(ith_emb)

        # Covariate embeddings for decoders
        # Dec_0 takes all covariates
        # Dec_i takes all covariates except the ith one
        dec_cats_in = [emb] + all_cats_but_one

        for dec_count in range(self.zs_num + 1):
            # Decoder_i
            x_decoder = self.x_decoders_list[dec_count]
            # Decoder_i covariates
            dec_covs = dec_cats_in[dec_count]
            # Decoder_i latent input
            x_decoder_input = z[dec_count]
            
            px_scale, px_r, px_rate, px_dropout = x_decoder(
                self.dispersion,
                torch.hstack((x_decoder_input, dec_covs)),
                library,
                # *dec_covs
            )
            px_r = torch.exp(self.px_r)

            # Use helper method to create distribution
            px = self._create_distribution(px_scale, px_r, px_rate, px_dropout)
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

        # Use helper methods for consistent preprocessing
        library = self._compute_library_size(x_)
        x_ = self._preprocess_input(x_)

        cat_in = torch.split(cat_covs, 1, dim=1)
        
        emb = []
        for i, embedding in enumerate(cat_covs.t()):
            emb.append(self.covars_embeddings[str(i)](embedding.long()))
        emb = torch.stack(emb, dim=0)
        full_embs_ubd = emb.clone() # unique_covs x batch_size x emb_dim
        emb = torch.permute(emb, (1, 0, 2))
        emb = self.pert_encoder(emb)
        emb = emb.reshape(emb.shape[0], -1)
        
        qz, z = (self.z_encoders_list[idx](torch.hstack((x_, emb))))
        
        if detach_z:
            z = z.detach()

        for i in range(self.zs_num):
            cov_indices = list(set(list(range(self.zs_num)))-set([idx-1]))
            ith_emb = full_embs_ubd[cov_indices, :, :]
            ith_emb = torch.permute(ith_emb, (1, 0, 2))
            ith_emb = ith_emb.reshape(ith_emb.shape[0], -1)    

        x_decoder = self.x_decoders_list[idx]

        px_scale, px_r, px_rate, px_dropout = x_decoder(
            self.dispersion,
            torch.hstack((z, ith_emb)),
            library,
            # *dec_cats
        )
        px_r = torch.exp(self.px_r)

        # Use helper method to create distribution
        px = self._create_distribution(px_scale, px_r, px_rate, px_dropout)
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

        # Use helper methods for consistent preprocessing
        library = self._compute_library_size(x_)
        x_ = self._preprocess_input(x_)

        emb = []
        for i, embedding in enumerate(cat_covs.t()):
            emb.append(self.covars_embeddings[str(i)](embedding.long()))
        emb = torch.stack(emb, dim=0)
        full_embs_ubd = emb.clone() # unique_covs x batch_size x emb_dim
        emb = torch.permute(emb, (1, 0, 2))
        emb = self.pert_encoder(emb)
        emb = emb.reshape(emb.shape[0], -1)

        qz, z = (self.z_encoders_list[idx](torch.hstack((x_, emb))))
        # TLDR: we encode the original gene expression and covariates of the cell using encoder idx
        # so z is the latent representation of the original cell, nothing about the counterfactual yet

        if detach_z:
            z = z.detach()

        if cat_covs_cf is None:
            cov_indices = list(set(range(self.zs_num)) - {idx - 1})
            ith_emb = full_embs_ubd[cov_indices, :, :]
            ith_emb = torch.permute(ith_emb, (1, 0, 2))
            ith_emb = self.pert_encoder(ith_emb)
            ith_emb = ith_emb.reshape(ith_emb.shape[0], -1)
        else:
            # Here's where the counterfactual decoding is happening
            ith_emb = []
            for i, embedding in enumerate(cat_covs_cf.t()):
                if i == idx-1:
                    continue
                ith_emb.append(self.covars_embeddings[str(i)](embedding.long()))
            ith_emb = torch.stack(ith_emb, dim=0) # (n_cat_covs-1) x batch_size x emb_dim
            ith_emb = torch.permute(ith_emb, (1, 0, 2)) # batch_size x (n_cat_covs-1) x emb_dim
            ith_emb = self.pert_encoder(ith_emb)
            ith_emb = ith_emb.reshape(ith_emb.shape[0], -1) # batch_size x ((n_cat_covs-1) * emb_dim)

        x_decoder = self.x_decoders_list[idx]

        px_scale, px_r, px_rate, px_dropout = x_decoder(
            self.dispersion,
            torch.hstack((z, ith_emb)),
            library,
            # *dec_cats
        )
        px_r = torch.exp(self.px_r)
        cf_difference = (cat_covs == cat_covs_cf).to(device)
        # px is of shape (batch_size, n_input)
        # cf_difference[:, idx-1] is a boolean tensor of shape (batch_size,) where True means
        # the covariate is the same as the original covariate in that cell(hasn't changed)
        # This is important:
        # We are currently in enc/dec idx, latent idx is aware of the covariate idx-1
        # so if covariate idx-1 has been changed in the cf, outputs from dec idx will be incorrect
        # because they are not getting the new covariate idx-1 value when decoding (decoder i gets covariates except i)
        # decoder idx is going to keep the original covariate idx-1 value
        # So we need to filter out the cells where covariate idx-1 has been changed
        px_scale = px_scale[cf_difference[:, idx-1]]
        px_rate = px_rate[cf_difference[:, idx-1]]
        px_dropout = px_dropout[cf_difference[:, idx-1]]
        if px_scale.shape[0] == 0:
            # if all cells in the batch have their covariate idx-1 changed
            # there won't be any output from the decoder idx
            return None

        # Use helper method to create distribution
        px = self._create_distribution(px_scale, px_r, px_rate, px_dropout)
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

        # Use helper methods for consistent preprocessing
        library = self._compute_library_size(x_)
        x_ = self._preprocess_input(x_)
        
        emb = []
        for i, embedding in enumerate(cat_covs.t()):
            emb.append(self.covars_embeddings[str(i)](embedding.long()))
        emb = torch.stack(emb, dim=0)
        full_embs_ubd = emb.clone() # unique_covs x batch_size x emb_dim
        emb = torch.permute(emb, (1, 0, 2))
        emb = self.pert_encoder(emb)
        emb = emb.reshape(emb.shape[0], -1)
        
        qz, z = (self.z_encoders_list[0](torch.hstack((x_, emb)))) # z0
        # TLDR: we encode the original gene expression and covariates of the cell using encoder 0
        # so z is the latent representation of the original cell, nothing about the counterfactual yet
        
        if detach_z:
            z = z.detach()

        if cat_covs_cf is None:
            cov_indices = list(set(list(range(self.zs_num))))
            ith_emb = full_embs_ubd[cov_indices, :, :]
            ith_emb = torch.permute(ith_emb, (1, 0, 2))
            ith_emb = self.pert_encoder(ith_emb)
            ith_emb = ith_emb.reshape(ith_emb.shape[0], -1)
        else:
            # Here's where the counterfactual decoding is happening
            ith_emb = []
            for i, embedding in enumerate(cat_covs_cf.t()):
                ith_emb.append(self.covars_embeddings[str(i)](embedding.long()))
            # decoder 0 takes all covariates because its latent is unaware of all covariates
            ith_emb = torch.stack(ith_emb, dim=0) # n_cat_covs x batch_size x emb_dim
            ith_emb = torch.permute(ith_emb, (1, 0, 2)) # batch_size x n_cat_covs x emb_dim
            ith_emb = self.pert_encoder(ith_emb)
            ith_emb = ith_emb.reshape(ith_emb.shape[0], -1) # batch_size x (n_cat_covs * emb_dim)

        # decoder 0
        x_decoder = self.x_decoders_list[0]

        # Note: decoder 0 takes all covariates and is unaware of them in its latent representation
        # therefore we can change all the covariates here while decoding without any issues
        # even if all the covariates are changed and we can't use decoders 1, ..., zs_num we can
        # always use decoder 0 to get the counterfactual gene expression without any issues.
        px_scale, px_r, px_rate, px_dropout = x_decoder(
            self.dispersion,
            torch.hstack((z, ith_emb)),
            library,
            # *dec_cats
        )
        px_r = torch.exp(self.px_r)

        # Use helper method to create distribution
        px = self._create_distribution(px_scale, px_r, px_rate, px_dropout)
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

        if self.gene_likelihood == 'normal':
            reconst_loss_x_list = [torch.mean((px.mean - x) ** 2) for px in generative_outputs["px"]]
        else:
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
                if self.gene_likelihood == 'normal':
                    mse_losses = []
                    for i, px_ in enumerate(pxs):
                        if px_ is not None:
                            x_cf_i = x_cf[cf_difference[:, i]]
                            mse_loss = torch.mean((px_.mean - x_cf_i) ** 2)
                            mse_losses.append(mse_loss)
                    if mse_losses:
                        reconst_loss_x_cf_list.append(torch.mean(torch.stack(mse_losses)))
                else:
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
