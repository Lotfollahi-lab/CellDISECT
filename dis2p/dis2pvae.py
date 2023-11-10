import random
from typing import Callable, Iterable, Literal, Optional, Union, Tuple

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

dim_indices = 0
device = 'cuda' if torch.cuda.is_available() else 'cpu'


class Dis2pVAE(BaseModuleClass):
    """
    Variational auto-encoder module.

    Parameters
    ----------
    n_input
        Number of input genes
    n_hidden
        Number of nodes per hidden layer
    n_latent_shared
        Dimensionality of the shared latent space (Z_{-s})
    n_latent_attribute
        Dimensionality of the latent space for each sensitive attributes (Z_{s_i})
    n_layers
        Number of hidden layers used for encoder and decoder NNs
    n_continuous_cov
        Number of continuous covarites
    n_cats_per_cov
        Number of categories for each extra categorical covariate
    dropout_rate
        Dropout rate for neural networks
    log_variational
        Log(data+1) prior to encoding for numerical stability. Not normalization.
    gene_likelihood
        One of
        * ``'nb'`` - Negative binomial distribution
        * ``'zinb'`` - Zero-inflated negative binomial distribution
        * ``'poisson'`` - Poisson distribution
    latent_distribution
        One of
        * ``'normal'`` - Isotropic normal
        * ``'ln'`` - Logistic normal with normal params N(0, 1)
    encode_covariates
        Whether to concatenate covariates to expression in encoder
    deeply_inject_covariates
        Whether to concatenate covariates into output of hidden layers in encoder/decoder. This option
        only applies when `n_layers` > 1. The covariates are concatenated to the input of subsequent hidden layers.
    use_batch_norm
        Whether to use batch norm in layers.
    use_layer_norm
        Whether to use layer norm in layers.
    var_activation
        Callable used to ensure positivity of the variational distributions' variance.
        When `None`, defaults to `torch.exp`.
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
            embeddings: Union[torch.Tensor, List[torch.Tensor]] = None
    ):
        super().__init__()
        self.dispersion = "gene"
        self.n_latent_shared = n_latent_shared
        self.n_latent_attribute = n_latent_attribute
        self.log_variational = log_variational
        self.gene_likelihood = gene_likelihood
        # Automatically deactivate if useless
        self.latent_distribution = latent_distribution

        self.px_r = torch.nn.Parameter(torch.randn(n_input)).to(device)

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
                    for key, embedding in enumerate(list(embeddings))
                }
            )
            drug_embeddings.weight.data.copy_(embeddings)
            drug_embeddings.weight.requires_grad = False
        else:
            self.covars_embeddings = nn.ModuleDict(
                {
                    str(key): torch.nn.Embedding(unique_covars, n_latent_shared)
                    for key, unique_covars in enumerate(self.n_cat_list)
                }
            )
        
        self.zs_num = len(self.n_cat_list)

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
                    n_latent_shared,
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
                ).to(device)
            )

    def _get_inference_input(self, tensors):

        cat_key = REGISTRY_KEYS.CAT_COVS_KEY
        cat_covs = tensors[cat_key]

        x = tensors[REGISTRY_KEYS.X_KEY]

        input_dict = {
            "x": x,
            "cat_covs": cat_covs,
        }
        return input_dict

    def _get_generative_input(self, tensors, inference_outputs):
        input_dict = {
            "z_shared": inference_outputs["z_shared"],
            "zs": inference_outputs["zs"],  # a list of all zs
            "library": inference_outputs["library"],
            "cat_covs": inference_outputs["cat_covs"],
        }
        return input_dict

    @auto_move_data
    def inference(self, x,
                  cat_covs,
                  nullify_cat_covs_indices: Optional[List[int]] = None,
                  nullify_shared: Optional[bool] = False,
                  ):

        nullify_cat_covs_indices = [] if nullify_cat_covs_indices is None else nullify_cat_covs_indices

        x_ = x
        library = torch.log(x.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)
        
        cat_in = torch.split(cat_covs, 1, dim=1)
        # z_shared
        emb = []
        for i, embedding in enumerate(cat_covs.t()):
            emb.append(self.covars_embeddings[str(i)](embedding.long()))
        prior_emb_in = emb[:]
        emb = torch.stack(emb, dim=0)
        emb = torch.permute(emb, (1, 0, 2))
        emb = emb.reshape(emb.shape[0], -1)
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
    def generative(self, z_shared,
                   zs,
                   library,
                   cat_covs,
                   ):

        output_dict = {"px": []}

        z = [z_shared] + zs

        cats_splits = torch.split(cat_covs, 1, dim=1)
        emb = []
        for i, embedding in enumerate(cat_covs.t()):
            emb.append(self.covars_embeddings[str(i)](embedding.long()))
        emb = torch.stack(emb, dim=0)
        full_embs_ubd = emb.clone() # unique_covs x batch_size x emb_dim
        emb = torch.permute(emb, (1, 0, 2))
        emb = emb.reshape(emb.shape[0], -1)
        all_cats_but_one = []
        for i in range(self.zs_num):
            cov_indices = list(set(list(range(self.zs_num)))-set([i]))
            ith_emb = full_embs_ubd[cov_indices, :, :]
            ith_emb = torch.permute(ith_emb, (1, 0, 2))
            ith_emb = ith_emb.reshape(ith_emb.shape[0], -1)
            all_cats_but_one.append(ith_emb)
        
        dec_cats_in = [emb] + all_cats_but_one

        for dec_count in range(self.zs_num + 1):

            x_decoder = self.x_decoders_list[dec_count]

            dec_covs = dec_cats_in[dec_count]

            x_decoder_input = z[dec_count]
            
            px_scale, px_r, px_rate, px_dropout = x_decoder(
                self.dispersion,
                torch.hstack((x_decoder_input, dec_covs)),
                library,
                # *dec_covs
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

        performs forward (inference + generative) only on enc/dec idx

        Parameters
        ----------
        idx
            index of enc/dec in [1, ..., self.zs_num]
        x
        cat_covs
        detach_x
        detach_z

        """
        x_ = x
        if detach_x:
            x_ = x.detach()

        library = torch.log(x_.sum(1)).unsqueeze(1)
        if self.log_variational:
            x_ = torch.log(1 + x_)

        cat_in = torch.split(cat_covs, 1, dim=1)
        
        emb = []
        for i, embedding in enumerate(cat_covs.t()):
            emb.append(self.covars_embeddings[str(i)](embedding.long()))
        emb = torch.stack(emb, dim=0)
        full_embs_ubd = emb.clone() # unique_covs x batch_size x emb_dim
        emb = torch.permute(emb, (1, 0, 2))
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
        zs = inference_outputs["zs"]
        logits = []
        for i in range(self.zs_num):
            s_i_classifier = self.s_classifiers_list[i]
            logits_i = s_i_classifier(zs[i])
            logits += [logits_i]

        return logits

    def compute_clf_metrics(self, logits, cat_covs):
        # CE, ACC, F1
        cats = torch.split(cat_covs, 1, dim=1)
        ce_losses = []
        accuracy_scores = []
        f1_scores = []
        for i in range(self.zs_num):
            s_i = one_hot_cat([self.n_cat_list[i]], cats[i]).to(device)
            ce_losses += [F.cross_entropy(logits[i], s_i)]
            kwargs = {"task": "multiclass", "num_classes": self.n_cat_list[i]}
            predicted_labels = torch.argmax(logits[i], dim=-1, keepdim=True).to(device)
            acc = Accuracy(**kwargs).to(device)
            accuracy_scores.append(acc(predicted_labels, cats[i]).to(device))
            F1 = F1Score(**kwargs).to(device)
            f1_scores.append(F1(predicted_labels, cats[i]).to(device))

        ce_loss_sum = sum(torch.mean(ce) for ce in ce_losses)
        accuracy = sum(accuracy_scores) / len(accuracy_scores)
        f1 = sum(f1_scores) / len(f1_scores)

        return ce_loss_sum, accuracy, f1

    def loss(
            self,
            tensors,
            inference_outputs,
            generative_outputs,
            cf_weight: Tunable[Union[float, int]],  # RECONST_LOSS_X_CF weight
            beta: Tunable[Union[float, int]],  # KL Zi weight
            clf_weight: Tunable[Union[float, int]],  # Si classifier weight
            n_cf: Tunable[int],  # number of X_cf recons (X_cf = a random permutation of X)
            kl_weight: float = 1.0,
    ):
        # reconstruction loss X

        x = tensors[REGISTRY_KEYS.X_KEY]

        reconst_loss_x_list = [-torch.mean(px.log_prob(x).sum(-1)) for px in generative_outputs["px"]]
        reconst_loss_x_dict = {'x_' + str(i): reconst_loss_x_list[i] for i in range(len(reconst_loss_x_list))}
        reconst_loss_x = sum(reconst_loss_x_list)

        # reconstruction loss X'

        cat_covs = tensors[REGISTRY_KEYS.CAT_COVS_KEY]
        batch_size = x.size(dim=0)

        reconst_loss_x_cf_list = []

        for _ in range(n_cf):

            # choose a random permutation of X as X_cf

            idx_shuffled = list(range(batch_size))
            random.shuffle(idx_shuffled)
            idx_shuffled = torch.tensor(idx_shuffled).to(device)

            x_ = x
            x_cf = torch.index_select(x, 0, idx_shuffled).to(device)

            cat_cov_ = cat_covs
            cat_cov_cf = torch.index_select(cat_covs, 0, idx_shuffled).to(device)
            cat_cov_cf_split = torch.split(cat_cov_cf, 1, dim=1)

            # a random ordering for diffusing through n VAEs

            perm = list(range(self.zs_num))
            random.shuffle(perm)

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

        # classification metrics: CE, ACC, F1

        logits = self.classification_logits(inference_outputs)
        ce_loss_sum, accuracy, f1 = self.compute_clf_metrics(logits, cat_covs)
        ce_loss_mean = ce_loss_sum / len(range(self.zs_num))

        # total loss
        loss = reconst_loss_x + \
               reconst_loss_x_cf * cf_weight + \
               sum(kl_z_list) * kl_weight * beta + \
               ce_loss_sum * clf_weight

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
