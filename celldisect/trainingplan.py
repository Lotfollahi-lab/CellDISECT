from collections import OrderedDict
from typing import Callable, Dict, Iterable, Literal, Optional, Union, Tuple

import optax
import torch
from torch import nn
from torch.optim.lr_scheduler import ReduceLROnPlateau

from scvi.module import Classifier
from scvi.module.base import BaseModuleClass, LossOutput
JaxOptimizerCreator = Callable[[], optax.GradientTransformation]
TorchOptimizerCreator = Callable[[Iterable[torch.Tensor]], torch.optim.Optimizer]
from scvi.train import TrainingPlan
from .utils import *
from scvi.train._metrics import ElboMetric

device = 'cuda' if torch.cuda.is_available() else 'cpu'
from scvi.autotune._types import Tunable

class CellDISECTTrainingPlan(TrainingPlan):
    """
    Train VAEs with adversarial loss option to encourage latent space mixing.

    Parameters
    ----------
    module : BaseModuleClass
        A module instance from class ``BaseModuleClass``.
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
    optimizer : Tunable[Literal["Adam", "AdamW", "Custom"]], optional
        One of "Adam" (:class:`~torch.optim.Adam`), "AdamW" (:class:`~torch.optim.AdamW`),
        or "Custom", which requires a custom optimizer creator callable to be passed via
        `optimizer_creator`. Default is "Adam".
    optimizer_creator : Optional[TorchOptimizerCreator], optional
        A callable taking in parameters and returning a :class:`~torch.optim.Optimizer`.
        This allows using any PyTorch optimizer with custom hyperparameters. Default is None.
    lr : Tunable[float], optional
        Learning rate used for optimization, when `optimizer_creator` is None. Default is 1e-3.
    weight_decay : Tunable[float], optional
        Weight decay used in optimization, when `optimizer_creator` is None. Default is 1e-6.
    n_steps_kl_warmup : Tunable[int], optional
        Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
        Only activated when `n_epochs_kl_warmup` is set to None. Default is None.
    n_epochs_kl_warmup : Tunable[int], optional
        Number of epochs to scale weight on KL divergences from 0 to 1.
        Overrides `n_steps_kl_warmup` when both are not `None`. Default is 400.
    n_epochs_pretrain_ae : Tunable[int], optional
        Number of epochs to pretrain the autoencoder. Default is 0.
    reduce_lr_on_plateau : Tunable[bool], optional
        Whether to monitor validation loss and reduce learning rate when validation set
        `lr_scheduler_metric` plateaus. Default is True.
    lr_factor : Tunable[float], optional
        Factor to reduce learning rate. Default is 0.6.
    lr_patience : Tunable[int], optional
        Number of epochs with no improvement after which learning rate will be reduced. Default is 30.
    lr_threshold : Tunable[float], optional
        Threshold for measuring the new optimum. Default is 0.0.
    lr_scheduler_metric : Literal["loss_validation"], optional
        Metric to monitor for learning rate reduction. Default is "loss_validation".
    lr_min : float, optional
        Minimum learning rate allowed. Default is 0.
    scale_adversarial_loss : Union[float, Literal["auto"]], optional
        Scaling factor on the adversarial components of the loss.
        By default, adversarial loss is scaled from 1 to 0 following opposite of
        kl warmup. Default is "auto".
    ensemble_method_cf : bool, optional
        Whether to use the new counterfactual method. Default is True.
    kappa_optimizer2 : bool, optional
        Whether to use the second kappa optimizer. Default is True.
    **loss_kwargs
        Keyword args to pass to the loss method of the `module`.
        `kl_weight` should not be passed here and is handled automatically.
    """

    def __init__(
        self,
        module: BaseModuleClass,
        *,
        recon_weight: Tunable[Union[float, int]] = 1.0,
        cf_weight: Tunable[Union[float, int]] = 1.0,
        beta: Tunable[Union[float, int]] = 1.0,
        clf_weight: Tunable[Union[float, int]] = 1.0,
        adv_clf_weight: Tunable[Union[float, int]] = 1.0,
        adv_period: Tunable[int] = 10,
        n_cf: Tunable[int] = 1,
        optimizer: Tunable[Literal["Adam", "AdamW", "Custom"]] = "Adam",
        optimizer_creator: Optional[TorchOptimizerCreator] = None,
        lr: Tunable[float] = 1e-3,
        weight_decay: Tunable[float] = 1e-6,
        n_steps_kl_warmup: Tunable[int] = None,
        n_epochs_kl_warmup: Tunable[int] = 400,
        n_epochs_pretrain_ae: Tunable[int] = 0,
        reduce_lr_on_plateau: Tunable[bool] = True,
        lr_factor: Tunable[float] = 0.6,
        lr_patience: Tunable[int] = 30,
        lr_threshold: Tunable[float] = 0.0,
        lr_scheduler_metric: Literal["loss_validation"] = "loss_validation",
        lr_min: float = 0,
        scale_adversarial_loss: Union[float, Literal["auto"]] = "auto",
        ensemble_method_cf: bool = True,
        kappa_optimizer2: bool = True,
        **loss_kwargs,
    ):
        """
        Initialize the CellDISECTTrainingPlan.

        Parameters
        ----------
        module : BaseModuleClass
            A module instance from class ``BaseModuleClass``.
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
        optimizer : Tunable[Literal["Adam", "AdamW", "Custom"]], optional
            One of "Adam" (:class:`~torch.optim.Adam`), "AdamW" (:class:`~torch.optim.AdamW`),
            or "Custom", which requires a custom optimizer creator callable to be passed via
            `optimizer_creator`. Default is "Adam".
        optimizer_creator : Optional[TorchOptimizerCreator], optional
            A callable taking in parameters and returning a :class:`~torch.optim.Optimizer`.
            This allows using any PyTorch optimizer with custom hyperparameters. Default is None.
        lr : Tunable[float], optional
            Learning rate used for optimization, when `optimizer_creator` is None. Default is 1e-3.
        weight_decay : Tunable[float], optional
            Weight decay used in optimization, when `optimizer_creator` is None. Default is 1e-6.
        n_steps_kl_warmup : Tunable[int], optional
            Number of training steps (minibatches) to scale weight on KL divergences from 0 to 1.
            Only activated when `n_epochs_kl_warmup` is set to None. Default is None.
        n_epochs_kl_warmup : Tunable[int], optional
            Number of epochs to scale weight on KL divergences from 0 to 1.
            Overrides `n_steps_kl_warmup` when both are not `None`. Default is 400.
        n_epochs_pretrain_ae : Tunable[int], optional
            Number of epochs to pretrain the autoencoder. Default is 0.
        reduce_lr_on_plateau : Tunable[bool], optional
            Whether to monitor validation loss and reduce learning rate when validation set
            `lr_scheduler_metric` plateaus. Default is True.
        lr_factor : Tunable[float], optional
            Factor to reduce learning rate. Default is 0.6.
        lr_patience : Tunable[int], optional
            Number of epochs with no improvement after which learning rate will be reduced. Default is 30.
        lr_threshold : Tunable[float], optional
            Threshold for measuring the new optimum. Default is 0.0.
        lr_scheduler_metric : Literal["loss_validation"], optional
            Which metric to track for learning rate reduction. Default is "loss_validation".
        lr_min : float, optional
            Minimum learning rate allowed. Default is 0.
        scale_adversarial_loss : Union[float, Literal["auto"]], optional
            Scaling factor on the adversarial components of the loss.
            By default, adversarial loss is scaled from 1 to 0 following opposite of
            kl warmup. Default is "auto".
        ensemble_method_cf : bool, optional
            Whether to use the new counterfactual method. Default is True.
        kappa_optimizer2 : bool, optional
            Whether to use the second kappa optimizer. Default is True.
        **loss_kwargs
            Keyword args to pass to the loss method of the `module`.
            `kl_weight` should not be passed here and is handled automatically.
        """
        super().__init__(
            module,
            optimizer=optimizer,
            optimizer_creator=optimizer_creator,
            lr=lr,
            weight_decay=weight_decay,
            n_steps_kl_warmup=n_steps_kl_warmup,
            n_epochs_kl_warmup=n_epochs_kl_warmup,
            reduce_lr_on_plateau=reduce_lr_on_plateau,
            lr_factor=lr_factor,
            lr_patience=lr_patience,
            lr_threshold=lr_threshold,
            lr_scheduler_metric=lr_scheduler_metric,
            lr_min=lr_min,
            **loss_kwargs,
        )

        self.adv_clf_weight = adv_clf_weight
        self.adv_period = adv_period
        self.n_epochs_pretrain_ae = n_epochs_pretrain_ae
        self.kappa_optimizer2 = kappa_optimizer2

        self.loss_kwargs.update({"recon_weight": recon_weight,
                                 "cf_weight": cf_weight,
                                 "beta": beta,
                                 "clf_weight": clf_weight,
                                 "n_cf": n_cf,
                                 "ensemble_method_cf": ensemble_method_cf,
                                })

        self.module = module
        self.zs_num = module.zs_num
        self.n_cat_list = module.n_cat_list
        # self.adv_input_size = module.n_latent_shared + module.n_latent_attribute * (module.zs_num - 1)
        self.adv_input_size_shared = module.n_latent_shared
        self.adv_input_size_attribute = module.n_latent_attribute

        self.adv_clf_list = nn.ModuleList([])
        for i in range(self.zs_num):
            for j in range(self.zs_num):
                if j == 0:
                    self.adv_clf_list.append(
                        Classifier(
                            n_input=self.adv_input_size_shared,
                            n_labels=self.n_cat_list[i],
                            logits=True,
                            use_layer_norm=True,
                            use_batch_norm=False,
                        ).to(device)
                    )
                else:
                    self.adv_clf_list.append(
                        Classifier(
                            n_input=self.adv_input_size_attribute,
                            n_labels=self.n_cat_list[i],
                            logits=True,
                            use_layer_norm=True,
                            use_batch_norm=False,
                        ).to(device)
                    )

        self.scale_adversarial_loss = scale_adversarial_loss
        self.automatic_optimization = False

    def compute_and_log_metrics(self, loss_output, metrics, mode):
        """
        Computes and logs metrics.

        Parameters
        ----------
        loss_output : LossOutput
            LossOutput object from scvi-tools module
        metrics : dict[str, ElboMetric]
            Dictionary of metrics to update
        mode : str
            Postfix string to add to the metric name of extra metrics
        """
        if isinstance(loss_output, LossOutput):
            # Handle LossOutput object
            loss = loss_output.loss
            reconstruction_loss = loss_output.reconstruction_loss
            kl_local = loss_output.kl_local
            kl_global = loss_output.kl_global
            n_obs_minibatch = loss_output.n_obs_minibatch
            extra_metrics = loss_output.extra_metrics
        else:
            # Handle dictionary for backward compatibility
            loss = loss_output.get(LOSS_KEYS.LOSS, None)
            reconstruction_loss = loss_output.get(LOSS_KEYS.RECONST_LOSS_X, None)
            kl_local = loss_output.get(LOSS_KEYS.KL_Z, None)
            kl_global = None
            n_obs_minibatch = 1
            extra_metrics = loss_output

        # Update metrics
        for metric in metrics.values():
            if metric.name == "elbo_train" or metric.name == "elbo_validation":
                metric.update(loss)
            elif metric.name == "reconstruction_loss_train" or metric.name == "reconstruction_loss_validation":
                if isinstance(reconstruction_loss, dict):
                    # If reconstruction_loss is a dictionary, use the sum of values
                    metric.update(sum(reconstruction_loss.values()))
                else:
                    metric.update(reconstruction_loss)
            elif metric.name == "kl_local_train" or metric.name == "kl_local_validation":
                if isinstance(kl_local, dict):
                    # If kl_local is a dictionary, use the sum of values
                    metric.update(sum(kl_local.values()))
                else:
                    metric.update(kl_local)
            elif metric.name == "kl_global_train" or metric.name == "kl_global_validation":
                metric.update(kl_global)

        # Log metrics
        for metric_name, metric_value in metrics.items():
            self.log(
                metric_name,
                metric_value,
                on_step=False,
                on_epoch=True,
                batch_size=n_obs_minibatch,
                sync_dist=self.use_sync_dist,
            )

        # Log extra metrics
        if extra_metrics is not None:
            for key, value in extra_metrics.items():
                if key not in [LOSS_KEYS.LOSS, LOSS_KEYS.RECONST_LOSS_X, LOSS_KEYS.KL_Z]:
                    if isinstance(value, torch.Tensor):
                        if value.shape != torch.Size([]):
                            continue  # Skip non-scalar tensors
                        value = value.detach()
                    self.log(
                        f"{key}_{mode}",
                        value,
                        on_step=False,
                        on_epoch=True,
                        batch_size=n_obs_minibatch,
                        sync_dist=self.use_sync_dist,
                    )

    def adv_classifier_metrics(self, inference_outputs, detach_z=True):
        """
        Computes the loss for the adversarial classifier.

        This function calculates the classification metrics for the adversarial classifier
        using the provided inference outputs.

        Parameters
        ----------
        inference_outputs : dict
            Dictionary containing the outputs from the inference step.
        detach_z : bool, optional
            Whether to detach the latent representation `z`, by default True.

        Returns
        -------
        tuple
            A tuple containing the mean CE loss, accuracy, and F1 score.
        """
        z_shared = inference_outputs["z_shared"]
        zs = inference_outputs["zs"]
        cat_covs = inference_outputs["cat_covs"]

        if detach_z:
            # Detach z
            zs = [zs_i.detach() for zs_i in zs]
            z_shared = z_shared.detach()

        logits = []
        for i in range(self.zs_num):
            for j in range(self.zs_num):
                if j == 0:
                    z = z_shared
                else:
                    z = zs[j-1]
                adv_clf_i = self.adv_clf_list[i*self.zs_num + j]  # Each covariate has n classifiers: Z0, Zi (i != covariate)
                logits_i = adv_clf_i(z)
                logits += [logits_i]

        return self.module.compute_clf_metrics(logits, cat_covs)


    def training_step(self, batch, batch_idx):
        """Training step for adversarial training."""
        # Get optimizers
        opts = self.optimizers()
        if not isinstance(opts, list):
            opt1 = opts
            opt2 = None
        else:
            opt1, opt2 = opts

        # kl annealing
        if "kl_weight" in self.loss_kwargs:
            self.loss_kwargs.update({"kl_weight": self.kl_weight})
        kappa = (
            self.kl_weight
            if self.scale_adversarial_loss == "auto"
            else self.scale_adversarial_loss
        )
        
        # Log kappa
        self.log("kl_weight", kappa, on_step=False, on_epoch=True)

        input_kwargs = {}
        input_kwargs.update(self.loss_kwargs)

        inference_outputs, _, scvi_loss = self.forward(
            batch, loss_kwargs=input_kwargs
        )

        # Handle LossOutput or dict
        if isinstance(scvi_loss, LossOutput):
            loss = scvi_loss.loss
            extra_metrics = scvi_loss.extra_metrics if scvi_loss.extra_metrics is not None else {}
            # Extract reconstruction loss for pretraining
            if LOSS_KEYS.RECONST_LOSS_X in extra_metrics:
                recon_loss_dict = extra_metrics[LOSS_KEYS.RECONST_LOSS_X]
                if isinstance(recon_loss_dict, dict):
                    recon_loss = sum(recon_loss_dict.values()) / len(recon_loss_dict)
                else:
                    recon_loss = recon_loss_dict
            else:
                recon_loss = scvi_loss.reconstruction_loss
        else:
            # For backward compatibility with dict
            loss = scvi_loss[LOSS_KEYS.LOSS]
            extra_metrics = scvi_loss
            # Extract reconstruction loss for pretraining
            recon_loss_dict = scvi_loss[LOSS_KEYS.RECONST_LOSS_X]
            if isinstance(recon_loss_dict, dict):
                recon_loss = sum(recon_loss_dict.values()) / len(recon_loss_dict)
            else:
                recon_loss = recon_loss_dict

        # train normally
        if self.n_epochs_pretrain_ae > 0 and self.current_epoch < self.n_epochs_pretrain_ae:
            opt1.zero_grad()
            self.manual_backward(recon_loss)
            opt1.step()

            ce_loss_mean, accuracy, f1 = self.adv_classifier_metrics(inference_outputs, True)

            # Update metrics with adversarial metrics
            adv_metrics = {'adv_ce': ce_loss_mean, 'adv_acc': accuracy, 'adv_f1': f1}
            
            if isinstance(scvi_loss, LossOutput):
                # Create a new LossOutput with updated extra_metrics
                if extra_metrics is not None:
                    extra_metrics.update(adv_metrics)
                else:
                    extra_metrics = adv_metrics
                    
                updated_loss = LossOutput(
                    loss=recon_loss,  # Use recon_loss as the main loss during pretraining
                    reconstruction_loss=scvi_loss.reconstruction_loss,
                    kl_local=scvi_loss.kl_local,
                    kl_global=scvi_loss.kl_global,
                    extra_metrics=extra_metrics
                )
            else:
                # Update the dictionary directly
                scvi_loss.update(adv_metrics)
                updated_loss = scvi_loss

            self.compute_and_log_metrics(updated_loss, self.train_metrics, "train")
            return updated_loss
        
        # Adversarial training
        if (self.current_epoch % self.adv_period == 0):
            # fool classifier if doing adversarial training
            if kappa > 0:
                ce_loss_mean, accuracy, f1 = self.adv_classifier_metrics(inference_outputs, False)
                if isinstance(scvi_loss, LossOutput):
                    # Adjust the loss for adversarial training
                    loss = loss - ce_loss_mean * kappa * self.adv_clf_weight
                else:
                    # For backward compatibility
                    loss = scvi_loss[LOSS_KEYS.LOSS] - ce_loss_mean * kappa * self.adv_clf_weight

            opt1.zero_grad()
            self.manual_backward(loss)
            opt1.step()

        # train adversarial classifier
        if opt2 is not None:
            ce_loss_mean, accuracy, f1 = self.adv_classifier_metrics(inference_outputs, True)
            if self.kappa_optimizer2:
                ce_loss_mean *= kappa
            opt2.zero_grad()
            self.manual_backward(ce_loss_mean)
            opt2.step()

        # Update metrics with adversarial metrics
        adv_metrics = {'adv_ce': ce_loss_mean, 'adv_acc': accuracy, 'adv_f1': f1}
        
        if isinstance(scvi_loss, LossOutput):
            # Create a new LossOutput with updated extra_metrics
            if extra_metrics is not None:
                extra_metrics.update(adv_metrics)
            else:
                extra_metrics = adv_metrics
                
            updated_loss = LossOutput(
                loss=loss,  # Use the potentially modified loss
                reconstruction_loss=scvi_loss.reconstruction_loss,
                kl_local=scvi_loss.kl_local,
                kl_global=scvi_loss.kl_global,
                extra_metrics=extra_metrics
            )
        else:
            # Update the dictionary directly
            scvi_loss.update(adv_metrics)
            # Update the loss value if it was modified
            scvi_loss[LOSS_KEYS.LOSS] = loss
            updated_loss = scvi_loss

        self.compute_and_log_metrics(updated_loss, self.train_metrics, "train")

        return updated_loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        input_kwargs = {}
        input_kwargs.update(self.loss_kwargs)

        inference_outputs, _, scvi_loss = self.forward(
            batch, loss_kwargs=input_kwargs
        )

        ce_loss_mean, accuracy, f1 = self.adv_classifier_metrics(inference_outputs, True)

        # Handle LossOutput or dict
        if isinstance(scvi_loss, LossOutput):
            extra_metrics = scvi_loss.extra_metrics if scvi_loss.extra_metrics is not None else {}
        else:
            # For backward compatibility with dict
            extra_metrics = scvi_loss

        # Update metrics with adversarial metrics
        adv_metrics = {'adv_ce': ce_loss_mean, 'adv_acc': accuracy, 'adv_f1': f1}
        
        if isinstance(scvi_loss, LossOutput):
            # Create a new LossOutput with updated extra_metrics
            if extra_metrics is not None:
                extra_metrics.update(adv_metrics)
            else:
                extra_metrics = adv_metrics
                
            updated_loss = LossOutput(
                loss=scvi_loss.loss,
                reconstruction_loss=scvi_loss.reconstruction_loss,
                kl_local=scvi_loss.kl_local,
                kl_global=scvi_loss.kl_global,
                extra_metrics=extra_metrics
            )
        else:
            # Update the dictionary directly
            scvi_loss.update(adv_metrics)
            updated_loss = scvi_loss
        
        self.compute_and_log_metrics(updated_loss, self.val_metrics, "validation")

        return updated_loss

    def on_train_epoch_end(self):
        """Update the learning rate via scheduler steps."""
        if "validation" in self.lr_scheduler_metric or not self.reduce_lr_on_plateau:
            return
        else:
            sch = self.lr_schedulers()
            sch.step(self.trainer.callback_metrics[self.lr_scheduler_metric])

    def on_validation_epoch_end(self) -> None:
        # Update the learning rate via scheduler steps.
        if (
            not self.reduce_lr_on_plateau
            or "validation" not in self.lr_scheduler_metric
        ):
            return
        else:
            sch = self.lr_schedulers()
            sch.step(self.trainer.callback_metrics[self.lr_scheduler_metric])
            # Log learning rate
            self.log(
                "learning_rate",
                sch.optimizer.param_groups[0]["lr"],
                on_step=False,
                on_epoch=True,
            )

    def configure_optimizers(self):
        """Configure optimizers for adversarial training."""
        params1 = filter(lambda p: p.requires_grad, self.module.parameters())
        optimizer1 = self.get_optimizer_creator()(params1)
        config1 = {"optimizer": optimizer1}
        if self.reduce_lr_on_plateau:
            scheduler1 = ReduceLROnPlateau(
                optimizer1,
                patience=self.lr_patience,
                factor=self.lr_factor,
                threshold=self.lr_threshold,
                min_lr=self.lr_min,
                threshold_mode="abs",
                verbose=True,
            )
            config1.update(
                {
                    "lr_scheduler": {
                        "scheduler": scheduler1,
                        "monitor": self.lr_scheduler_metric,
                    },
                },
            )

        params2 = filter(
            lambda p: p.requires_grad, self.adv_clf_list.parameters()
        )
        optimizer2 = torch.optim.Adam(
            params2, lr=1e-3, eps=0.01, weight_decay=self.weight_decay
        )
        config2 = {"optimizer": optimizer2}

        # pytorch lightning requires this way to return
        opts = [config1.pop("optimizer"), config2["optimizer"]]
        if "lr_scheduler" in config1:
            scheds = [config1["lr_scheduler"]]
            return opts, scheds
        else:
            return opts

    @staticmethod
    def _create_elbo_metric_components(mode: str, n_total: Optional[int] = None):
        """Initialize metrics and the metric collection."""
        metrics_list = [ElboMetric(met_name, mode, "obs") for met_name in LOSS_KEYS_LIST]
        collection = OrderedDict([(metric.name, metric) for metric in metrics_list])
        return metrics_list, collection

    def initialize_train_metrics(self):
        """Initialize train related metrics."""
        self.elbo_metrics_list_train, self.train_metrics = \
            self._create_elbo_metric_components(mode="train", n_total=self.n_obs_training)

    def initialize_val_metrics(self):
        """Initialize val related metrics."""
        self.elbo_metrics_list_val, self.val_metrics = \
            self._create_elbo_metric_components(mode="validation", n_total=self.n_obs_validation)
