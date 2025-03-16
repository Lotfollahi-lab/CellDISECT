from typing import Optional

from scvi import settings
from scvi.data import AnnDataManager
from scvi.dataloaders import DataSplitter, AnnDataLoader
import torch


class AnnDataSplitter(DataSplitter):
    def __init__(
            self,
            adata_manager: AnnDataManager,
            train_indices,
            valid_indices,
            test_indices,
            use_gpu: bool = False,
            **kwargs,
    ):
        super().__init__(adata_manager)
        self.data_loader_kwargs = kwargs
        self.use_gpu = use_gpu
        self.train_idx = train_indices
        self.val_idx = valid_indices
        self.test_idx = test_indices

    def setup(self, stage: Optional[str] = None):
        # Handle device selection in a way compatible with newer scvi-tools
        if self.use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            accelerator = "gpu"
        else:
            self.device = torch.device("cpu")
            accelerator = "cpu"
        # Set pin_memory directly based on accelerator type
        # Not using deprecated settings.dl_pin_memory_gpu_training
        self.pin_memory = accelerator == "gpu"

    def train_dataloader(self):
        if len(self.train_idx) > 0:
            return AnnDataLoader(
                self.adata_manager,
                indices=self.train_idx,
                shuffle=True,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass

    def val_dataloader(self):
        if len(self.val_idx) > 0:
            data_loader_kwargs = self.data_loader_kwargs.copy()
            return AnnDataLoader(
                self.adata_manager,
                indices=self.val_idx,
                shuffle=True,
                pin_memory=self.pin_memory,
                **data_loader_kwargs,
            )
        else:
            pass

    def test_dataloader(self):
        if len(self.test_idx) > 0:
            return AnnDataLoader(
                self.adata_manager,
                indices=self.test_idx,
                shuffle=True,
                pin_memory=self.pin_memory,
                **self.data_loader_kwargs,
            )
        else:
            pass
