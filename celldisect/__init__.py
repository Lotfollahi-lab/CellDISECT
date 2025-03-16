import warnings
import logging
import os

warnings.simplefilter('ignore')

# Configure global logging to prevent duplicate messages
# Set up logging to avoid duplicates
logging.basicConfig(level=logging.INFO, format='%(message)s')
logging.getLogger('lightning.pytorch.utilities.rank_zero').setLevel(logging.WARNING)

from ._model import CellDISECT
from .tuner_base import run_autotune
from ._module import CellDISECTModule
from .trainingplan import CellDISECTTrainingPlan
from .data import AnnDataSplitter

from importlib.metadata import version

package_name = "celldisect"
__version__ = version(package_name)

__all__ = [
    "CellDISECT",
    "run_autotune",
    "CellDISECTModule",
    "CellDISECTTrainingPlan",
    "AnnDataSplitter",
]