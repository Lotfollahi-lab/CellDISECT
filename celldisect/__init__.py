import warnings

warnings.simplefilter('ignore')

from ._model import CellDISECT
from ._module import CellDISECTModule, PerturbationEmbedding
from .utils import perturbation_metrics

try:
    from importlib.metadata import version
except ImportError:
    from importlib_metadata import version

package_name = "celldisect"
try:
    __version__ = version(package_name)
except Exception:
    __version__ = "0.1.6"

__all__ = [
    "CellDISECT",
    "CellDISECTModule",
    "PerturbationEmbedding",
    "perturbation_metrics",
]