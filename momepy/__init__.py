import contextlib
from importlib.metadata import PackageNotFoundError, version

from . import datasets  # noqa
from .coins import *
from .dimension import *
from .distribution import *
from .diversity import *
from .elements import *
from .functional._dimension import *
from .functional._distribution import *
from .functional._diversity import *
from .functional._elements import *
from .functional._intensity import *
from .functional._shape import *
from .graph import *
from .intensity import *
from .preprocessing import *
from .shape import *
from .utils import *
from .weights import *

__author__ = "Martin Fleischmann"
__author_email__ = "martin@martinfleischmann.net"

with contextlib.suppress(PackageNotFoundError):
    __version__ = version("momepy")
