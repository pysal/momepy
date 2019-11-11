from .dimension import *
from .intensity import *
from .utils import *
from .elements import *
from .diversity import *
from .distribution import *
from .graph import *
from .shape import *
import momepy.datasets

__author__ = "Martin Fleischmann"
__author_email__ = "martin@martinfleischmann.net"

from ._version import get_versions

__version__ = get_versions()["version"]
del get_versions
