from importlib.util import find_spec

from .kompyle import *


def is_package_installed(name):
    return find_spec(name) is not None


if is_package_installed("torch"):
    from .torch_tools import *

# __doc__ = kompyle.__doc__
# if hasattr(kompyle, "__all__"):
#     __all__ = kompyle.__all__