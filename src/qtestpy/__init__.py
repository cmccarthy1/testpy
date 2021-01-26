from functools import wraps
from pathlib import Path
import sys

with open(Path(__file__).parent.absolute()/'VERSION') as version_file:
    __version__ = version_file.read().strip()

import qtestpy.adapt as adapt                        # noqa: F401
from qtestpy.kdb import kdb                          # noqa: F401

__all__ = sorted([
    'adapt', 'kdb',
])

if sys.version_info >= (3, 7):
    __all__ = sorted(['q', *__all__])

def __dir__():
    return sorted([
        *__all__, '__all__', '__doc__', '__name__', '__spec__', '__dir__',
        '__path__', '__file__', '__package__'
    ])
