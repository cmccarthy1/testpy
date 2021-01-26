from pathlib import Path
import random
import string
from typing import List, Optional, Type, Union

import qtestpy.adapt as adapt


class Singleton(type):
    def __init__(cls, name, bases, _dict):
        super().__init__(name, bases, _dict)
        cls.instance = None

    def __call__(cls, **kwargs):
        if cls.instance is None:
            cls.instance = super().__call__(**kwargs)
        return cls.instance


class kdb(metaclass=Singleton):
    def __init__(self,
                 *,  # kwargs only
                 argv: Optional[List[str]] = (),
                 module_paths: Optional[List[Union[str, Path]]] = None):
	adapt.init(self)
	self.K = adapt.adapt

    def __repr__(self):
        return f'kdb( )'

    def __call__(self, query: str = '', mode: str = '') -> Type[k.K]:
	return(1)

    def __getattr__(self, key):
	return(1)
    def __delattr__(self, key):
        raise PyKdbException('Cannot delete from global kdb context.')

