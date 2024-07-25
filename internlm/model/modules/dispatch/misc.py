import importlib
from collections import abc
from typing import Any, Optional, Type, Union


# adapted from https://github.com/open-mmlab/mmengine/blob/main/mmengine/config/lazy.py#L8
class LazyObject:
    """LazyObject is used to lazily initialize the imported module during
    parsing the configuration file.

    During parsing process, the syntax like:

    Examples:
        >>> import torch.nn as nn
        >>> from mmdet.models import RetinaNet
        >>> import mmcls.models
        >>> import mmcls.datasets
        >>> import mmcls

    Will be parsed as:

    Examples:
        >>> # import torch.nn as nn
        >>> nn = lazyObject('torch.nn')
        >>> # from mmdet.models import RetinaNet
        >>> RetinaNet = lazyObject('mmdet.models', 'RetinaNet')
        >>> # import mmcls.models; import mmcls.datasets; import mmcls
        >>> mmcls = lazyObject(['mmcls', 'mmcls.datasets', 'mmcls.models'])

    ``LazyObject`` records all module information and will be further
    referenced by the configuration file.

    Args:
        module (str or list or tuple): The module name to be imported.
        imported (str, optional): The imported module name. Defaults to None.
        location (str, optional): The filename and line number of the imported
            module statement happened.
    """

    def __init__(self, module: Union[str, list, tuple], imported: Optional[str] = None, location: Optional[str] = None):
        if not isinstance(module, str) and not is_seq_of(module, str):
            raise TypeError(
                "module should be `str`, `list`, or `tuple`"
                f"but got {type(module)}, this might be "
                "a bug of MMEngine, please report it to "
                "https://github.com/open-mmlab/mmengine/issues"
            )
        self._module: Union[str, list, tuple] = module

        if not isinstance(imported, str) and imported is not None:
            raise TypeError(
                "imported should be `str` or None, but got "
                f"{type(imported)}, this might be "
                "a bug of MMEngine, please report it to "
                "https://github.com/open-mmlab/mmengine/issues"
            )
        self._imported = imported
        self.location = location

    def build(self) -> Any:
        if isinstance(self._module, str):
            try:
                module = importlib.import_module(self._module)
            except Exception as e:
                raise type(e)(f"Failed to import {self._module} " f"in {self.location} for {e}")

            if self._imported is not None:
                if hasattr(module, self._imported):
                    module = getattr(module, self._imported)
                else:
                    raise ImportError(f"Failed to import {self._imported} " f"from {self._module} in {self.location}")

            return module
        else:
            try:
                for module in self._module:
                    importlib.import_module(module)  # type: ignore
                module_name = self._module[0].split(".")[0]
                return importlib.import_module(module_name)
            except Exception as e:
                raise type(e)(f"Failed to import {self.module} " f"in {self.location} for {e}")

    @property
    def module(self):
        if isinstance(self._module, str):
            return self._module
        return self._module[0].split(".")[0]

    def __call__(self, *args, **kwargs):
        raise RuntimeError()

    def __deepcopy__(self, memo):
        return LazyObject(self._module, self._imported, self.location)

    def __getattr__(self, name):
        if self.location is not None:
            location = self.location.split(", line")[0]
        else:
            location = self.location
        return LazyAttr(name, self, location)

    def __str__(self) -> str:
        if self._imported is not None:
            return self._imported
        return self.module

    __repr__ = __str__

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state


# adapted from https://github.com/open-mmlab/mmengine/blob/main/mmengine/config/lazy.py#L135
class LazyAttr:
    """The attribute of the LazyObject.

    When parsing the configuration file, the imported syntax will be
    parsed as the assignment ``LazyObject``. During the subsequent parsing
    process, users may reference the attributes of the LazyObject.
    To ensure that these attributes also contain information needed to
    reconstruct the attribute itself, LazyAttr was introduced.

    Examples:
        >>> models = LazyObject(['mmdet.models'])
        >>> model = dict(type=models.RetinaNet)
        >>> print(type(model['type']))  # <class 'mmengine.config.lazy.LazyAttr'>
        >>> print(model['type'].build())  # <class 'mmdet.models.detectors.retinanet.RetinaNet'>
    """  # noqa: E501

    def __init__(self, name: str, source: Union["LazyObject", "LazyAttr"], location=None):
        self.name = name
        self.source: Union[LazyAttr, LazyObject] = source

        if isinstance(self.source, LazyObject):
            if isinstance(self.source._module, str):
                if self.source._imported is None:
                    self._module = self.source._module
                else:
                    self._module = f"{self.source._module}.{self.source}"
            else:
                self._module = str(self.source)
        elif isinstance(self.source, LazyAttr):
            self._module = f"{self.source._module}.{self.source.name}"
        self.location = location

    @property
    def module(self):
        return self._module

    def __call__(self, *args, **kwargs: Any) -> Any:
        raise RuntimeError()

    def __getattr__(self, name: str) -> "LazyAttr":
        return LazyAttr(name, self)

    def __deepcopy__(self, memo):
        return LazyAttr(self.name, self.source)

    def build(self) -> Any:
        obj = self.source.build()
        try:
            return getattr(obj, self.name)
        except AttributeError:
            raise ImportError(f"Failed to import {self.module}.{self.name} in " f"{self.location}")
        except ImportError as e:
            raise e

    def __str__(self) -> str:
        return self.name

    __repr__ = __str__

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__ = state


# adapt from https://github.com/open-mmlab/mmengine/blob/main/mmengine/utils/misc.py#L132
def is_seq_of(seq: Any, expected_type: Union[Type, tuple], seq_type: Type = None) -> bool:
    if seq_type is None:
        exp_seq_type = abc.Sequence
    else:
        assert isinstance(seq_type, type)
        exp_seq_type = seq_type
    if not isinstance(seq, exp_seq_type):
        return False
    for item in seq:
        if not isinstance(item, expected_type):
            return False
    return True