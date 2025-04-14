from abc import ABC, abstractmethod
from typing import Any, Iterable, Self

from mininn.parameter import Parameter


class Module(ABC):
    _is_training: bool = True

    @abstractmethod
    def forward(self, *args: Any, **kwargs: Any) -> Any:
        pass

    @abstractmethod
    def backward(self, *gradients: Any, **kwargs: Any) -> Any:
        pass

    def parameters(self, recurse: bool = True) -> Iterable[Parameter]:
        return iter([])

    def set_training(self, value: bool) -> Self:
        self._is_training = value
        return self

    def is_training(self) -> bool:
        return self._is_training

    def __call__(self, *args: Any, **kwargs: Any) -> Any:
        return self.forward(*args, **kwargs)
