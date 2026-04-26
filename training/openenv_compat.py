"""Training-only compatibility shims for direct local environment runs.

The OpenEnv server stack is the production interface for this project.  The
training script only needs the model base classes and State container, so this
module installs tiny stand-ins when OpenEnv or Pydantic are not available in a
local hackathon notebook / shell.
"""

from __future__ import annotations

import copy
import sys
import types
from dataclasses import dataclass
from typing import Any, Callable


class _FieldInfo:
    def __init__(
        self,
        default: Any = ...,
        *,
        default_factory: Callable[[], Any] | None = None,
        **_: Any,
    ) -> None:
        self.default = default
        self.default_factory = default_factory

    def make_default(self) -> Any:
        if self.default_factory is not None:
            return self.default_factory()
        if self.default is ...:
            return None
        return copy.deepcopy(self.default)


def _field(
    default: Any = ...,
    *,
    default_factory: Callable[[], Any] | None = None,
    **kwargs: Any,
) -> _FieldInfo:
    return _FieldInfo(default, default_factory=default_factory, **kwargs)


class _SimpleModel:
    def __init__(self, **kwargs: Any) -> None:
        fields: dict[str, Any] = {}
        for cls in reversed(self.__class__.mro()):
            fields.update(getattr(cls, "__annotations__", {}))

        for name in fields:
            if name.startswith("_"):
                continue
            if name in kwargs:
                value = kwargs.pop(name)
            else:
                value = self._default_for(name)
            setattr(self, name, value)

        for name, value in kwargs.items():
            setattr(self, name, value)

    def _default_for(self, name: str) -> Any:
        for cls in self.__class__.mro():
            if name not in cls.__dict__:
                continue
            default = cls.__dict__[name]
            if isinstance(default, _FieldInfo):
                return default.make_default()
            if hasattr(default, "default") or hasattr(default, "default_factory"):
                factory = getattr(default, "default_factory", None)
                if factory is not None:
                    return factory()
                raw_default = getattr(default, "default", None)
                if raw_default is ... or "Undefined" in repr(raw_default):
                    return None
                return copy.deepcopy(raw_default)
            if default is ...:
                return None
            return copy.deepcopy(default)
        return None

    def model_dump(self) -> dict[str, Any]:
        return dict(self.__dict__)

    def dict(self) -> dict[str, Any]:
        return self.model_dump()

    def __repr__(self) -> str:
        args = ", ".join(f"{key}={value!r}" for key, value in self.__dict__.items())
        return f"{self.__class__.__name__}({args})"


class _Action(_SimpleModel):
    pass


class _Observation(_SimpleModel):
    done: bool = False
    reward: float | None = None
    metadata: dict[str, Any] = _FieldInfo(default_factory=dict)


@dataclass
class _State:
    episode_id: str | None = None
    step_count: int = 0


class _Environment:
    pass


class _EnvClient:
    def __class_getitem__(cls, _item: Any) -> type["_EnvClient"]:
        return cls


@dataclass
class _StepResult:
    observation: Any
    reward: float | None = None
    done: bool = False

    def __class_getitem__(cls, _item: Any) -> type["_StepResult"]:
        return cls


def _ensure_pydantic_stub() -> None:
    try:
        import pydantic  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    pydantic_mod = types.ModuleType("pydantic")
    pydantic_mod.Field = _field
    sys.modules.setdefault("pydantic", pydantic_mod)


def _ensure_openenv_stub() -> None:
    try:
        from openenv.core.env_server.types import Action, Observation, State  # noqa: F401
        from openenv.core.env_server.interfaces import Environment  # noqa: F401
        return
    except ModuleNotFoundError:
        pass

    openenv_mod = sys.modules.setdefault("openenv", types.ModuleType("openenv"))
    core_mod = sys.modules.setdefault("openenv.core", types.ModuleType("openenv.core"))
    env_server_mod = sys.modules.setdefault(
        "openenv.core.env_server",
        types.ModuleType("openenv.core.env_server"),
    )
    types_mod = sys.modules.setdefault(
        "openenv.core.env_server.types",
        types.ModuleType("openenv.core.env_server.types"),
    )
    interfaces_mod = sys.modules.setdefault(
        "openenv.core.env_server.interfaces",
        types.ModuleType("openenv.core.env_server.interfaces"),
    )
    client_types_mod = sys.modules.setdefault(
        "openenv.core.client_types",
        types.ModuleType("openenv.core.client_types"),
    )

    types_mod.Action = _Action
    types_mod.Observation = _Observation
    types_mod.State = _State
    interfaces_mod.Environment = _Environment
    core_mod.EnvClient = _EnvClient
    client_types_mod.StepResult = _StepResult

    openenv_mod.core = core_mod
    core_mod.env_server = env_server_mod
    core_mod.client_types = client_types_mod
    env_server_mod.types = types_mod
    env_server_mod.interfaces = interfaces_mod


def ensure_training_runtime() -> None:
    """Make direct training imports work with or without OpenEnv installed."""

    _ensure_pydantic_stub()
    _ensure_openenv_stub()
