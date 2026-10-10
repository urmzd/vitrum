"""Vitrum core — public API.

``vitrum`` is a namespace package; the importable surface lives in the
member distributions (``vitrum.core``, ``vitrum.architectures``,
``vitrum.data``, ``vitrum.training``, ``vitrum.cli``). This
module re-exports the pieces most callers need:

    from vitrum.core import AgiModel, build_model, get_registry, register
"""
from vitrum.core.base import AgiModel, CommonSettings
from vitrum.core.registry import build_model, get_registry, register

__all__ = [
    "AgiModel",
    "CommonSettings",
    "build_model",
    "get_registry",
    "register",
]
