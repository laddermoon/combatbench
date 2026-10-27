"""Humanoid21 adapter for the point5 business language.

This is the ONLY package under mocap/point5 allowed to import simulator code.
"""
from .adapter import point5_from_view

__all__ = ["point5_from_view"]
