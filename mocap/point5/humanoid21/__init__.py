"""Humanoid21 adapter for the point5 business language.

This is the ONLY package under mocap/point5 allowed to import simulator code.
"""
from .adapter import measure_point5

__all__ = ["measure_point5"]
