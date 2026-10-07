"""Hilbert bytes.

Hilbert Bytes is a python library for converting to and from points in
d-dimensions and their corresponding index on a hilbert curve. It's similar to
`hilbertcurve <https://pypi.org/project/hilbertcurve/>`_ and
`numpy-hilbert-curve <https://pypi.org/project/numpy-hilbert-curve/>`_ but is
faster and more space efficient than either by keeping manipulations at the byte
level, and using numba to compile the results. It also uses arbitrary precision
integers, allowing you to make the grid arbitrarily fine.

Use `encode` and `decode` to convert between the spaces.
"""

from ._hilbert import decode, encode

__all__ = ("decode", "encode")
