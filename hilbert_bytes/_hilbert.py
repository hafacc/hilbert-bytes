"""Hilbert curve conversions on big-endian byte arrays."""

from operator import index

import numba as nb
import numpy as np
from numpy.typing import NDArray

# NOTE the signatures only match writable c-contiguous arrays, so the public
# functions copy anything else before calling the compiled ones


@nb.jit(nb.void(nb.uint8[:, :], nb.int64, nb.int64, nb.int64), cache=True, nogil=True)
def _fold(
    point: NDArray[np.uint8], dim: int, byte: int, bitmask: int
) -> None:  # pragma: no cover
    """Invert or exchange the bits of a point below one bit of one dimension."""
    _, nbytes = point.shape
    low = bitmask - 1
    if point[dim, byte] & bitmask:
        # the bit is on, so invert the lower bits of the first dimension
        point[0, byte] ^= low
        for rest in range(byte + 1, nbytes):
            point[0, rest] ^= 255
    else:
        # the bit is off, so exchange the lower bits with the first dimension
        flip = (point[0, byte] ^ point[dim, byte]) & low
        point[0, byte] ^= flip
        point[dim, byte] ^= flip
        for rest in range(byte + 1, nbytes):
            flip = point[0, rest] ^ point[dim, rest]
            point[0, rest] ^= flip
            point[dim, rest] ^= flip


@nb.jit(nb.void(nb.uint8[:, :, ::1], nb.uint8[:, ::1]), cache=True, nogil=True)
def _encode(
    points: NDArray[np.uint8], indices: NDArray[np.uint8]
) -> None:  # pragma: no cover
    """Fill zeroed indices from points, overwriting the points."""
    _, ndim, nbytes = points.shape
    # numba doesn't support strict
    for point, index_bytes in zip(points, indices):  # noqa: B905
        for byte in range(nbytes):
            for bit in range(8):
                bitmask = 128 >> bit
                for dim in range(ndim):
                    _fold(point, dim, byte, bitmask)

        # interleave the bits of the dimensions; the running xor undoes the
        # gray code of the interleaved number
        parity = 0
        position = 0
        for byte in range(nbytes):
            for bit in range(8):
                for dim in range(ndim):
                    parity ^= (point[dim, byte] >> (7 - bit)) & 1
                    index_bytes[position >> 3] |= parity << (7 - (position & 7))
                    position += 1


@nb.jit(nb.void(nb.uint8[:, ::1], nb.uint8[:, :, ::1]), cache=True, nogil=True)
def _decode(
    indices: NDArray[np.uint8], points: NDArray[np.uint8]
) -> None:  # pragma: no cover
    """Fill zeroed points from indices."""
    _, ndim, nbytes = points.shape
    # numba doesn't support strict
    for index_bytes, point in zip(indices, points):  # noqa: B905
        # gray code the index while dealing its bits out to the dimensions
        previous = 0
        position = 0
        for index_byte in index_bytes:
            for bit in range(8):
                current = (index_byte >> (7 - bit)) & 1
                depth, dim = divmod(position, ndim)
                point[dim, depth >> 3] |= (current ^ previous) << (7 - (depth & 7))
                previous = current
                position += 1

        for byte in range(nbytes - 1, -1, -1):
            for bit in range(8):
                bitmask = 1 << bit
                for dim in range(ndim - 1, -1, -1):
                    _fold(point, dim, byte, bitmask)


def _as_bytes(array: NDArray[np.uint8], name: str, ndim: int) -> NDArray[np.uint8]:
    """Check that an argument is a uint8 array with ndim dimensions."""
    result = np.asarray(array)
    if result.dtype != np.uint8:
        raise TypeError(
            f"{name} must have dtype uint8, but got {result.dtype}; cast wider "
            'integers to big-endian and view them as bytes, e.g. `.astype(">u8").view("u1")`'
        )
    elif result.ndim != ndim:
        raise ValueError(
            f"{name} must have {ndim} dimensions, but got shape {result.shape}"
        )
    else:
        return result


def encode(points: NDArray[np.uint8]) -> NDArray[np.uint8]:
    """Encode d-dimensional points into their indices on a hilbert curve.

    This function takes points in a d-dimensional space, and converts them to
    their index on the hilbert curve. All numbers are represented as arbitrary
    precision integers in big-endian form.

    Example
    -------
    If you want to use this with native multi-byte integers, you can first cast
    them to a big-endian variant, then view it as bytes.

    ::

        points = ...
        point_bytes = points[..., None].astype(">u8").view("u1")
        res = hilbert_bytes.encode(point_bytes)

    Parameters
    ----------
    points : (n, d, p)
        A collection of n, d-dimensional points stored as p-byte big-endian
        unsigned integers.

    Returns
    -------
    indices : (n, dp)
        A collection of n big-endian unsigned integers that correspond to the
        index along the hilbert-curve for the input points.

    Raises
    ------
    TypeError
        If `points` isn't a uint8 array.
    ValueError
        If `points` doesn't have three dimensions.
    """
    # always copy because the compiled loop overwrites its input
    work = np.array(_as_bytes(points, "points", 3), order="C")
    num, ndim, nbytes = work.shape
    indices = np.zeros((num, ndim * nbytes), np.uint8)
    _encode(work, indices)
    return indices


def decode(indices: NDArray[np.uint8], ndim: int) -> NDArray[np.uint8]:
    """Decode dp-byte indices into d-dimensional points.

    This function takes indices on the hilbert curve, and the output dimension
    and converts them to their corresponding points. All numbers are represented
    as arbitrary precision integers in big-endian form.

    `ndim` must divide the last dimension. If it doesn't this will error. You
    may want to treat the input as a smaller number in a higher dimensional
    space, in which case you just need to prefix with correct number of zero
    bytes so that `ndim` does divide.

    Example
    -------
    If you want to use this with native multi-byte integers, you can first cast
    them to a big-endian variant, then view it as bytes.

    ::

        indices = ...
        index_bytes = indices[..., None].astype(">u8").view("u1")
        res = hilbert_bytes.decode(index_bytes, 2)

    Parameters
    ----------
    indices : (n, dp)
        A collection of n indices stored as dp-byte big-endian unsigned
        integers.
    ndim : d
        The dimension of points to decode into. It must divide dp, but you can
        always zero pad the left of indices.

    Returns
    -------
    points : (n, d, p)
        A collection of n d-dimensional points that correspond to the indices
        along the hilbert-curve.

    Raises
    ------
    TypeError
        If `indices` isn't a uint8 array, or `ndim` isn't an integer.
    ValueError
        If `indices` doesn't have two dimensions, or `ndim` isn't positive or
        doesn't divide dp.
    """
    num_dims = index(ndim)
    if num_dims < 1:
        raise ValueError(f"ndim must be positive, but got {num_dims}")
    source = np.require(_as_bytes(indices, "indices", 2), requirements=["C", "W"])
    num, total_bytes = source.shape
    nbytes, extra = divmod(total_bytes, num_dims)
    if extra:
        raise ValueError(
            f"ndim ({num_dims}) must evenly divide the number of index bytes ({total_bytes})"
        )
    points = np.zeros((num, num_dims, nbytes), np.uint8)
    _decode(source, points)
    return points
