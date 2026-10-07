"""Test for hilbert_bytes."""

from concurrent.futures import ThreadPoolExecutor

import hilbert
import numpy as np
import pytest
from numpy.typing import NDArray

import hilbert_bytes


def to_ints(byte_array: NDArray[np.uint8]) -> NDArray[np.uint64]:
    """Convert up to eight trailing big-endian bytes into native integers."""
    *lead, nbytes = byte_array.shape
    padded = np.zeros((*lead, 8), "u1")
    padded[..., 8 - nbytes :] = byte_array
    return padded.view(">u8")[..., 0].astype("u8")


def test_single_dim() -> None:
    """Test hilbert index on single dimension.

    This should be a noop.
    """
    nums = np.arange(0, 1 << 16, dtype="u2")
    byte_nums = nums[..., None].astype(">u2").view("u1")

    expected_points = hilbert.decode(nums, 1, 16)
    byte_points = hilbert_bytes.decode(byte_nums, 1)
    actual_points = byte_points.view(">u2").astype("u8")[..., 0]
    assert np.all(expected_points == actual_points)

    expected_nums = hilbert.encode(expected_points, 1, 16)
    assert np.all(nums == expected_nums)
    actual_nums = hilbert_bytes.encode(byte_points)
    assert np.all(byte_nums == actual_nums)


def test_single_dim_single_byte() -> None:
    """Test single-dimension single-byte indices."""
    nums = np.arange(0, 1 << 8, dtype="u1")
    byte_nums = nums[..., None]

    expected_points = hilbert.decode(nums, 1, 8)
    byte_points = hilbert_bytes.decode(byte_nums, 1)
    actual_points = byte_points.view("u1").astype("u8")[..., 0]
    assert np.all(expected_points == actual_points)

    actual_nums = hilbert_bytes.encode(byte_points)
    assert np.all(byte_nums == actual_nums)


def test_two_dims() -> None:
    """Test hilbert bytes on two dimensions."""
    nums = np.arange(0, 1 << 16, dtype="u2")
    byte_nums = nums[..., None].astype(">u2").view("u1")

    expected_points = hilbert.decode(nums, 2, 8)
    byte_points = hilbert_bytes.decode(byte_nums, 2)
    assert np.all(expected_points == byte_points[..., 0])

    expected_nums = hilbert.encode(expected_points, 2, 8)
    assert np.all(nums == expected_nums)
    actual_nums = hilbert_bytes.encode(byte_points)
    assert np.all(byte_nums == actual_nums)


def test_three_dims() -> None:
    """Test with three dimensions."""
    nums = np.arange(0, 1 << 24, 53, dtype="u4")
    byte_nums = nums[..., None].astype(">u4").view("u1")[:, 1:]

    expected_points = hilbert.decode(nums, 3, 8)
    byte_points = hilbert_bytes.decode(byte_nums, 3)
    assert np.all(expected_points == byte_points[..., 0])

    expected_nums = hilbert.encode(expected_points, 3, 8)
    assert np.all(nums == expected_nums)
    actual_nums = hilbert_bytes.encode(byte_points)
    assert np.all(byte_nums == actual_nums)


@pytest.mark.parametrize(
    ("ndim", "nbytes"), [(2, 2), (2, 3), (2, 4), (3, 2), (4, 2), (5, 1), (8, 1)]
)
def test_matches_reference(ndim: int, nbytes: int) -> None:
    """Test random multi-byte indices against numpy-hilbert-curve."""
    rng = np.random.default_rng(0)
    byte_nums = rng.integers(0, 256, size=(500, ndim * nbytes), dtype="u1")
    nums = to_ints(byte_nums)

    expected_points = hilbert.decode(nums, ndim, nbytes * 8)
    byte_points = hilbert_bytes.decode(byte_nums, ndim)
    assert np.all(expected_points == to_ints(byte_points))

    actual_nums = hilbert_bytes.encode(byte_points)
    assert np.all(byte_nums == actual_nums)


@pytest.mark.parametrize(("ndim", "nbytes"), [(1, 20), (2, 16), (3, 11), (10, 5)])
def test_wide_round_trip(ndim: int, nbytes: int) -> None:
    """Test indices too wide for the reference library."""
    rng = np.random.default_rng(0)
    byte_nums = rng.integers(0, 256, size=(100, ndim * nbytes), dtype="u1")
    # clear the last bit so the next index is one bit away
    byte_nums[:, -1] &= 0xFE
    next_nums = byte_nums.copy()
    next_nums[:, -1] |= 1

    byte_points = hilbert_bytes.decode(byte_nums, ndim)
    assert np.all(byte_nums == hilbert_bytes.encode(byte_points))

    # neighbors on the curve are neighbors in space
    next_points = hilbert_bytes.decode(next_nums, ndim)
    diffs = byte_points[..., -1].astype("i8") - next_points[..., -1]
    assert np.all(byte_points[..., :-1] == next_points[..., :-1])
    assert np.all(np.abs(diffs).sum(1) == 1)


def test_encode_does_not_mutate_input() -> None:
    """Encode must not clobber the caller's input array."""
    nums = np.arange(0, 1 << 16, dtype="u2")
    byte_points = nums[..., None].astype(">u2").view("u1")[:, None, :]
    original = byte_points.copy()

    first = hilbert_bytes.encode(byte_points)
    assert np.array_equal(byte_points, original)

    # encoding again yields the same result rather than garbage
    second = hilbert_bytes.encode(byte_points)
    assert np.array_equal(first, second)


def test_read_only_and_strided() -> None:
    """Test inputs that are read-only or not contiguous."""
    rng = np.random.default_rng(0)
    data = rng.integers(0, 256, size=24, dtype="u1").tobytes()
    byte_points = np.frombuffer(data, "u1").reshape(4, 3, 2)
    assert not byte_points.flags.writeable

    byte_nums = hilbert_bytes.encode(byte_points)
    byte_nums.setflags(write=False)
    assert np.all(byte_points == hilbert_bytes.decode(byte_nums, 3))

    reversed_nums = hilbert_bytes.encode(byte_points[::-1, ::-1])
    assert np.all(reversed_nums == hilbert_bytes.encode(byte_points[::-1, ::-1].copy()))
    assert np.all(
        hilbert_bytes.decode(byte_nums[::2], 3)
        == hilbert_bytes.decode(byte_nums, 3)[::2]
    )


def test_decode_is_contiguous() -> None:
    """Decoded points can be viewed as multi-byte integers."""
    byte_nums = np.arange(32, dtype="u1").reshape(2, 16)
    byte_points = hilbert_bytes.decode(byte_nums, 2)
    assert byte_points.view(">u8").shape == (2, 2, 1)


def test_threads() -> None:
    """Test calling from several threads at once."""
    rng = np.random.default_rng(0)
    byte_points = rng.integers(0, 256, size=(1000, 3, 4), dtype="u1")
    expected = hilbert_bytes.encode(byte_points)
    with ThreadPoolExecutor(8) as pool:
        results = pool.map(hilbert_bytes.encode, [byte_points] * 32)
        assert all(np.array_equal(expected, result) for result in results)


def test_bad_arrays() -> None:
    """Test exceptions for arrays of the wrong type or shape."""
    byte_points = np.zeros((4, 3, 2), "u1")
    with pytest.raises(TypeError, match="points must have dtype uint8"):
        hilbert_bytes.encode(byte_points.astype("i8"))  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="points must have 3 dimensions"):
        hilbert_bytes.encode(byte_points[0])
    with pytest.raises(TypeError, match="indices must have dtype uint8"):
        hilbert_bytes.decode(byte_points[0].astype("u2"), 2)  # pyright: ignore[reportArgumentType]
    with pytest.raises(ValueError, match="indices must have 2 dimensions"):
        hilbert_bytes.decode(byte_points, 2)


def test_bad_ndim() -> None:
    """Test exceptions when ndim is invalid or bytes don't align."""
    nums = np.arange(0, 1 << 8, dtype="u1")[:, None]
    with pytest.raises(ValueError, match="evenly divide"):
        hilbert_bytes.decode(nums, 2)
    with pytest.raises(ValueError, match="positive"):
        hilbert_bytes.decode(nums, 0)
    with pytest.raises(ValueError, match="positive"):
        hilbert_bytes.decode(nums, -1)
    with pytest.raises(TypeError):
        hilbert_bytes.decode(nums, 1.0)  # pyright: ignore[reportArgumentType]
