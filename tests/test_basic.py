from pathlib import Path

import numpy as np
import pytest
from numpy.testing import assert_allclose

DATA_DIRECTORY = Path(__file__).parent / "data"


def test_math_util_angle_domain():
    assert not np.isfinite(
        great_circle_arc.angle([[0, 0, 0]], [[0, 0, 0]], [[0, 0, 0]])[0]
    )


def test_math_util_length_domain():
    with pytest.raises(ValueError):
        great_circle_arc.length([[np.nan, 0, 0]], [[0, 0, np.inf]])


def test_math_util_angle_nearly_coplanar_vec():
    # test from issue #222 + extra values
    vectors = [
        5 * [[1.0, 1.0, 1.0]],
        5 * [[1, 0.9999999, 1]],
        [[1, 0.5, 1], [1, 0.15, 1], [1, 0.001, 1], [1, 0.15, 1], [-1, 0.1, -1]],
    ]
    angles = great_circle_arc.angle(*vectors)

    assert_allclose(angles[:-1], np.pi, rtol=0, atol=1e-16)
    assert_allclose(angles[-1], 0, rtol=0, atol=1e-32)


def test_inner1d():
    vectors = [
        [1.0, 1.0, 1.0],
        3 * [1.0 / np.sqrt(3)],
        [1, 0.5, 1],
        [1, 0.15, 1],
        [-1, 0.1, -1],
    ]
    lengths = great_circle_arc._inner1d_np(vectors, vectors)

    assert_allclose(lengths, [3.0, 1.0, 2.25, 2.0225, 2.01], rtol=0, atol=1e-15)


@pytest.mark.skipif(math_util is None, reason="math_util C-ext is missing")
def test_math_util_inner1d():
    vectors = [
        [1.0, 1.0, 1.0],
        3 * [1.0 / np.sqrt(3)],
        [1, 0.5, 1],
        [1, 0.15, 1],
        [-1, 0.1, -1],
    ]
    lengths = math_util.inner1d(vectors, vectors)

    assert_allclose(lengths, [3.0, 1.0, 2.25, 2.0225, 2.01], rtol=0, atol=1e-15)
