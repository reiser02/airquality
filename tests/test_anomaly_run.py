from __future__ import annotations

from argparse import ArgumentTypeError

import pytest

from airquality.anomaly.run import _parse_sub_pca_components


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        ("all", (None,)),
        ("none", (None,)),
        ("4", (4,)),
        (8, (8,)),
        ("all,4,8", (None, 4, 8)),
        (["all", "4", "8"], (None, 4, 8)),
    ],
)
def test_parse_sub_pca_components(value, expected):
    assert _parse_sub_pca_components(value) == expected


@pytest.mark.parametrize("value", ["0", "-1", "four"])
def test_parse_sub_pca_components_rejects_invalid_values(value):
    with pytest.raises(ArgumentTypeError):
        _parse_sub_pca_components(value)
