# SPDX-FileCopyrightText: 2022 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

"""Tests for Pillow decompression bomb handling and --max-image-mpixels."""

from __future__ import annotations

import pytest

from .conftest import run_ocrmypdf_api


def test_decompression_bomb_error(resources, outpdf, caplog):
    # ccitt.pdf is 2550x3300 (8.4 MP), which is more than 2x the 1 MP limit set
    # here, so Pillow raises DecompressionBombError at the same rasterization
    # call site that hugemono.pdf reaches - but far more cheaply.
    run_ocrmypdf_api(resources / 'ccitt.pdf', outpdf, '--max-image-mpixels', '1')
    assert 'decompression bomb' in caplog.text
    assert 'max-image-mpixels' in caplog.text


@pytest.mark.slow
def test_decompression_bomb_succeeds(resources, outpdf):
    exitcode = run_ocrmypdf_api(
        resources / 'hugemono.pdf', outpdf, '--max-image-mpixels', '2000'
    )
    assert exitcode == 0
