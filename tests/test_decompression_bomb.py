# SPDX-FileCopyrightText: 2022 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

"""Tests for Pillow decompression bomb handling and --max-image-mpixels."""

from __future__ import annotations

import pytest

from .conftest import run_ocrmypdf_api


def test_decompression_bomb_error(resources, outpdf, caplog):
    run_ocrmypdf_api(resources / 'hugemono.pdf', outpdf)
    assert 'decompression bomb' in caplog.text
    assert 'max-image-mpixels' in caplog.text


@pytest.mark.slow
def test_decompression_bomb_succeeds(resources, outpdf):
    exitcode = run_ocrmypdf_api(
        resources / 'hugemono.pdf', outpdf, '--max-image-mpixels', '2000'
    )
    assert exitcode == 0
