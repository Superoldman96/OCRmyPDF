# SPDX-FileCopyrightText: 2022 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

"""Deliberate whole-pipeline canaries."""

from __future__ import annotations

import pytest

from .conftest import RENDERERS, check_ocrmypdf, have_unpaper


def test_quick(resources, outpdf):
    check_ocrmypdf(
        resources / 'ccitt.pdf', outpdf, '--plugin', 'tests/plugins/tesseract_cache.py'
    )


@pytest.mark.parametrize('renderer', RENDERERS)
@pytest.mark.parametrize('output_type', ['pdf', 'pdfa'])
def test_maximum_options(renderer, output_type, multipage, outpdf):
    check_ocrmypdf(
        multipage,
        outpdf,
        '-d',
        '-ci' if have_unpaper() else None,
        '-f',
        '-k',
        '--oversample',
        '300',
        '--skip-big',
        '10',
        '--title',
        'Too Many Weird Files',
        '--author',
        'py.test',
        '--pdf-renderer',
        renderer,
        '--output-type',
        output_type,
        '--plugin',
        'tests/plugins/tesseract_cache.py',
    )
