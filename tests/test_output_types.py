# SPDX-FileCopyrightText: 2022 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

"""Tests for --output-type and related output packaging options."""

from __future__ import annotations

import pikepdf
import pytest

from ocrmypdf.exceptions import ExitCode
from ocrmypdf.pdfa import file_claims_pdfa

from .conftest import check_ocrmypdf, run_ocrmypdf, run_ocrmypdf_api


def test_outputtype_none(resources, outtxt):
    exitcode = run_ocrmypdf_api(
        resources / 'trivial.pdf',
        '-',
        '--output-type=none',
        '--sidecar',
        outtxt,
        '--plugin',
        'tests/plugins/tesseract_noop.py',
    )
    assert exitcode == ExitCode.ok
    assert outtxt.exists()


def test_outputtype_none_bad_setup(resources, outpdf):
    p = run_ocrmypdf(
        resources / 'trivial.pdf',
        outpdf,
        '--output-type=none',
        '--plugin',
        'tests/plugins/tesseract_noop.py',
    )
    assert p.returncode == ExitCode.bad_args
    assert 'Set the output file to' in p.stderr


@pytest.mark.parametrize('pdfa_level', ['1', '2', '3'])
def test_pdfa_n(pdfa_level, resources, outpdf):
    check_ocrmypdf(
        resources / 'ccitt.pdf',
        outpdf,
        '--output-type',
        'pdfa-' + pdfa_level,
        '--plugin',
        'tests/plugins/tesseract_cache.py',
    )

    pdfa_info = file_claims_pdfa(outpdf)
    assert pdfa_info['conformance'] == f'PDF/A-{pdfa_level}b'


@pytest.mark.parametrize(
    'threshold, optimize, output_type, expected',
    [
        [1.0, 0, 'pdfa', False],
        [1.0, 0, 'pdf', False],
        [0.0, 0, 'pdfa', True],
        [0.0, 0, 'pdf', True],
        [1.0, 1, 'pdfa', False],
        [1.0, 1, 'pdf', False],
        [0.0, 1, 'pdfa', True],
        [0.0, 1, 'pdf', True],
    ],
)
def test_fast_web_view(resources, outpdf, threshold, optimize, output_type, expected):
    check_ocrmypdf(
        resources / 'trivial.pdf',
        outpdf,
        '--fast-web-view',
        threshold,
        '--optimize',
        optimize,
        '--output-type',
        output_type,
        '--plugin',
        'tests/plugins/tesseract_noop.py',
    )
    with pikepdf.open(outpdf) as pdf:
        assert pdf.is_linearized == expected
