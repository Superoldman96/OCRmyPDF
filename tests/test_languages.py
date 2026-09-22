# SPDX-FileCopyrightText: 2022 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

"""Tests for language selection and user word lists."""

from __future__ import annotations

import pikepdf
import pytest

from ocrmypdf._exec import tesseract
from ocrmypdf._metadata import _set_language
from ocrmypdf.exceptions import MissingDependencyError
from ocrmypdf.languages import iso_639_2_from_3

from .conftest import check_ocrmypdf, is_macos, run_ocrmypdf_api


@pytest.mark.skipif(
    is_macos(),
    reason="takes too long to install language packs in macOS homebrew",
)
def test_german(resources, outdir):
    # Produce a sidecar too - implicit test that system locale is set up
    # properly. It is fine that we are testing -l deu on a French file because
    # we are exercising the functionality not going for accuracy.
    sidecar = outdir / 'francais.txt'
    try:
        check_ocrmypdf(
            resources / 'francais.pdf',
            outdir / 'francais.pdf',
            '-l',
            'deu',  # more commonly installed
            '--sidecar',
            sidecar,
            '--output-type',
            'pdf',
            '--plugin',
            'tests/plugins/tesseract_cache.py',
        )
    except MissingDependencyError:
        if 'deu' not in tesseract.get_languages():
            pytest.xfail(reason="tesseract-deu language pack not installed")
        raise


def test_klingon(resources, outpdf):
    with pytest.raises(MissingDependencyError):
        run_ocrmypdf_api(resources / 'francais.pdf', outpdf, '-l', 'klz')


def test_user_words_ocr(resources, outdir):
    # Does not actually test if --user-words causes output to differ
    word_list = outdir / 'wordlist.txt'
    sidecar_after = outdir / 'sidecar.txt'

    with word_list.open('w') as f:
        f.write('cromulent\n')  # a perfectly cromulent word

    check_ocrmypdf(
        resources / 'crom.png',
        outdir / 'out.pdf',
        '--image-dpi',
        150,
        '--sidecar',
        sidecar_after,
        '--user-words',
        word_list,
    )


@pytest.mark.parametrize(
    'iso3, expected',
    [
        # ISO 639-2/T terminology codes (the ones Tesseract uses)
        ('deu', 'de'),
        ('fra', 'fr'),
        ('nld', 'nl'),
        ('ces', 'cs'),
        ('ell', 'el'),
        ('zho', 'zh'),
        # ISO 639-2/B bibliographic codes
        ('ger', 'de'),
        ('fre', 'fr'),
        ('dut', 'nl'),
        ('cze', 'cs'),
        ('gre', 'el'),
        ('chi', 'zh'),
        # Languages with a single 3-letter code and an alpha-2 code
        ('eng', 'en'),
        ('spa', 'es'),
        ('san', 'sa'),
        # Languages with no alpha-2 code: BCP 47 says use the 3-letter code
        ('ast', 'ast'),
        ('chr', 'chr'),
        # Unknown codes
        ('chi_sim', ''),
        ('klz', ''),
        ('', ''),
    ],
)
def test_iso_639_2_from_3(iso3, expected):
    assert iso_639_2_from_3(iso3) == expected


def test_set_language():
    pdf = pikepdf.new()
    _set_language(pdf, ['deu'])
    assert pdf.Root.Lang == 'de'


def test_set_language_does_not_overwrite():
    pdf = pikepdf.new()
    pdf.Root.Lang = pikepdf.String('en-CA')
    _set_language(pdf, ['deu'])
    assert pdf.Root.Lang == 'en-CA'


def test_set_language_unknown():
    pdf = pikepdf.new()
    _set_language(pdf, ['chi_sim'])
    assert pikepdf.Name.Lang not in pdf.Root
