# SPDX-FileCopyrightText: 2022 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import logging
import os
import subprocess
from os import fspath
from pathlib import Path

import pytest

from ocrmypdf import pdfinfo
from ocrmypdf._exec import tesseract
from ocrmypdf.exceptions import BadArgsError, ExitCode, MissingDependencyError

from .conftest import RENDERERS, check_ocrmypdf, run_ocrmypdf, run_ocrmypdf_api

# pylint: disable=redefined-outer-name


@pytest.mark.parametrize('basename', ['graph_ocred.pdf', 'cardinal.pdf'])
def test_skip_pages_does_not_replicate(resources, basename, outdir):
    infile = resources / basename
    outpdf = outdir / basename

    check_ocrmypdf(
        infile,
        outpdf,
        '--pdf-renderer',
        'sandwich',
        '--force-ocr',
        '--tesseract-timeout',
        '0',
    )

    info_in = pdfinfo.PdfInfo(infile)

    info = pdfinfo.PdfInfo(outpdf)
    for page in info:
        assert len(page.images) == 1, "skipped page was replicated"

    for n, info_out_n in enumerate(info):
        assert info_out_n.width_inches == info_in[n].width_inches, "output resized"
        assert info_out_n.height_inches == info_in[n].height_inches, "output resized"


def test_content_preservation(resources, outpdf):
    infile = resources / 'masks.pdf'

    check_ocrmypdf(
        infile, outpdf, '--pdf-renderer', 'fpdf2', '--tesseract-timeout', '0'
    )

    info = pdfinfo.PdfInfo(outpdf)
    page = info[0]
    assert len(page.images) > 1, "masks were rasterized"


@pytest.mark.skipif(
    tesseract.version() >= tesseract.TesseractVersion('5'), reason="doesn't fool Tess 5"
)
def test_no_languages(tmp_path, monkeypatch):
    (tmp_path / 'tessdata').mkdir()
    monkeypatch.setenv('TESSDATA_PREFIX', fspath(tmp_path))
    with pytest.raises(MissingDependencyError):
        tesseract.get_languages()


def test_image_too_large_hocr(monkeypatch, resources, outdir):
    def dummy_run(args, *, env=None, **kwargs):
        raise subprocess.CalledProcessError(1, 'tesseract', output=b'Image too large')

    monkeypatch.setattr(tesseract, 'run', dummy_run)
    tesseract.generate_hocr(
        input_file=resources / 'crom.png',
        output_hocr=outdir / 'out.hocr',
        output_text=outdir / 'out.txt',
        languages=['eng'],
        engine_mode=None,
        tessconfig=[],
        timeout=180.0,
        pagesegmode=None,
        thresholding=0,
        user_words=None,
        user_patterns=None,
    )
    assert Path(outdir / 'out.hocr').read_text() == ''


def test_image_too_large_pdf(monkeypatch, resources, outdir):
    def dummy_run(args, *, env=None, **kwargs):
        raise subprocess.CalledProcessError(1, 'tesseract', output=b'Image too large')

    monkeypatch.setattr(tesseract, 'run', dummy_run)
    tesseract.generate_pdf(
        input_file=resources / 'crom.png',
        output_pdf=outdir / 'pdf.pdf',
        output_text=outdir / 'txt.txt',
        languages=['eng'],
        engine_mode=None,
        tessconfig=[],
        timeout=180.0,
        pagesegmode=None,
        thresholding=0,
        user_words=None,
        user_patterns=None,
    )
    assert Path(outdir / 'txt.txt').read_text() == '[skipped page]'
    if os.name != 'nt':  # different semantics
        assert Path(outdir / 'pdf.pdf').stat().st_size == 0


def test_timeout(caplog):
    tesseract.page_timedout(5)
    assert "took too long" in caplog.text


@pytest.mark.parametrize(
    'in_, logged',
    [
        (b'Tesseract Open Source', ''),
        (b'lots of diacritics blah blah', 'diacritics'),
        (b'Warning in pixReadMem', ''),
        (b'OSD: Weak margin', 'unsure about page orientation'),
        (b'Error in pixScanForForeground', ''),
        (b'Error in boxClipToRectangle', ''),
        (b'an unexpected error', 'an unexpected error'),
        (b'a dire warning', 'a dire warning'),
        (b'an innocent message', 'innocent'),
        (b'\x7f\x7f\x80innocent unicode failure', 'innocent'),
    ],
)
def test_tesseract_log_output(caplog, in_, logged):
    caplog.set_level(logging.INFO)
    tesseract.tesseract_log_output(in_)
    if logged == '':
        assert caplog.text == ''
    else:
        assert logged in caplog.text


def test_tesseract_log_output_diacritics_raw(caplog):
    """Diacritics branch keeps the interpreted hint and surfaces raw (#1566)."""
    caplog.set_level(logging.DEBUG)
    tesseract.tesseract_log_output(b'lots of diacritics blah blah')
    assert 'possibly poor OCR' in caplog.text  # interpreted hint retained
    assert 'lots of diacritics blah blah' in caplog.text  # raw message surfaced


def test_tesseract_log_output_raises(caplog):
    with pytest.raises(tesseract.TesseractConfigError):
        tesseract.tesseract_log_output(b'parameter not found: moo')
    assert 'not found' in caplog.text


def test_tesseract_log_output_raises_on_missing_config(caplog):
    with pytest.raises(tesseract.TesseractConfigError) as excinfo:
        tesseract.tesseract_log_output(b"read_params_file: Can't open hocr")
    assert 'hocr' in excinfo.value.args[0]
    assert 'read_params_file' in caplog.text


def test_blocked_language(resources, no_outpdf):
    infile = resources / 'masks.pdf'
    for bad_lang in ['osd', 'equ']:
        with pytest.raises(BadArgsError):
            run_ocrmypdf_api(infile, no_outpdf, '-l', bad_lang)


@pytest.mark.parametrize('renderer', RENDERERS)
def test_tesseract_crash(renderer, resources, no_outpdf, caplog):
    exitcode = run_ocrmypdf_api(
        resources / 'ccitt.pdf',
        no_outpdf,
        '-v',
        '1',
        '--pdf-renderer',
        renderer,
        '--plugin',
        'tests/plugins/tesseract_crash.py',
    )
    assert exitcode == ExitCode.child_process_error
    assert not no_outpdf.exists()
    assert "SubprocessOutputError" in caplog.text


def test_tesseract_crash_autorotate(resources, no_outpdf, caplog):
    exitcode = run_ocrmypdf_api(
        resources / 'ccitt.pdf',
        no_outpdf,
        '-r',
        '--plugin',
        'tests/plugins/tesseract_crash.py',
    )
    assert exitcode == ExitCode.child_process_error
    assert not no_outpdf.exists()
    assert "uncaught exception" in caplog.text


@pytest.mark.parametrize('renderer', RENDERERS)
@pytest.mark.slow
def test_tesseract_image_too_big(renderer, resources, outpdf):
    check_ocrmypdf(
        resources / 'hugemono.pdf',
        outpdf,
        '-r',
        '--pdf-renderer',
        renderer,
        '--max-image-mpixels',
        '0',
        '--plugin',
        'tests/plugins/tesseract_big_image_error.py',
    )


@pytest.fixture
def valid_tess_config(outdir):
    cfg_file = outdir / 'test.cfg'
    with cfg_file.open('w') as f:
        f.write(
            '''\
load_system_dawg 0
language_model_penalty_non_dict_word 0
language_model_penalty_non_freq_dict_word 0
'''
        )
    yield cfg_file


def test_tesseract_config_valid(resources, valid_tess_config, outpdf):
    check_ocrmypdf(
        resources / '3small.pdf',
        outpdf,
        '--tesseract-config',
        valid_tess_config,
        '--pages',
        '1',
    )


@pytest.fixture
def invalid_tess_config(outdir):
    cfg_file = outdir / 'test.cfg'
    with cfg_file.open('w') as f:
        f.write(
            '''\
THIS FILE IS INVALID
'''
        )
    yield cfg_file


@pytest.mark.slow  # This test sometimes times out in CI
@pytest.mark.parametrize('renderer', RENDERERS)
def test_tesseract_config_invalid(renderer, resources, invalid_tess_config, outpdf):
    p = run_ocrmypdf(
        resources / 'ccitt.pdf',
        outpdf,
        '--pdf-renderer',
        renderer,
        '--tesseract-config',
        invalid_tess_config,
    )
    assert (
        "parameter not found" in p.stderr.lower()
        or "error occurred while parsing" in p.stderr.lower()
    ), "No error message"
    assert p.returncode == ExitCode.invalid_config


@pytest.mark.parametrize('value', ['auto', 'otsu', 'adaptive-otsu', 'sauvola'])
def test_tesseract_thresholding(value, resources, outpdf):
    check_ocrmypdf(
        resources / 'trivial.pdf',
        outpdf,
        '--tesseract-thresholding',
        value,
        '--plugin',
        'tests/plugins/tesseract_cache.py',
    )


@pytest.mark.parametrize('value', ['abcxyz'])
def test_tesseract_thresholding_invalid(value, resources, no_outpdf):
    with pytest.raises(SystemExit, match='2'):
        run_ocrmypdf_api(
            resources / 'trivial.pdf',
            no_outpdf,
            '--tesseract-thresholding',
            value,
            '--plugin',
            'tests/plugins/tesseract_cache.py',
        )


@pytest.mark.skipif(
    tesseract.version() >= tesseract.TesseractVersion('5'),
    reason="tess 5 tries harder to find its files",
)
def test_tesseract_missing_tessdata(monkeypatch, resources, no_outpdf, tmpdir):
    monkeypatch.setenv("TESSDATA_PREFIX", os.fspath(tmpdir))
    with pytest.raises(MissingDependencyError):
        run_ocrmypdf_api(resources / 'graph.pdf', no_outpdf, '-v', '1', '--skip-text')


def test_tesseract_oem(resources, outpdf):
    check_ocrmypdf(
        resources / 'trivial.pdf',
        outpdf,
        '--tesseract-oem',
        '1',
        '--plugin',
        'tests/plugins/tesseract_cache.py',
    )


@pytest.mark.parametrize('renderer', RENDERERS)
def test_pagesegmode(renderer, resources, outpdf):
    check_ocrmypdf(
        resources / 'skew.pdf',
        outpdf,
        '--tesseract-pagesegmode',
        '7',
        '-v',
        '1',
        '--pdf-renderer',
        renderer,
        '--plugin',
        'tests/plugins/tesseract_cache.py',
    )
