# SPDX-FileCopyrightText: 2026 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

"""Tests for create_ocr_image, especially when it can avoid re-encoding."""

from __future__ import annotations

import os
from types import SimpleNamespace

import pytest
from PIL import Image

from ocrmypdf._options import ProcessingMode
from ocrmypdf._pipeline import create_ocr_image


@pytest.fixture
def raster(tmp_path):
    """A page image of the kind rasterize() produces."""
    path = tmp_path / 'rasterize.png'
    # Black, so that a white mask rectangle is detectable.
    Image.new('RGB', (300, 400), 'black').save(path, dpi=(150.0, 150.0))
    return path


def make_context(tmp_path, *, textareas=(), mode=ProcessingMode.skip, filter_ocr_image):
    """Build the smallest context create_ocr_image needs."""
    return SimpleNamespace(
        options=SimpleNamespace(mode=mode, max_ocr_image_mpixels=None),
        pageinfo=SimpleNamespace(
            get_textareas=lambda visible, corrupt: iter(textareas)
        ),
        plugin_manager=SimpleNamespace(filter_ocr_image=filter_ocr_image),
        get_path=lambda name: tmp_path / name,
    )


def test_links_instead_of_reencoding_when_nothing_changes(tmp_path, raster):
    """No masking and no filtering means the raster already is the OCR image."""
    ctx = make_context(tmp_path, filter_ocr_image=lambda page, image: None)

    out = create_ocr_image(raster, ctx)

    assert out.samefile(raster), "should point at the raster, not a re-encoding"
    if os.name != 'nt':
        assert out.is_symlink()


def test_reencodes_when_a_text_area_is_masked(tmp_path, raster):
    ctx = make_context(
        tmp_path,
        textareas=[(10, 10, 50, 50)],
        mode=ProcessingMode.redo,
        filter_ocr_image=lambda page, image: None,
    )

    out = create_ocr_image(raster, ctx)

    assert not out.samefile(raster)
    with Image.open(out) as im:
        # 150 dpi means PDF points scale by 150/72, and y is measured from the
        # bottom, so the box at (10, 10, 50, 50) pt covers roughly x 21..104,
        # y 296..379 in pixels.
        assert im.getpixel((60, 340)) == (255, 255, 255), "text area not blanked"
        assert im.getpixel((200, 50)) == (0, 0, 0), "blanked outside the text area"


def test_reencodes_when_a_plugin_replaces_the_image(tmp_path, raster):
    def replace(page, image):
        return image.convert('L')

    ctx = make_context(tmp_path, filter_ocr_image=replace)

    out = create_ocr_image(raster, ctx)

    assert not out.samefile(raster)
    with Image.open(out) as im:
        assert im.mode == 'L'


def test_reencodes_when_a_plugin_reads_the_image(tmp_path, raster):
    """A filter that inspects pixels and returns the same object still decodes."""
    seen = []

    def inspect(page, image):
        seen.append(image.getpixel((0, 0)))
        return image

    ctx = make_context(tmp_path, filter_ocr_image=inspect)

    out = create_ocr_image(raster, ctx)

    assert seen, "the filter should have run"
    assert not out.samefile(raster), (
        "decoding the image means we can no longer prove the file is unchanged"
    )
