# SPDX-FileCopyrightText: 2026 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

"""Tolerance for PDFs that store the wrong type at a structural key.

Every scan in OCRmyPDF walks paths like ``/Resources /XObject`` or
``/Root /AcroForm /SigFlags`` through files it did not write. A malformed
producer may put an array, a name, or nothing at all at any step, and none of
those may abort the run: the answer is "no XObjects", not a traceback. These
tests pin that down for each traversal, since the well-formed fixtures in the
rest of the suite exercise none of it.
"""

from __future__ import annotations

from decimal import Decimal

import pikepdf
import pytest
from pikepdf import Array, Dictionary, Name, Stream

from ocrmypdf._annots import remove_broken_goto_annotations
from ocrmypdf._graft import discard_text_search_index
from ocrmypdf.builtin_plugins.ghostscript import _collect_dctdecode_images
from ocrmypdf.optimize import extract_image_filter
from ocrmypdf.pdfa import find_nonembedded_cid_fonts
from ocrmypdf.pdfinfo import PdfInfo

#: Values a malformed PDF may store where a dictionary or number belongs.
WRONG_TYPES = [
    pytest.param(Array([1, 2]), id='array'),
    pytest.param(Name.Nope, id='name'),
    pytest.param(pikepdf.String('nope'), id='string'),
    pytest.param(42, id='integer'),
]

#: The subset that is not a number either, for keys that hold a number.
NON_NUMERIC_TYPES = WRONG_TYPES[:3]


def _empty_name_tree(pdf) -> None:
    """Give *pdf* a valid but empty /Names /Dests tree.

    pikepdf's NameTree requires the dictionary it wraps to be owned by the Pdf,
    so both levels have to be indirect objects.
    """
    pdf.Root[Name.Names] = pdf.make_indirect(
        Dictionary(Dests=pdf.make_indirect(Dictionary(Names=Array([]))))
    )


@pytest.fixture(params=['explicit', 'implicit'])
def blank_pdf(request):
    """A one-page Pdf, in each conversion mode.

    OCRmyPDF opens every file in explicit mode, but the scans below must not
    depend on it, since a library caller may hand them a Pdf of its own.
    """
    pdf = pikepdf.Pdf.new(conversion_mode=request.param)
    pdf.add_blank_page(page_size=(200, 200))
    return pdf


class TestMalformedResources:
    """/Resources or /Resources /XObject holding something that is not a dict."""

    @pytest.mark.parametrize('wrong', WRONG_TYPES)
    def test_pdfinfo_scans_page(self, blank_pdf, outdir, wrong):
        blank_pdf.pages[0].obj[Name.Resources] = wrong
        target = outdir / 'bad_resources.pdf'
        blank_pdf.save(target)
        assert PdfInfo(target).pages[0].images == []

    @pytest.mark.parametrize('wrong', WRONG_TYPES)
    def test_pdfinfo_scans_nested_xobject(self, blank_pdf, outdir, wrong):
        blank_pdf.pages[0].obj[Name.Resources] = Dictionary(XObject=wrong)
        target = outdir / 'bad_xobject.pdf'
        blank_pdf.save(target)
        assert PdfInfo(target).pages[0].images == []

    @pytest.mark.parametrize('wrong', WRONG_TYPES)
    def test_ghostscript_jpeg_scan(self, blank_pdf, wrong):
        blank_pdf.pages[0].obj[Name.Resources] = Dictionary(XObject=wrong)
        assert _collect_dctdecode_images(blank_pdf) == {}

    @pytest.mark.parametrize('wrong', WRONG_TYPES)
    def test_ghostscript_jpeg_scan_through_form(self, blank_pdf, wrong):
        form = blank_pdf.make_stream(b'', Subtype=Name.Form, Resources=wrong)
        blank_pdf.pages[0].obj[Name.Resources] = Dictionary(
            XObject=Dictionary(Fm0=form)
        )
        assert _collect_dctdecode_images(blank_pdf) == {}

    @pytest.mark.parametrize('wrong', WRONG_TYPES)
    def test_pdfa_font_scan(self, blank_pdf, wrong):
        blank_pdf.pages[0].obj[Name.Resources] = Dictionary(Font=wrong)
        assert find_nonembedded_cid_fonts(blank_pdf) == set()

    @pytest.mark.parametrize('wrong', WRONG_TYPES)
    def test_pdfa_font_scan_through_form(self, blank_pdf, wrong):
        form = blank_pdf.make_stream(b'', Subtype=Name.Form, Resources=wrong)
        blank_pdf.pages[0].obj[Name.Resources] = Dictionary(
            XObject=Dictionary(Fm0=form)
        )
        assert find_nonembedded_cid_fonts(blank_pdf) == set()

    def test_pdfa_font_descriptor_is_not_a_dict(self, blank_pdf):
        descendant = Dictionary(FontDescriptor=Array([1, 2]))
        font = Dictionary(
            Subtype=Name.Type0,
            BaseFont=Name.Broken,
            DescendantFonts=Array([descendant]),
        )
        blank_pdf.pages[0].obj[Name.Resources] = Dictionary(Font=Dictionary(F0=font))
        # No embedded FontFile is reachable, so the font blocks PDF/A.
        assert find_nonembedded_cid_fonts(blank_pdf) == {'Broken'}


class TestMalformedCatalog:
    """Catalog entries PdfInfo reads before any OCR work begins."""

    @pytest.mark.parametrize('wrong', WRONG_TYPES)
    def test_acroform_is_not_a_dict(self, blank_pdf, outdir, wrong):
        blank_pdf.Root[Name.AcroForm] = wrong
        target = outdir / 'bad_acroform.pdf'
        blank_pdf.save(target)
        info = PdfInfo(target)
        assert info.has_acroform is False
        assert info.has_signature is False

    @pytest.mark.parametrize('wrong', WRONG_TYPES)
    def test_markinfo_is_not_a_dict(self, blank_pdf, outdir, wrong):
        blank_pdf.Root[Name.MarkInfo] = wrong
        target = outdir / 'bad_markinfo.pdf'
        blank_pdf.save(target)
        assert PdfInfo(target).is_tagged is False

    def test_marked_written_as_an_integer(self, blank_pdf, outdir):
        """A /Marked stored as 1 rather than true still reads as tagged."""
        blank_pdf.Root[Name.MarkInfo] = blank_pdf.make_indirect(Dictionary(Marked=1))
        target = outdir / 'marked_int.pdf'
        blank_pdf.save(target)
        assert PdfInfo(target).is_tagged is True

    @pytest.mark.parametrize('wrong', NON_NUMERIC_TYPES)
    def test_userunit_is_not_a_number(self, blank_pdf, outdir, wrong):
        blank_pdf.pages[0].obj[Name.UserUnit] = wrong
        target = outdir / 'bad_userunit.pdf'
        blank_pdf.save(target)
        assert PdfInfo(target).pages[0].userunit == Decimal(1)

    def test_rotate_written_as_a_real(self, blank_pdf, outdir):
        """A /Rotate stored as 90.0 rather than 90 still reads as a rotation."""
        blank_pdf.pages[0].obj[Name.Rotate] = Decimal('90.0')
        target = outdir / 'real_rotate.pdf'
        blank_pdf.save(target)
        assert PdfInfo(target).pages[0].rotation == 90

    @pytest.mark.parametrize('wrong', NON_NUMERIC_TYPES)
    def test_rotate_is_not_a_number(self, blank_pdf, outdir, wrong):
        blank_pdf.pages[0].obj[Name.Rotate] = wrong
        target = outdir / 'bad_rotate.pdf'
        blank_pdf.save(target)
        assert PdfInfo(target).pages[0].rotation == 0

    @pytest.mark.parametrize('wrong', WRONG_TYPES)
    def test_pieceinfo_is_not_a_dict(self, blank_pdf, wrong):
        blank_pdf.Root[Name.PieceInfo] = wrong
        assert discard_text_search_index(blank_pdf) is False

    def test_pieceinfo_search_index_is_discarded(self, blank_pdf):
        blank_pdf.Root[Name.PieceInfo] = blank_pdf.make_indirect(
            Dictionary(SearchIndex=Dictionary(Private=1))
        )
        assert discard_text_search_index(blank_pdf) is True
        assert Name.PieceInfo not in blank_pdf.Root


class TestMalformedAnnotations:
    """The named-destination cleanup walks /Names /Dests and /Annots /A /D."""

    @pytest.mark.parametrize('wrong', WRONG_TYPES)
    def test_names_is_not_a_dict(self, blank_pdf, wrong):
        blank_pdf.Root[Name.Names] = wrong
        assert remove_broken_goto_annotations(blank_pdf) is False

    @pytest.mark.parametrize('wrong', WRONG_TYPES)
    def test_dests_is_not_a_dict(self, blank_pdf, wrong):
        blank_pdf.Root[Name.Names] = blank_pdf.make_indirect(Dictionary(Dests=wrong))
        assert remove_broken_goto_annotations(blank_pdf) is False

    @pytest.mark.parametrize('wrong', WRONG_TYPES)
    def test_annotation_action_is_not_a_dict(self, blank_pdf, wrong):
        _empty_name_tree(blank_pdf)
        annot = Dictionary(A=wrong)
        blank_pdf.pages[0].obj[Name.Annots] = Array([annot])
        assert remove_broken_goto_annotations(blank_pdf) is False

    def test_broken_destination_is_disabled(self, blank_pdf):
        _empty_name_tree(blank_pdf)
        annot = blank_pdf.make_indirect(
            Dictionary(A=Dictionary(D=pikepdf.String('nowhere')))
        )
        blank_pdf.pages[0].obj[Name.Annots] = Array([annot])
        assert remove_broken_goto_annotations(blank_pdf) is True
        assert Name.D not in annot[Name.A]


class TestMalformedImageStream:
    """Image dictionaries the optimizer inspects before transcoding."""

    def test_image_without_subtype(self, blank_pdf):
        image = Stream(blank_pdf, b'x' * 1000, Width=100, Height=100)
        assert extract_image_filter(image, 1) is None

    @pytest.mark.parametrize('wrong', NON_NUMERIC_TYPES)
    def test_image_dimensions_not_integers(self, blank_pdf, wrong):
        image = Stream(
            blank_pdf,
            b'x' * 1000,
            Subtype=Name.Image,
            Width=wrong,
            Height=wrong,
        )
        assert extract_image_filter(image, 1) is None

    @pytest.mark.parametrize('wrong', WRONG_TYPES)
    def test_smask_is_not_a_dict(self, blank_pdf, wrong):
        """A /SMask that is not a dictionary has no /Matte to worry about."""
        image = Stream(
            blank_pdf,
            b'x' * 1000,
            Subtype=Name.Image,
            Width=100,
            Height=100,
            SMask=wrong,
        )
        # Reaches the /SMask /Matte check without raising; an uncompressed
        # image is then skipped for want of a filter.
        assert extract_image_filter(image, 1) is None


def _image(pdf, **entries):
    """An image dictionary owned by *pdf*, large enough to be optimized."""
    return pdf.make_indirect(
        Dictionary(
            Subtype=Name.Image,
            Length=1000,
            Width=100,
            Height=100,
            ColorSpace=Name.DeviceGray,
            **entries,
        )
    )


#: Filter parameters that the PDF reference types as integer, stored with a
#: type that leaves their value unknown. A Real is only unknown when it is not
#: integral: the sign of /K -0.5 differs from that of its truncation.
NON_INTEGER_PARAMS = [
    pytest.param(Name.Nope, id='name'),
    pytest.param(pikepdf.String('1'), id='string'),
    pytest.param(True, id='boolean'),
    pytest.param(Array([1]), id='array'),
    pytest.param(Decimal('-0.5'), id='fractional'),
]


class TestMalformedFilterParameters:
    """/DecodeParms entries the optimizer reads to choose a transcoding.

    ISO 32000-2 types /Predictor (Table 8) and /K (Table 11) as integers, and
    a null entry is the same as a missing one (7.3.9), which takes the
    default. It defines nothing for a value of another type, so the encoding
    is unknown and the image is left alone.
    """

    def _flate_jpeg(self, pdf, flate_parms):
        return _image(
            pdf,
            BitsPerComponent=8,
            Filter=Array([Name.FlateDecode, Name.DCTDecode]),
            DecodeParms=Array([flate_parms, None]),
        )

    def test_flate_jpeg_with_null_parms(self, blank_pdf):
        """A null in /DecodeParms means the filter uses its defaults."""
        image = self._flate_jpeg(blank_pdf, None)
        assert extract_image_filter(image, 1) is not None

    @pytest.mark.parametrize('predictor', [1, Decimal('1.0')])
    def test_flate_jpeg_without_prediction(self, blank_pdf, predictor):
        image = self._flate_jpeg(blank_pdf, Dictionary(Predictor=predictor))
        assert extract_image_filter(image, 1) is not None

    @pytest.mark.parametrize('predictor', NON_INTEGER_PARAMS)
    def test_flate_jpeg_predictor_not_an_integer(self, blank_pdf, predictor):
        image = self._flate_jpeg(blank_pdf, Dictionary(Predictor=predictor))
        assert extract_image_filter(image, 1) is None

    @pytest.mark.parametrize('parms', WRONG_TYPES)
    def test_flate_jpeg_parms_not_a_dict(self, blank_pdf, parms):
        image = self._flate_jpeg(blank_pdf, parms)
        assert extract_image_filter(image, 1) is None

    def _ccitt(self, pdf, **entries):
        return _image(pdf, BitsPerComponent=1, Filter=Name.CCITTFaxDecode, **entries)

    @pytest.mark.parametrize('k', [-1, Decimal('-1.0')])
    def test_ccitt_group4(self, blank_pdf, k):
        image = self._ccitt(blank_pdf, DecodeParms=Dictionary(K=k))
        assert extract_image_filter(image, 1) is not None

    @pytest.mark.parametrize('k', [None, 0, 1])
    def test_ccitt_group3(self, blank_pdf, k):
        """Group 3, which is also what a missing /K means, is not supported."""
        parms = Dictionary() if k is None else Dictionary(K=k)
        image = self._ccitt(blank_pdf, DecodeParms=parms)
        assert extract_image_filter(image, 1) is None

    def test_ccitt_without_parms(self, blank_pdf):
        assert extract_image_filter(self._ccitt(blank_pdf), 1) is None

    @pytest.mark.parametrize('k', NON_INTEGER_PARAMS)
    def test_ccitt_k_not_an_integer(self, blank_pdf, k):
        image = self._ccitt(blank_pdf, DecodeParms=Dictionary(K=k))
        assert extract_image_filter(image, 1) is None

    @pytest.mark.parametrize('parms', WRONG_TYPES)
    def test_ccitt_parms_not_a_dict(self, blank_pdf, parms):
        image = self._ccitt(blank_pdf, DecodeParms=parms)
        assert extract_image_filter(image, 1) is None
