# SPDX-FileCopyrightText: 2022 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

from unittest.mock import patch

import pikepdf

import ocrmypdf


def test_no_glyphless_graft(resources, outdir):
    with (
        pikepdf.open(resources / 'francais.pdf') as pdf,
        pikepdf.open(resources / 'aspect.pdf') as pdf_aspect,
        pikepdf.open(resources / 'cmyk.pdf') as pdf_cmyk,
    ):
        pdf.pages.extend(pdf_aspect.pages)
        pdf.pages.extend(pdf_cmyk.pages)
        pdf.save(outdir / 'test.pdf')

    with patch('ocrmypdf._graft.MAX_REPLACE_PAGES', 2):
        ocrmypdf.ocr(
            outdir / 'test.pdf',
            outdir / 'out.pdf',
            deskew=True,
            tesseract_timeout=0,
            force_ocr=True,
        )
    # This test needs asserts


def test_links(resources, outpdf):
    ocrmypdf.ocr(
        resources / 'link.pdf', outpdf, redo_ocr=True, oversample=200, output_type='pdf'
    )
    with pikepdf.open(outpdf) as pdf:
        p1 = pdf.pages[0]
        p2 = pdf.pages[1]
        assert p1.Annots[0].A.D[0].objgen == p2.objgen
        assert p2.Annots[0].A.D[0].objgen == p1.objgen


def test_redo_ocr_with_offset_mediabox(resources, outdir):
    """Test that --redo-ocr handles non-zero mediabox origins correctly.

    Regression test for issue #1630 where PDFs with mediabox origins like
    [0, 100, width, height+100] (common in cropped/JSTOR-style PDFs)
    would have OCR text shifted vertically because the text layer CTM
    did not account for the page origin offset.
    """
    # Create a PDF with a non-zero mediabox origin
    input_pdf = outdir / 'offset_mediabox_input.pdf'
    y_offset = 100

    with pikepdf.open(resources / 'graph_ocred.pdf') as pdf:
        page = pdf.pages[0]
        original_mb = list(page.MediaBox)

        # Shift mediabox Y origin to simulate cropped/JSTOR-style PDFs
        page.MediaBox = [
            original_mb[0],
            original_mb[1] + y_offset,
            original_mb[2],
            original_mb[3] + y_offset,
        ]

        pdf.save(input_pdf)

    # Run --redo-ocr (this is where the bug occurred)
    output_pdf = outdir / 'offset_redo_ocr.pdf'
    ocrmypdf.ocr(input_pdf, output_pdf, redo_ocr=True)

    # Verify the output
    with pikepdf.open(output_pdf) as pdf:
        page = pdf.pages[0]
        mediabox = list(page.MediaBox)

        # MediaBox origin should be preserved
        assert float(mediabox[1]) == y_offset, (
            f"MediaBox Y origin should be preserved at {y_offset}, got {mediabox[1]}"
        )

        # The content stream should include a CTM with the Y origin translation.
        # Without the fix, the CTM was omitted for rotation==0, causing a shift.
        content = page.Contents.read_bytes()
        assert b'cm' in content, (
            "Content stream should include a CTM to translate by the page origin"
        )


def test_strip_invisble_text():
    pdf = pikepdf.Pdf.new()
    print(pikepdf.parse_content_stream(pikepdf.Stream(pdf, b'3 Tr')))
    page = pdf.add_blank_page()
    visible_text = [
        pikepdf.ContentStreamInstruction((), pikepdf.Operator('BT')),
        pikepdf.ContentStreamInstruction(
            (pikepdf.Name('/F0'), 12), pikepdf.Operator('Tf')
        ),
        pikepdf.ContentStreamInstruction((288, 720), pikepdf.Operator('Td')),
        pikepdf.ContentStreamInstruction(
            (pikepdf.String('visible'),), pikepdf.Operator('Tj')
        ),
        pikepdf.ContentStreamInstruction((), pikepdf.Operator('ET')),
    ]
    invisible_text = [
        pikepdf.ContentStreamInstruction((), pikepdf.Operator('BT')),
        pikepdf.ContentStreamInstruction(
            (pikepdf.Name('/F0'), 12), pikepdf.Operator('Tf')
        ),
        pikepdf.ContentStreamInstruction((288, 720), pikepdf.Operator('Td')),
        pikepdf.ContentStreamInstruction(
            (pikepdf.String('invisible'),), pikepdf.Operator('Tj')
        ),
        pikepdf.ContentStreamInstruction((), pikepdf.Operator('ET')),
    ]
    invisible_text_setting_tr = [
        pikepdf.ContentStreamInstruction((), pikepdf.Operator('BT')),
        pikepdf.ContentStreamInstruction([3], pikepdf.Operator('Tr')),
        pikepdf.ContentStreamInstruction(
            (pikepdf.Name('/F0'), 12), pikepdf.Operator('Tf')
        ),
        pikepdf.ContentStreamInstruction((288, 720), pikepdf.Operator('Td')),
        pikepdf.ContentStreamInstruction(
            (pikepdf.String('invisible'),), pikepdf.Operator('Tj')
        ),
        pikepdf.ContentStreamInstruction((), pikepdf.Operator('ET')),
    ]
    stream = [
        pikepdf.ContentStreamInstruction([], pikepdf.Operator('q')),
        pikepdf.ContentStreamInstruction([3], pikepdf.Operator('Tr')),
        *invisible_text,
        pikepdf.ContentStreamInstruction([], pikepdf.Operator('Q')),
        *visible_text,
        *invisible_text_setting_tr,
        *invisible_text,
    ]
    content_stream = pikepdf.unparse_content_stream(stream)
    page.Contents = pikepdf.Stream(pdf, content_stream)

    def count(string, page):
        return len(
            [
                True
                for operands, operator in pikepdf.parse_content_stream(page)
                if operator == pikepdf.Operator('Tj')
                and operands[0] == pikepdf.String(string)
            ]
        )

    nr_visible_pre = count('visible', page)
    ocrmypdf._graft.strip_invisible_text(pdf, page)
    nr_visible_post = count('visible', page)
    assert nr_visible_pre == nr_visible_post, (
        'Number of visible text elements did not change'
    )
    assert count('invisible', page) == 0, 'No invisible elems left'


def test_strip_invisible_text_in_form_xobject():
    """Invisible text inside a Form XObject is stripped too.

    Regression test for #1730. OCRmyPDF grafts its own OCR text layer as a
    Form XObject and appends ``/OCR-xxxx Do`` to the page (see
    ``OcrGrafter._graft_fpdf2_text_layer``), so on OCRmyPDF's own output the
    invisible text is not in the page content stream at all and ``--mode
    strip`` used to be a no-op.
    """
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page()
    font = pdf.make_indirect(
        pikepdf.Dictionary(
            Type=pikepdf.Name.Font,
            Subtype=pikepdf.Name.Type1,
            BaseFont=pikepdf.Name.Helvetica,
        )
    )

    def text(string, render_mode):
        return b'BT\n%d Tr\n/F0 12 Tf\n72 720 Td\n(%s) Tj\nET\n' % (
            render_mode,
            string,
        )

    # The grafted text layer: invisible OCR text plus a visible run that must
    # survive, exactly as they would sit inside an /OCR-... Form XObject.
    xobj = pdf.make_stream(text(b'invisible', 3) + text(b'visible-in-form', 0))
    xobj.Type = pikepdf.Name.XObject
    xobj.Subtype = pikepdf.Name.Form
    xobj.FormType = 1
    xobj.BBox = pikepdf.Array([0, 0, 612, 792])
    xobj.Resources = pikepdf.Dictionary(Font=pikepdf.Dictionary(F0=font))
    page.obj[pikepdf.Name.Resources] = pikepdf.Dictionary(
        Font=pikepdf.Dictionary(F0=font),
        XObject=pikepdf.Dictionary({'/OCR-abc': xobj}),
    )
    page.Contents = pikepdf.Stream(pdf, text(b'visible', 0) + b'q\n/OCR-abc Do\nQ\n')

    def strings(obj):
        return [
            bytes(operands[0])
            for operands, operator in pikepdf.parse_content_stream(obj, '')
            if operator == pikepdf.Operator('Tj')
        ]

    assert strings(page) == [b'visible']
    assert strings(xobj) == [b'invisible', b'visible-in-form']

    ocrmypdf._graft.strip_invisible_text(pdf, page)

    assert strings(page) == [b'visible'], 'page text must be untouched'
    assert strings(xobj) == [b'visible-in-form'], (
        'invisible text in the Form XObject must be removed, visible text kept'
    )


def test_strip_invisible_text_with_self_referential_form():
    """A Form XObject that draws itself must not recurse forever (#1730)."""
    pdf = pikepdf.Pdf.new()
    page = pdf.add_blank_page()
    xobj = pdf.make_stream(b'BT\n3 Tr\n/F0 12 Tf\n(invisible) Tj\nET\n/Self Do\n')
    xobj.Type = pikepdf.Name.XObject
    xobj.Subtype = pikepdf.Name.Form
    xobj.FormType = 1
    xobj.BBox = pikepdf.Array([0, 0, 612, 792])
    xobj.Resources = pikepdf.Dictionary(XObject=pikepdf.Dictionary({'/Self': xobj}))
    page.obj[pikepdf.Name.Resources] = pikepdf.Dictionary(
        XObject=pikepdf.Dictionary({'/Self': xobj})
    )
    page.Contents = pikepdf.Stream(pdf, b'/Self Do\n')

    ocrmypdf._graft.strip_invisible_text(pdf, page)

    assert b'invisible' not in xobj.read_bytes()
