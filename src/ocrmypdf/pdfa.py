# SPDX-FileCopyrightText: 2022 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

"""Utilities for PDF/A production, with pikepdf or Ghostscript."""

from __future__ import annotations

import base64
import logging
from collections.abc import Iterator
from importlib.resources import files as package_files
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pikepdf
from pikepdf import Name, Object, Pdf

from ocrmypdf.helpers import pikepdf_get_dict

if TYPE_CHECKING:
    from pikepdf.pdfa import Flavour, PrepareResult, Report

log = logging.getLogger(__name__)

SRGB_ICC_PROFILE_NAME = 'sRGB.icc'


def _postscript_objdef(
    alias: str,
    dictionary: dict[str, str],
    *,
    stream_name: str | None = None,
    stream_data: bytes | None = None,
) -> Iterator[str]:
    assert (stream_name is None) == (stream_data is None)

    objtype = '/stream' if stream_name else '/dict'

    if stream_name:
        assert stream_data is not None
        a85_data = base64.a85encode(stream_data, adobe=True).decode('ascii')
        yield f'{stream_name} ' + a85_data
        yield 'def'

    if alias != '{Catalog}':  # Catalog needs no definition
        yield f'[/_objdef {alias} /type {objtype} /OBJ pdfmark'

    yield f'[{alias} <<'
    for key, val in dictionary.items():
        yield f'  {key} {val}'
    yield '>> /PUT pdfmark'

    if stream_name:
        yield f'[{alias} {stream_name[1:]} /PUT pdfmark'


def _make_postscript(icc_name: str, icc_data: bytes, colors: int) -> Iterator[str]:
    yield '%!'
    yield from _postscript_objdef(
        '{icc_PDFA}',  # Not an f-string
        {'/N': str(colors)},
        stream_name='/ICCProfile',
        stream_data=icc_data,
    )
    yield ''
    yield from _postscript_objdef(
        '{OutputIntent_PDFA}',
        {
            '/Type': '/OutputIntent',
            '/S': '/GTS_PDFA1',
            '/DestOutputProfile': '{icc_PDFA}',
            '/OutputConditionIdentifier': f'({icc_name})',  # Only f-string
        },
    )
    yield ''
    yield from _postscript_objdef(
        '{Catalog}', {'/OutputIntents': '[ {OutputIntent_PDFA} ]'}
    )


def generate_pdfa_ps(target_filename: Path, icc: str = 'sRGB'):
    """Create a Postscript PDFMARK file for Ghostscript PDF/A conversion.

    pdfmark is an extension to the Postscript language that describes some PDF
    features like bookmarks and annotations. It was originally specified Adobe
    Distiller, for Postscript to PDF conversion.

    Ghostscript uses pdfmark for PDF to PDF/A conversion as well. To use Ghostscript
    to create a PDF/A, we need to create a pdfmark file with the necessary metadata.

    This function takes care of the many version-specific bugs and peculiarities in
    Ghostscript's handling of pdfmark.

    The only information we put in specifies that we want the file to be a
    PDF/A, and we want to Ghostscript to convert objects to the sRGB colorspace
    if it runs into any object that it decides must be converted.

    Arguments:
        target_filename: filename to save
        icc: ICC identifier such as 'sRGB'
    References:
        Adobe PDFMARK Reference:
        https://opensource.adobe.com/dc-acrobat-sdk-docs/library/pdfmark/
    """
    if icc != 'sRGB':
        raise NotImplementedError("Only supporting sRGB")

    bytes_icc_profile = (
        package_files('ocrmypdf.data') / SRGB_ICC_PROFILE_NAME
    ).read_bytes()
    postscript = '\n'.join(_make_postscript(icc, bytes_icc_profile, 3))

    # We should have encoded everything to pure ASCII by this point, and
    # to be safe, only allow ASCII in PostScript
    Path(target_filename).write_text(postscript, encoding='ascii')
    return target_filename


def file_claims_pdfa(filename: Path):
    """Determines if the file claims to be PDF/A compliant.

    This only checks if the XMP metadata contains a PDF/A marker. It does not
    do full PDF/A validation.
    """
    with pikepdf.open(filename) as pdf:
        pdfmeta = pdf.open_metadata()
        if not pdfmeta.pdfa_status:
            return {
                'pass': False,
                'output': 'pdf',
                'conformance': 'No PDF/A metadata in XMP',
            }
        valid_part_conforms = {'1a', '1b', '2a', '2b', '2u', '3a', '3b', '3u'}
        # Raw value in XMP metadata returned by pikepdf is uppercase, but ISO
        # uses lower case for conformance levels.
        pdfa_status_iso = pdfmeta.pdfa_status.lower()
        conformance = f'PDF/A-{pdfa_status_iso}'
        pdfa_dict: dict[str, str | bool] = {}
        if pdfa_status_iso in valid_part_conforms:
            pdfa_dict['pass'] = True
            pdfa_dict['output'] = 'pdfa'
        pdfa_dict['conformance'] = conformance
    return pdfa_dict


def _cid_font_is_embedded(type0_font: Object) -> bool:
    """Return True if a Type0 font's CID descendant carries embedded glyphs."""
    for descendant in type0_font.get(Name.DescendantFonts, []):
        # A malformed PDF may store a non-dictionary here; `key in descriptor`
        # raises on those, so reduce anything that is not a dictionary to an
        # empty one before probing it.
        descriptor = pikepdf_get_dict(descendant, Name.FontDescriptor)
        if any(
            key in descriptor for key in (Name.FontFile, Name.FontFile2, Name.FontFile3)
        ):
            return True
    return False


def find_nonembedded_cid_fonts(pdf: Pdf) -> set[str]:
    """Find CID-keyed (Type0) fonts that lack embedded glyph data.

    PDF/A requires every font to be embedded. When Ghostscript converts a PDF
    to PDF/A it must substitute and embed a replacement for any non-embedded
    font. For CID-keyed fonts -- which is how CJK text is encoded, including the
    OCR text layers produced by Adobe Acrobat -- this substitution routinely
    corrupts the character-to-Unicode mapping, silently destroying the
    searchable text. Detecting these fonts lets the caller refuse PDF/A
    conversion rather than emit corrupted output.

    Simple (non-CID) non-embedded fonts are not reported: Ghostscript
    substitutes standard encodings for them without corrupting the text, and
    they are far too common to treat as conversion blockers.

    Args:
        pdf: An open ``pikepdf.Pdf`` to scan.

    Returns:
        The set of ``BaseFont`` names of non-embedded CID fonts found.
    """
    found: set[str] = set()

    def scan_resources(resources: Object, depth: int = 0) -> None:
        if depth > 10:
            return
        # A well-formed PDF stores dictionaries under /Font and /XObject, but a
        # malformed one (common in OCR workloads) may store an array, a name, or
        # another non-dictionary object. pikepdf_get_dict reduces every one of
        # those to "no fonts" rather than let the scan crash (issue #1713).
        for font in pikepdf_get_dict(resources, Name.Font).as_dict().values():
            try:
                if font.get(Name.Subtype) != Name.Type0:
                    continue
                if not _cid_font_is_embedded(font):
                    name = font.get(Name.BaseFont, Name('/(unnamed)'))
                    try:
                        basefont = str(name)
                    except UnicodeDecodeError:
                        # Name objects are byte sequences with no mandated
                        # encoding; e.g. CJK foundry font names are often
                        # GBK, which is not valid UTF-8 (issue #1727). Fall
                        # back to the hex-escaped PDF syntax form. Do not
                        # skip the font: it is still non-embedded and must
                        # block PDF/A conversion.
                        basefont = name.unparse().decode('ascii', 'replace')
                    found.add(basefont.lstrip('/'))
            except (AttributeError, TypeError, KeyError):
                continue
        for xobj in pikepdf_get_dict(resources, Name.XObject).as_dict().values():
            if xobj.get(Name.Subtype) == Name.Form:
                scan_resources(pikepdf_get_dict(xobj, Name.Resources), depth + 1)

    for page in pdf.pages:
        scan_resources(pikepdf_get_dict(page.obj, Name.Resources))
    return found


# PDF/A flavour for each --output-type that produces PDF/A. 'auto' (and any
# other value) makes PDF/A-2b, the same default as Ghostscript conversion.
_OUTPUT_TYPE_FLAVOURS = {
    'pdfa': '2b',
    'pdfa-1': '1b',
    'pdfa-2': '2b',
    'pdfa-3': '3b',
}


def output_type_to_flavour(output_type: str) -> Flavour:
    """Map an ``--output-type`` value to the PDF/A flavour it produces.

    Args:
        output_type: One of 'pdfa', 'pdfa-1', 'pdfa-2', 'pdfa-3' or 'auto'.

    Returns:
        The pikepdf PDF/A flavour; PDF/A-2b for 'auto' or an unknown value.
    """
    from pikepdf.pdfa import Flavour

    return Flavour(_OUTPUT_TYPE_FLAVOURS.get(output_type, '2b'))


def get_pdf_save_settings(output_type: str) -> dict[str, Any]:
    """Get pikepdf.Pdf.save settings for the given output type.

    For the PDF/A output types these are the complete settings pikepdf
    resolves for the flavour, with settings that would make the file
    invalid, or different from what was validated, pinned
    (`pikepdf.pdfa.resolve_save_kwargs`). Callers may change only the
    settings pikepdf leaves to the user, such as ``linearize`` and
    ``progress``.

    Args:
        output_type: 'pdf', or one of the PDF/A output types. For 'auto',
            pass the output type achieved, 'pdfa' or 'pdf'.
    """
    if output_type.startswith('pdfa'):
        from pikepdf.pdfa import resolve_save_kwargs

        return resolve_save_kwargs(
            output_type_to_flavour(output_type), compress_streams=True
        )
    return dict(
        preserve_pdfa=True,
        compress_streams=True,
        object_stream_mode=pikepdf.ObjectStreamMode.generate,
    )


def log_prepare_result(result: PrepareResult | None) -> None:
    """Log what `pikepdf.pdfa.prepare` changed.

    Removing hidden annotations discards content the user may care about, so
    it is a warning. Other changes to what the reader might notice are logged
    at info level, and the rest at debug level.
    """
    if result is None:
        return
    from pikepdf.pdfa import PrepareResult

    warnings = PrepareResult(
        annotations_removed=result.annotations_removed,
        annotations_removed_pages=result.annotations_removed_pages,
    )
    infos = PrepareResult(
        print_flags_set=result.print_flags_set,
        xmp_unreadable=result.xmp_unreadable,
        xmp_dropped=result.xmp_dropped,
    )
    shown = set()
    for line in warnings.describe():
        log.warning('%s', line)
        shown.add(line)
    for line in infos.describe():
        log.info('%s', line)
        shown.add(line)
    for line in result.describe():
        if line not in shown:
            log.debug('%s', line)


def prepare_pdfa(pdf: Pdf, output_type: str) -> PrepareResult:
    """Declare PDF/A in an open PDF that is to be saved as PDF/A.

    Runs `pikepdf.pdfa.prepare`, keeping the document's output intents,
    which were installed earlier by speculative conversion or by
    Ghostscript: it rewrites the XMP packet in the canonical form pikepdf's
    validator accepts, declares PDF/A conformance, sets DocInfo to agree
    with XMP, and repeats the structural repairs, which change nothing on a
    file that was already prepared. It is safe to call again after editing
    the metadata.

    Args:
        pdf: An open pikepdf.Pdf object
        output_type: One of 'pdfa', 'pdfa-1', 'pdfa-2', 'pdfa-3'
    """
    from pikepdf.pdfa import prepare

    result = prepare(pdf, output_type_to_flavour(output_type), output_intent=None)
    log_prepare_result(result)
    return result


def speculative_pdfa_conversion(
    input_file: Path,
    output_file: Path,
    output_type: str,
) -> Report:
    """Attempt to convert a PDF to PDF/A by adding and repairing structures.

    `pikepdf.pdfa.save` replaces the output intents with an sRGB PDF/A
    intent, removes image interpolation, removes annotations that are hidden
    or not viewable and sets the Print flag on the others, adds the /CIDSet
    that PDF/A-1 requires on subset CIDFonts, rewrites the XMP packet with
    only what PDF/A permits, declares PDF/A conformance, and saves with the
    settings of the flavour. It then validates the bytes written, and moves
    them to *output_file* only if they pass.

    This works for PDFs that are already mostly PDF/A compliant but lack the
    formal declarations. It does NOT perform color conversion, font
    embedding, or other transformations that Ghostscript does.

    Args:
        input_file: Path to input PDF
        output_file: Path where output PDF should be written
        output_type: One of 'pdfa', 'pdfa-1', 'pdfa-2', 'pdfa-3'

    Returns:
        pikepdf's validation report on the file written, which passed.

    Raises:
        pikepdf.pdfa.PdfaError: If the file written did not pass validation;
            ``e.report`` explains why. *output_file* is not written.
        pikepdf.PdfError: If the PDF cannot be opened or modified
    """
    from pikepdf.pdfa import save

    flavour = output_type_to_flavour(output_type)
    with Pdf.open(input_file) as pdf:
        report = save(pdf, output_file, flavour, output_intent='sRGB')
    log_prepare_result(report.prepared)

    log.debug('Speculative PDF/A conversion complete: %s', output_file)
    return report
