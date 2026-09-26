# SPDX-FileCopyrightText: 2026 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

"""Every Pdf OCRmyPDF opens or creates uses explicit conversion mode.

The mode is set per document rather than globally, because OCRmyPDF is also
used as a library and must not change how pikepdf behaves for its host.
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

SRC = Path(__file__).parent.parent / 'src' / 'ocrmypdf'

#: Callables that create a Pdf: ``pikepdf.open``, ``Pdf.open``, ``Pdf.new``
#: and their fully qualified spellings.
_OPENERS = {('pikepdf', 'open'), ('Pdf', 'open'), ('Pdf', 'new')}


def _opener_calls(tree: ast.AST):
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call) or not isinstance(node.func, ast.Attribute):
            continue
        owner = node.func.value
        owner_name = (
            owner.id
            if isinstance(owner, ast.Name)
            else owner.attr
            if isinstance(owner, ast.Attribute)
            else None
        )
        if (owner_name, node.func.attr) in _OPENERS:
            yield node


def _is_explicit(call: ast.Call) -> bool:
    return any(
        kw.arg == 'conversion_mode'
        and isinstance(kw.value, ast.Constant)
        and kw.value.value == 'explicit'
        for kw in call.keywords
    )


def _source_files():
    return sorted(SRC.rglob('*.py'))


@pytest.mark.parametrize('path', _source_files(), ids=lambda p: str(p.relative_to(SRC)))
def test_every_pdf_is_opened_in_explicit_mode(path):
    tree = ast.parse(path.read_text(encoding='utf-8'), filename=str(path))
    implicit = [
        f'{path.relative_to(SRC)}:{call.lineno}'
        for call in _opener_calls(tree)
        if not _is_explicit(call)
    ]
    assert not implicit, "Pdf opened without conversion_mode='explicit'"


def test_no_global_conversion_mode_change():
    for path in _source_files():
        text = path.read_text(encoding='utf-8')
        assert 'set_object_conversion_mode' not in text, path
