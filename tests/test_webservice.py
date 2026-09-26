# SPDX-FileCopyrightText: 2026 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import os
import runpy
import sys
import types
from pathlib import Path

import pytest

MISC = Path(__file__).parent.parent / 'misc'
WEBSERVICE = MISC / 'webservice.py'


def test_webservice_launches_from_any_directory(tmp_path, monkeypatch):
    # The Docker image runs /app/webservice.py (a symlink) from /data, so the
    # Streamlit script must be found relative to the launcher, not the cwd.
    link = tmp_path / 'webservice.py'
    try:
        link.symlink_to(WEBSERVICE)
    except OSError:
        link = WEBSERVICE
    workdir = tmp_path / 'data'
    workdir.mkdir()
    monkeypatch.chdir(workdir)
    monkeypatch.setitem(sys.modules, 'streamlit', types.ModuleType('streamlit'))
    monkeypatch.setattr(sys, 'argv', [str(link), '--server.port', '5000'])

    calls = []

    def fake_execvp(file, args):
        calls.append(args)
        raise SystemExit(0)

    monkeypatch.setattr(os, 'execvp', fake_execvp)
    with pytest.raises(SystemExit):
        runpy.run_path(str(link), run_name='__main__')

    (args,) = calls
    assert args[1:4] == ['-m', 'streamlit', 'run']
    script = Path(args[4])
    assert script.is_absolute()
    assert script == (MISC / '_webservice.py').resolve()
    assert args[5:] == ['--server.port', '5000']
