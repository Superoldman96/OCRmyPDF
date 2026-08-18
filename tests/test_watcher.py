from __future__ import annotations

import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from pathlib import Path

import pikepdf
import pytest

watchfiles = pytest.importorskip('watchfiles')

WATCHER = Path(__file__).parent.parent / 'misc' / 'watcher.py'

# Printed by the watcher immediately before it enters the watch loop.
WATCHING_MESSAGE = 'for new PDFs'

# Basename stem of the throwaway entries used to prove the watch is live.
PROBE_NAME = 'watchprobe'

# Generous, so a loaded CI runner cannot flake. Nothing pays for it in the
# common case because every wait below is event-driven.
TIMEOUT = 60.0


def _spawn_watcher(input_dir, output_dir, processed_dir, cwd, log_path, env_extra=None):
    """Launch watcher.py as a subprocess, cwd disjoint from the data dirs.

    The startup check treats the working directory as part of the code zone, so
    the subprocess must run from a directory that is neither an ancestor nor a
    descendant of the watched data directories.

    Both output streams are merged into ``log_path``. A file rather than a pipe,
    so the child can never block on a full pipe buffer, and ``PYTHONUNBUFFERED``
    keeps its stdout from sitting in a block buffer until exit; the tests read
    the log to learn what the watcher has done so far.
    """
    with log_path.open('wb') as log_file:
        return subprocess.Popen(
            [
                sys.executable,
                str(WATCHER),
                str(input_dir),
                str(output_dir),
                str(processed_dir),
            ],
            cwd=str(cwd),
            env=os.environ.copy() | {'PYTHONUNBUFFERED': '1'} | (env_extra or {}),
            stdout=log_file,
            stderr=subprocess.STDOUT,
        )


def _read_log(log_path: Path) -> str:
    try:
        return log_path.read_text(encoding='utf-8', errors='replace')
    except FileNotFoundError:
        return ''


def _wait_for(
    predicate: Callable[[], bool],
    what: str,
    log_path: Path | None = None,
    timeout: float = TIMEOUT,
    interval: float = 0.05,
) -> None:
    """Poll ``predicate`` until true, else fail with the watcher's log attached."""
    deadline = time.monotonic() + timeout
    while not predicate():
        if time.monotonic() > deadline:
            log = f"\nWatcher log:\n{_read_log(log_path)}" if log_path else ''
            raise AssertionError(f"Timed out after {timeout}s waiting for {what}.{log}")
        time.sleep(interval)


def _wait_for_log(log_path: Path, needle: str, what: str) -> None:
    _wait_for(lambda: needle in _read_log(log_path), what, log_path)


def _wait_for_watch_live(input_dir: Path, log_path: Path) -> None:
    """Wait until the watcher is really watching, not merely started.

    The readiness banner is printed just before the watch loop is entered, so an
    entry created the instant it appears can land before the filesystem watch
    exists and be missed entirely. Keep creating a throwaway probe until one of
    them shows up in the log; from then on no event is lost.

    The probe is a *directory* with a PDF name: it matches the watched pattern,
    yet the watcher rejects it at once as a non-regular file, so it costs no OCR
    run, leaves nothing in the output directory, and needs no POSIX-only trick.
    """
    _wait_for_log(log_path, WATCHING_MESSAGE, "the watcher to start up")

    attempt = 0

    def probe_seen() -> bool:
        nonlocal attempt
        if PROBE_NAME in _read_log(log_path):
            return True
        (input_dir / f'{PROBE_NAME}{attempt}.pdf').mkdir()
        attempt += 1
        return False

    _wait_for(
        probe_seen,
        "the watcher's filesystem watch to go live",
        log_path,
        interval=0.2,
    )


def _wait_for_file(path: Path, log_path: Path) -> None:
    _wait_for(path.exists, f"{path} to be created", log_path)


@pytest.fixture
def data_dirs(tmp_path):
    input_dir = tmp_path / 'input'
    input_dir.mkdir()
    output_dir = tmp_path / 'output'
    output_dir.mkdir()
    processed_dir = tmp_path / 'processed'
    processed_dir.mkdir()
    work_dir = tmp_path / 'work'  # cwd, disjoint from the data dirs
    work_dir.mkdir()
    return input_dir, output_dir, processed_dir, work_dir


@pytest.fixture
def watcher_log(tmp_path):
    # Outside the watched data dirs, so it is never mistaken for input.
    return tmp_path / 'watcher.log'


@pytest.mark.parametrize('year_month', [True, False])
def test_watcher(data_dirs, watcher_log, resources, year_month):
    input_dir, output_dir, processed_dir, work_dir = data_dirs

    env_extra = {'OCR_OUTPUT_DIRECTORY_YEAR_MONTH': '1'} if year_month else {}
    proc = _spawn_watcher(
        input_dir,
        output_dir,
        processed_dir,
        work_dir,
        watcher_log,
        env_extra=env_extra,
    )
    _wait_for_watch_live(input_dir, watcher_log)

    shutil.copy(resources / 'trivial.pdf', input_dir / 'trivial.pdf')

    if year_month:
        expected = (
            output_dir
            / f'{dt.date.today().year}'
            / f'{dt.date.today().month:02d}'
            / 'trivial.pdf'
        )
    else:
        expected = output_dir / 'trivial.pdf'
    _wait_for_file(expected, watcher_log)

    proc.terminate()
    proc.wait()


def test_watcher_aborts_when_input_dir_in_code_zone(tmp_path, watcher_log):
    # Point the input directory at the interpreter prefix (a code zone). The
    # watcher must refuse to run and never enter the watch loop.
    input_dir = Path(sys.prefix)
    output_dir = tmp_path / 'output'
    output_dir.mkdir()
    processed_dir = tmp_path / 'processed'
    processed_dir.mkdir()
    work_dir = tmp_path / 'work'
    work_dir.mkdir()

    proc = _spawn_watcher(input_dir, output_dir, processed_dir, work_dir, watcher_log)
    assert proc.wait(timeout=30) == 9  # ExitCode.invalid_config


def test_watcher_aborts_when_output_under_input(tmp_path, watcher_log):
    # Output nested inside the watched input directory would loop forever.
    input_dir = tmp_path / 'input'
    input_dir.mkdir()
    output_dir = input_dir / 'output'
    output_dir.mkdir()
    processed_dir = tmp_path / 'processed'
    processed_dir.mkdir()
    work_dir = tmp_path / 'work'
    work_dir.mkdir()

    proc = _spawn_watcher(input_dir, output_dir, processed_dir, work_dir, watcher_log)
    assert proc.wait(timeout=30) == 9


def test_watcher_rejects_world_writable_settings_file(data_dirs, watcher_log, tmp_path):
    input_dir, output_dir, processed_dir, work_dir = data_dirs
    settings = tmp_path / 'settings.json'
    settings.write_text('{}')
    settings.chmod(0o666)

    proc = _spawn_watcher(
        input_dir,
        output_dir,
        processed_dir,
        work_dir,
        watcher_log,
        env_extra={'OCR_JSON_SETTINGS': str(settings)},
    )
    assert proc.wait(timeout=30) == 9


def test_watcher_rejects_plugin_in_data_dir(data_dirs, watcher_log):
    input_dir, output_dir, processed_dir, work_dir = data_dirs
    settings = json.dumps({'plugins': [str(input_dir / 'evil.py')]})

    proc = _spawn_watcher(
        input_dir,
        output_dir,
        processed_dir,
        work_dir,
        watcher_log,
        env_extra={'OCR_JSON_SETTINGS': settings},
    )
    assert proc.wait(timeout=30) == 9


@pytest.mark.skipif(os.name == 'nt', reason="POSIX fifo semantics")
def test_watcher_skips_fifo_input(data_dirs, watcher_log, resources):
    input_dir, output_dir, processed_dir, work_dir = data_dirs

    proc = _spawn_watcher(input_dir, output_dir, processed_dir, work_dir, watcher_log)
    _wait_for_watch_live(input_dir, watcher_log)

    # A fifo named like a PDF must be ignored, not OCR'd or crash the watcher.
    os.mkfifo(input_dir / 'pipe.pdf')
    _wait_for_log(watcher_log, 'pipe.pdf', "the watcher to notice the fifo")
    # Then a genuine PDF should still be processed. The watcher handles files
    # one at a time, so the fifo is fully dealt with by the time this appears.
    shutil.copy(resources / 'trivial.pdf', input_dir / 'trivial.pdf')
    _wait_for_file(output_dir / 'trivial.pdf', watcher_log)

    assert not (output_dir / 'pipe.pdf').exists()
    assert proc.poll() is None  # still running

    proc.terminate()
    proc.wait()


def test_watcher_survives_encrypted_pdf(data_dirs, watcher_log, resources, tmp_path):
    # pikepdf.PasswordError does not derive from pikepdf.PdfError, so an
    # encrypted PDF used to escape wait_for_file_ready() and tear down the
    # watch loop, leaving later files unprocessed.
    input_dir, output_dir, processed_dir, work_dir = data_dirs

    encrypted = tmp_path / 'encrypted.pdf'
    with pikepdf.open(resources / 'trivial.pdf') as pdf:
        pdf.save(
            encrypted, encryption=pikepdf.Encryption(owner='owner', user='user', R=6)
        )

    proc = _spawn_watcher(input_dir, output_dir, processed_dir, work_dir, watcher_log)
    _wait_for_watch_live(input_dir, watcher_log)

    shutil.copy(encrypted, input_dir / 'encrypted.pdf')
    _wait_for_log(
        watcher_log, 'encrypted.pdf', "the watcher to pick up the encrypted PDF"
    )
    # A PDF arriving after the encrypted one must still be processed. The
    # watcher handles files one at a time, so the encrypted one is fully dealt
    # with by the time this output appears.
    shutil.copy(resources / 'trivial.pdf', input_dir / 'trivial.pdf')
    _wait_for_file(output_dir / 'trivial.pdf', watcher_log)

    assert not (output_dir / 'encrypted.pdf').exists()
    assert proc.poll() is None  # still running

    proc.terminate()
    proc.wait()


def test_watcher_survives_ocr_exception(data_dirs, watcher_log, resources):
    # Any per-file error, not just PasswordError, must be contained: a file
    # that makes ocrmypdf.ocr() raise must not stop the watch loop.
    input_dir, output_dir, processed_dir, work_dir = data_dirs

    # An unreadable-by-ocrmypdf file that still passes wait_for_file_ready() is
    # hard to synthesize, so drive the failure through OCR_JSON_SETTINGS: an
    # unsupported language makes ocrmypdf.ocr() raise MissingDependencyError.
    proc = _spawn_watcher(
        input_dir,
        output_dir,
        processed_dir,
        work_dir,
        watcher_log,
        env_extra={'OCR_JSON_SETTINGS': json.dumps({'language': ['zzz']})},
    )
    _wait_for_watch_live(input_dir, watcher_log)

    shutil.copy(resources / 'trivial.pdf', input_dir / 'bad.pdf')
    _wait_for_log(
        watcher_log,
        'Error while processing',
        "the watcher to log and contain the OCR failure",
    )

    assert not (output_dir / 'bad.pdf').exists()
    assert proc.poll() is None  # error contained, watcher still running

    proc.terminate()
    proc.wait()
