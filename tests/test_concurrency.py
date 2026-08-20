# SPDX-FileCopyrightText: 2022 James R. Barlow
# SPDX-License-Identifier: MPL-2.0

from __future__ import annotations

import itertools
import os
import platform
import threading

import pikepdf
import pytest

import ocrmypdf
import ocrmypdf.api
from ocrmypdf import ExitCode
from ocrmypdf._concurrent import SerialExecutor
from ocrmypdf.builtin_plugins.concurrency import get_progressbar_class

from .conftest import run_ocrmypdf_api

NOOP_PLUGIN = 'tests/plugins/tesseract_noop.py'


@pytest.mark.skipif(os.name == 'nt', reason="Windows doesn't have SIGKILL")
@pytest.mark.skipif(
    platform.python_version_tuple() >= ('3', '12'), reason="can deadlock due to fork"
)
def test_simulate_oom_killer(multipage, no_outpdf):
    exitcode = run_ocrmypdf_api(
        multipage,
        no_outpdf,
        '--force-ocr',
        '--no-use-threads',
        '--plugin',
        'tests/plugins/tesseract_simulate_oom_killer.py',
    )
    assert exitcode == ExitCode.child_process_error


def _run_in_threads(target, count=2):
    """Run ``target`` in ``count`` threads, returning any exceptions raised."""
    errors: list[BaseException] = []

    def wrapper():
        try:
            target()
        except BaseException as e:  # noqa: BLE001 - report, don't swallow
            errors.append(e)

    threads = [threading.Thread(target=wrapper) for _ in range(count)]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=180)
    return errors


def test_executors_do_not_serialize_across_instances():
    """Two Executor instances must be able to run tasks at the same time.

    ``Executor`` used to hold a lock shared by every instance and subclass in
    the process, so a second executor could not start until the first finished.
    """
    barrier = threading.Barrier(2, timeout=30)

    def run():
        # A distinct instance per thread - the point is that they must not
        # share a class-level lock.
        SerialExecutor()(
            use_threads=True,
            max_workers=1,
            progress_kwargs=dict(total=1, desc='test', unit='task', disable=True),
            task=lambda _: barrier.wait(),
            task_arguments=[(1,)],
        )

    assert not _run_in_threads(run)


def test_progress_bar_console_ownership():
    """Only one progress bar at a time may render on the shared console."""
    pbar_class = get_progressbar_class()
    kwargs = dict(total=10, unit='page', disable=False)

    with pbar_class(desc='first', **kwargs) as first:
        assert first.progress.disable is False
        with pbar_class(desc='second', **kwargs) as second:
            assert second.progress.disable is True  # lost the console race

    # Console released - a later bar renders again. Each job creates several
    # bars in sequence, so this must work repeatedly.
    with pbar_class(desc='third', **kwargs) as third:
        assert third.progress.disable is False


def test_progress_bar_releases_console_on_exception():
    """A progress bar that raises must not strand ownership of the console."""
    pbar_class = get_progressbar_class()
    kwargs = dict(total=1, unit='page', disable=False)

    with pytest.raises(ZeroDivisionError), pbar_class(desc='doomed', **kwargs):
        raise ZeroDivisionError

    with pbar_class(desc='next', **kwargs) as pbar:
        assert pbar.progress.disable is False


def test_two_api_jobs_overlap_in_pipeline(monkeypatch, resources, tmp_path):
    """Two OCR jobs in one process must be able to be in the pipeline at once.

    The barrier only completes if both jobs reach ``run_pipeline`` together, so
    this asserts the property directly rather than timing anything.
    """
    barrier = threading.Barrier(2, timeout=60)
    real_run_pipeline = ocrmypdf.api.run_pipeline

    def spy_run_pipeline(*args, **kwargs):
        barrier.wait()
        return real_run_pipeline(*args, **kwargs)

    monkeypatch.setattr(ocrmypdf.api, 'run_pipeline', spy_run_pipeline)

    counter = itertools.count()

    def job():
        ocrmypdf.ocr(
            resources / 'trivial.pdf',
            tmp_path / f'out{next(counter)}.pdf',
            jobs=1,
            optimize=0,
            plugins=[NOOP_PLUGIN],
            progress_bar=False,
        )

    assert not _run_in_threads(job)


def test_two_api_jobs_produce_valid_output(resources, tmp_path):
    """Two concurrent jobs must both produce a valid PDF."""
    counter = itertools.count()

    def job():
        n = next(counter)
        ocrmypdf.ocr(
            resources / 'trivial.pdf',
            tmp_path / f'valid{n}.pdf',
            jobs=1,
            optimize=0,
            plugins=[NOOP_PLUGIN],
            progress_bar=False,
        )

    assert not _run_in_threads(job)
    outputs = sorted(tmp_path.glob('valid*.pdf'))
    assert len(outputs) == 2
    for output in outputs:
        with pikepdf.open(output) as pdf:
            assert len(pdf.pages) >= 1
