"""
Keep docs/report.html honest against README.md.

The report restates measured figures that live in README.md. Duplication is the
point — the report is the document you hand someone — but it means the two can
drift, and a report contradicting the README is worse than shipping no report:
it turns a strength (everything is measured) into evidence of carelessness.

These tests fail when a headline number appears in one and not the other. They
are stdlib-only and offline, so CI runs them in seconds alongside the metrics
tests.

If a number legitimately changes: update README.md, update docs/report.html,
then regenerate the PDF with ./scripts/make_report_pdf.sh.
"""
import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
REPORT = ROOT / "docs" / "report.html"

# Headline measurements. Each must appear in BOTH documents. Kept deliberately
# short: these are the numbers someone would quote, not every figure reported.
HEADLINE = [
    "11.54",     # retrieval P50, ms
    "53.57",     # retrieval P100, ms
    "103,068",   # serving-index chunks
    "0.551",     # MRR@10
    "0.744",     # Recall@5
    "14.7%",     # ungrounded answers, cross-encoder gate
    "63.75",     # rank_bm25 per-query cost, ms
    "0.9983",    # bm25s rank correlation
    "21,657",    # eval-index chunks
    "1,219",     # gold-labelled eval queries
]


def _readme() -> str:
    return README.read_text(encoding="utf-8")


def _report() -> str:
    return REPORT.read_text(encoding="utf-8")


def test_report_source_exists():
    """The PDF is generated from this file; without it the script cannot run."""
    assert REPORT.is_file(), (
        "docs/report.html is missing — scripts/make_report_pdf.sh builds from it"
    )


@pytest.mark.parametrize("value", HEADLINE)
def test_headline_number_present_in_readme(value):
    assert value in _readme(), (
        f"{value!r} is quoted in docs/report.html but no longer appears in "
        "README.md. If the measurement changed, update both — a report that "
        "contradicts the README undermines every other number in it."
    )


@pytest.mark.parametrize("value", HEADLINE)
def test_headline_number_present_in_report(value):
    assert value in _report(), (
        f"{value!r} appears in README.md but not in docs/report.html. Update the "
        "report, then regenerate the PDF with ./scripts/make_report_pdf.sh."
    )


def test_report_has_numbered_sections_for_contents():
    """make_report_pdf.sh builds the PDF contents page from these headings."""
    sections = re.findall(r'<h2><span class="num">(\d+)</span>', _report())
    assert len(sections) >= 10, (
        f"expected the report's numbered sections, found {len(sections)} — the "
        "PDF contents page is generated from them and would come out empty"
    )


def test_report_declares_a_title():
    """The <title> names the artifact in the browser tab and gallery."""
    assert re.search(r"<title>.+?</title>", _report()), "docs/report.html has no <title>"


def test_no_local_paths_leaked_into_report():
    """A scratchpad path in a shared document is an obvious tell."""
    for probe in ("/tmp/claude", "/home/lokesh"):
        assert probe not in _report(), f"local path {probe!r} leaked into docs/report.html"
