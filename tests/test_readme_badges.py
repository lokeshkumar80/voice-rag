"""
Guard the README's CI badge against a silent break.

A GitHub Actions badge is built from the workflow *filename*:

    https://github.com/<owner>/<repo>/actions/workflows/<file>/badge.svg

Rename or move the workflow and the badge does not error — it quietly renders
"no status" forever, while CI itself keeps passing. Nothing in the repo would
otherwise catch that, and a stale badge on a portfolio project reads as broken
tests.

These tests are stdlib-only and offline: they assert the badge URL and the
workflow file agree with each other, not that GitHub is reachable.
"""
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
README = ROOT / "README.md"
WORKFLOW_DIR = ROOT / ".github" / "workflows"

# .../actions/workflows/<name>.yml  (badge.svg and the plain link both match)
BADGE_RE = re.compile(r"actions/workflows/([A-Za-z0-9_.-]+\.ya?ml)")


def _referenced_workflows() -> set[str]:
    return set(BADGE_RE.findall(README.read_text(encoding="utf-8")))


def test_readme_references_at_least_one_workflow():
    """If the badge disappears entirely, that is also a regression."""
    assert _referenced_workflows(), (
        "README references no Actions workflow — the CI badge was removed or its "
        "URL changed shape"
    )


def test_every_referenced_workflow_file_exists():
    """The badge filename must match a real workflow, or the badge goes blank."""
    missing = sorted(w for w in _referenced_workflows() if not (WORKFLOW_DIR / w).is_file())
    assert not missing, (
        f"README points at workflow(s) that do not exist: {missing}. "
        f"Present: {sorted(p.name for p in WORKFLOW_DIR.glob('*.y*ml'))}. "
        "Renaming a workflow silently blanks its badge — update the README URL too."
    )


def test_badge_and_link_target_the_same_workflow():
    """The badge image and the link it wraps should point at one workflow."""
    refs = _referenced_workflows()
    assert len(refs) == 1, (
        f"expected the badge image and its link to name the same workflow, got {sorted(refs)}"
    )


def test_workflow_is_not_orphaned():
    """Every workflow present should be surfaced by a badge, so CI stays visible."""
    present = {p.name for p in WORKFLOW_DIR.glob("*.y*ml")}
    unreferenced = sorted(present - _referenced_workflows())
    assert not unreferenced, (
        f"workflow(s) with no README badge: {unreferenced} — CI runs but nobody sees it"
    )
