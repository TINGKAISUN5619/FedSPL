"""Locate experiment records in the packaged directory layout."""
import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PREFIX = 'results/reviewer_revision_20260831/'
GROUPS = ('corrected', 'official_audit', 'historical_task')


def source(name):
    starts = [name.index(marker) for marker in ('results/', 'paper_acs_latex/', 'TEST/') if marker in name]
    relative = Path(name[min(starts):] if starts else name)
    if relative.is_absolute() or '..' in relative.parts:
        raise ValueError('Expected a relative result path')
    matches = [ROOT / group / 'records' / relative for group in GROUPS]
    matches = [path for path in matches if path.exists()]
    if len(matches) != 1:
        raise FileNotFoundError('Expected one packaged result for: ' + str(relative))
    return matches[0]


def data(name):
    with source(PREFIX + name).open(newline='', encoding='utf-8-sig') as stream:
        return list(csv.DictReader(stream))
