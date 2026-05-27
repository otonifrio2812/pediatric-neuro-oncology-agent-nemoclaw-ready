# -*- coding: utf-8 -*-
"""Simple clinical trial matcher for hackathon demo.

The matcher uses local illustrative trials plus optionally refreshed ClinicalTrials.gov
records in rag_sources/clinicaltrials_latest.jsonl. Results are preliminary and require
human verification.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List


SAMPLE_TRIALS = [
    {
        'id': 'SAMPLE-DMG-02',
        'title': 'Targeted agent for H3 K27M-altered diffuse midline glioma',
        'phase': 'Phase II',
        'age_min': 2,
        'age_max': 25,
        'tumor_types': ['diffuse midline glioma', 'brainstem glioma', 'high grade glioma'],
        'markers': ['h3k27m', 'h3 k27'],
        'source': 'ClinicalTrials.gov (illustrative sample)',
    },
    {
        'id': 'SAMPLE-MB-01',
        'title': 'Risk-adapted therapy for newly diagnosed medulloblastoma',
        'phase': 'Phase III',
        'age_min': 3,
        'age_max': 21,
        'tumor_types': ['medulloblastoma', 'posterior fossa tumor'],
        'markers': [],
        'source': 'ClinicalTrials.gov (illustrative sample)',
    },
    {
        'id': 'SAMPLE-LGG-03',
        'title': 'BRAF/MEK inhibition in pediatric low-grade glioma',
        'phase': 'Phase II',
        'age_min': 1,
        'age_max': 18,
        'tumor_types': ['low grade glioma', 'pilocytic astrocytoma', 'glioma'],
        'markers': ['braf'],
        'source': 'ClinicalTrials.gov (illustrative sample)',
    },
    {
        'id': 'SAMPLE-EP-04',
        'title': 'Adjuvant therapy intensification for pediatric ependymoma',
        'phase': 'Phase III',
        'age_min': 1,
        'age_max': 21,
        'tumor_types': ['ependymoma', 'posterior fossa tumor'],
        'markers': [],
        'source': 'ClinicalTrials.gov (illustrative sample)',
    },
]


def _norm(x: Any) -> str:
    return str(x or '').lower()


def _case_markers(case: Dict[str, Any]) -> str:
    markers = case.get('molecular_markers') or {}
    if isinstance(markers, dict):
        return ' '.join(f'{k} {v}' for k, v in markers.items()).lower()
    return str(markers).lower()


def load_trials(path: str = 'clinical_trials/sample_trials.json') -> List[Dict[str, Any]]:
    p = Path(path)
    trials = list(SAMPLE_TRIALS)
    if p.exists():
        try:
            data = json.loads(p.read_text(encoding='utf-8'))
            if isinstance(data, list):
                trials = data + trials
        except Exception:
            pass
    # Optional refreshed ClinicalTrials.gov summaries.
    refreshed = Path('rag_sources/clinicaltrials_latest.jsonl')
    if refreshed.exists():
        with refreshed.open(encoding='utf-8', errors='ignore') as f:
            for line in f:
                try:
                    obj = json.loads(line)
                except Exception:
                    continue
                trials.append({
                    'id': obj.get('nct_id') or obj.get('id'),
                    'title': obj.get('title'),
                    'phase': obj.get('study_type', 'Unknown'),
                    'age_min': 0,
                    'age_max': 99,
                    'tumor_types': obj.get('conditions', []),
                    'markers': [],
                    'source': 'ClinicalTrials.gov refreshed',
                    'status': obj.get('overall_status'),
                })
    return trials


def match_trials(case: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
    age = case.get('age')
    try:
        age = int(age)
    except Exception:
        age = None
    tumor_text = ' '.join([
        _norm(case.get('tumor_type')),
        _norm(case.get('pathology')),
        _norm(case.get('tumor_location')),
        _norm(case.get('imaging_description')),
    ])
    markers = _case_markers(case)

    matches = []
    for t in load_trials():
        score = 0
        notes = []
        if age is not None and t.get('age_min', 0) <= age <= t.get('age_max', 99):
            score += 2
            notes.append(f'age {age} within {t.get("age_min")}-{t.get("age_max")}')
        elif age is not None:
            notes.append(f'age {age} outside nominal range {t.get("age_min")}-{t.get("age_max")}')

        tumor_types = [_norm(x) for x in t.get('tumor_types', [])]
        if any(tt and tt in tumor_text for tt in tumor_types):
            score += 3
            notes.append('tumor type/location text overlaps inclusion keywords')
        else:
            notes.append('tumor type not confirmed against inclusion list')

        req_markers = [_norm(x) for x in t.get('markers', [])]
        if req_markers:
            if any(m and m in markers for m in req_markers):
                score += 2
                notes.append('required molecular marker appears present')
            else:
                notes.append('required marker pending/absent: ' + ', '.join(req_markers))

        matches.append({**t, 'score': score, 'notes': notes})

    matches.sort(key=lambda x: x.get('score', 0), reverse=True)
    return matches[:top_k]


def trials_to_markdown(matches: List[Dict[str, Any]]) -> str:
    if not matches:
        return 'No trial candidates found. Manual ClinicalTrials.gov review recommended.'
    lines = []
    for m in matches:
        lines.append(f"### {m.get('id')} — {m.get('title')} ({m.get('phase')})")
        prefix = '✅ Match' if m.get('score', 0) >= 4 else '❓ To confirm'
        lines.append(f"- {prefix}: " + '; '.join(m.get('notes', [])))
        if m.get('status'):
            lines.append(f"- Status: {m.get('status')}")
        lines.append(f"- Source: {m.get('source')}")
    return '\n'.join(lines)
