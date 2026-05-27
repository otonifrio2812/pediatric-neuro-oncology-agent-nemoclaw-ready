# -*- coding: utf-8 -*-
"""Lightweight de-identification helper for demo purposes."""
from __future__ import annotations

import re
from typing import Dict, Any, Tuple

PHI_PATTERNS = [
    (re.compile(r"\bMRN[:\s]*[A-Za-z0-9_-]+", re.I), "MRN:[REDACTED]"),
    (re.compile(r"\b(patient\s*name|name)[:\s]+[A-Z][A-Za-z ,.'-]+", re.I), r"\1: [REDACTED]"),
    (re.compile(r"\b\d{3}[- ]?\d{2}[- ]?\d{4}\b"), "[REDACTED_ID]"),
    (re.compile(r"\b\d{4}[-/]\d{1,2}[-/]\d{1,2}\b"), "[REDACTED_DATE]"),
    (re.compile(r"\b\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\b"), "[REDACTED_DATE]"),
    (re.compile(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"), "[REDACTED_EMAIL]"),
    (re.compile(r"\b(?:\+?\d{1,3}[-.\s]?)?(?:\(?\d{2,4}\)?[-.\s]?)?\d{3,4}[-.\s]?\d{4}\b"), "[REDACTED_PHONE]"),
]


def deidentify_text(text: str) -> Tuple[str, list[str]]:
    text = text or ""
    findings = []
    out = text
    for pat, repl in PHI_PATTERNS:
        if pat.search(out):
            findings.append(pat.pattern)
            out = pat.sub(repl, out)
    return out, findings


def deidentify_case(case: Dict[str, Any]) -> Tuple[Dict[str, Any], list[str]]:
    case = dict(case or {})
    findings = []
    for key in ["patient_name", "name", "mrn", "date_of_birth"]:
        if key in case and case[key]:
            case[key] = "[REDACTED]"
            findings.append(key)
    for key, value in list(case.items()):
        if isinstance(value, str):
            cleaned, f = deidentify_text(value)
            case[key] = cleaned
            findings.extend([f"{key}:{x}" for x in f])
    return case, findings
