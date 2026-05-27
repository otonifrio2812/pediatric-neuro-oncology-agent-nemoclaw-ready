# -*- coding: utf-8 -*-
"""Autonomous PubMed / ClinicalTrials updater for RAG sources."""
from __future__ import annotations

import datetime as dt
import hashlib
import json
import os
import time
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Dict, Any, List

import requests

DEFAULT_QUERY = (
    '("pediatric brain tumor" OR "pediatric neuro-oncology" OR '
    '"diffuse midline glioma" OR "H3 K27-altered" OR "brainstem glioma" OR '
    '"posterior fossa tumor")'
)
PUBMED_ESEARCH = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi'
PUBMED_EFETCH = 'https://eutils.ncbi.nlm.nih.gov/entrez/eutils/efetch.fcgi'
CTGOV_STUDIES = 'https://clinicaltrials.gov/api/v2/studies'


def now_iso() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + 'Z'


def stable_id(prefix: str, text: str) -> str:
    return prefix + '_' + hashlib.sha256(text.encode('utf-8', errors='ignore')).hexdigest()[:16]


def append_jsonl(path: str, records: List[Dict[str, Any]]) -> int:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    existing = set()
    if p.exists():
        for line in p.read_text(encoding='utf-8', errors='ignore').splitlines():
            try:
                existing.add(json.loads(line).get('id'))
            except Exception:
                pass
    n = 0
    with p.open('a', encoding='utf-8') as f:
        for r in records:
            rid = r.get('id')
            if not rid or rid in existing:
                continue
            f.write(json.dumps(r, ensure_ascii=False) + '\n')
            existing.add(rid)
            n += 1
    return n


def pubmed_search_pmids(query: str = DEFAULT_QUERY, days: int = 30, retmax: int = 50) -> List[str]:
    params = {
        'db': 'pubmed', 'term': query, 'reldate': str(days), 'datetype': 'edat',
        'retmax': str(retmax), 'retmode': 'json', 'sort': 'pub_date',
        'tool': os.getenv('NCBI_TOOL', 'pediatric_neuro_oncology_agent'),
        'email': os.getenv('NCBI_EMAIL', 'your_email@example.com'),
    }
    if os.getenv('NCBI_API_KEY'):
        params['api_key'] = os.getenv('NCBI_API_KEY')
    r = requests.get(PUBMED_ESEARCH, params=params, timeout=30)
    r.raise_for_status()
    return r.json().get('esearchresult', {}).get('idlist', [])


def pubmed_fetch_details(pmids: List[str]) -> List[Dict[str, Any]]:
    if not pmids:
        return []
    params = {
        'db': 'pubmed', 'id': ','.join(pmids), 'retmode': 'xml',
        'tool': os.getenv('NCBI_TOOL', 'pediatric_neuro_oncology_agent'),
        'email': os.getenv('NCBI_EMAIL', 'your_email@example.com'),
    }
    if os.getenv('NCBI_API_KEY'):
        params['api_key'] = os.getenv('NCBI_API_KEY')
    r = requests.get(PUBMED_EFETCH, params=params, timeout=60)
    r.raise_for_status()
    root = ET.fromstring(r.text)
    records = []
    for art in root.findall('.//PubmedArticle'):
        pmid_el = art.find('.//PMID')
        pmid = pmid_el.text if pmid_el is not None else None
        title_el = art.find('.//ArticleTitle')
        title = ''.join(title_el.itertext()).strip() if title_el is not None else ''
        abstract = '\n'.join(''.join(x.itertext()).strip() for x in art.findall('.//AbstractText'))
        journal_el = art.find('.//Journal/Title')
        journal = journal_el.text if journal_el is not None else ''
        year_el = art.find('.//PubDate/Year')
        year = year_el.text if year_el is not None else ''
        records.append({
            'id': f'PMID:{pmid}' if pmid else stable_id('pubmed', title + abstract),
            'source': 'PubMed', 'pmid': pmid, 'title': title, 'abstract': abstract,
            'journal': journal, 'year': year, 'fetched_at': now_iso(),
            'rag_text': f'Title: {title}\nJournal: {journal} {year}\nAbstract: {abstract}',
        })
    return records


def clinicaltrials_search(query: str = 'pediatric diffuse midline glioma OR pediatric brain tumor', page_size: int = 25, max_pages: int = 2) -> List[Dict[str, Any]]:
    out = []
    token = None
    for _ in range(max_pages):
        params = {'query.term': query, 'pageSize': str(page_size), 'format': 'json'}
        if token:
            params['pageToken'] = token
        r = requests.get(CTGOV_STUDIES, params=params, timeout=30)
        r.raise_for_status()
        data = r.json()
        for study in data.get('studies', []):
            protocol = study.get('protocolSection', {})
            ident = protocol.get('identificationModule', {})
            status = protocol.get('statusModule', {})
            desc = protocol.get('descriptionModule', {})
            cond = protocol.get('conditionsModule', {})
            arms = protocol.get('armsInterventionsModule', {})
            nct = ident.get('nctId')
            title = ident.get('briefTitle') or ident.get('officialTitle') or ''
            summary = desc.get('briefSummary', '')
            conditions = cond.get('conditions', [])
            interventions = [x.get('name') for x in arms.get('interventions', []) if isinstance(x, dict) and x.get('name')]
            out.append({
                'id': f'NCT:{nct}' if nct else stable_id('trial', title + summary),
                'source': 'ClinicalTrials.gov', 'nct_id': nct, 'title': title,
                'overall_status': status.get('overallStatus'), 'conditions': conditions,
                'interventions': interventions, 'summary': summary, 'fetched_at': now_iso(),
                'rag_text': f'NCT ID: {nct}\nTitle: {title}\nStatus: {status.get("overallStatus")}\nConditions: {", ".join(conditions)}\nInterventions: {", ".join(interventions)}\nSummary: {summary}',
            })
        token = data.get('nextPageToken')
        if not token:
            break
        time.sleep(0.5)
    return out


def refresh_evidence_sources(out_dir: str = 'rag_sources', pubmed_days: int = 30, pubmed_retmax: int = 50, query: str = DEFAULT_QUERY) -> Dict[str, Any]:
    Path(out_dir).mkdir(parents=True, exist_ok=True)
    try:
        pmids = pubmed_search_pmids(query=query, days=pubmed_days, retmax=pubmed_retmax)
        pubmed_records = pubmed_fetch_details(pmids)
    except Exception as exc:
        pubmed_records = []
        pubmed_error = str(exc)
    else:
        pubmed_error = None
    try:
        trial_records = clinicaltrials_search()
    except Exception as exc:
        trial_records = []
        trial_error = str(exc)
    else:
        trial_error = None
    new_pubmed = append_jsonl(str(Path(out_dir) / 'pubmed_latest.jsonl'), pubmed_records)
    new_trials = append_jsonl(str(Path(out_dir) / 'clinicaltrials_latest.jsonl'), trial_records)
    manifest = {
        'refreshed_at': now_iso(), 'query': query, 'pubmed_days': pubmed_days,
        'pubmed_found': len(pubmed_records), 'pubmed_new': new_pubmed, 'pubmed_error': pubmed_error,
        'trials_found': len(trial_records), 'trials_new': new_trials, 'trial_error': trial_error,
    }
    Path(out_dir, 'refresh_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding='utf-8')
    Path('logs').mkdir(exist_ok=True)
    with open('logs/autonomous_runs.log', 'a', encoding='utf-8') as f:
        f.write(f"{now_iso()} EVIDENCE_REFRESH pubmed_new={new_pubmed} trials_new={new_trials} pubmed_found={len(pubmed_records)} trials_found={len(trial_records)}\n")
    return manifest


if __name__ == '__main__':
    print(json.dumps(refresh_evidence_sources(), ensure_ascii=False, indent=2))
