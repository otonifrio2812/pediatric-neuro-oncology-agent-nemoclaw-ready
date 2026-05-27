#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Autonomous once/loop runner for evidence refresh + case execution."""
from __future__ import annotations

import argparse
import datetime as dt
import subprocess
import time
from pathlib import Path


def ts() -> str:
    return dt.datetime.now().isoformat(timespec='seconds')


def run_once(case_file: str = 'sample_cases/case_003_diffuse_midline_glioma.txt') -> None:
    Path('logs').mkdir(exist_ok=True)
    # Refresh evidence from local knowledge base status; external updater can be called separately.
    try:
        from rag import SimpleRAG
        rag = SimpleRAG()
        status = rag.status()
        with open('logs/autonomous_runs.log', 'a', encoding='utf-8') as f:
            f.write(f"{ts()} KB_REFRESH new_items=0 indexed_chunks={status.get('chunks')}\n")
        print(f"[{ts()}] autonomous refresh: indexed {status.get('chunks')} chunks")
    except Exception as exc:
        print('KB refresh skipped:', exc)

    if Path(case_file).exists():
        subprocess.run(['python', 'run_demo.py', case_file], check=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--once', action='store_true')
    p.add_argument('--interval-seconds', type=int, default=3600)
    p.add_argument('--case-file', default='sample_cases/case_003_diffuse_midline_glioma.txt')
    args = p.parse_args()
    if args.once:
        run_once(args.case_file)
        return
    while True:
        run_once(args.case_file)
        time.sleep(args.interval_seconds)


if __name__ == '__main__':
    main()
