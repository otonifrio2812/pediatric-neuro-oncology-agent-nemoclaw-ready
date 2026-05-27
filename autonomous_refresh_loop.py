#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Persistent loop: update PubMed/ClinicalTrials evidence, then run watcher.py."""
from __future__ import annotations

import argparse
import subprocess
import time
from pathlib import Path

from literature_trial_updater import refresh_evidence_sources


def run_once(pubmed_days: int = 30, pubmed_retmax: int = 50) -> None:
    print(refresh_evidence_sources(pubmed_days=pubmed_days, pubmed_retmax=pubmed_retmax))
    if Path('watcher.py').exists():
        subprocess.run(['python', 'watcher.py', '--once'], check=False)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument('--once', action='store_true')
    p.add_argument('--interval-hours', type=float, default=6.0)
    p.add_argument('--pubmed-days', type=int, default=30)
    p.add_argument('--pubmed-retmax', type=int, default=50)
    args = p.parse_args()
    if args.once:
        run_once(args.pubmed_days, args.pubmed_retmax)
        return
    while True:
        try:
            run_once(args.pubmed_days, args.pubmed_retmax)
        except Exception as exc:
            print('loop error:', exc)
        time.sleep(int(args.interval_hours * 3600))


if __name__ == '__main__':
    main()
