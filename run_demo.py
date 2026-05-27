#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Run a single pediatric neuro-oncology agent demo."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from agent import SurgicalPlanningAgent


def load_structured(path: str | None) -> dict:
    if not path:
        return {}
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(path)
    return json.loads(p.read_text(encoding='utf-8'))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('case_file', help='Path to raw case text file')
    parser.add_argument('--structured-json', default=None, help='Optional structured case JSON')
    parser.add_argument('--image', default=None, help='Optional JPG/PNG/DICOM/NIfTI path')
    parser.add_argument('--images', nargs='*', default=None, help='Optional multiple images or medical files')
    parser.add_argument('--medical-study', default=None, help='Optional DICOM folder/zip or NIfTI file/folder')
    parser.add_argument('--enable-drug-ranking', action='store_true')
    parser.add_argument('--attach-architecture', action='store_true')
    parser.add_argument('--output-dir', default='outputs')
    args = parser.parse_args()

    raw = Path(args.case_file).read_text(encoding='utf-8', errors='ignore')
    structured = load_structured(args.structured_json)
    if args.medical_study:
        structured['medical_study_path'] = args.medical_study
    elif args.images:
        structured['images'] = args.images
    elif args.image:
        structured['image_path'] = args.image
    if args.enable_drug_ranking:
        structured['enable_drug_ranking'] = True
        os.environ['ENABLE_DRUG_RANKING'] = '1'

    agent = SurgicalPlanningAgent(output_dir=args.output_dir)
    result = agent.run(raw, structured)
    output_path = result['output_path']

    if args.attach_architecture:
        try:
            from architecture_report_integration import install_architecture_asset, write_enhanced_report
            arch = install_architecture_asset()
            if arch.get('status') == 'ok':
                enhanced = write_enhanced_report(
                    output_path=str(Path(args.output_dir) / ('enhanced_' + Path(output_path).name)),
                    base_report_path=output_path,
                    architecture_image_path=arch.get('output_path'),
                    drug_result=result['case'].get('drug_ranking_result'),
                    imaging_result=result['case'].get('advanced_imaging_result') or result['case'].get('image_analysis_result'),
                )
                output_path = enhanced
        except Exception as exc:
            print('Architecture attachment skipped:', exc)

    print(f"Reasoning mode: {agent.nemotron.mode}")
    print(f"Report written: {output_path}")
    print('\n' + Path(output_path).read_text(encoding='utf-8')[:4000])


if __name__ == '__main__':
    main()
