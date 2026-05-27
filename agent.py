# -*- coding: utf-8 -*-
"""Pediatric Neuro-Oncology Surgical Planning Agent.

Autonomous long-agent flow:
1. De-identification
2. Completeness check
3. Optional image analysis (JPG/PNG VLM or DICOM/NIfTI advanced imaging)
4. RAG retrieval
5. Nemotron 3 Super core reasoning
6. Clinical-trial matching
7. Optional preclinical drug-ranking appendix
8. Policy-based medical guardrails
9. MDT report generation
"""
from __future__ import annotations

import json
import os
import datetime as dt
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

from deid import deidentify_text, deidentify_case
from guardrails import MedicalGuardrails
from nemotron_client import NemotronClient
from rag import SimpleRAG
from clinical_trials import match_trials, trials_to_markdown


SAFETY_DISCLAIMER = (
    "AI 影像與臨床資料輔助分析，僅供 MDT 討論；不可作為確定性診斷、治療或手術決策唯一依據。"
)


def now_tag() -> str:
    return dt.datetime.now().strftime('%Y%m%d_%H%M%S')


def _is_volume_input(path: str) -> bool:
    if not path:
        return False
    p = str(path).lower()
    return os.path.isdir(path) or p.endswith('.nii') or p.endswith('.nii.gz') or p.endswith('.dcm') or p.endswith('.dicom') or p.endswith('.zip')


def _is_raster_input(path: str) -> bool:
    return str(path).lower().endswith(('.png', '.jpg', '.jpeg', '.webp'))


class SurgicalPlanningAgent:
    def __init__(
        self,
        nemotron_client: Optional[NemotronClient] = None,
        rag: Optional[SimpleRAG] = None,
        guardrails: Optional[MedicalGuardrails] = None,
        output_dir: str = 'outputs',
    ) -> None:
        self.nemotron = nemotron_client or NemotronClient()
        self.rag = rag or SimpleRAG()
        self.guardrails = guardrails or MedicalGuardrails()
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def parse_case_text(self, raw_case_text: str) -> Dict[str, Any]:
        """Tiny heuristic parser for demo text files. Structured dict may override these values."""
        lower = (raw_case_text or '').lower()
        case: Dict[str, Any] = {}
        import re
        m = re.search(r'\b(\d{1,2})\s*(?:year|yr|y/o|yo|歲)', lower)
        if m:
            case['age'] = int(m.group(1))
        elif '8-year' in lower or '8 year' in lower:
            case['age'] = 8
        elif '7-year' in lower or '7 year' in lower:
            case['age'] = 7

        if 'diffuse midline glioma' in lower or 'h3 k27' in lower or 'h3k27' in lower:
            case['tumor_type'] = 'diffuse midline glioma'
            case['pathology'] = 'possible H3 K27-altered diffuse midline glioma'
        elif 'posterior fossa' in lower:
            case['tumor_type'] = 'posterior fossa tumor'
        elif 'medulloblastoma' in lower:
            case['tumor_type'] = 'medulloblastoma'
        elif 'ependymoma' in lower:
            case['tumor_type'] = 'ependymoma'

        loc_map = {
            'pons': 'pons', 'pontine': 'pons', 'brainstem': 'brainstem',
            'midbrain': 'midbrain', 'medulla': 'medulla', 'thalamus': 'thalamus',
            'spinal cord': 'spinal cord', 'posterior fossa': 'posterior fossa', 'cerebellum': 'cerebellum',
        }
        for key, loc in loc_map.items():
            if key in lower:
                case['tumor_location'] = loc
                break

        markers = {}
        if 'h3k27m' in lower or 'h3 k27m' in lower or 'h3 k27' in lower:
            markers['H3K27M'] = 'positive/suspected'
        if 'braf' in lower:
            markers['BRAF'] = 'mentioned'
        if markers:
            case['molecular_markers'] = markers

        symptoms = []
        for s in ['headache', 'vomiting', 'ataxia', 'gait instability', 'diplopia', 'weakness', 'hydrocephalus']:
            if s in lower:
                symptoms.append(s)
        if symptoms:
            case['symptoms'] = symptoms

        # Keep a compact imaging sentence if obvious.
        imaging_lines = [ln.strip() for ln in (raw_case_text or '').splitlines() if any(k in ln.lower() for k in ['mri', 'ct', 'imaging', 'flair', 'enhancement', 'hydrocephalus', 'pons'])]
        if imaging_lines:
            case['imaging_description'] = ' '.join(imaging_lines[:4])
        return case

    def check_completeness(self, case: Dict[str, Any]) -> List[str]:
        required = ['age', 'tumor_type', 'tumor_location', 'imaging_description']
        missing = [k for k in required if not case.get(k)]
        return missing

    def maybe_analyze_images(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """Optional image analysis with graceful fallback."""
        case = dict(case)
        image_input = case.get('medical_study_path') or case.get('image_dir') or case.get('image_path')
        images = case.get('images')
        if not image_input and images:
            image_input = images
        if not image_input:
            return case

        # DICOM/NIfTI advanced imaging route.
        try:
            if isinstance(image_input, (list, tuple)) or _is_volume_input(str(image_input)):
                from advanced_medical_imaging import integrate_advanced_imaging_into_case
                updated = integrate_advanced_imaging_into_case(case, image_input, output_dir=str(self.output_dir))
                return updated
        except Exception as exc:
            case['advanced_imaging_warning'] = f'Advanced imaging failed gracefully: {exc}'

        # JPG/PNG VLM route.
        try:
            first = image_input[0] if isinstance(image_input, (list, tuple)) else image_input
            if first and _is_raster_input(str(first)):
                from image_analysis import describe_image, image_description_to_text
                img = describe_image(str(first))
                case['image_analysis_result'] = img
                if img.get('status') == 'ok':
                    case['imaging_description'] = image_description_to_text(img)
                else:
                    case['image_analysis_warning'] = 'Image VLM unavailable; keeping manual imaging_description.'
        except Exception as exc:
            case['image_analysis_warning'] = f'Raster image analysis failed gracefully: {exc}'
        return case

    def build_prompt(self, case: Dict[str, Any], evidence: List[Dict[str, Any]], trials_md: str) -> str:
        evidence_md = '\n\n'.join(
            f"[{e.get('id')}] {e.get('source')} score={e.get('score', 0):.3f}\n{e.get('text')}"
            for e in evidence
        )
        return f"""
Structured pediatric neuro-oncology case:
{json.dumps(case, ensure_ascii=False, indent=2)}

Retrieved evidence snippets:
{evidence_md}

Preliminary trial matches:
{trials_md}

Task:
Generate a concise pre-operative MDT planning analysis for pediatric neuro-oncology.
Do NOT make a definitive diagnosis. Do NOT prescribe treatment. Distinguish evidence-based points from uncertainties.
Address: differential considerations, GTR vs STR considerations, high-risk anatomy, hydrocephalus/CSF concerns, molecular/pathology needs, and MDT next steps.
""".strip()

    def run(self, raw_case_text: str, structured: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        trace = []
        raw_case_text = raw_case_text or ''
        cleaned_text, phi_hits = deidentify_text(raw_case_text)
        trace.append(f'De-identification: removed {len(phi_hits)} text pattern(s)')

        parsed = self.parse_case_text(cleaned_text)
        case = {**parsed, **(structured or {})}
        case, case_phi_hits = deidentify_case(case)
        if case_phi_hits:
            trace.append(f'De-identification: removed structured fields/patterns {case_phi_hits}')

        case = self.maybe_analyze_images(case)
        if case.get('advanced_imaging_result'):
            trace.append(f"Advanced imaging: {case['advanced_imaging_result'].get('status')}")
        elif case.get('image_analysis_result'):
            trace.append(f"Image analysis: {case['image_analysis_result'].get('status')}")
        else:
            trace.append('Image analysis: not provided; using manual imaging_description if present')

        missing = self.check_completeness(case)
        trace.append('Completeness: missing ' + (', '.join(missing) if missing else 'nothing critical'))

        query = '\n'.join([cleaned_text, json.dumps(case, ensure_ascii=False)])
        evidence = self.rag.retrieve(query, top_k=int(os.getenv('RAG_TOP_K', '5')))
        trace.append(f'Retrieved {len(evidence)} evidence chunks')

        trials = match_trials(case, top_k=4)
        trials_md = trials_to_markdown(trials)
        trace.append(f'Matched {len(trials)} candidate trial(s)')

        system = (
            'You are NVIDIA Nemotron 3 Super acting as the core reasoning model for a pediatric neuro-oncology MDT planning agent. '
            'You must be cautious, non-diagnostic, and policy-safe. Use only decision-support language.'
        )
        user = self.build_prompt(case, evidence, trials_md)
        reasoning = self.nemotron.complete(system=system, user=user)
        trace.append(f'Nemotron reasoning generated ({self.nemotron.mode})')

        drug_md = ''
        if os.getenv('ENABLE_DRUG_RANKING', '0') == '1' or case.get('enable_drug_ranking'):
            try:
                from drug_ranking_adapter import rank_drugs_for_case, drug_ranking_to_markdown
                drug_result = rank_drugs_for_case(case, top_k=10)
                case['drug_ranking_result'] = drug_result
                drug_md = drug_ranking_to_markdown(drug_result)
                trace.append(f"Drug ranking appendix: {drug_result.get('status')}")
            except Exception as exc:
                drug_md = f"## Preclinical Drug Ranking\n\nModule unavailable: {exc}\n"
                trace.append('Drug ranking appendix: unavailable')

        report = self.compose_report(
            raw_case_text=cleaned_text,
            case=case,
            missing=missing,
            evidence=evidence,
            reasoning=reasoning,
            trials_md=trials_md,
            drug_md=drug_md,
            trace=trace,
        )

        gr = self.guardrails.check(report)
        trace.append(f"Guardrails: passed={gr.passed}, blocked={len(gr.blocked)}, flagged={gr.flagged}")
        final_report = gr.safe_text

        output_name = f"case_report_{now_tag()}.md"
        output_path = self.output_dir / output_name
        output_path.write_text(final_report, encoding='utf-8')

        try:
            Path('logs').mkdir(exist_ok=True)
            with open('logs/autonomous_runs.log', 'a', encoding='utf-8') as f:
                f.write(f"{dt.datetime.now().isoformat()} CASE_RUN mode={self.nemotron.mode} missing={len(missing)} trials={len(trials)} guardrail_pass={gr.passed}\n")
        except Exception:
            pass

        return {
            'case': case,
            'missing': missing,
            'evidence': evidence,
            'trials': trials,
            'reasoning': reasoning,
            'guardrails': gr.__dict__,
            'report': final_report,
            'output_path': str(output_path),
            'trace': trace,
        }

    def compose_report(self, raw_case_text: str, case: Dict[str, Any], missing: List[str], evidence: List[Dict[str, Any]], reasoning: str, trials_md: str, drug_md: str, trace: List[str]) -> str:
        lines = []
        lines.append('# Pediatric Neuro-Oncology Surgical Planning MDT Report')
        lines.append('')
        lines.append(f'Generated: {dt.datetime.now().isoformat(timespec="seconds")}')
        lines.append(f'Core reasoning model: **{self.nemotron.mode}**')
        lines.append('')
        lines.append('## 1. Structured Case Snapshot')
        lines.append('```json')
        safe_case = {k: v for k, v in case.items() if k not in {'advanced_imaging_result', 'image_analysis_result', 'drug_ranking_result'}}
        lines.append(json.dumps(safe_case, ensure_ascii=False, indent=2))
        lines.append('```')
        lines.append('')
        lines.append('## 2. Completeness Check')
        lines.append('- Missing critical fields: ' + (', '.join(missing) if missing else 'none'))
        lines.append('')
        if case.get('advanced_imaging_result'):
            lines.append('## 3. Advanced Imaging Findings (DICOM/NIfTI)')
            lines.append(case['advanced_imaging_result'].get('imaging_description_text', ''))
            if case.get('surgical_risk_map_path'):
                lines.append(f"\nSurgical risk map: `{case.get('surgical_risk_map_path')}`")
            lines.append('')
        elif case.get('image_analysis_result'):
            lines.append('## 3. Image-to-Text Findings')
            lines.append(json.dumps(case.get('image_analysis_result'), ensure_ascii=False, indent=2))
            lines.append('')
        else:
            lines.append('## 3. Imaging Description')
            lines.append(str(case.get('imaging_description', 'Not provided.')))
            lines.append('')

        lines.append('## 4. Nemotron Clinical Reasoning')
        lines.append(reasoning)
        lines.append('')
        lines.append('## 5. Retrieved Evidence')
        for e in evidence:
            lines.append(f"- **[{e.get('id')}]** `{e.get('source')}` ({e.get('category')}, score {e.get('score', 0):.3f}): {str(e.get('text', ''))[:400]}...")
        lines.append('')
        lines.append('## 6. Clinical Trial Matching (preliminary candidates)')
        lines.append(trials_md)
        lines.append('')
        if drug_md:
            lines.append(drug_md)
            lines.append('')
        lines.append('## 7. Autonomous Agent Trace')
        for t in trace:
            lines.append(f'- {t}')
        lines.append('')
        lines.append('## 8. Safety Scope')
        lines.append(SAFETY_DISCLAIMER)
        return '\n'.join(lines)
