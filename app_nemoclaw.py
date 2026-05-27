# -*- coding: utf-8 -*-
"""
app_nemoclaw.py

NemoClaw-ready HTTP service wrapper for the Pediatric Neuro-Oncology Surgical Planning Agent.

This keeps the existing project architecture intact:
- NemotronClient remains the core reasoning interface.
- guardrails.py remains the medical safety backstop.
- image_analysis.py / advanced_medical_imaging.py / drug_ranking_adapter.py are optional tools.
- This file exposes the agent as a deployable service that a NemoClaw runner can call.

Security note:
- Do not send identifiable DICOM metadata to cloud services.
- Use de-identified image folders/files or local NIM/VLM deployment for real clinical data.
"""
from __future__ import annotations

import os
import json
import datetime as dt
from pathlib import Path
from typing import Any, Dict, Optional, List

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent import SurgicalPlanningAgent
from deid import deidentify_text, deidentify_case

APP_NAME = "pediatric-neuro-oncology-nemoclaw-service"
DEFAULT_OUTPUT_DIR = os.getenv("OUTPUT_DIR", "outputs")

app = FastAPI(
    title="Pediatric Neuro-Oncology Surgical Planning Agent - NemoClaw Service",
    version="1.0.0",
    description=(
        "Deployable long-agent service for pediatric neuro-oncology MDT planning. "
        "Nemotron is the core reasoning model; policy-based medical guardrails are applied to outputs."
    ),
)


class CaseRequest(BaseModel):
    raw_case_text: str = Field(..., description="Raw clinical case text. PHI should be removed before submission.")
    structured: Dict[str, Any] = Field(default_factory=dict, description="Structured case dict: age, tumor_type, markers, etc.")
    image_path: Optional[str] = Field(default=None, description="Optional JPG/PNG path visible inside the container.")
    medical_study_path: Optional[str] = Field(default=None, description="Optional DICOM folder/zip or NIfTI path visible inside the container.")
    enable_drug_ranking: bool = Field(default=False, description="Enable preclinical drug-ranking appendix; not treatment advice.")
    attach_architecture: bool = Field(default=True, description="Attach architecture diagram to the final report if available.")
    refresh_evidence_before_run: bool = Field(default=False, description="Refresh PubMed/ClinicalTrials evidence before running agent.")
    output_dir: str = Field(default=DEFAULT_OUTPUT_DIR, description="Output directory inside the container.")


class CaseResponse(BaseModel):
    status: str
    reasoning_mode: str
    output_path: Optional[str]
    enhanced_output_path: Optional[str] = None
    missing: List[str] = Field(default_factory=list)
    trace: List[str] = Field(default_factory=list)
    guardrails: Dict[str, Any] = Field(default_factory=dict)
    report_excerpt: str = ""
    generated_at: str


class RefreshRequest(BaseModel):
    pubmed_days: int = 30
    pubmed_retmax: int = 50
    out_dir: str = "rag_sources"


def _utc_now() -> str:
    return dt.datetime.utcnow().replace(microsecond=0).isoformat() + "Z"


def _preflight_case(raw_case_text: str, structured: Dict[str, Any]) -> tuple[str, Dict[str, Any], List[str]]:
    """Run lightweight de-identification before the long-agent pipeline.

    The main agent also runs de-identification; this preflight creates an audit signal
    for a NemoClaw policy layer without changing the existing agent API.
    """
    cleaned_text, text_hits = deidentify_text(raw_case_text or "")
    cleaned_case, case_hits = deidentify_case(structured or {})
    notes = []
    if text_hits:
        notes.append(f"preflight_deid_text_hits={len(text_hits)}")
    if case_hits:
        notes.append(f"preflight_deid_structured_hits={case_hits}")
    return cleaned_text, cleaned_case, notes


def _attach_architecture_if_requested(base_output_path: str, result_case: Dict[str, Any], output_dir: str) -> Optional[str]:
    try:
        from architecture_report_integration import install_architecture_asset, write_enhanced_report

        arch = install_architecture_asset(output_dir=output_dir)
        if arch.get("status") != "ok":
            return None
        enhanced = write_enhanced_report(
            output_path=str(Path(output_dir) / ("enhanced_" + Path(base_output_path).name)),
            base_report_path=base_output_path,
            architecture_image_path=arch.get("output_path"),
            drug_result=result_case.get("drug_ranking_result"),
            imaging_result=result_case.get("advanced_imaging_result") or result_case.get("image_analysis_result"),
        )
        return enhanced
    except Exception as exc:
        Path("logs").mkdir(exist_ok=True)
        with open("logs/nemoclaw_service.log", "a", encoding="utf-8") as f:
            f.write(f"{_utc_now()} ARCHITECTURE_ATTACH_SKIPPED {exc}\n")
        return None


@app.get("/health")
def health() -> Dict[str, Any]:
    return {
        "status": "ok",
        "service": APP_NAME,
        "nemotron_model": os.getenv("NEMOTRON_MODEL", "nvidia/nemotron-3-super-120b-a12b"),
        "nvidia_base_url": os.getenv("NVIDIA_BASE_URL", "https://integrate.api.nvidia.com/v1"),
        "output_dir": DEFAULT_OUTPUT_DIR,
        "timestamp": _utc_now(),
    }


@app.post("/refresh_once")
def refresh_once(req: RefreshRequest) -> Dict[str, Any]:
    try:
        from literature_trial_updater import refresh_evidence_sources

        manifest = refresh_evidence_sources(
            out_dir=req.out_dir,
            pubmed_days=req.pubmed_days,
            pubmed_retmax=req.pubmed_retmax,
        )
        return {"status": "ok", "manifest": manifest}
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"refresh failed: {exc}")


@app.post("/run_case", response_model=CaseResponse)
def run_case(req: CaseRequest) -> CaseResponse:
    try:
        if req.refresh_evidence_before_run:
            # Best-effort evidence refresh. Failure should not block urgent MDT report generation.
            try:
                from literature_trial_updater import refresh_evidence_sources
                refresh_evidence_sources(out_dir="rag_sources", pubmed_days=30, pubmed_retmax=30)
            except Exception as exc:
                Path("logs").mkdir(exist_ok=True)
                with open("logs/nemoclaw_service.log", "a", encoding="utf-8") as f:
                    f.write(f"{_utc_now()} EVIDENCE_REFRESH_SKIPPED {exc}\n")

        raw, structured, preflight_notes = _preflight_case(req.raw_case_text, req.structured)
        structured = dict(structured)
        if req.medical_study_path:
            structured["medical_study_path"] = req.medical_study_path
        elif req.image_path:
            structured["image_path"] = req.image_path
        if req.enable_drug_ranking:
            structured["enable_drug_ranking"] = True
            os.environ["ENABLE_DRUG_RANKING"] = "1"

        Path(req.output_dir).mkdir(parents=True, exist_ok=True)
        agent = SurgicalPlanningAgent(output_dir=req.output_dir)
        result = agent.run(raw, structured)
        result.setdefault("trace", []).extend(preflight_notes)

        output_path = result.get("output_path")
        enhanced_path = None
        if req.attach_architecture and output_path:
            enhanced_path = _attach_architecture_if_requested(output_path, result.get("case", {}), req.output_dir)

        report_path = enhanced_path or output_path
        report_excerpt = ""
        if report_path and Path(report_path).exists():
            report_excerpt = Path(report_path).read_text(encoding="utf-8", errors="ignore")[:4000]

        return CaseResponse(
            status="ok",
            reasoning_mode=agent.nemotron.mode,
            output_path=output_path,
            enhanced_output_path=enhanced_path,
            missing=result.get("missing", []),
            trace=result.get("trace", []),
            guardrails=result.get("guardrails", {}),
            report_excerpt=report_excerpt,
            generated_at=_utc_now(),
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"agent run failed: {exc}")
