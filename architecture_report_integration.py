# -*- coding: utf-8 -*-
"""
architecture_report_integration.py

Utilities to add `腦瘤架構圖.jpg` and optional drug-ranking results
to the Pediatric Neuro-Oncology Surgical Planning Agent outputs.

This is intentionally independent from agent.py:
- It can be called from Colab after run_demo.py finishes.
- It can be called inside MDT report generation later.
- It does not change Nemotron core reasoning.
"""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Any, Dict, Optional


DEFAULT_ARCHITECTURE_FILENAME = "腦瘤架構圖.jpg"


def find_architecture_image(search_roots=None) -> Optional[str]:
    """
    Find architecture JPG in common locations.
    """
    if search_roots is None:
        search_roots = [".", "assets", "outputs", "/content", "/content/pediatric-neuro-oncology-agent"]

    names = [
        DEFAULT_ARCHITECTURE_FILENAME,
        "brain_tumor_architecture.jpg",
        "architecture.jpg",
        "architecture.png",
    ]

    for root in search_roots:
        root = Path(root)
        for name in names:
            p = root / name
            if p.exists():
                return str(p)

    # recursive but bounded
    for root in search_roots:
        root = Path(root)
        if not root.exists():
            continue
        try:
            for p in root.rglob("*"):
                if p.is_file() and p.name in names:
                    return str(p)
        except Exception:
            pass

    return None


def install_architecture_asset(
    input_path: Optional[str] = None,
    asset_dir: str = "assets",
    output_dir: str = "outputs",
    filename: str = DEFAULT_ARCHITECTURE_FILENAME,
) -> Dict[str, Any]:
    """
    Copy the architecture figure into assets/ and outputs/.

    Returns:
      {
        "status": "ok" / "missing" / "error",
        "asset_path": "...",
        "output_path": "...",
        "markdown": "![...](...)"
      }
    """
    try:
        src = input_path or find_architecture_image()
        if not src:
            return {
                "status": "missing",
                "error": "Architecture image not found. Upload 腦瘤架構圖.jpg or architecture.jpg.",
                "asset_path": None,
                "output_path": None,
                "markdown": "",
            }

        src = Path(src)
        Path(asset_dir).mkdir(parents=True, exist_ok=True)
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        asset_path = Path(asset_dir) / filename
        output_path = Path(output_dir) / filename

        # Avoid SameFileError when the source is already assets/ or outputs/.
        if src.resolve() != asset_path.resolve():
            shutil.copyfile(src, asset_path)
        if src.resolve() != output_path.resolve():
            shutil.copyfile(src, output_path)

        return {
            "status": "ok",
            "source_path": str(src),
            "asset_path": str(asset_path),
            "output_path": str(output_path),
            "markdown": architecture_markdown(str(output_path)),
        }

    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "asset_path": None,
            "output_path": None,
            "markdown": "",
        }


def architecture_markdown(image_path: str = f"outputs/{DEFAULT_ARCHITECTURE_FILENAME}") -> str:
    return (
        "## Agent Architecture\n\n"
        "The following figure summarizes the autonomous long-agent workflow: "
        "de-identification guardrail, RAG, Nemotron core reasoning, surgical outputs, "
        "clinical-trial matching, optional imaging analysis, optional drug-ranking appendix, "
        "and final medical safety guardrails.\n\n"
        f"![Pediatric Neuro-Oncology Agent Architecture]({image_path})\n"
    )


def enhanced_report_markdown(
    base_report_text: str = "",
    architecture_image_path: Optional[str] = None,
    drug_result: Optional[Dict[str, Any]] = None,
    imaging_result: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Combine existing MDT report text + architecture figure + optional imaging/drug appendices.
    """
    sections = []

    if base_report_text:
        sections.append(base_report_text.strip())

    if architecture_image_path:
        sections.append(architecture_markdown(architecture_image_path))

    if imaging_result:
        sections.append("## Imaging Module Audit Trail\n")
        sections.append(f"- Status: `{imaging_result.get('status', 'unknown')}`")
        if imaging_result.get("risk_map_path"):
            sections.append(f"- Surgical risk map: `{imaging_result.get('risk_map_path')}`")
        if imaging_result.get("disclaimer"):
            sections.append(f"- Disclaimer: {imaging_result.get('disclaimer')}")

    if drug_result:
        try:
            from drug_ranking_adapter import drug_ranking_to_markdown
            sections.append(drug_ranking_to_markdown(drug_result))
        except Exception:
            sections.append("## Preclinical Drug Ranking\n")
            sections.append(str(drug_result))

    return "\n\n".join(sections).strip() + "\n"


def write_enhanced_report(
    output_path: str = "outputs/enhanced_mdt_report.md",
    base_report_path: Optional[str] = None,
    base_report_text: str = "",
    architecture_image_path: Optional[str] = None,
    drug_result: Optional[Dict[str, Any]] = None,
    imaging_result: Optional[Dict[str, Any]] = None,
) -> str:
    """
    Write a combined report.

    If base_report_path is provided, read it first and append architecture/drug sections.
    """
    if base_report_path and Path(base_report_path).exists():
        base_report_text = Path(base_report_path).read_text(encoding="utf-8", errors="ignore")

    text = enhanced_report_markdown(
        base_report_text=base_report_text,
        architecture_image_path=architecture_image_path,
        drug_result=drug_result,
        imaging_result=imaging_result,
    )

    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    Path(output_path).write_text(text, encoding="utf-8")
    return output_path


def display_architecture(image_path: Optional[str] = None):
    """
    Colab/Jupyter display helper.
    """
    image_path = image_path or find_architecture_image()
    if not image_path:
        print("Architecture image not found.")
        return None

    try:
        from IPython.display import Image, display
        display(Image(filename=image_path))
    except Exception as exc:
        print(f"Could not display image: {exc}")
    return image_path


if __name__ == "__main__":
    result = install_architecture_asset()
    print(result)
