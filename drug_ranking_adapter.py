# -*- coding: utf-8 -*-
"""
drug_ranking_adapter.py

Adapter layer to plug `06_drug_ranking_widget.ipynb` into the
Pediatric Neuro-Oncology Surgical Planning Agent.

It keeps the drug system optional:
- If artifacts are available, it produces preclinical drug sensitivity rankings.
- If artifacts / GitHub / intermediates are unavailable, it returns a safe fallback.
- Output is explicitly NOT a treatment recommendation.

Expected external repo from the original widget notebook:
    https://github.com/otonifrio2812/pediatric-bt-drug-prediction
Expected API in that repo:
    from src.drug_ranking import load_artifacts, list_cells_by_cancer_type, predict_drug_ranking
"""

from __future__ import annotations

import os
import sys
import json
import subprocess
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


DRUG_RANKING_DISCLAIMER = (
    "AI preclinical drug-ranking module for research triage only. "
    "It is not a treatment recommendation, prescription, or substitute for "
    "pediatric neuro-oncology / pharmacy / molecular tumor board review."
)

DEFAULT_GITHUB_USER = "otonifrio2812"
DEFAULT_REPO_NAME = "pediatric-bt-drug-prediction"
DEFAULT_RELEASE_TAG = "v1.0.1"


def _run(cmd: List[str], cwd: Optional[str] = None) -> None:
    """Run a shell command safely from Python."""
    print(" ".join(cmd))
    subprocess.check_call(cmd, cwd=cwd)


def setup_drug_ranking_repo(
    workdir: str = "/content",
    github_user: str = DEFAULT_GITHUB_USER,
    repo_name: str = DEFAULT_REPO_NAME,
    release_tag: str = DEFAULT_RELEASE_TAG,
    install_requirements: bool = True,
) -> str:
    """
    Colab helper:
    - clone pediatric-bt-drug-prediction
    - install its requirements
    - download intermediates.zip from GitHub Release

    Returns repo_dir.

    NOTE:
    The original widget notebook used NumPy<2 for pickle compatibility.
    If Colab already loaded NumPy 2.x, restart runtime after downgrading.
    """
    workdir_path = Path(workdir)
    repo_dir = workdir_path / repo_name
    workdir_path.mkdir(parents=True, exist_ok=True)

    if not repo_dir.exists():
        _run([
            "git", "clone", "-q",
            f"https://github.com/{github_user}/{repo_name}.git",
            str(repo_dir),
        ])

    if install_requirements and (repo_dir / "requirements.txt").exists():
        _run([sys.executable, "-m", "pip", "install", "-q", "-r", "requirements.txt"], cwd=str(repo_dir))

    intermediates_dir = repo_dir / "intermediates"
    pkl_path = intermediates_dir / "stage6_ensemble_models.pkl"
    if not pkl_path.exists():
        intermediates_dir.mkdir(parents=True, exist_ok=True)
        release_url = (
            f"https://github.com/{github_user}/{repo_name}/"
            f"releases/download/{release_tag}/intermediates.zip"
        )
        zip_path = repo_dir / "intermediates.zip"
        _run(["wget", "-q", release_url, "-O", str(zip_path)])
        _run(["unzip", "-q", str(zip_path), "-d", str(intermediates_dir)])

    if str(repo_dir) not in sys.path:
        sys.path.insert(0, str(repo_dir))

    return str(repo_dir)


def ensure_numpy_pickle_compatibility() -> Dict[str, Any]:
    """
    Helper for Colab before loading the external pickle artifacts.
    Returns a dict; if needs_restart=True, restart runtime then run cells again.
    """
    import numpy as np

    version = np.__version__
    out = {
        "numpy_version": version,
        "needs_restart": False,
        "message": f"numpy={version} is compatible.",
    }

    if version.startswith("2"):
        subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "numpy<2.0", "scipy<1.13"])
        out["needs_restart"] = True
        out["message"] = (
            f"Current numpy={version}. Downgraded to numpy<2.0. "
            "Please restart runtime/session, then rerun the setup cell."
        )
    return out


def _import_drug_api(repo_dir: Optional[str] = None):
    """Import the external repo's drug-ranking API."""
    if repo_dir and repo_dir not in sys.path:
        sys.path.insert(0, repo_dir)
    try:
        from src.drug_ranking import load_artifacts, list_cells_by_cancer_type, predict_drug_ranking
        return load_artifacts, list_cells_by_cancer_type, predict_drug_ranking
    except Exception as exc:
        raise ImportError(
            "Could not import src.drug_ranking. Run setup_drug_ranking_repo() first "
            "or execute the original 06_drug_ranking_widget.ipynb setup cells."
        ) from exc


def load_drug_ranking_artifacts(
    repo_dir: Optional[str] = None,
    intermediates_dir: str = "intermediates/",
):
    """Load external drug-ranking artifacts."""
    load_artifacts, _, _ = _import_drug_api(repo_dir)
    return load_artifacts(intermediates_dir=intermediates_dir)


def list_available_cancer_types(artifacts, repo_dir: Optional[str] = None) -> Dict[str, List[Tuple[str, str]]]:
    """Return {cancer_type: [(cell_id, cell_name), ...]}."""
    _, list_cells_by_cancer_type, _ = _import_drug_api(repo_dir)
    return list_cells_by_cancer_type(artifacts)


def infer_cancer_type_from_case(
    structured_case: Dict[str, Any],
    available_types: List[str],
) -> str:
    """
    Heuristic mapping from clinical case text to the drug-ranking repo's cancer_type.

    The drug-ranking model is cell-line based, not patient-specific.
    We therefore choose the closest available cancer type as a surrogate.
    """
    text = " ".join([
        str(structured_case.get("tumor_type", "")),
        str(structured_case.get("pathology", "")),
        str(structured_case.get("diagnosis", "")),
        str(structured_case.get("tumor_location", "")),
        str(structured_case.get("imaging_description", "")),
    ]).lower()

    available_lc = {x.lower(): x for x in available_types}

    def choose(candidates: List[str]) -> Optional[str]:
        for c in candidates:
            if c.lower() in available_lc:
                return available_lc[c.lower()]
        return None

    if any(k in text for k in ["neuroblastoma"]):
        picked = choose(["Neuroblastoma"])
        if picked:
            return picked

    if any(k in text for k in ["glioblastoma", "gbm"]):
        picked = choose(["Glioblastoma", "Glioma"])
        if picked:
            return picked

    if any(k in text for k in ["diffuse midline", "dmg", "dipg", "glioma", "astrocytoma"]):
        picked = choose(["Glioma", "Glioblastoma"])
        if picked:
            return picked

    if "medulloblastoma" in text:
        picked = choose(["Medulloblastoma", "Glioma"])
        if picked:
            return picked

    if "ependymoma" in text:
        picked = choose(["Ependymoma", "Glioma"])
        if picked:
            return picked

    # Safe default for CNS tumor demo.
    picked = choose(["Glioblastoma", "Glioma", "Neuroblastoma"])
    if picked:
        return picked

    return sorted(available_types)[0]


def choose_surrogate_cell_line(
    structured_case: Dict[str, Any],
    artifacts,
    cells_by_type: Dict[str, List[Tuple[str, str]]],
    cancer_type: str,
) -> Tuple[Optional[str], Optional[str], List[str]]:
    """
    Select cell line:
    1) use structured_case["cell_line_id"] if provided and valid
    2) otherwise choose the first cell line under inferred cancer_type
    """
    warnings: List[str] = []
    requested = structured_case.get("cell_line_id") or structured_case.get("drug_cell_line_id")
    lookup = getattr(artifacts, "cell_metadata_lookup", {})

    if requested and requested in lookup:
        name = lookup[requested].get("CELL_LINE_NAME", requested)
        return requested, name, warnings

    if requested and requested not in lookup:
        warnings.append(f"Requested cell_line_id={requested} not found; using surrogate cell line.")

    cells = cells_by_type.get(cancer_type, [])
    if not cells:
        warnings.append(f"No cell lines available for cancer_type={cancer_type}.")
        return None, None, warnings

    cell_id, cell_name = cells[0]
    warnings.append(
        f"No patient-specific cell line was provided. Using surrogate cell line {cell_id} ({cell_name}) "
        f"for cancer_type={cancer_type}."
    )
    return cell_id, cell_name, warnings


def _safe_float(x: Any) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def _records_from_ranking_df(ranking_df, top_k: int) -> List[Dict[str, Any]]:
    rows = []
    for _, r in ranking_df.head(top_k).iterrows():
        rows.append({
            "drug_name": str(r.get("drug_name", "")),
            "P_sens": _safe_float(r.get("P_sens")),
            "CI_lo": _safe_float(r.get("CI_lo")),
            "CI_hi": _safe_float(r.get("CI_hi")),
            "target": str(r.get("target", "")),
            "pathway": str(r.get("pathway", "")),
        })
    return rows


def rank_drugs_for_case(
    structured_case: Dict[str, Any],
    artifacts=None,
    repo_dir: Optional[str] = None,
    top_k: int = 10,
    with_ci: bool = True,
) -> Dict[str, Any]:
    """
    Main integration entry point.

    Returns a dict that can be inserted into case["drug_ranking_result"].
    Gracefully degrades if the external model/artifacts are unavailable.
    """
    structured_case = structured_case or {}

    if artifacts is None:
        try:
            artifacts = load_drug_ranking_artifacts(repo_dir=repo_dir)
        except Exception as exc:
            return {
                "status": "unavailable",
                "error": str(exc),
                "selected_cancer_type": None,
                "selected_cell_line_id": None,
                "selected_cell_line_name": None,
                "top_drugs": [],
                "warnings": ["Drug-ranking artifacts are unavailable; skipped optional drug-ranking module."],
                "disclaimer": DRUG_RANKING_DISCLAIMER,
            }

    try:
        _, list_cells_by_cancer_type, predict_drug_ranking = _import_drug_api(repo_dir)
        cells_by_type = list_cells_by_cancer_type(artifacts)
        cancer_type = structured_case.get("drug_cancer_type") or infer_cancer_type_from_case(
            structured_case, list(cells_by_type.keys())
        )
        cell_id, cell_name, warnings = choose_surrogate_cell_line(
            structured_case, artifacts, cells_by_type, cancer_type
        )

        if not cell_id:
            return {
                "status": "unavailable",
                "error": "No valid cell line could be selected.",
                "selected_cancer_type": cancer_type,
                "selected_cell_line_id": None,
                "selected_cell_line_name": None,
                "top_drugs": [],
                "warnings": warnings,
                "disclaimer": DRUG_RANKING_DISCLAIMER,
            }

        ranking = predict_drug_ranking(cell_id, artifacts, top_k=top_k, with_ci=with_ci)
        top_drugs = _records_from_ranking_df(ranking, top_k=top_k)

        return {
            "status": "ok",
            "selected_cancer_type": cancer_type,
            "selected_cell_line_id": cell_id,
            "selected_cell_line_name": cell_name,
            "top_drugs": top_drugs,
            "warnings": warnings,
            "disclaimer": DRUG_RANKING_DISCLAIMER,
            "source": "pediatric-bt-drug-prediction / 06_drug_ranking_widget adapter",
        }

    except Exception as exc:
        return {
            "status": "error",
            "error": str(exc),
            "selected_cancer_type": None,
            "selected_cell_line_id": None,
            "selected_cell_line_name": None,
            "top_drugs": [],
            "warnings": ["Drug-ranking module failed; continuing without drug-ranking output."],
            "disclaimer": DRUG_RANKING_DISCLAIMER,
        }


def drug_ranking_to_markdown(result: Dict[str, Any], title: str = "Preclinical Drug Ranking") -> str:
    """Convert drug-ranking result into a safe MDT-report markdown section."""
    result = result or {}
    lines = [f"## {title}", ""]
    lines.append(f"**Safety note:** {result.get('disclaimer', DRUG_RANKING_DISCLAIMER)}")
    lines.append("")

    status = result.get("status", "unknown")
    lines.append(f"- Status: `{status}`")
    if status != "ok":
        lines.append(f"- Reason: {result.get('error', 'not available')}")
        for w in result.get("warnings", []):
            lines.append(f"- Warning: {w}")
        return "\n".join(lines)

    lines.extend([
        f"- Selected cancer type: `{result.get('selected_cancer_type')}`",
        f"- Surrogate cell line: `{result.get('selected_cell_line_id')}` "
        f"({result.get('selected_cell_line_name')})",
    ])
    for w in result.get("warnings", []):
        lines.append(f"- Warning: {w}")

    lines.append("")
    lines.append("| Rank | Drug | P_sens | 95% CI | Target | Pathway |")
    lines.append("|---:|---|---:|---|---|---|")
    for i, d in enumerate(result.get("top_drugs", []), start=1):
        ps = d.get("P_sens")
        lo = d.get("CI_lo")
        hi = d.get("CI_hi")
        ps_s = "" if ps is None else f"{ps:.3f}"
        ci_s = "" if lo is None or hi is None else f"[{lo:.3f}, {hi:.3f}]"
        lines.append(
            f"| {i} | {d.get('drug_name','')} | {ps_s} | {ci_s} | "
            f"{d.get('target','')} | {d.get('pathway','')} |"
        )

    lines.append("")
    lines.append(
        "> This table is intended for hypothesis generation / trial discussion only. "
        "It must pass the existing medical guardrails and MDT review before being shown as a clinical-facing appendix."
    )
    return "\n".join(lines)


def launch_drug_ranking_widget(artifacts, repo_dir: Optional[str] = None):
    """
    Interactive Colab/Jupyter widget copied from 06_drug_ranking_widget.ipynb,
    wrapped as a reusable function.
    """
    _, list_cells_by_cancer_type, predict_drug_ranking = _import_drug_api(repo_dir)
    cells_by_type = list_cells_by_cancer_type(artifacts)

    import ipywidgets as widgets
    from IPython.display import clear_output, display

    cancer_dropdown = widgets.Dropdown(
        options=sorted(cells_by_type.keys()),
        value="Glioblastoma" if "Glioblastoma" in cells_by_type else sorted(cells_by_type.keys())[0],
        description="Cancer:",
    )

    initial_cells = cells_by_type[cancer_dropdown.value]
    cell_dropdown = widgets.Dropdown(
        options=[f"{cid} ({name})" for cid, name in initial_cells],
        description="Cell:",
    )

    top_k_slider = widgets.IntSlider(value=10, min=5, max=30, step=5, description="Top K:")
    output = widgets.Output()

    def on_cancer_change(change):
        ctype = change["new"]
        cell_dropdown.options = [f"{cid} ({name})" for cid, name in cells_by_type[ctype]]

    def update_ranking(*args):
        cell_id = cell_dropdown.value.split(" ")[0]
        top_k = top_k_slider.value
        with output:
            clear_output()
            try:
                ranking = predict_drug_ranking(cell_id, artifacts, top_k=top_k, with_ci=True)
                meta = artifacts.cell_metadata_lookup[cell_id]
                print("=" * 70)
                print(f"Cell: {cell_id} ({meta['CELL_LINE_NAME']})")
                print(f"Cancer type: {meta['CANCER_TYPE']}")
                print("=" * 70)
                display_df = ranking.copy()
                display_df["P_sens"] = display_df["P_sens"].apply(lambda x: f"{x:.3f}")
                display_df["CI"] = display_df.apply(
                    lambda r: f"[{r['CI_lo']:.3f}, {r['CI_hi']:.3f}]", axis=1
                )
                display_df["target"] = display_df["target"].apply(
                    lambda t: str(t)[:25] + ".." if len(str(t)) > 27 else str(t)
                )
                display_df["pathway"] = display_df["pathway"].apply(
                    lambda p: str(p)[:25] + ".." if len(str(p)) > 27 else str(p)
                )
                display(display_df[["drug_name", "P_sens", "CI", "target", "pathway"]])
                print("\nSafety note:", DRUG_RANKING_DISCLAIMER)
            except Exception as exc:
                print(f"Error: {exc}")

    cancer_dropdown.observe(on_cancer_change, names="value")
    cancer_dropdown.observe(update_ranking, names="value")
    cell_dropdown.observe(update_ranking, names="value")
    top_k_slider.observe(update_ranking, names="value")

    ui = widgets.VBox([cancer_dropdown, cell_dropdown, top_k_slider, output])
    display(ui)
    update_ranking()
    return ui


def attach_drug_ranking_to_case(
    structured_case: Dict[str, Any],
    artifacts=None,
    repo_dir: Optional[str] = None,
    top_k: int = 10,
) -> Dict[str, Any]:
    """Minimal helper for agent.py integration."""
    case = dict(structured_case or {})
    result = rank_drugs_for_case(case, artifacts=artifacts, repo_dir=repo_dir, top_k=top_k)
    case["drug_ranking_result"] = result
    case["drug_ranking_report_section"] = drug_ranking_to_markdown(result)
    return case


if __name__ == "__main__":
    # Demo fallback: will print unavailable unless artifacts are already installed.
    case = {
        "tumor_type": "diffuse midline glioma",
        "pathology": "H3 K27-altered diffuse midline glioma",
        "tumor_location": "pons",
    }
    print(json.dumps(rank_drugs_for_case(case, top_k=5), ensure_ascii=False, indent=2))
