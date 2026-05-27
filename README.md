# Pediatric Neuro-Oncology Surgical Planning Agent

A runnable hackathon project for an autonomous long-agent workflow for pediatric neuro-oncology pre-operative MDT planning.

## Core design

Nemotron 3 Super is the core clinical reasoning model. Other modules are upstream tools:

- `image_analysis.py`: optional JPG/PNG image-to-text using NVIDIA OpenAI-compatible VLM endpoint.
- `advanced_medical_imaging.py`: optional DICOM/NIfTI research-prototype advanced imaging module.
- `drug_ranking_adapter.py`: optional preclinical drug-ranking appendix adapter.
- `literature_trial_updater.py`: autonomous PubMed / ClinicalTrials.gov evidence refresh.
- `rag.py`: retrieval-augmented generation (RAG) layer; retrieves evidence chunks from `knowledge_base/` and `rag_sources/` for Nemotron reasoning and the MDT report.
- `watcher.py` / `autonomous_refresh_loop.py`: autonomous watcher that refreshes the knowledge base and evidence, then auto-triggers an agent run and report generation.
- `guardrails.py`: policy-based medical safety checks.

## Retrieval-Augmented Generation (RAG)

The project includes a RAG retrieval layer (`rag.py`). On each case run it retrieves the most relevant evidence chunks from the local `knowledge_base/` (curated guideline / abstract summaries) and `rag_sources/` (autonomously refreshed evidence), and supplies them to **Nemotron 3 Super** as grounding context for clinical reasoning. The retrieved chunks are cited in the generated **MDT report** (see the "Retrieved Evidence" section of each report) so every evidence item is traceable. Retrieval uses TF-IDF when scikit-learn is available and falls back to a lightweight keyword scorer otherwise, so it runs with or without optional dependencies.

## Safety scope

This repository is a research prototype for hackathon demonstration. It does not provide a definitive diagnosis, prescription, or operative plan. All outputs require independent review by radiology, neurosurgery, pediatric neuro-oncology, pharmacy, and the MDT.

## Quick start in Colab

Upload this zip in Step 1 of the notebook. The notebook looks for a file matching:

```python
/content/*pediatric-neuro-oncology-agent*.zip
```

Then it extracts to:

```text
/content/pediatric-neuro-oncology-agent
```

## Local quick start

```bash
pip install -r requirements.txt
python run_demo.py sample_cases/case_003_diffuse_midline_glioma.txt
```

Without `NVIDIA_API_KEY`, the project runs in MOCK mode. With an API key:

```bash
export NVIDIA_API_KEY="nvapi-..."
export NEMOTRON_MODEL="nvidia/nemotron-3-super-120b-a12b"
python run_demo.py sample_cases/case_003_diffuse_midline_glioma.txt
```

## DICOM / NIfTI advanced imaging

```bash
python run_demo.py sample_cases/case_003_diffuse_midline_glioma.txt --medical-study medical_inputs/my_study_or_nii
```

Or directly:

```bash
python advanced_medical_imaging.py medical_inputs/my_study_or_nii --output-dir outputs
```

The segmentation and anatomic landmarks are heuristic placeholders, not validated clinical models. Replace with MONAI/nnUNet/atlas registration before any formal research use.

## Autonomous watcher & refresh loop

The project ships an autonomous watcher component (`watcher.py`) plus a scheduler/driver (`autonomous_refresh_loop.py`). Together they perform knowledge-base and evidence refresh (PubMed / ClinicalTrials.gov) and then **automatically trigger an agent run and MDT report generation** — demonstrating end-to-end autonomous long-agent behavior with no human in the loop.

```bash
python autonomous_refresh_loop.py --once
# or persistent
python autonomous_refresh_loop.py --interval-hours 6
```

This refreshes PubMed / ClinicalTrials.gov evidence sources (updating `rag_sources/`, which the RAG layer retrieves from) and then runs the agent watcher, which picks up new inputs and generates reports automatically.

## Hackathon fit

The workflow supports autonomous operation, Nemotron core reasoning, real tasks (retrieval, automation, analysis, orchestration, reporting), deployability in Colab/local/cloud, and policy-based guardrails.
