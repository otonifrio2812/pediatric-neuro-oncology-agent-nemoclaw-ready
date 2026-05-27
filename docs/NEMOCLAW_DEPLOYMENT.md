# NemoClaw / NeMo Guardrails Deployment Guide

This project is packaged as a **NemoClaw-ready containerized long agent**. It keeps NVIDIA Nemotron 3 Super as the core reasoning model and exposes the workflow through a small HTTP service that a NemoClaw runner can call.

## What is deployed

- `app_nemoclaw.py`: FastAPI service wrapper.
- `nemoclaw/agent.yaml`: agent/tool manifest for a NemoClaw-style registry.
- `nemoclaw/policies.yaml`: explicit policy-based guardrails.
- `nemo_guardrails/`: optional NeMo Guardrails configuration skeleton.
- `Dockerfile` and `docker-compose.yml`: persistent local/cloud deployment.

## Why this satisfies the hackathon direction

The long-agent pipeline autonomously performs:

1. de-identification,
2. optional image analysis from JPG/PNG or DICOM/NIfTI,
3. RAG retrieval,
4. Nemotron 3 Super clinical reasoning,
5. clinical-trial matching,
6. optional preclinical drug-ranking appendix,
7. policy-based medical guardrails,
8. MDT report generation,
9. autonomous evidence refresh.

The output is decision support for clinicians, not a diagnosis or treatment order.

## Local Docker deployment

```bash
git clone https://github.com/otonifrio2812/pediatric-neuro-oncology-agent.git
cd pediatric-neuro-oncology-agent
cp .env.nemoclaw.example .env
# Edit .env and set NVIDIA_API_KEY

docker compose up --build -d
curl http://localhost:8080/health
```

Run one sample case:

```bash
curl -X POST http://localhost:8080/run_case \
  -H "Content-Type: application/json" \
  -d @examples/run_case_payload.json
```

If you do not have `examples/run_case_payload.json`, use:

```bash
cat > /tmp/run_case_payload.json <<'JSON'
{
  "raw_case_text": "8-year-old child with progressive gait instability and vomiting. MRI describes an expansile pontine lesion with concern for diffuse midline glioma. H3 K27-altered pathology is suspected. No patient identifiers included.",
  "structured": {
    "age": 8,
    "tumor_type": "diffuse midline glioma",
    "tumor_location": "pons",
    "pathology": "possible H3 K27-altered diffuse midline glioma",
    "molecular_markers": {"H3K27M": "suspected"},
    "symptoms": ["gait instability", "vomiting"]
  },
  "attach_architecture": true,
  "enable_drug_ranking": false,
  "refresh_evidence_before_run": false
}
JSON

curl -X POST http://localhost:8080/run_case \
  -H "Content-Type: application/json" \
  -d @/tmp/run_case_payload.json
```

Outputs are persisted in:

- `outputs/`
- `logs/`
- `rag_sources/`

## Persistent evidence refresh

```bash
docker compose exec pediatric-neuro-agent \
  python autonomous_refresh_loop.py --once --pubmed-days 30 --pubmed-retmax 30
```

For a long-running VM:

```bash
docker compose exec -d pediatric-neuro-agent \
  python autonomous_refresh_loop.py --interval-hours 6 --pubmed-days 30 --pubmed-retmax 50
```

## NemoClaw registration pattern

If your NemoClaw runtime accepts a manifest, register:

```yaml
manifest: nemoclaw/agent.yaml
policies: nemoclaw/policies.yaml
service_url: http://pediatric-neuro-agent:8080
health: /health
tools:
  - /run_case
  - /refresh_once
```

The minimal tool call is `POST /run_case`. The policy layer should enforce:

- de-identification before model calls,
- no definitive diagnosis,
- no treatment orders,
- imaging is auxiliary/non-diagnostic,
- drug ranking is research appendix only,
- clinician-review disclaimer,
- high-risk anatomy flags,
- auditable logs.

## Optional NeMo Guardrails layer

`nemo_guardrails/` is a starter configuration. Keep `guardrails.py` enabled regardless; it is the project-specific medical safety backstop.
