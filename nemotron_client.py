# -*- coding: utf-8 -*-
"""NVIDIA Nemotron client.

Core requirement for the hackathon:
- Nemotron is the core clinical reasoning model.
- This client uses NVIDIA's OpenAI-compatible endpoint when NVIDIA_API_KEY is set.
- Without an API key, it falls back to deterministic MOCK mode so the project remains runnable.
"""
from __future__ import annotations

import os
from dataclasses import dataclass

try:
    from openai import OpenAI
except Exception:  # pragma: no cover
    OpenAI = None


DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NEMOTRON_MODEL = "nvidia/nemotron-3-super-120b-a12b"


@dataclass
class NemotronClient:
    model: str | None = None
    api_key: str | None = None
    base_url: str | None = None
    timeout: float | None = None

    def __post_init__(self) -> None:
        self.model = self.model or os.getenv("NEMOTRON_MODEL", DEFAULT_NEMOTRON_MODEL)
        self.api_key = self.api_key or os.getenv("NVIDIA_API_KEY")
        self.base_url = self.base_url or os.getenv("NVIDIA_BASE_URL", DEFAULT_BASE_URL)
        self.timeout = float(self.timeout or os.getenv("NEMOTRON_TIMEOUT", "60"))
        self.mock = not bool(self.api_key) or OpenAI is None
        self.client = None
        if not self.mock:
            self.client = OpenAI(api_key=self.api_key, base_url=self.base_url, timeout=self.timeout)

    @property
    def mode(self) -> str:
        if self.mock:
            return f"MOCK Nemotron ({self.model})"
        return f"Nemotron live ({self.model})"

    def complete(self, system: str, user: str) -> str:
        """Return plain-text completion. Keeps compatibility with existing project interface."""
        if self.mock:
            return self._mock_complete(system=system, user=user)

        resp = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            temperature=float(os.getenv("NEMOTRON_TEMPERATURE", "0.2")),
            max_tokens=int(os.getenv("NEMOTRON_MAX_TOKENS", "1800")),
        )
        return resp.choices[0].message.content or ""

    def _mock_complete(self, system: str, user: str) -> str:
        """Deterministic fallback for Colab/demo without NVIDIA_API_KEY."""
        lower = user.lower()
        flags = []
        if any(k in lower for k in ["pons", "brainstem", "basilar", "fourth ventricle"]):
            flags.append("high-risk brainstem/posterior fossa anatomy")
        if any(k in lower for k in ["hydrocephalus", "ventriculomegaly"]):
            flags.append("possible CSF obstruction/hydrocephalus")
        if any(k in lower for k in ["h3 k27", "h3k27", "diffuse midline"]):
            flags.append("molecular/pathology concern for diffuse midline glioma pathway")
        if not flags:
            flags.append("requires multidisciplinary review")

        return (
            "## Nemotron Clinical Reasoning (MOCK fallback)\n"
            "Nemotron API key was not available, so this deterministic mock response was generated. "
            "In live mode, the same prompt is sent to NVIDIA Nemotron 3 Super.\n\n"
            "### Key risk signals\n"
            + "\n".join(f"- {x}" for x in flags)
            + "\n\n### Planning stance\n"
            "- Treat this as decision support for MDT discussion, not as a definitive diagnosis.\n"
            "- Prioritize radiology/neurosurgery review of lesion location, vascular proximity, cranial nerve risk, and CSF obstruction.\n"
            "- Discuss biopsy vs resection strategy based on tumor location, neurologic deficits, molecular testing plan, and safe surgical corridor.\n"
            "- Use RAG evidence and trial matching as preliminary prompts for clinician verification.\n"
        )
