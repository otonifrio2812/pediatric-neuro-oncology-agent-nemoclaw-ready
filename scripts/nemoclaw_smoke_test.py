#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""Local import smoke test for NemoClaw deployment files."""
from pathlib import Path

required = [
    "app_nemoclaw.py",
    "Dockerfile",
    "docker-compose.yml",
    "nemoclaw/agent.yaml",
    "nemoclaw/policies.yaml",
    "nemo_guardrails/config.yml",
    "nemo_guardrails/rails.co",
    "examples/run_case_payload.json",
]
missing = [p for p in required if not Path(p).exists()]
if missing:
    raise SystemExit("Missing files: " + ", ".join(missing))

# Syntax-level check without starting the server.
import py_compile
for p in ["app_nemoclaw.py"]:
    py_compile.compile(p, doraise=True)
print("OK NemoClaw deployment files are present and Python syntax is valid.")
