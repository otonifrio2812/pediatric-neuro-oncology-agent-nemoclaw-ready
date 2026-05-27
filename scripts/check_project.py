#!/usr/bin/env python
from pathlib import Path
files = [
    'agent.py','nemotron_client.py','guardrails.py','run_demo.py','watcher.py','image_analysis.py',
    'advanced_medical_imaging.py','drug_ranking_adapter.py','architecture_report_integration.py',
    'literature_trial_updater.py','autonomous_refresh_loop.py','assets/腦瘤架構圖.jpg',
    'sample_cases/case_003_diffuse_midline_glioma.txt','knowledge_base/posterior_fossa_guideline.md'
]
root = Path(__file__).resolve().parents[1]
print('Project root:', root)
for f in files:
    p = root / f
    print(('OK     ' if p.exists() else 'MISSING') + ' ' + f)
