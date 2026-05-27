"""
image_analysis.py

Optional medical image-to-text module for the Pediatric Neuro-Oncology
Surgical Planning Agent.

Design notes / 設計重點:
1. Nemotron 3 Super remains the text-only core reasoning model.
2. This module uses a vision-language model (VLM) only to convert an image
   into a cautious, non-diagnostic structured description.
3. The resulting text can be inserted into case["imaging_description"] before
   RAG retrieval and Nemotron clinical reasoning.

NVIDIA API notes:
- Hosted OpenAI-compatible endpoint:
    base_url = "https://integrate.api.nvidia.com/v1"
- Default VLM model string:
    "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
- Alternative lighter VLM:
    "nvidia/nemotron-nano-12b-v2-vl"
- Authentication uses the same NVIDIA_API_KEY used by NemotronClient.
  Set NVIDIA_VLM_MODEL to switch model without code changes.

Medical safety:
- This module MUST NOT produce a definitive diagnosis, tumor grade,
  histology, molecular subtype, segmentation, or automated measurement.
- Output is an auxiliary observation only and must be reviewed by clinicians.
"""

from __future__ import annotations

import base64
import io
import json
import logging
import os
import re
import warnings
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple, Union

try:
    from PIL import Image, ImageOps, UnidentifiedImageError
except Exception:  # pragma: no cover - handled at runtime
    Image = None
    ImageOps = None
    UnidentifiedImageError = Exception

try:
    from openai import OpenAI
except Exception:  # pragma: no cover - handled at runtime
    OpenAI = None


LOGGER = logging.getLogger(__name__)

DEFAULT_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_VLM_MODEL = "nvidia/nemotron-3-nano-omni-30b-a3b-reasoning"
FALLBACK_VLM_MODEL = "nvidia/nemotron-nano-12b-v2-vl"

DISCLAIMER = (
    "AI 影像輔助觀察，最終判讀須由放射科/神經外科醫師負責；"
    "本描述不得作為確定性診斷或治療決策的唯一依據。"
)

PRIVACY_NOTE = (
    "送出影像到雲端 VLM 前，請確認已去識別化：DICOM metadata、"
    "影像邊框燒錄姓名/病歷號/日期等個資需移除。此模組會重新編碼影像以移除一般檔案 metadata，"
    "但無法保證移除燒錄在像素內的文字。"
)

SUPPORTED_RASTER_EXTENSIONS = {".png", ".jpg", ".jpeg", ".webp"}
SUPPORTED_DICOM_EXTENSIONS = {".dcm", ".dicom"}

ALLOWED_MODALITIES = {"X-ray", "CT", "MRI", "PET", "Unknown"}

# Terms that should not be emitted by the image module as a finding.
# 影像模組只保留非診斷性觀察，不輸出病理/腫瘤類型推論。
DIAGNOSTIC_TERMS = {
    "medulloblastoma",
    "ependymoma",
    "glioblastoma",
    "glioma",
    "astrocytoma",
    "oligodendroglioma",
    "lymphoma",
    "metastasis",
    "metastatic",
    "meningioma",
    "craniopharyngioma",
    "diffuse midline glioma",
    "dipg",
    "at/rt",
    "atypical teratoid",
    "rhabdoid",
    "who grade",
    "molecular subtype",
    "h3k27",
    "braf",
    "idh",
}

BANNED_FINDING_KEYS = {
    "diagnosis",
    "diagnoses",
    "differential",
    "differential_diagnosis",
    "tumor_type",
    "histology",
    "pathology",
    "grade",
    "molecular",
    "molecular_markers",
    "treatment",
    "management",
    "surgical_plan",
    "recommendation",
}

MAX_IMAGE_SIDE = int(os.getenv("IMAGE_ANALYSIS_MAX_SIDE", "1024"))

# NVIDIA serverless docs recommend asset upload for larger media payloads.
# For a hackathon demo, inline base64 is simpler; keep it compact.
MAX_INLINE_IMAGE_BYTES = int(os.getenv("NVIDIA_MAX_INLINE_IMAGE_BYTES", "180000"))


def describe_image(image_path: Union[str, Path]) -> Dict[str, Any]:
    """Describe a brain image using an optional NVIDIA-hosted VLM.

    Args:
        image_path: PNG/JPG/JPEG/WEBP image path, or optional DICOM path.

    Returns:
        dict with at least:
        - modality: one of X-ray/CT/MRI/PET/Unknown
        - findings: structured non-diagnostic observations
        - disclaimer: required medical safety disclaimer

    Graceful degradation:
    - If NVIDIA_API_KEY is absent, returns MOCK placeholder.
    - If image loading or API call fails, returns an error fallback.
    - The caller should keep the physician-provided imaging_description
      whenever status != "ok".
    """
    path = Path(image_path)

    if not path.exists():
        return _fallback_result(
            status="error",
            path=path,
            reason=f"Image file not found: {path}",
            source="file_check",
        )

    api_key = os.getenv("NVIDIA_API_KEY", "").strip()
    if _is_missing_or_placeholder_key(api_key):
        return _fallback_result(
            status="mock",
            path=path,
            reason="NVIDIA_API_KEY is not set; returning demonstration placeholder.",
            source="mock_no_api_key",
        )

    if OpenAI is None:
        return _fallback_result(
            status="error",
            path=path,
            reason="openai package is not installed. Add openai to requirements.txt.",
            source="dependency_error",
        )

    try:
        warnings.warn(PRIVACY_NOTE, RuntimeWarning, stacklevel=2)
        data_url, local_modality_hint = _image_to_data_url(path)

        raw_text = _call_nvidia_vlm(
            data_url=data_url,
            local_modality_hint=local_modality_hint,
            api_key=api_key,
        )
        parsed = _extract_json_object(raw_text)
        return _normalize_vlm_payload(
            parsed=parsed,
            raw_text=raw_text,
            path=path,
            local_modality_hint=local_modality_hint,
        )
    except Exception as exc:
        LOGGER.exception("Image analysis failed; falling back to manual imaging_description.")
        return _fallback_result(
            status="error",
            path=path,
            reason=str(exc),
            source="exception_fallback",
        )


def image_description_to_text(result: Dict[str, Any]) -> str:
    """Convert describe_image() output into safe text for case['imaging_description'].

    This helper is intentionally simple so agent.py can add the image module
    with minimal changes.
    """
    modality = result.get("modality", "Unknown")
    findings = result.get("findings") or {}
    observations = _as_list(findings.get("observations"))
    limitations = _as_list(findings.get("limitations"))
    uncertainty = _as_list(findings.get("uncertain_or_not_assessable"))

    lines = [
        result.get("disclaimer", DISCLAIMER),
        f"影像來源: AI-assisted image-to-text module; status={result.get('status', 'unknown')}",
        f"推定影像類型 modality: {modality}",
        "非診斷性影像觀察 findings:",
    ]

    if observations:
        lines.extend([f"- {obs}" for obs in observations])
    else:
        lines.append("- 未能產生可靠的非診斷性觀察；請以人工影像判讀為準。")

    if uncertainty:
        lines.append("不確定或不可評估項目:")
        lines.extend([f"- {item}" for item in uncertainty])

    if limitations:
        lines.append("限制 limitations:")
        lines.extend([f"- {item}" for item in limitations])

    lines.append("注意: 未執行腫瘤分割、自動量測、病理分類或確定性診斷。")
    return "\n".join(lines)


def _call_nvidia_vlm(
    *,
    data_url: str,
    local_modality_hint: str,
    api_key: str,
) -> str:
    model = os.getenv("NVIDIA_VLM_MODEL", DEFAULT_VLM_MODEL).strip() or DEFAULT_VLM_MODEL
    base_url = os.getenv("NVIDIA_BASE_URL", DEFAULT_BASE_URL).strip() or DEFAULT_BASE_URL
    timeout = float(os.getenv("NVIDIA_VLM_TIMEOUT", "60"))

    # Same NVIDIA_API_KEY, OpenAI-compatible client.
    # 同一把 NVIDIA_API_KEY，同一個 base_url，只是 model 換成能讀圖的 VLM。
    client = OpenAI(api_key=api_key, base_url=base_url, timeout=timeout)

    system_prompt = (
        "You are a cautious radiology image-to-text assistant for pediatric "
        "neuro-oncology preoperative planning. You only describe visible image "
        "features. You do NOT diagnose tumor type, grade, histology, molecular "
        "subtype, prognosis, or treatment. You do NOT perform segmentation or "
        "automatic measurement. Return ONLY valid compact JSON."
    )

    user_prompt = f"""
Analyze the uploaded brain image or image screenshot.

Local file-derived modality hint, if any: {local_modality_hint}

Return ONLY this JSON schema:
{{
  "modality": "X-ray | CT | MRI | PET | Unknown",
  "findings": {{
    "observations": [
      "Cautious visible observation without definitive diagnosis"
    ],
    "uncertain_or_not_assessable": [
      "Items not assessable from a single image or screenshot"
    ]
  }},
  "limitations": [
    "Single uploaded image/screenshot may not represent the complete study"
  ],
  "safety_flags": [
    "No definitive diagnosis",
    "No segmentation",
    "No automated measurement"
  ]
}}

Rules:
- Use hedging language: "appears", "possible", "not assessable".
- Do not infer histology such as glioma, medulloblastoma, ependymoma, etc.
- Do not state benign/malignant, WHO grade, molecular subtype, or treatment plan.
- If the image is not a medical brain image, set modality to "Unknown" and explain.
""".strip()

    messages = [
        {"role": "system", "content": system_prompt},
        {
            "role": "user",
            "content": [
                {"type": "text", "text": user_prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ],
        },
    ]

    kwargs = dict(
        model=model,
        messages=messages,
        temperature=0,
        max_tokens=900,
        stream=False,
    )

    try:
        response = client.chat.completions.create(
            **kwargs,
            extra_body={"chat_template_kwargs": {"enable_thinking": False}},
        )
    except Exception:
        # Some hosted models may reject model-specific extra_body fields.
        # Retry once with strict OpenAI-compatible fields only.
        response = client.chat.completions.create(**kwargs)

    content = response.choices[0].message.content
    if not content:
        raise RuntimeError("VLM returned an empty response.")
    return str(content)


def _image_to_data_url(path: Path) -> Tuple[str, str]:
    if _looks_like_dicom(path):
        jpeg_bytes, modality_hint = _dicom_to_jpeg_bytes(path)
    else:
        jpeg_bytes, modality_hint = _raster_to_jpeg_bytes(path)

    b64 = base64.b64encode(jpeg_bytes).decode("utf-8")
    return f"data:image/jpeg;base64,{b64}", modality_hint


def _raster_to_jpeg_bytes(path: Path) -> Tuple[bytes, str]:
    if Image is None:
        raise RuntimeError("pillow is not installed. Add pillow to requirements.txt.")

    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_RASTER_EXTENSIONS:
        raise ValueError(
            f"Unsupported image extension {suffix!r}. Use PNG/JPG/JPEG now; DICOM is optional."
        )

    try:
        with Image.open(path) as img:
            img = ImageOps.exif_transpose(img)
            img = _ensure_rgb(img)
            img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), _pil_lanczos())
            return _encode_jpeg_under_limit(img), _modality_hint_from_filename(path)
    except UnidentifiedImageError as exc:
        raise ValueError(f"Cannot identify image file: {path}") from exc


def _dicom_to_jpeg_bytes(path: Path) -> Tuple[bytes, str]:
    if Image is None:
        raise RuntimeError("pillow is not installed. Add pillow to requirements.txt.")

    try:
        import numpy as np
        import pydicom
    except Exception as exc:
        raise RuntimeError(
            "DICOM support requires pydicom and numpy. Add pydicom numpy to requirements.txt."
        ) from exc

    # Read the pixel array and render it to a fresh JPEG.
    # This avoids sending DICOM metadata to the cloud VLM, but does NOT remove
    # burned-in annotations already present in the pixels.
    ds = pydicom.dcmread(str(path), force=True)
    raw_modality = str(getattr(ds, "Modality", "")).upper().strip()
    modality_hint = _dicom_modality_to_label(raw_modality)

    arr = ds.pixel_array
    arr = np.asarray(arr)

    # Multi-frame: use first frame for demo. Production can loop over key slices.
    if arr.ndim == 4:
        arr = arr[0, :, :, 0]
    elif arr.ndim == 3:
        # Could be frames x H x W or H x W x channels.
        if arr.shape[-1] in (3, 4):
            arr = arr[:, :, :3]
        else:
            arr = arr[0]

    if arr.ndim == 3 and arr.shape[-1] in (3, 4):
        img = Image.fromarray(_scale_rgb_to_uint8(arr))
        img = _ensure_rgb(img)
    else:
        arr = arr.astype("float32")
        slope = float(getattr(ds, "RescaleSlope", 1) or 1)
        intercept = float(getattr(ds, "RescaleIntercept", 0) or 0)
        arr = arr * slope + intercept

        low, high = _dicom_window(ds, arr)
        if high <= low:
            low, high = float(arr.min()), float(arr.max())
        if high <= low:
            high = low + 1.0

        arr = (arr - low) / (high - low)
        arr = np.clip(arr, 0, 1) * 255.0

        if str(getattr(ds, "PhotometricInterpretation", "")).upper() == "MONOCHROME1":
            arr = 255.0 - arr

        img = Image.fromarray(arr.astype("uint8")).convert("RGB")

    img.thumbnail((MAX_IMAGE_SIDE, MAX_IMAGE_SIDE), _pil_lanczos())
    return _encode_jpeg_under_limit(img), modality_hint


def _dicom_window(ds: Any, arr: Any) -> Tuple[float, float]:
    def first_float(value: Any) -> Optional[float]:
        if value is None:
            return None
        if isinstance(value, (list, tuple)):
            return first_float(value[0]) if value else None
        try:
            # pydicom MultiValue supports indexing but may not be list/tuple.
            if hasattr(value, "__len__") and not isinstance(value, (str, bytes)):
                return float(value[0])
        except Exception:
            pass
        try:
            return float(value)
        except Exception:
            return None

    center = first_float(getattr(ds, "WindowCenter", None))
    width = first_float(getattr(ds, "WindowWidth", None))
    if center is not None and width is not None and width > 0:
        return center - width / 2.0, center + width / 2.0

    import numpy as np

    return tuple(float(x) for x in np.percentile(arr, [1, 99]))


def _scale_rgb_to_uint8(arr: Any) -> Any:
    import numpy as np

    arr = arr.astype("float32")
    low, high = np.percentile(arr, [1, 99])
    if high <= low:
        high = low + 1
    return np.clip((arr - low) / (high - low) * 255.0, 0, 255).astype("uint8")


def _encode_jpeg_under_limit(img: Any) -> bytes:
    qualities = [88, 80, 72, 64, 56, 48]
    working = img

    for _ in range(6):
        for quality in qualities:
            buffer = io.BytesIO()
            working.save(buffer, format="JPEG", quality=quality, optimize=True)
            data = buffer.getvalue()
            if len(data) <= MAX_INLINE_IMAGE_BYTES:
                return data

        w, h = working.size
        if min(w, h) <= 384:
            return data

        new_size = (max(384, int(w * 0.8)), max(384, int(h * 0.8)))
        working = working.resize(new_size, _pil_lanczos())

    return data


def _ensure_rgb(img: Any) -> Any:
    if img.mode in ("RGBA", "LA"):
        background = Image.new("RGB", img.size, (255, 255, 255))
        alpha = img.split()[-1]
        background.paste(img.convert("RGBA"), mask=alpha)
        return background
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def _pil_lanczos() -> Any:
    if Image is None:
        return None
    return getattr(getattr(Image, "Resampling", Image), "LANCZOS")


def _looks_like_dicom(path: Path) -> bool:
    if path.suffix.lower() in SUPPORTED_DICOM_EXTENSIONS:
        return True
    try:
        with path.open("rb") as f:
            header = f.read(132)
        return len(header) >= 132 and header[128:132] == b"DICM"
    except Exception:
        return False


def _dicom_modality_to_label(raw: str) -> str:
    mapping = {
        "CT": "CT",
        "MR": "MRI",
        "MRI": "MRI",
        "PT": "PET",
        "PET": "PET",
        "CR": "X-ray",
        "DX": "X-ray",
        "DR": "X-ray",
        "XA": "X-ray",
        "RF": "X-ray",
    }
    return mapping.get(raw.upper(), "Unknown")


def _modality_hint_from_filename(path: Path) -> str:
    name = path.name.lower()
    if "mri" in name or "_mr" in name or "magnetic" in name:
        return "MRI"
    if "ct" in name:
        return "CT"
    if "pet" in name:
        return "PET"
    if "xray" in name or "x-ray" in name or "xr" in name:
        return "X-ray"
    return "Unknown"


def _extract_json_object(text: str) -> Dict[str, Any]:
    cleaned = text.strip()

    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        value = json.loads(cleaned)
        if isinstance(value, dict):
            return value
    except Exception:
        pass

    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        try:
            value = json.loads(match.group(0))
            if isinstance(value, dict):
                return value
        except Exception:
            pass

    return {
        "modality": "Unknown",
        "findings": {
            "observations": [
                "VLM response could not be parsed as JSON; manual clinician review is required."
            ],
            "uncertain_or_not_assessable": ["Structured extraction failed."],
        },
        "limitations": ["Raw VLM output was not used to avoid unsafe free-text carryover."],
        "safety_flags": ["No definitive diagnosis"],
    }


def _normalize_vlm_payload(
    *,
    parsed: Dict[str, Any],
    raw_text: str,
    path: Path,
    local_modality_hint: str,
) -> Dict[str, Any]:
    modality = _normalize_modality(parsed.get("modality") or local_modality_hint)

    findings, safety_flags = _normalize_findings(parsed)
    limitations = _as_list(parsed.get("limitations"))

    if limitations:
        findings.setdefault("limitations", [])
        findings["limitations"].extend(_clean_text_list(limitations))

    if not findings.get("observations"):
        findings["observations"] = [
            "影像可被模型讀取，但未能抽取足夠可靠的非診斷性觀察；請以人工影像判讀為準。"
        ]

    findings.setdefault("limitations", [])
    if "Single image/screenshot may not represent the complete imaging study." not in findings["limitations"]:
        findings["limitations"].append("Single image/screenshot may not represent the complete imaging study.")

    safety_flags = list(dict.fromkeys(safety_flags + _as_list(parsed.get("safety_flags"))))
    safety_flags.extend(["No definitive diagnosis", "No segmentation", "No automated measurement"])
    safety_flags = list(dict.fromkeys(_clean_text_list(safety_flags)))

    return {
        "status": "ok",
        "modality": modality,
        "findings": findings,
        "disclaimer": DISCLAIMER,
        "privacy_note": PRIVACY_NOTE,
        "source": "nvidia_vlm",
        "model": os.getenv("NVIDIA_VLM_MODEL", DEFAULT_VLM_MODEL).strip() or DEFAULT_VLM_MODEL,
        "used_mock": False,
        "error": None,
        "safety_flags": safety_flags,
        "image_path": str(path),
    }


def _normalize_findings(parsed: Dict[str, Any]) -> Tuple[Dict[str, Any], List[str]]:
    source = parsed.get("findings", parsed)
    observations: List[str] = []
    uncertain: List[str] = []
    limitations: List[str] = []
    safety_flags: List[str] = []
    removed_count = 0

    def add_observation(value: Any, prefix: Optional[str] = None) -> None:
        nonlocal removed_count
        for item in _as_list(value):
            text = str(item).strip()
            if prefix and prefix.lower() not in {"observations", "observation"}:
                text = f"{prefix}: {text}"
            cleaned = _sanitize_observation(text)
            if cleaned:
                observations.append(cleaned)
            elif text:
                removed_count += 1

    if isinstance(source, dict):
        for key, value in source.items():
            key_str = str(key).strip()
            key_lower = key_str.lower()

            if key_lower in BANNED_FINDING_KEYS:
                removed_count += len(_as_list(value)) or 1
                continue

            if key_lower in {"uncertain", "uncertainty", "uncertain_or_not_assessable", "not_assessable"}:
                uncertain.extend(_clean_text_list(_as_list(value)))
            elif key_lower in {"limitations", "limitation"}:
                limitations.extend(_clean_text_list(_as_list(value)))
            elif key_lower in {"safety_flags", "safety"}:
                safety_flags.extend(_clean_text_list(_as_list(value)))
            else:
                add_observation(value, prefix=key_str)
    else:
        add_observation(source)

    if removed_count:
        safety_flags.append(
            f"Removed {removed_count} potential diagnostic/pathology/treatment claim(s) from VLM output."
        )

    findings = {
        "observations": list(dict.fromkeys(observations)),
        "uncertain_or_not_assessable": list(dict.fromkeys(uncertain)),
        "limitations": list(dict.fromkeys(limitations)),
    }
    return findings, safety_flags


def _sanitize_observation(text: str) -> Optional[str]:
    text = re.sub(r"\s+", " ", str(text)).strip()
    if not text:
        return None

    lowered = text.lower()
    if any(term in lowered for term in DIAGNOSTIC_TERMS):
        return None

    # Soften overconfident wording if a model produces it.
    replacements = [
        (r"\bdefinitely\b", "possibly"),
        (r"\bcertainly\b", "possibly"),
        (r"\bpathognomonic for\b", "non-specific for"),
        (r"\bdiagnostic of\b", "non-specific for"),
        (r"\bproves\b", "may suggest"),
    ]
    for pattern, repl in replacements:
        text = re.sub(pattern, repl, text, flags=re.IGNORECASE)

    return text


def _clean_text_list(values: Iterable[Any]) -> List[str]:
    cleaned: List[str] = []
    for value in values:
        text = re.sub(r"\s+", " ", str(value)).strip()
        if text:
            cleaned.append(text)
    return cleaned


def _as_list(value: Any) -> List[Any]:
    if value is None:
        return []
    if isinstance(value, list):
        out: List[Any] = []
        for item in value:
            if isinstance(item, list):
                out.extend(item)
            else:
                out.append(item)
        return out
    if isinstance(value, tuple):
        return list(value)
    if isinstance(value, dict):
        return [f"{k}: {v}" for k, v in value.items()]
    return [value]


def _normalize_modality(value: Any) -> str:
    text = str(value or "").strip().lower()
    text = text.replace("_", "-").replace("xray", "x-ray")
    if "mri" in text or text in {"mr"} or "magnetic resonance" in text:
        return "MRI"
    if text == "ct" or "computed tomography" in text or re.search(r"\bct\b", text):
        return "CT"
    if "pet" in text or "positron" in text:
        return "PET"
    if "x-ray" in text or "radiograph" in text:
        return "X-ray"
    return "Unknown"


def _is_missing_or_placeholder_key(api_key: str) -> bool:
    if not api_key:
        return True
    lowered = api_key.lower()
    placeholders = ["your_api_key", "paste", "replace_me", "changeme", "nvapi-..."]
    return any(token in lowered for token in placeholders)


def _fallback_result(
    *,
    status: str,
    path: Path,
    reason: str,
    source: str,
) -> Dict[str, Any]:
    is_mock = status == "mock"
    if is_mock:
        observations = [
            "示範用佔位（MOCK）：影像模組已被呼叫，但未進行真實雲端 VLM 影像描述。",
            "正式版請設定 NVIDIA_API_KEY，並由 NVIDIA VLM 產生非診斷性影像觀察。",
        ]
    else:
        observations = [
            "影像辨識失敗；主流程應保留醫師手動輸入的 imaging_description 作為 fallback。",
        ]

    return {
        "status": status,
        "modality": "Unknown",
        "findings": {
            "observations": observations,
            "uncertain_or_not_assessable": [
                "目前沒有可靠的 AI 影像觀察可供臨床推理使用。"
            ],
            "limitations": [
                "This is a fallback result, not a clinical image interpretation.",
                reason,
            ],
        },
        "disclaimer": DISCLAIMER,
        "privacy_note": PRIVACY_NOTE,
        "source": source,
        "model": os.getenv("NVIDIA_VLM_MODEL", DEFAULT_VLM_MODEL).strip() or DEFAULT_VLM_MODEL,
        "used_mock": is_mock,
        "error": None if is_mock else reason,
        "safety_flags": [
            "No definitive diagnosis",
            "No segmentation",
            "No automated measurement",
            "Use manual imaging_description fallback when status is not ok.",
        ],
        "image_path": str(path),
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Describe a medical image using NVIDIA VLM.")
    parser.add_argument("image_path", help="Path to PNG/JPG/JPEG/WEBP or DICOM image.")
    args = parser.parse_args()

    result = describe_image(args.image_path)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    print("\n--- imaging_description text ---")
    print(image_description_to_text(result))
