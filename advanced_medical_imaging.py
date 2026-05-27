# -*- coding: utf-8 -*-
"""
advanced_medical_imaging.py

Advanced medical imaging helper for the Pediatric Neuro-Oncology Surgical
Planning Agent. Designed for Colab and hackathon demo workflows.

What it does:
- Reads DICOM folders / DICOM files / NIfTI (.nii, .nii.gz) volumes.
- Auto-detects MRI sequences: T1, T1C, T2, FLAIR, DWI, ADC, SWI/GRE.
- Produces a DEMO tumor mask using a heuristic fallback segmenter.
- Classifies rough tumor location: pons, midbrain, medulla, thalamus,
  spinal_cord, other, indeterminate.
- Extracts structured fields: enhancement, necrosis, hemorrhage,
  diffusion restriction, hydrocephalus.
- Cross-checks imaging findings against case text.
- Produces a neurosurgery-facing surgical risk map JPG.

Safety scope:
- This module is a research prototype / hackathon component.
- It must NOT be used for definitive diagnosis, treatment, or operative planning.
- The segmentation and anatomy landmarks are heuristic placeholders.
- Formal use should replace segment_tumor() and landmark logic with validated
  MONAI / nnUNet / atlas-registration components.
- DICOM must be de-identified before cloud upload or external processing.

中文重點：
- 可跑 DICOM / NIfTI，但 segmentation 是示範用 heuristic，不是正式醫療模型。
- 只產生「輔助觀察」和結構化欄位，不輸出確定診斷。
"""

from __future__ import annotations

import json
import math
import os
import re
import shutil
import warnings
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np

try:
    import nibabel as nib
except Exception:  # pragma: no cover
    nib = None

try:
    import pydicom
except Exception:  # pragma: no cover
    pydicom = None

try:
    from scipy import ndimage as ndi
except Exception:  # pragma: no cover
    ndi = None

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


DISCLAIMER = (
    "AI 影像輔助分析（research prototype），不可作為確定性診斷、治療或手術決策依據；"
    "最終判讀與手術決策須由放射科／神經外科／兒童神經腫瘤 MDT 負責。"
)

PRIVACY_NOTE = (
    "DICOM / NIfTI 影像在上傳雲端或外部服務前必須先去識別化；"
    "需移除 DICOM metadata 中的姓名、病歷號、生日、檢查日期等 PHI，"
    "並檢查影像邊框是否有 burned-in 文字。"
)

SEQUENCE_ORDER = ["T1C", "FLAIR", "T2", "DWI", "T1", "ADC", "SWI/GRE", "CT", "PET", "UNKNOWN"]
LOCATION_CLASSES = ["pons", "midbrain", "medulla", "thalamus", "spinal_cord", "other", "indeterminate"]


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------

def _safe_str(x: Any) -> str:
    if x is None:
        return ""
    try:
        return str(x)
    except Exception:
        return ""


def _lower(x: Any) -> str:
    return _safe_str(x).lower()


def _json_safe(obj: Any) -> Any:
    """Convert NumPy / Path / tuple objects to JSON-safe values."""
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_json_safe(v) for v in obj]
    if isinstance(obj, Path):
        return str(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, (np.bool_,)):
        return bool(obj)
    return obj


def _ensure_dir(path: Union[str, Path]) -> str:
    Path(path).mkdir(parents=True, exist_ok=True)
    return str(path)


def _percentile_normalize(arr: np.ndarray, p_low: float = 1, p_high: float = 99) -> np.ndarray:
    arr = np.asarray(arr, dtype=np.float32)
    arr = np.nan_to_num(arr, nan=0.0, posinf=0.0, neginf=0.0)
    if arr.size == 0:
        return arr
    lo, hi = np.percentile(arr, [p_low, p_high])
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo + 1e-6:
        return np.zeros_like(arr, dtype=np.float32)
    arr = np.clip(arr, lo, hi)
    return ((arr - lo) / (hi - lo)).astype(np.float32)


def _maybe_open_zip(input_path: Union[str, Path], work_dir: str = "medical_inputs_unzipped") -> Union[str, Path]:
    p = Path(input_path)
    if p.is_file() and p.suffix.lower() == ".zip":
        out = Path(work_dir)
        out.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(str(p), "r") as zf:
            zf.extractall(str(out))
        return out
    return input_path


def _collect_input_files(path_or_paths: Union[str, Path, Sequence[Union[str, Path]]]) -> List[str]:
    if isinstance(path_or_paths, (list, tuple, set)):
        files: List[str] = []
        for x in path_or_paths:
            files.extend(_collect_input_files(x))
        return sorted(set(files))

    p = Path(path_or_paths)
    p = Path(_maybe_open_zip(p))
    if p.is_file():
        return [str(p)]
    if p.is_dir():
        files = []
        for root, _, names in os.walk(str(p)):
            for name in names:
                if name.startswith("."):
                    continue
                files.append(str(Path(root) / name))
        return sorted(files)
    return []


def _is_nifti(path: Union[str, Path]) -> bool:
    s = str(path).lower()
    return s.endswith(".nii") or s.endswith(".nii.gz")


def _dicom_available() -> bool:
    return pydicom is not None


def _is_dicom(path: Union[str, Path]) -> bool:
    if pydicom is None:
        return False
    s = str(path).lower()
    if s.endswith(".dcm") or s.endswith(".dicom"):
        return True
    try:
        pydicom.dcmread(str(path), stop_before_pixels=True, force=True)
        return True
    except Exception:
        return False


def _status(status: str, evidence: str, confidence: str = "low") -> Dict[str, str]:
    return {"status": status, "evidence": evidence, "confidence": confidence}


# -----------------------------------------------------------------------------
# Sequence detection
# -----------------------------------------------------------------------------

def infer_sequence_label(text: str) -> str:
    """Infer sequence label from DICOM tags or filename."""
    s = _lower(text).replace("_", " ").replace("-", " ")

    # Post-contrast T1 / T1C must be checked before plain T1.
    if any(k in s for k in [
        "t1c", "t1 c", "t1+c", "t1 + c", "t1 post", "post t1", "postcontrast",
        "post contrast", "post gad", "gadolinium", "gd", "t1ce", "t1 ce", "ce t1",
        "mprage post", "bravo post", "spgr post",
    ]):
        return "T1C"
    if "adc" in s or "apparent diffusion" in s:
        return "ADC"
    if any(k in s for k in ["dwi", "diffusion", "trace", "b1000", "b 1000", "b800", "b 800"]):
        return "DWI"
    if "flair" in s:
        return "FLAIR"
    if any(k in s for k in ["swi", "susceptibility", "gre", "t2*", "t2 star", "t2star", "hemosiderin"]):
        return "SWI/GRE"
    if re.search(r"\bt2\b", s) or "tse t2" in s or "frfse" in s:
        return "T2"
    if re.search(r"\bt1\b", s) or "mprage" in s or "bravo" in s or "spgr" in s:
        return "T1"
    if re.search(r"\bct\b", s) or "computed tomography" in s:
        return "CT"
    if re.search(r"\bpet\b", s):
        return "PET"
    return "UNKNOWN"


def _unique_sequence_key(existing: Dict[str, Any], seq: str) -> str:
    if seq not in existing:
        return seq
    i = 2
    while f"{seq}_{i}" in existing:
        i += 1
    return f"{seq}_{i}"


def canonical_sequence(seq_key: str) -> str:
    """Return canonical base sequence without _2 suffix."""
    return re.sub(r"_\d+$", "", seq_key)


# -----------------------------------------------------------------------------
# Study loaders
# -----------------------------------------------------------------------------

@dataclass
class SeriesVolume:
    key: str
    sequence: str
    volume: np.ndarray
    source_type: str
    path: str
    spacing: Tuple[float, float, float]
    affine: np.ndarray
    metadata: Dict[str, Any]

    def as_public_dict(self) -> Dict[str, Any]:
        return {
            "key": self.key,
            "sequence": self.sequence,
            "source_type": self.source_type,
            "path": self.path,
            "shape": list(self.volume.shape),
            "spacing": list(self.spacing),
            "metadata": self.metadata,
        }


def _read_nifti(path: str) -> Optional[SeriesVolume]:
    if nib is None:
        raise ImportError("nibabel is required for NIfTI support. Please install nibabel.")
    img = nib.load(path)
    data = img.get_fdata(dtype=np.float32)
    if data.ndim == 4:
        # Demo: use first volume in 4D file. Formal pipeline should handle multi-volume DWI correctly.
        data = data[..., 0]
    # Nibabel returns X,Y,Z; transpose to Z,Y,X for consistent display.
    if data.ndim != 3:
        raise ValueError(f"NIfTI volume must be 3D or 4D, got shape {data.shape}: {path}")
    vol = np.transpose(data, (2, 1, 0)).astype(np.float32)
    zooms = img.header.get_zooms()[:3]
    spacing_xyz = tuple(float(x) for x in zooms) if len(zooms) >= 3 else (1.0, 1.0, 1.0)
    spacing_zyx = (spacing_xyz[2], spacing_xyz[1], spacing_xyz[0])
    seq = infer_sequence_label(Path(path).name)
    return SeriesVolume(
        key=seq,
        sequence=seq,
        volume=vol,
        source_type="NIFTI",
        path=str(path),
        spacing=spacing_zyx,
        affine=np.asarray(img.affine, dtype=np.float32),
        metadata={"filename": Path(path).name},
    )


def _dicom_sort_key(ds: Any) -> float:
    ipp = getattr(ds, "ImagePositionPatient", None)
    if ipp is not None and len(ipp) >= 3:
        try:
            return float(ipp[2])
        except Exception:
            pass
    inst = getattr(ds, "InstanceNumber", None)
    try:
        return float(inst)
    except Exception:
        return 0.0


def _get_dicom_spacing(first: Any) -> Tuple[float, float, float]:
    px = getattr(first, "PixelSpacing", [1.0, 1.0])
    try:
        row = float(px[0])
        col = float(px[1])
    except Exception:
        row, col = 1.0, 1.0
    try:
        z = float(getattr(first, "SpacingBetweenSlices", getattr(first, "SliceThickness", 1.0)))
    except Exception:
        z = 1.0
    return (z, row, col)


def _read_dicom_series(dicom_files: List[str]) -> Dict[str, SeriesVolume]:
    if pydicom is None:
        raise ImportError("pydicom is required for DICOM support. Please install pydicom.")

    grouped: Dict[str, List[Any]] = {}
    source_paths: Dict[str, List[str]] = {}

    for fp in dicom_files:
        try:
            ds = pydicom.dcmread(fp, force=True)
            if not hasattr(ds, "PixelData"):
                continue
            uid = _safe_str(getattr(ds, "SeriesInstanceUID", "")) or _safe_str(getattr(ds, "SeriesNumber", "")) or f"series_{len(grouped)+1}"
            grouped.setdefault(uid, []).append(ds)
            source_paths.setdefault(uid, []).append(fp)
        except Exception:
            continue

    series: Dict[str, SeriesVolume] = {}
    for uid, dss in grouped.items():
        if not dss:
            continue
        dss = sorted(dss, key=_dicom_sort_key)
        first = dss[0]
        desc_parts = [
            getattr(first, "SeriesDescription", ""),
            getattr(first, "ProtocolName", ""),
            getattr(first, "SequenceName", ""),
            getattr(first, "ScanningSequence", ""),
            getattr(first, "MRAcquisitionType", ""),
            Path(source_paths.get(uid, [""])[0]).name,
        ]
        seq = infer_sequence_label(" ".join(_safe_str(x) for x in desc_parts))
        key = _unique_sequence_key(series, seq)

        slices: List[np.ndarray] = []
        for ds in dss:
            try:
                arr = ds.pixel_array.astype(np.float32)
                slope = float(getattr(ds, "RescaleSlope", 1.0))
                intercept = float(getattr(ds, "RescaleIntercept", 0.0))
                arr = arr * slope + intercept
                if arr.ndim == 2:
                    slices.append(arr)
                elif arr.ndim == 3:
                    # multi-frame DICOM fallback
                    for i in range(arr.shape[0]):
                        slices.append(arr[i])
            except Exception as exc:
                warnings.warn(f"Could not read pixel array for one DICOM slice: {exc}")

        if not slices:
            continue
        if len(slices) == 1:
            vol = slices[0][None, :, :]
        else:
            try:
                vol = np.stack(slices, axis=0)
            except Exception:
                # fallback: keep only compatible slices
                shapes = {}
                for sl in slices:
                    shapes[sl.shape] = shapes.get(sl.shape, 0) + 1
                keep_shape = sorted(shapes.items(), key=lambda kv: kv[1], reverse=True)[0][0]
                vol = np.stack([sl for sl in slices if sl.shape == keep_shape], axis=0)

        metadata = {
            "SeriesDescription": _safe_str(getattr(first, "SeriesDescription", "")),
            "ProtocolName": _safe_str(getattr(first, "ProtocolName", "")),
            "SequenceName": _safe_str(getattr(first, "SequenceName", "")),
            "Modality": _safe_str(getattr(first, "Modality", "")),
            "SeriesInstanceUID": uid,
            "n_slices": int(vol.shape[0]),
        }
        series[key] = SeriesVolume(
            key=key,
            sequence=canonical_sequence(key),
            volume=vol.astype(np.float32),
            source_type="DICOM",
            path=f"DICOM_SERIES:{uid}",
            spacing=_get_dicom_spacing(first),
            affine=np.eye(4, dtype=np.float32),
            metadata=metadata,
        )

    return series


def load_study(path_or_paths: Union[str, Path, Sequence[Union[str, Path]]]) -> Dict[str, SeriesVolume]:
    """Load DICOM and/or NIfTI study into sequence-keyed volumes."""
    files = _collect_input_files(path_or_paths)
    if not files:
        raise FileNotFoundError(f"No input files found: {path_or_paths}")

    series: Dict[str, SeriesVolume] = {}

    for fp in files:
        if _is_nifti(fp):
            sv = _read_nifti(fp)
            if sv is not None:
                key = _unique_sequence_key(series, sv.sequence)
                sv.key = key
                series[key] = sv

    dicom_files = [fp for fp in files if _is_dicom(fp)]
    if dicom_files:
        series.update(_read_dicom_series(dicom_files))

    if not series:
        raise ValueError("No readable DICOM or NIfTI series found. Please upload .nii/.nii.gz or DICOM folder/zip.")

    return series


# -----------------------------------------------------------------------------
# Segmentation and measurement helpers
# -----------------------------------------------------------------------------

def _binary_ops_available() -> bool:
    return ndi is not None


def _simple_brain_mask(vol: np.ndarray) -> np.ndarray:
    v = _percentile_normalize(vol)
    if ndi is None:
        return v > max(0.05, float(np.percentile(v, 20)))
    mask = v > max(0.05, float(np.percentile(v, 20)))
    try:
        mask = ndi.binary_fill_holes(mask)
        mask = ndi.binary_opening(mask, structure=np.ones((3, 3, 3)))
        mask = ndi.binary_closing(mask, structure=np.ones((3, 3, 3)))
        labels, n = ndi.label(mask)
        if n > 0:
            sizes = ndi.sum(mask, labels, index=np.arange(1, n + 1))
            largest = int(np.argmax(sizes)) + 1
            mask = labels == largest
    except Exception:
        pass
    return mask.astype(bool)


def _choose_sequence(series: Dict[str, SeriesVolume], preference: Sequence[str] = ("T1C", "FLAIR", "T2", "DWI", "T1", "ADC", "SWI/GRE")) -> Tuple[str, SeriesVolume]:
    for pref in preference:
        for key, sv in series.items():
            if canonical_sequence(key) == pref:
                return key, sv
    key = list(series.keys())[0]
    return key, series[key]


def _largest_reasonable_component(mask: np.ndarray) -> np.ndarray:
    if mask.sum() == 0 or ndi is None:
        return mask.astype(bool)
    labels, n = ndi.label(mask)
    if n == 0:
        return np.zeros_like(mask, dtype=bool)
    sizes = ndi.sum(mask, labels, index=np.arange(1, n + 1))
    shape = np.asarray(mask.shape, dtype=np.float32)
    center = (shape - 1) / 2.0
    best_label = None
    best_score = -1e18
    for idx, size in enumerate(sizes, start=1):
        if size < 20:
            continue
        comp = labels == idx
        try:
            com = np.asarray(ndi.center_of_mass(comp), dtype=np.float32)
        except Exception:
            com = center
        dist = float(np.linalg.norm((com - center) / np.maximum(shape, 1)))
        # bias toward larger and central components; useful only for demo fallback.
        score = float(size) - 500.0 * dist
        if score > best_score:
            best_score = score
            best_label = idx
    if best_label is None:
        return np.zeros_like(mask, dtype=bool)
    return (labels == best_label)


def segment_tumor(series: Dict[str, SeriesVolume]) -> Dict[str, Any]:
    """
    DEMO heuristic tumor segmentation.

    Formal replacement target:
    - MONAI / nnUNet / BraTS-like model trained or validated for pediatric tumors.
    - Atlas registration / sequence-aware fusion.
    """
    seq_key, sv = _choose_sequence(series)
    vol = sv.volume
    v = _percentile_normalize(vol)
    brain = _simple_brain_mask(vol)

    vals = v[brain]
    if vals.size < 50:
        mask = np.zeros_like(v, dtype=bool)
    else:
        mu = float(vals.mean())
        sd = float(vals.std()) + 1e-6
        base_seq = canonical_sequence(seq_key)
        if base_seq in {"FLAIR", "T2", "T1C", "DWI"}:
            threshold = max(mu + 1.15 * sd, float(np.percentile(vals, 88)))
        else:
            threshold = max(mu + 1.45 * sd, float(np.percentile(vals, 92)))
        mask = (v > threshold) & brain
        if ndi is not None:
            try:
                mask = ndi.binary_opening(mask, structure=np.ones((2, 2, 2)))
                mask = ndi.binary_closing(mask, structure=np.ones((3, 3, 3)))
                mask = ndi.binary_fill_holes(mask)
            except Exception:
                pass
        mask = _largest_reasonable_component(mask)

    spacing = np.asarray(sv.spacing, dtype=np.float32)
    voxel_volume_mm3 = float(np.prod(spacing)) if spacing.size == 3 else 1.0
    volume_voxels = int(mask.sum())
    volume_ml = float(volume_voxels * voxel_volume_mm3 / 1000.0)

    return {
        "mask": mask.astype(bool),
        "source_sequence_key": seq_key,
        "source_sequence": canonical_sequence(seq_key),
        "method": "heuristic_fallback_demo_not_validated",
        "confidence": "low",
        "volume_voxels": volume_voxels,
        "volume_ml_rough": round(volume_ml, 3),
        "voxel_spacing_zyx_mm": [float(x) for x in spacing.tolist()],
    }


def _centroid(mask: np.ndarray) -> Optional[np.ndarray]:
    if mask.sum() == 0:
        return None
    if ndi is not None:
        try:
            return np.asarray(ndi.center_of_mass(mask), dtype=np.float32)
        except Exception:
            pass
    coords = np.argwhere(mask)
    if coords.size == 0:
        return None
    return coords.mean(axis=0).astype(np.float32)


def _normalized_centroid(mask: np.ndarray) -> Optional[np.ndarray]:
    c = _centroid(mask)
    if c is None:
        return None
    shape = np.asarray(mask.shape, dtype=np.float32)
    return c / np.maximum(shape - 1, 1)


def classify_location(mask: np.ndarray) -> Dict[str, Any]:
    """
    Rough location classifier based on normalized centroid.

    This is a placeholder. Production should use atlas registration or a validated classifier.
    """
    c = _normalized_centroid(mask)
    if c is None:
        return {"primary": "indeterminate", "confidence": "low", "centroid_normalized_zyx": None}

    z, y, x = [float(v) for v in c]
    # Heuristic for demo: posterior fossa / midline structure bands.
    if z < 0.24:
        loc = "thalamus"
    elif z < 0.38:
        loc = "midbrain"
    elif z < 0.55:
        loc = "pons"
    elif z < 0.72:
        loc = "medulla"
    else:
        loc = "spinal_cord"

    # If very lateral, reduce confidence and mark other.
    if abs(x - 0.5) > 0.30:
        loc = "other"

    return {
        "primary": loc,
        "confidence": "low",
        "centroid_normalized_zyx": [round(z, 3), round(y, 3), round(x, 3)],
        "note": "Heuristic location only; formal use requires atlas registration or expert review.",
    }


# -----------------------------------------------------------------------------
# Structured findings
# -----------------------------------------------------------------------------

def _find_sequence(series: Dict[str, SeriesVolume], seq: str) -> Optional[SeriesVolume]:
    for key, sv in series.items():
        if canonical_sequence(key) == seq:
            return sv
    return None


def _same_shape(a: np.ndarray, b: np.ndarray) -> bool:
    return tuple(a.shape) == tuple(b.shape)


def _masked_mean(vol: np.ndarray, mask: np.ndarray) -> Optional[float]:
    if mask.sum() == 0 or not _same_shape(vol, mask):
        return None
    vals = vol[mask]
    if vals.size == 0:
        return None
    return float(np.mean(vals))


def assess_enhancement(series: Dict[str, SeriesVolume], mask: np.ndarray) -> Dict[str, str]:
    t1 = _find_sequence(series, "T1")
    t1c = _find_sequence(series, "T1C")
    if t1 is None or t1c is None:
        return _status("indeterminate", "Needs paired T1 and T1C sequences.")
    if not _same_shape(t1.volume, t1c.volume) or not _same_shape(t1.volume, mask):
        return _status("indeterminate", "T1/T1C/mask shapes differ; registration not implemented in demo.")
    t1v = _percentile_normalize(t1.volume)
    t1cv = _percentile_normalize(t1c.volume)
    m1 = _masked_mean(t1v, mask)
    m2 = _masked_mean(t1cv, mask)
    if m1 is None or m2 is None:
        return _status("indeterminate", "No tumor mask statistics available.")
    diff = m2 - m1
    if diff > 0.12:
        return _status("possible_present", f"T1C mean exceeds T1 by {diff:.3f} in heuristic mask.")
    if diff > 0.05:
        return _status("mild_or_equivocal", f"T1C-T1 difference {diff:.3f}.")
    return _status("not_prominent", f"T1C-T1 difference {diff:.3f}.")


def assess_necrosis(series: Dict[str, SeriesVolume], mask: np.ndarray) -> Dict[str, str]:
    sv = _find_sequence(series, "T1C") or _find_sequence(series, "FLAIR") or _find_sequence(series, "T2")
    if sv is None:
        return _status("indeterminate", "Needs T1C, FLAIR, or T2 sequence.")
    if mask.sum() == 0 or not _same_shape(sv.volume, mask):
        return _status("indeterminate", "Tumor mask unavailable or unregistered.")
    v = _percentile_normalize(sv.volume)
    vals = v[mask]
    if vals.size < 20:
        return _status("indeterminate", "Tumor mask too small for necrosis surrogate.")
    low_frac = float((vals < np.percentile(vals, 20)).mean())
    if low_frac > 0.30:
        return _status("possible", f"Internal low-signal fraction {low_frac:.2f} in {sv.sequence} mask.")
    return _status("not_prominent", f"Internal low-signal fraction {low_frac:.2f}.")


def assess_hemorrhage(series: Dict[str, SeriesVolume], mask: np.ndarray) -> Dict[str, str]:
    swi = _find_sequence(series, "SWI/GRE")
    if swi is None:
        return _status("indeterminate", "Needs SWI/GRE or CT for hemorrhage surrogate.")
    if mask.sum() == 0 or not _same_shape(swi.volume, mask):
        return _status("indeterminate", "Tumor mask unavailable or unregistered.")
    v = _percentile_normalize(swi.volume)
    vals = v[mask]
    if vals.size < 20:
        return _status("indeterminate", "Tumor mask too small.")
    low_global = float(np.percentile(v, 10))
    low_frac = float((vals < low_global).mean())
    if low_frac > 0.15:
        return _status("possible", f"Low-signal foci fraction {low_frac:.2f} on SWI/GRE surrogate.")
    return _status("not_obvious", f"Low-signal foci fraction {low_frac:.2f}.")


def assess_diffusion_restriction(series: Dict[str, SeriesVolume], mask: np.ndarray) -> Dict[str, str]:
    dwi = _find_sequence(series, "DWI")
    adc = _find_sequence(series, "ADC")
    if dwi is None or adc is None:
        return _status("indeterminate", "Needs both DWI and ADC.")
    if not _same_shape(dwi.volume, adc.volume) or not _same_shape(dwi.volume, mask):
        return _status("indeterminate", "DWI/ADC/mask shapes differ; registration not implemented in demo.")
    dwi_v = _percentile_normalize(dwi.volume)
    adc_v = _percentile_normalize(adc.volume)
    dwi_mean = _masked_mean(dwi_v, mask)
    adc_mean = _masked_mean(adc_v, mask)
    if dwi_mean is None or adc_mean is None:
        return _status("indeterminate", "No tumor mask statistics available.")
    if dwi_mean > 0.60 and adc_mean < 0.45:
        return _status("possible_present", f"DWI mean {dwi_mean:.2f}, ADC mean {adc_mean:.2f}.")
    return _status("not_prominent", f"DWI mean {dwi_mean:.2f}, ADC mean {adc_mean:.2f}.")


def assess_hydrocephalus(series: Dict[str, SeriesVolume], mask: np.ndarray, location: Dict[str, Any]) -> Dict[str, str]:
    c = _normalized_centroid(mask)
    if c is None:
        return _status("indeterminate", "No tumor centroid.")
    loc = location.get("primary", "indeterminate")
    # Approximate fourth ventricle region in normalized ZYX coordinate for demo.
    fourth_ventricle = np.array([0.54, 0.52, 0.50], dtype=np.float32)
    dist = float(np.linalg.norm(c - fourth_ventricle))
    if loc in {"pons", "midbrain", "medulla"} and dist < 0.18:
        return _status("possible", f"Brainstem/posterior-fossa mass near fourth-ventricle surrogate ROI; distance={dist:.2f}.")
    return _status("not_obvious", f"No strong fourth-ventricle obstruction surrogate; distance={dist:.2f}.")


def extract_structured_findings(series: Dict[str, SeriesVolume], segmentation: Dict[str, Any], location: Dict[str, Any]) -> Dict[str, Any]:
    mask = segmentation["mask"]
    return {
        "tumor_detected": bool(mask.sum() > 0),
        "tumor_volume_voxels": int(segmentation.get("volume_voxels", 0)),
        "tumor_volume_ml_rough": float(segmentation.get("volume_ml_rough", 0.0)),
        "source_sequence_for_segmentation": segmentation.get("source_sequence"),
        "segmentation_method": segmentation.get("method"),
        "segmentation_confidence": segmentation.get("confidence"),
        "location": location,
        "enhancement": assess_enhancement(series, mask),
        "necrosis": assess_necrosis(series, mask),
        "hemorrhage": assess_hemorrhage(series, mask),
        "diffusion_restriction": assess_diffusion_restriction(series, mask),
        "hydrocephalus": assess_hydrocephalus(series, mask, location),
    }


# -----------------------------------------------------------------------------
# Cross-check with case text
# -----------------------------------------------------------------------------

def _case_text(structured_case: Optional[Dict[str, Any]]) -> str:
    if not structured_case:
        return ""
    chunks = []
    for key in [
        "age", "tumor_type", "tumor_location", "pathology", "symptoms",
        "imaging_description", "molecular_markers", "raw_case_text",
    ]:
        value = structured_case.get(key)
        if value is None:
            continue
        if isinstance(value, (dict, list, tuple)):
            chunks.append(json.dumps(value, ensure_ascii=False))
        else:
            chunks.append(str(value))
    return "\n".join(chunks).lower()


def cross_check_case(structured_case: Optional[Dict[str, Any]], findings: Dict[str, Any]) -> Dict[str, List[str]]:
    text = _case_text(structured_case)
    concordant: List[str] = []
    discrepant: List[str] = []
    notes: List[str] = []

    loc = findings.get("location", {}).get("primary", "indeterminate")
    if loc != "indeterminate":
        if loc in text:
            concordant.append(f"Case text explicitly mentions imaging-compatible location: {loc}.")
        elif structured_case and structured_case.get("tumor_location"):
            discrepant.append(
                f"Case tumor_location='{structured_case.get('tumor_location')}' does not explicitly match imaging-derived location='{loc}'."
            )
        else:
            notes.append(f"Imaging-derived location surrogate is '{loc}', not stated in case text.")

    field_terms = {
        "hydrocephalus": ["hydrocephalus", "ventriculomegaly", "ventricular enlargement"],
        "hemorrhage": ["hemorrhage", "haemorrhage", "bleed", "swi", "gre"],
        "diffusion_restriction": ["diffusion restriction", "restricted diffusion", "adc", "dwi"],
        "enhancement": ["enhancement", "enhancing", "contrast"],
        "necrosis": ["necrosis", "necrotic"],
    }
    for field, terms in field_terms.items():
        status = findings.get(field, {}).get("status", "")
        if status in {"possible", "possible_present", "present"} and not any(t in text for t in terms):
            notes.append(f"Imaging module flags {field} as {status}, but the case text does not clearly mention it.")

    pathology = ""
    if structured_case:
        pathology = _lower(structured_case.get("pathology", "")) + " " + _lower(structured_case.get("tumor_type", ""))
    if "diffuse midline" in pathology or "h3 k27" in pathology or "h3k27" in pathology:
        if loc in {"pons", "midbrain", "medulla", "thalamus", "spinal_cord"}:
            concordant.append("Pathology/tumor-type text is broadly compatible with a midline CNS location.")
        elif loc != "indeterminate":
            discrepant.append("Case suggests diffuse midline/H3 K27-altered tumor, but imaging location surrogate is not midline-compatible.")

    return {"concordant": concordant, "discrepant": discrepant, "notes": notes}


# -----------------------------------------------------------------------------
# Surgical risk map
# -----------------------------------------------------------------------------

def _slice_views(vol: np.ndarray, center_zyx: Tuple[int, int, int]) -> Dict[str, np.ndarray]:
    z, y, x = center_zyx
    z = int(np.clip(z, 0, vol.shape[0] - 1))
    y = int(np.clip(y, 0, vol.shape[1] - 1))
    x = int(np.clip(x, 0, vol.shape[2] - 1))
    return {
        "Axial": vol[z, :, :],
        "Coronal": vol[:, y, :],
        "Sagittal": vol[:, :, x],
    }


def _risk_level(distance_vox: float, shape: Sequence[int]) -> str:
    diag = float(np.linalg.norm(np.asarray(shape, dtype=np.float32)))
    ratio = distance_vox / max(diag, 1.0)
    if ratio < 0.08:
        return "HIGH"
    if ratio < 0.16:
        return "MEDIUM"
    return "LOW"


def _landmark_points(shape: Sequence[int]) -> Dict[str, Tuple[int, int, int]]:
    shape_arr = np.asarray(shape, dtype=np.float32)
    # Demo normalized landmarks, not anatomical ground truth.
    pts_norm = {
        "basilar_artery_surrogate": np.array([0.48, 0.60, 0.50], dtype=np.float32),
        "cranial_nerve_nuclei_surrogate": np.array([0.50, 0.48, 0.50], dtype=np.float32),
        "fourth_ventricle_surrogate": np.array([0.54, 0.52, 0.50], dtype=np.float32),
    }
    return {k: tuple(np.round(v * (shape_arr - 1)).astype(int).tolist()) for k, v in pts_norm.items()}


def create_surgical_risk_map(
    series: Dict[str, SeriesVolume],
    segmentation: Dict[str, Any],
    findings: Dict[str, Any],
    output_path: str = "outputs/surgical_risk_map.jpg",
) -> str:
    if plt is None:
        raise ImportError("matplotlib is required to create surgical risk map.")

    _ensure_dir(Path(output_path).parent)
    seq_key, sv = _choose_sequence(series, preference=("T1C", "FLAIR", "T2", "T1", "DWI", "ADC", "SWI/GRE"))
    vol = _percentile_normalize(sv.volume)
    mask = segmentation["mask"]

    if mask.sum() > 0 and _same_shape(vol, mask):
        center_arr = _centroid(mask)
        center = tuple(np.round(center_arr).astype(int).tolist()) if center_arr is not None else (vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2)
    else:
        center = (vol.shape[0] // 2, vol.shape[1] // 2, vol.shape[2] // 2)

    views = _slice_views(vol, center)
    mask_views = _slice_views(mask.astype(float), center) if _same_shape(vol, mask) else {k: np.zeros_like(v) for k, v in views.items()}

    coords = np.argwhere(mask) if mask.sum() > 0 else np.empty((0, 3), dtype=np.float32)
    risk_rows = []
    for name, pt in _landmark_points(mask.shape if mask.ndim == 3 else vol.shape).items():
        if coords.size:
            dists = np.linalg.norm(coords.astype(np.float32) - np.asarray(pt, dtype=np.float32)[None, :], axis=1)
            d = float(dists.min())
        else:
            d = math.inf
        risk_rows.append((name, d, "indeterminate" if not np.isfinite(d) else _risk_level(d, vol.shape)))

    fig = plt.figure(figsize=(16, 9))
    gs = fig.add_gridspec(2, 3, width_ratios=[1, 1, 1.25], height_ratios=[1, 1])
    ax1 = fig.add_subplot(gs[0, 0])
    ax2 = fig.add_subplot(gs[0, 1])
    ax3 = fig.add_subplot(gs[1, 0])
    ax_text = fig.add_subplot(gs[:, 2])
    axes = [ax1, ax2, ax3]

    for ax, title in zip(axes, ["Axial", "Coronal", "Sagittal"]):
        ax.imshow(views[title], cmap="gray")
        mv = mask_views[title]
        if np.asarray(mv).sum() > 0:
            try:
                ax.contour(mv, levels=[0.5], colors="red", linewidths=1.5)
            except Exception:
                pass
        ax.set_title(title)
        ax.axis("off")

    loc = findings.get("location", {}).get("primary", "indeterminate")
    hydro = findings.get("hydrocephalus", {}).get("status", "indeterminate")

    ax_text.axis("off")
    y = 0.98
    ax_text.text(0.0, y, "Surgical Risk Map", fontsize=18, fontweight="bold", va="top")
    y -= 0.07
    ax_text.text(0.0, y, f"Display sequence: {canonical_sequence(seq_key)}", fontsize=11, va="top")
    y -= 0.045
    ax_text.text(0.0, y, f"Tumor location surrogate: {loc}", fontsize=11, va="top")
    y -= 0.045
    ax_text.text(0.0, y, f"Hydrocephalus surrogate: {hydro}", fontsize=11, va="top")
    y -= 0.065

    ax_text.text(0.0, y, "Critical structure proximity", fontsize=13, fontweight="bold", va="top")
    y -= 0.048
    for name, dist, level in risk_rows:
        d_txt = "NA" if not np.isfinite(dist) else f"{dist:.1f} voxels"
        ax_text.text(0.0, y, f"- {name}: {level} ({d_txt})", fontsize=10, va="top")
        y -= 0.04

    y -= 0.035
    ax_text.text(0.0, y, "Structured findings", fontsize=13, fontweight="bold", va="top")
    y -= 0.048
    for key in ["enhancement", "necrosis", "hemorrhage", "diffusion_restriction", "hydrocephalus"]:
        block = findings.get(key, {})
        ax_text.text(0.0, y, f"- {key}: {block.get('status', 'indeterminate')}", fontsize=10, va="top")
        y -= 0.04

    y -= 0.035
    ax_text.text(0.0, y, "Safety", fontsize=13, fontweight="bold", va="top")
    y -= 0.048
    ax_text.text(0.0, y, DISCLAIMER, fontsize=9, va="top", wrap=True)

    plt.tight_layout()
    plt.savefig(output_path, dpi=160, bbox_inches="tight")
    plt.close(fig)
    return output_path


# -----------------------------------------------------------------------------
# Text output for Nemotron input and report appendix
# -----------------------------------------------------------------------------

def build_imaging_description(result: Dict[str, Any]) -> str:
    findings = result.get("structured_findings", {})
    location = findings.get("location", {})
    cross = result.get("cross_check", {})

    lines = [
        "Structured imaging summary (AI-assisted, non-diagnostic):",
        f"- Input type: {result.get('input_type', 'DICOM/NIfTI')}",
        f"- Series detected: {', '.join(result.get('series_detected', [])) or 'none'}",
        f"- Tumor detected by demo segmenter: {findings.get('tumor_detected', False)}",
        f"- Segmentation method: {findings.get('segmentation_method', 'NA')}",
        f"- Segmentation confidence: {findings.get('segmentation_confidence', 'low')}",
        f"- Rough tumor volume: {findings.get('tumor_volume_voxels', 0)} voxels / {findings.get('tumor_volume_ml_rough', 0.0)} mL",
        f"- Location surrogate: {location.get('primary', 'indeterminate')} ({location.get('confidence', 'low')} confidence)",
    ]
    for key in ["enhancement", "necrosis", "hemorrhage", "diffusion_restriction", "hydrocephalus"]:
        block = findings.get(key, {})
        lines.append(f"- {key}: {block.get('status', 'indeterminate')} — {block.get('evidence', '')}")

    if cross.get("concordant"):
        lines.append("- Concordant points with case text:")
        lines.extend([f"  * {x}" for x in cross.get("concordant", [])])
    if cross.get("discrepant"):
        lines.append("- Potential discrepancies against case text:")
        lines.extend([f"  * {x}" for x in cross.get("discrepant", [])])
    if cross.get("notes"):
        lines.append("- Additional cross-check notes:")
        lines.extend([f"  * {x}" for x in cross.get("notes", [])])

    lines.append(f"- Safety disclaimer: {DISCLAIMER}")
    lines.append(f"- Privacy note: {PRIVACY_NOTE}")
    return "\n".join(lines)


def save_mask(mask: np.ndarray, output_path: str = "outputs/advanced_tumor_mask.npy") -> str:
    _ensure_dir(Path(output_path).parent)
    np.save(output_path, mask.astype(np.uint8))
    return output_path


def analyze_medical_study(
    path_or_paths: Union[str, Path, Sequence[Union[str, Path]]],
    structured_case: Optional[Dict[str, Any]] = None,
    output_dir: str = "outputs",
) -> Dict[str, Any]:
    """Main entry point for DICOM/NIfTI advanced imaging analysis."""
    try:
        _ensure_dir(output_dir)
        series = load_study(path_or_paths)
        segmentation = segment_tumor(series)
        location = classify_location(segmentation["mask"])
        findings = extract_structured_findings(series, segmentation, location)
        cross = cross_check_case(structured_case, findings)
        risk_map_path = create_surgical_risk_map(
            series=series,
            segmentation=segmentation,
            findings=findings,
            output_path=str(Path(output_dir) / "surgical_risk_map.jpg"),
        )
        mask_path = save_mask(segmentation["mask"], str(Path(output_dir) / "advanced_tumor_mask.npy"))

        result = {
            "status": "ok",
            "input_type": "DICOM/NIfTI",
            "series_detected": [f"{k}:{sv.sequence}" for k, sv in series.items()],
            "series_metadata": [sv.as_public_dict() for sv in series.values()],
            "structured_findings": findings,
            "cross_check": cross,
            "risk_map_path": risk_map_path,
            "mask_path": mask_path,
            "disclaimer": DISCLAIMER,
            "privacy_note": PRIVACY_NOTE,
            "safety_flags": [
                "Research prototype only",
                "Heuristic segmentation fallback, not validated",
                "No definitive diagnosis",
                "No direct treatment recommendation",
                "Requires radiology/neurosurgery review",
            ],
        }
        result["imaging_description_text"] = build_imaging_description(result)
        return _json_safe(result)

    except Exception as exc:
        return _json_safe({
            "status": "error",
            "error": str(exc),
            "input_type": "DICOM/NIfTI",
            "series_detected": [],
            "structured_findings": {},
            "cross_check": {},
            "risk_map_path": None,
            "mask_path": None,
            "imaging_description_text": "",
            "disclaimer": DISCLAIMER,
            "privacy_note": PRIVACY_NOTE,
            "safety_flags": [
                "Advanced imaging failed gracefully",
                "Use physician-provided imaging_description fallback",
                "No definitive diagnosis",
            ],
        })


def integrate_advanced_imaging_into_case(
    case: Dict[str, Any],
    path_or_paths: Union[str, Path, Sequence[Union[str, Path]]],
    output_dir: str = "outputs",
) -> Dict[str, Any]:
    """Small helper for agent.py / notebook integration."""
    updated = dict(case or {})
    result = analyze_medical_study(path_or_paths, structured_case=updated, output_dir=output_dir)
    updated["advanced_imaging_result"] = result
    if result.get("status") == "ok":
        updated["imaging_description"] = result.get("imaging_description_text", updated.get("imaging_description", ""))
        updated["surgical_risk_map_path"] = result.get("risk_map_path")
    else:
        updated["advanced_imaging_warning"] = "Advanced DICOM/NIfTI analysis unavailable; use manual imaging_description fallback."
    return updated


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Run advanced DICOM/NIfTI imaging analysis.")
    parser.add_argument("input", help="DICOM folder/zip/file or NIfTI file/folder")
    parser.add_argument("--output-dir", default="outputs")
    parser.add_argument("--case-json", default=None, help="Optional structured case JSON file")
    args = parser.parse_args()

    case = None
    if args.case_json:
        case = json.loads(Path(args.case_json).read_text(encoding="utf-8"))
    out = analyze_medical_study(args.input, structured_case=case, output_dir=args.output_dir)
    print(json.dumps(out, ensure_ascii=False, indent=2))
