from __future__ import annotations

import hashlib
import re
import subprocess
import tempfile
from io import BytesIO
from pathlib import Path

from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError


class OCRProcessingError(RuntimeError):
    """Raised when OCR extraction fails or the report is unreadable."""


SWIFT_OCR_SOURCE = r'''
import Foundation
import Vision

let arguments = CommandLine.arguments
guard arguments.count > 1 else {
    fputs("missing-image-path\n", stderr)
    exit(1)
}

let imageURL = URL(fileURLWithPath: arguments[1])
let request = VNRecognizeTextRequest()
request.recognitionLevel = .accurate
request.usesLanguageCorrection = false
request.recognitionLanguages = ["en-US"]

let handler = VNImageRequestHandler(url: imageURL, options: [:])

do {
    try handler.perform([request])
    let lines = request.results?.compactMap { observation in
        observation.topCandidates(1).first?.string
    } ?? []
    print(lines.joined(separator: "\n"))
} catch {
    fputs("ocr-failed: \(error.localizedDescription)\n", stderr)
    exit(2)
}
'''

FIELD_ALIASES = {
    "hemoglobin": [r"hemoglobin", r"haemoglobin", r"\bhgb\b", r"\bhb\b"],
    "mcv": [r"\bmcv\b", r"mean\s+corpuscular\s+volume"],
    "mch": [r"\bmch\b", r"mean\s+corpuscular\s+hemoglobin"],
    "mchc": [r"\bmchc\b", r"mean\s+corpuscular\s+hemoglobin\s+concentration"],
    "rbc": [r"\brbc\b", r"red\s+blood\s+cell(?:s)?(?:\s+count)?"],
}

CORE_FIELDS = ("hemoglobin", "mcv", "mch", "mchc")


def _ocr_workspace() -> Path:
    workspace = Path(tempfile.gettempdir()) / "haemascan_local_ocr"
    workspace.mkdir(parents=True, exist_ok=True)
    return workspace


def _ensure_ocr_binary() -> Path:
    workspace = _ocr_workspace()
    source_hash = hashlib.sha256(SWIFT_OCR_SOURCE.encode("utf-8")).hexdigest()[:12]
    source_path = workspace / f"ocr_{source_hash}.swift"
    binary_path = workspace / f"ocr_{source_hash}"
    module_cache_path = workspace / "module-cache"
    module_cache_path.mkdir(parents=True, exist_ok=True)

    if binary_path.exists():
        return binary_path

    source_path.write_text(SWIFT_OCR_SOURCE, encoding="utf-8")
    command = [
        "swiftc",
        "-module-cache-path",
        str(module_cache_path),
        str(source_path),
        "-o",
        str(binary_path),
    ]

    try:
        subprocess.run(command, capture_output=True, text=True, check=True, timeout=90)
    except FileNotFoundError as exc:
        raise OCRProcessingError("Swift compiler is unavailable on this system.") from exc
    except subprocess.SubprocessError as exc:
        raise OCRProcessingError("The local OCR helper could not be prepared.") from exc

    return binary_path


def _save_candidate_images(image_bytes: bytes) -> list[Path]:
    workspace = _ocr_workspace()
    image = Image.open(BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image).convert("RGB")

    resized = image.copy()
    if resized.width < 1600:
        scale = 1600 / float(resized.width)
        resized = resized.resize((1600, int(resized.height * scale)))

    grayscale = ImageOps.autocontrast(resized.convert("L"))
    sharpened = grayscale.filter(ImageFilter.SHARPEN)
    thresholded = sharpened.point(lambda value: 255 if value > 175 else 0)

    candidates = [
        (workspace / "candidate_original.png", resized),
        (workspace / "candidate_grayscale.png", grayscale),
        (workspace / "candidate_threshold.png", thresholded),
    ]

    saved_paths: list[Path] = []
    for path, candidate in candidates:
        candidate.save(path)
        saved_paths.append(path)

    return saved_paths


def _run_ocr(image_path: Path) -> str:
    binary_path = _ensure_ocr_binary()
    try:
        response = subprocess.run(
            [str(binary_path), str(image_path)],
            capture_output=True,
            text=True,
            check=True,
            timeout=30,
        )
    except subprocess.SubprocessError as exc:
        raise OCRProcessingError("Could not read the uploaded report.") from exc

    return response.stdout.strip()


def _clean_text(text: str) -> str:
    normalized = text.replace("\r", "\n")
    normalized = re.sub(r"(?<=\d),(?=\d)", ".", normalized)
    normalized = re.sub(r"[|]", " ", normalized)
    cleaned_lines = [re.sub(r"\s+", " ", line).strip() for line in normalized.splitlines()]
    cleaned_lines = [line for line in cleaned_lines if line]
    return "\n".join(cleaned_lines)


def _extract_numeric_value(text: str, aliases: list[str]) -> float | None:
    alias_pattern = "(?:" + "|".join(aliases) + ")"
    pattern = re.compile(
        rf"{alias_pattern}(?:\s*[:=\-]?\s*|\s+)(\d{{1,3}}(?:\.\d{{1,2}})?)",
        re.IGNORECASE,
    )
    match = pattern.search(text)
    if match:
        try:
            return round(float(match.group(1)), 2)
        except ValueError:
            return None

    for line in text.splitlines():
        if re.search(alias_pattern, line, re.IGNORECASE):
            numbers = re.findall(r"\d{1,3}(?:\.\d{1,2})?", line)
            for number in numbers:
                try:
                    return round(float(number), 2)
                except ValueError:
                    continue
    return None


def _extract_gender(text: str) -> str | None:
    match = re.search(r"(?:gender|sex)\s*[:=\-]?\s*(male|female|m|f)\b", text, re.IGNORECASE)
    if match:
        token = match.group(1).lower()
        return "Male" if token in {"male", "m"} else "Female"

    fallback = re.search(r"\b(male|female)\b", text, re.IGNORECASE)
    if fallback:
        return fallback.group(1).title()
    return None


def _score_fields(fields: dict[str, float | str | None]) -> float:
    score = sum(1 for field_name in CORE_FIELDS if fields.get(field_name) is not None)
    if fields.get("gender"):
        score += 0.5
    if fields.get("rbc") is not None:
        score += 0.25
    return score


def _parse_report_text(text: str) -> dict[str, float | str | None]:
    cleaned = _clean_text(text)
    fields: dict[str, float | str | None] = {
        "gender": _extract_gender(cleaned),
        "hemoglobin": None,
        "mcv": None,
        "mch": None,
        "mchc": None,
        "rbc": None,
    }

    for field_name, aliases in FIELD_ALIASES.items():
        fields[field_name] = _extract_numeric_value(cleaned, aliases)

    return fields


def process_image(image_data: bytes) -> dict[str, float | str | None]:
    try:
        candidate_paths = _save_candidate_images(image_data)
    except UnidentifiedImageError as exc:
        raise OCRProcessingError("The uploaded file is not a valid image.") from exc
    except Exception as exc:
        raise OCRProcessingError("The image could not be prepared for OCR.") from exc

    best_fields: dict[str, float | str | None] | None = None
    best_score = -1.0

    for image_path in candidate_paths:
        text = _run_ocr(image_path)
        if not text.strip():
            continue

        candidate_fields = _parse_report_text(text)
        candidate_score = _score_fields(candidate_fields)
        if candidate_score > best_score:
            best_score = candidate_score
            best_fields = candidate_fields

        if candidate_score >= 4.5:
            break

    if not best_fields or best_score < 2:
        raise OCRProcessingError("Could not read report. Please enter values manually.")

    return best_fields
