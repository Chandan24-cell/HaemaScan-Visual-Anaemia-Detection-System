from __future__ import annotations

import os
from io import BytesIO
from typing import Any

import numpy as np
from dotenv import load_dotenv
from flask import Flask, jsonify, redirect, render_template, request, session, url_for
from PIL import Image, ImageOps, UnidentifiedImageError

from process_image import OCRProcessingError, process_image

load_dotenv(override=True)

USE_AUTH = False
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "dev-secret-key")
app.config["MAX_CONTENT_LENGTH"] = 12 * 1024 * 1024

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
VISION_MODEL_HINT = os.getenv("HAEMASCAN_VISION_BACKEND", "heuristic")


class UserInputError(ValueError):
    """Raised when user-provided input is invalid."""


def is_publishable_key(key: str) -> bool:
    return key.startswith("sb_publishable_")


def create_supabase_client():
    if not USE_AUTH:
        return None

    try:
        from supabase import create_client
    except Exception:
        return None

    if not SUPABASE_URL or not SUPABASE_KEY or is_publishable_key(SUPABASE_KEY):
        return None

    try:
        return create_client(SUPABASE_URL, SUPABASE_KEY)
    except Exception:
        return None


supabase = create_supabase_client()


CBC_FIELD_LIMITS = {
    "hemoglobin": (2.0, 25.0),
    "mcv": (40.0, 130.0),
    "mch": (10.0, 45.0),
    "mchc": (20.0, 40.0),
    "rbc": (1.0, 9.0),
}

VISION_CONFIG = {
    "palm": {
        "weight": 0.45,
        "red_range": (0.12, 0.25),
        "sat_range": (0.15, 0.35),
        "bright_range": (0.45, 0.85),
    },
    "nail": {
        "weight": 0.35,
        "red_range": (0.10, 0.22),
        "sat_range": (0.12, 0.32),
        "bright_range": (0.35, 0.80),
    },
    "conjunctiva": {
        "weight": 0.20,
        "red_range": (0.18, 0.35),
        "sat_range": (0.18, 0.45),
        "bright_range": (0.30, 0.82),
    },
}


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def normalize_range(value: float, low: float, high: float) -> float:
    if high <= low:
        return 0.0
    return clamp((value - low) / (high - low), 0.0, 1.0)


def parse_numeric(form_data: Any, field_name: str, *, required: bool = True) -> float | None:
    raw_value = str(form_data.get(field_name, "") or "").strip()
    if not raw_value:
        if required:
            raise UserInputError(f"{field_name.replace('_', ' ').title()} is required.")
        return None

    try:
        value = float(raw_value)
    except ValueError as exc:
        raise UserInputError(f"{field_name.replace('_', ' ').title()} must be numeric.") from exc

    low, high = CBC_FIELD_LIMITS[field_name]
    if not (low <= value <= high):
        raise UserInputError(
            f"{field_name.replace('_', ' ').title()} must be between {low:g} and {high:g}."
        )
    return round(value, 2)


def normalize_gender(raw_gender: str | None) -> str | None:
    value = (raw_gender or "").strip().lower()
    if value in {"male", "m"}:
        return "Male"
    if value in {"female", "f"}:
        return "Female"
    return None


def validate_manual_payload(form_data: Any) -> dict[str, float | str | None]:
    gender = normalize_gender(form_data.get("gender"))
    if not gender:
        raise UserInputError("Please select a valid gender before analyzing.")

    payload = {
        "gender": gender,
        "hemoglobin": parse_numeric(form_data, "hemoglobin"),
        "mcv": parse_numeric(form_data, "mcv"),
        "mch": parse_numeric(form_data, "mch"),
        "mchc": parse_numeric(form_data, "mchc"),
        "rbc": parse_numeric(form_data, "rbc", required=False),
    }
    return payload


def infer_possible_type(
    *,
    is_anemic: bool,
    mcv: float,
    mch: float,
    mchc: float,
    rbc: float | None,
) -> tuple[str, float | None, str | None]:
    mentzer_index = round(mcv / rbc, 2) if rbc else None
    note = None

    if not is_anemic:
        if mcv < 80 or mch < 27 or mchc < 32:
            note = "Indices are slightly low; correlate with iron studies if symptoms persist."
            return "Borderline microcytic pattern", mentzer_index, note
        return "No anemia pattern detected", mentzer_index, note

    if mcv < 80:
        if mentzer_index is not None:
            if mentzer_index < 13:
                return "Microcytic pattern, possible thalassemia trait", mentzer_index, note
            return "Microcytic hypochromic pattern, likely iron deficiency anemia", mentzer_index, note
        note = "Add RBC count to calculate the Mentzer index and refine the subtype."
        return "Microcytic hypochromic pattern, likely iron deficiency anemia", mentzer_index, note

    if mcv > 100:
        return "Macrocytic pattern, consider vitamin B12 or folate deficiency", mentzer_index, note

    if mch < 27 or mchc < 32:
        return "Hypochromic pattern, possible iron deficiency anemia", mentzer_index, note

    return "Normocytic pattern, consider chronic disease or acute blood loss", mentzer_index, note


def analyze_cbc(*, hemoglobin: float, mcv: float, mch: float, mchc: float, gender: str, rbc: float | None = None) -> dict[str, Any]:
    normal_cutoff = 13.5 if gender == "Male" else 12.0

    if hemoglobin >= normal_cutoff:
        classification = "Normal"
        is_anemic = False
    elif hemoglobin < 8.0:
        classification = "Severe Anemia"
        is_anemic = True
    elif hemoglobin < 11.0:
        classification = "Moderate Anemia"
        is_anemic = True
    else:
        classification = "Mild Anemia"
        is_anemic = True

    possible_type, mentzer_index, note = infer_possible_type(
        is_anemic=is_anemic,
        mcv=mcv,
        mch=mch,
        mchc=mchc,
        rbc=rbc,
    )

    headline = classification if is_anemic else "Normal CBC Pattern"
    summary = (
        f"{classification} detected with a likely {possible_type.lower()}."
        if is_anemic
        else "Values are within the expected anemia screening range."
    )

    return {
        "headline": headline,
        "classification": classification,
        "anemia_status": "Anemic" if is_anemic else "Non-Anemic",
        "is_anemic": is_anemic,
        "possible_type": possible_type,
        "summary": summary,
        "guidance": note,
        "gender": gender,
        "hemoglobin_cutoff": normal_cutoff,
        "indices": {
            "hemoglobin": hemoglobin,
            "mcv": mcv,
            "mch": mch,
            "mchc": mchc,
            "rbc": rbc,
            "mentzer_index": mentzer_index,
        },
    }


def extract_visual_metrics(image_bytes: bytes) -> dict[str, float]:
    image = Image.open(BytesIO(image_bytes))
    image = ImageOps.exif_transpose(image).convert("RGB")

    width, height = image.size
    left = int(width * 0.15)
    right = int(width * 0.85)
    top = int(height * 0.15)
    bottom = int(height * 0.85)
    cropped = image.crop((left, top, right, bottom))

    array = np.asarray(cropped, dtype=np.float32) / 255.0
    red = array[..., 0]
    green = array[..., 1]
    blue = array[..., 2]
    max_channel = np.max(array, axis=2)
    min_channel = np.min(array, axis=2)
    saturation = (max_channel - min_channel) / (max_channel + 1e-6)

    skin_mask = (red > 0.2) & (red > green * 0.9) & (red > blue * 0.9) & (saturation > 0.05)
    if float(np.mean(skin_mask)) < 0.05:
        skin_mask = np.ones_like(red, dtype=bool)

    pallor = red - ((green + blue) / 2.0)

    return {
        "redness": float(np.mean(pallor[skin_mask])),
        "saturation": float(np.mean(saturation[skin_mask])),
        "brightness": float(np.mean(max_channel[skin_mask])),
    }


def estimate_region_probability(region: str, image_bytes: bytes) -> dict[str, Any]:
    metrics = extract_visual_metrics(image_bytes)
    config = VISION_CONFIG[region]

    redness_score = 1.0 - normalize_range(metrics["redness"], *config["red_range"])
    saturation_score = 1.0 - normalize_range(metrics["saturation"], *config["sat_range"])
    brightness_score = normalize_range(metrics["brightness"], *config["bright_range"])
    probability = clamp((0.60 * redness_score) + (0.25 * saturation_score) + (0.15 * brightness_score), 0.03, 0.97)

    return {
        "region": region,
        "probability": round(float(probability), 4),
        "metrics": {name: round(value, 4) for name, value in metrics.items()},
    }


def json_error(message: str, status: int = 400, **extra: Any):
    payload = {"success": False, "error": message}
    payload.update(extra)
    return jsonify(payload), status


@app.route('/login/email-password', methods=['GET', 'POST'])
def login_email_password():
    if supabase is None:
        return render_template('login.html', error='Authentication is disabled for this deployment.')

    if request.method == 'POST':
        email = request.form['email']
        password = request.form['password']

        try:
            response = supabase.auth.sign_in_with_password({
                "email": email,
                "password": password,
            })
            session['user'] = response.user.email
            return redirect(url_for('dashboard'))
        except Exception:
            return render_template('login.html', error='Invalid email or password')

    return render_template('login.html')


@app.route('/login/google')
def login_google():
    if not USE_AUTH or supabase is None:
        return redirect('/')

    response = supabase.auth.sign_in_with_oauth({
        "provider": "google",
        "options": {
            "redirect_to": url_for("auth_callback", _external=True),
        },
    })
    return redirect(response.url)


@app.route('/login')
def login():
    if not USE_AUTH:
        return redirect('/')
    return render_template('login.html')


@app.route('/auth/callback')
def auth_callback():
    session["logged_in"] = True
    session["user"] = "google_user"
    return redirect(url_for("dashboard"))


@app.route('/')
def dashboard():
    if USE_AUTH and 'user' not in session:
        return redirect(url_for('login'))
    return render_template('index.html', auth_enabled=USE_AUTH)


@app.route('/api/manual-predict', methods=['POST'])
def manual_predict():
    try:
        payload = validate_manual_payload(request.form)
        analysis = analyze_cbc(**payload)
        return jsonify({"success": True, "analysis": analysis})
    except UserInputError as exc:
        return json_error(str(exc), 400)
    except Exception:
        return json_error('We could not analyze the CBC values right now. Please try again.', 500)


@app.route("/logout")
def logout():
    session.clear()
    return redirect(url_for("dashboard"))


@app.route('/predict-image', methods=['POST'])
def upload_image():
    uploaded_file = request.files.get('image')
    if uploaded_file is None or uploaded_file.filename == '':
        return json_error('Please upload a blood report image to continue.', 400)

    try:
        extracted = process_image(uploaded_file.read())
    except OCRProcessingError:
        return json_error('Could not read report. Please enter values manually.', 422)
    except Exception:
        return json_error('The report could not be processed. Please try another image.', 500)

    fields = {
        "gender": extracted.get("gender") or "",
        "hemoglobin": extracted.get("hemoglobin"),
        "mcv": extracted.get("mcv"),
        "mch": extracted.get("mch"),
        "mchc": extracted.get("mchc"),
        "rbc": extracted.get("rbc"),
    }

    missing = [
        field_name
        for field_name in ("hemoglobin", "mcv", "mch", "mchc", "gender")
        if not fields.get(field_name)
    ]

    analysis = None
    message = 'CBC values extracted. Please review the fields and continue.'
    if missing:
        if len(missing) == 1 and missing[0] == "gender":
            message = 'CBC values were extracted, but gender was not detected. Please select it manually.'
        else:
            message = 'Some values could not be read confidently. Please review and complete the missing fields manually.'
    else:
        analysis = analyze_cbc(
            hemoglobin=float(fields["hemoglobin"]),
            mcv=float(fields["mcv"]),
            mch=float(fields["mch"]),
            mchc=float(fields["mchc"]),
            gender=str(fields["gender"]),
            rbc=float(fields["rbc"]) if fields["rbc"] is not None else None,
        )
        message = 'Report values extracted and copied to the CBC form for verification.'

    return jsonify({
        "success": True,
        "fields": fields,
        "analysis": analysis,
        "missing_fields": missing,
        "message": message,
    })


@app.route('/api/vision-predict', methods=['POST'])
def vision_predict():
    uploads = {
        "palm": request.files.get('palm'),
        "nail": request.files.get('nail'),
        "conjunctiva": request.files.get('conjunctiva'),
    }
    provided_files = {name: upload for name, upload in uploads.items() if upload and upload.filename}

    if not provided_files:
        return json_error('Upload at least one palm, nail, or conjunctiva image.', 400)

    individual_results: dict[str, Any] = {}
    weighted_probability = 0.0
    applied_weight = 0.0

    for region, file_storage in provided_files.items():
        try:
            result = estimate_region_probability(region, file_storage.read())
        except UnidentifiedImageError:
            return json_error(f'The {region} file is not a valid image.', 400)
        except Exception:
            return json_error(f'We could not analyze the {region} image. Please try another photo.', 500)

        individual_results[region] = result
        region_weight = VISION_CONFIG[region]["weight"]
        weighted_probability += result["probability"] * region_weight
        applied_weight += region_weight

    combined_probability = weighted_probability / applied_weight if applied_weight else 0.0
    is_anemic = combined_probability >= 0.5
    confidence = clamp(0.55 + abs(combined_probability - 0.5) * 0.9 + (0.05 * max(len(provided_files) - 1, 0)), 0.55, 0.97)

    return jsonify({
        "success": True,
        "prediction": 'Anemic' if is_anemic else 'Non-Anemic',
        "combined_probability": round(float(combined_probability), 4),
        "confidence": round(float(confidence), 4),
        "individual_results": individual_results,
        "backend": VISION_MODEL_HINT,
        "input_count": len(provided_files),
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5001))
    app.run(host="0.0.0.0", port=port, debug=True)
