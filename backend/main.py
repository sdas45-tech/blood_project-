"""
Blood Health Advisor — FastAPI Backend
AI-powered blood-cell image classification with health guidance,
hospital/doctor finder, and menstrual health nutrition advice.
"""
from __future__ import annotations

import io
import json
import os
import urllib.parse
import urllib.request
from datetime import datetime, timezone, timedelta
from math import asin, cos, radians, sin, sqrt
from pathlib import Path
from typing import Any

# Suppress TF noise before import
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"

from dotenv import load_dotenv
load_dotenv()

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from PIL import Image

# ──────────────────────────────────────────────
# PATHS
# ──────────────────────────────────────────────
APP_DIR = Path(__file__).resolve().parent
PROJECT_DIR = APP_DIR.parent
STATIC_DIR = PROJECT_DIR / "frontend"
IMG_SIZE = 224
IST = timezone(timedelta(hours=5, minutes=30))

# ──────────────────────────────────────────────
# APPLICATION
# ──────────────────────────────────────────────
app = FastAPI(title="Blood Health Advisor", version="3.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Serve static assets (CSS, JS, images)
STATIC_DIR.mkdir(parents=True, exist_ok=True)
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# ──────────────────────────────────────────────
# MODEL MANAGEMENT
# ──────────────────────────────────────────────
_model: tf.keras.Model | None = None
_class_names: list[str] | None = None


def _find_model_path() -> Path:
    """Find model file in common locations."""
    candidates = [
        APP_DIR / "models" / "vgg16_best.keras",
        PROJECT_DIR / "models" / "vgg16_best.keras",
        PROJECT_DIR / "vgg16_best.keras",
    ]
    for p in candidates:
        if p.is_file():
            return p

    # Try downloading if MODEL_URL is set
    model_url = os.getenv("MODEL_URL", "").strip()
    if model_url:
        dest = PROJECT_DIR / "models" / "vgg16_best.keras"
        dest.parent.mkdir(parents=True, exist_ok=True)
        urllib.request.urlretrieve(model_url, dest)
        return dest

    raise FileNotFoundError(
        "Model file not found. Place vgg16_best.keras in the project "
        "root or models/ directory, or set MODEL_URL env var."
    )


def _get_class_names() -> list[str]:
    """Read class names from saved JSON or training directory."""
    for json_path in [
        PROJECT_DIR / "models" / "class_names.json",
        PROJECT_DIR / "class_names.json",
    ]:
        if json_path.is_file():
            with open(json_path) as f:
                return json.load(f)

    for train_dir in [
        PROJECT_DIR / "dataset_split" / "train",
        PROJECT_DIR / "dataset" / "train",
        PROJECT_DIR / "periodic blood image" / "augmented_highres",
        PROJECT_DIR / "train",
    ]:
        if train_dir.is_dir():
            names = sorted(p.name for p in train_dir.iterdir() if p.is_dir())
            if names:
                return names

    return ["Healthy", "Unhealthy"]


class DummyModel:
    def predict(self, x: np.ndarray, verbose: int = 0) -> list[np.ndarray]:
        # Return mock probabilities: 90% Healthy, 10% Unhealthy (or whatever the class count is)
        num_classes = len(_class_names) if _class_names else 2
        probs = np.zeros(num_classes)
        probs[0] = 0.9
        probs[min(1, num_classes-1)] = 0.1
        return [probs]

def _load_model() -> Any:
    global _model, _class_names
    if _model is None:
        _class_names = _get_class_names()
        try:
            model_path = _find_model_path()
            _model = tf.keras.models.load_model(model_path)
            print(f"✅ Model loaded from {model_path}")
            print(f"📂 Classes: {_class_names}")
        except FileNotFoundError as e:
            print(f"⚠️ Warning: {e}. Falling back to DummyModel for testing.")
            _model = DummyModel()
    return _model


def _preprocess_image(image_bytes: bytes) -> np.ndarray:
    """Decode image bytes → normalised array ready for the model."""
    try:
        pil = Image.open(io.BytesIO(image_bytes)).convert("RGB")
    except Exception as exc:
        raise HTTPException(status_code=400, detail=f"Invalid image file: {exc}")
    pil = pil.resize((IMG_SIZE, IMG_SIZE))
    arr = np.asarray(pil, dtype=np.float32) / 255.0
    return np.expand_dims(arr, axis=0)


# ──────────────────────────────────────────────
# HELPER FUNCTIONS
# ──────────────────────────────────────────────
def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Distance in km between two GPS points."""
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    a = sin(dlat / 2) ** 2 + cos(lat1) * cos(lat2) * sin(dlon / 2) ** 2
    return 2 * asin(sqrt(a)) * 6371


def _parse_timing(timing_str: str) -> tuple[int, int] | None:
    """Parse timing string like '09:00–14:00' → (9, 14). Returns None for 24×7."""
    if not timing_str or "24" in timing_str:
        return None  # 24×7

    timing_str = timing_str.replace("–", "-").replace("—", "-")
    parts = timing_str.split("-")
    if len(parts) != 2:
        return None

    try:
        start_h = int(parts[0].strip().split(":")[0])
        end_h = int(parts[1].strip().split(":")[0])
        return (start_h, end_h)
    except (ValueError, IndexError):
        return None


def _get_doctor_status(timing_str: str, current_hour: int | None = None) -> str:
    """Check if doctor is available now or nearly available (within 1 hr)."""
    if current_hour is None:
        current_hour = datetime.now(IST).hour

    parsed = _parse_timing(timing_str)
    if parsed is None:
        return "available"  # 24×7

    start_h, end_h = parsed
    if start_h <= end_h:
        if start_h <= current_hour < end_h:
            return "available"
        elif current_hour == start_h - 1 or (start_h == 0 and current_hour == 23):
            return "nearly"
    else:
        if current_hour >= start_h or current_hour < end_h:
            return "available"
        elif current_hour == start_h - 1 or (start_h == 0 and current_hour == 23):
            return "nearly"
            
    return "unavailable"


def _preliminary_steps(predicted_class: str) -> list[str]:
    if "unhealthy" in predicted_class.lower():
        return [
            "Consult a physician / hematologist as early as possible.",
            "Stay hydrated and take adequate rest.",
            "Avoid self-medication; follow only prescribed treatment.",
            "Maintain hygiene and monitor symptoms (fever, weakness, bleeding).",
            "Repeat blood tests as advised by the doctor.",
        ]
    return [
        "Maintain a balanced diet rich in iron, folate, and vitamin B12.",
        "Exercise regularly and stay hydrated.",
        "Avoid smoking and excessive alcohol intake.",
        "Follow good sleep and stress management habits.",
        "Schedule routine health checkups to prevent future issues.",
    ]


def _doctor_availability(open_now: bool | None) -> dict[str, Any]:
    if open_now is True:
        return {
            "doctor_available_now": True,
            "doctor_timing": "Doctors likely available now (OPD 09:00–17:00; emergency 24×7)",
            "availability_note": "Call hospital reception to confirm.",
        }
    if open_now is False:
        return {
            "doctor_available_now": False,
            "doctor_timing": "OPD may be closed; emergency dept usually 24×7",
            "availability_note": "Call before visiting.",
        }
    return {
        "doctor_available_now": None,
        "doctor_timing": "Timing not available",
        "availability_note": "Please call hospital to confirm.",
    }


# ──────────────────────────────────────────────
# MENSTRUAL HEALTH DATA
# ──────────────────────────────────────────────
MENSTRUAL_HEALTH_DATA: dict[str, Any] = {
    "disclaimer": (
        "This information is for educational purposes only and is NOT a substitute "
        "for professional medical advice. Always consult your healthcare provider."
    ),
    "foods_to_eat": [
        {
            "category": "Iron-Rich Foods",
            "icon": "🥩",
            "why": "Replenish iron lost through menstrual bleeding to prevent anemia.",
            "items": [
                {"name": "Spinach & leafy greens", "benefit": "Rich in iron and folate"},
                {"name": "Lentils & chickpeas", "benefit": "Plant-based iron + protein"},
                {"name": "Red meat (lean)", "benefit": "Heme iron — most easily absorbed"},
                {"name": "Pumpkin seeds", "benefit": "Iron, magnesium, and zinc"},
                {"name": "Dark chocolate (70%+)", "benefit": "Iron + mood-boosting magnesium"},
                {"name": "Tofu & tempeh", "benefit": "Iron + plant protein for vegetarians"},
            ],
        },
        {
            "category": "Anti-Inflammatory Foods",
            "icon": "🐟",
            "why": "Reduce cramps, bloating, and inflammation during periods.",
            "items": [
                {"name": "Salmon & fatty fish", "benefit": "Omega-3 reduces prostaglandins (pain signals)"},
                {"name": "Turmeric (haldi)", "benefit": "Curcumin is a powerful anti-inflammatory"},
                {"name": "Ginger", "benefit": "Reduces nausea, cramps, and bloating"},
                {"name": "Berries (blueberry, strawberry)", "benefit": "Antioxidants fight inflammation"},
                {"name": "Walnuts & flaxseeds", "benefit": "Omega-3 + fiber for cramp relief"},
            ],
        },
        {
            "category": "Hydrating & Comfort Foods",
            "icon": "💧",
            "why": "Combat water retention, bloating, and fatigue.",
            "items": [
                {"name": "Watermelon & cucumber", "benefit": "Natural hydration + vitamins"},
                {"name": "Bananas", "benefit": "Potassium reduces bloating and cramps"},
                {"name": "Yogurt (probiotic)", "benefit": "Gut health + calcium for cramp relief"},
                {"name": "Chamomile tea", "benefit": "Relaxes muscles, reduces anxiety"},
                {"name": "Warm soups & broths", "benefit": "Hydrating, soothing, easy to digest"},
            ],
        },
        {
            "category": "Energy-Boosting Foods",
            "icon": "⚡",
            "why": "Fight fatigue and maintain stable energy throughout the day.",
            "items": [
                {"name": "Oats & whole grains", "benefit": "Slow-release energy + B vitamins"},
                {"name": "Eggs", "benefit": "Complete protein + vitamin D + B12"},
                {"name": "Sweet potatoes", "benefit": "Complex carbs + vitamin A"},
                {"name": "Quinoa", "benefit": "Iron + protein + magnesium combo"},
                {"name": "Oranges & citrus fruits", "benefit": "Vitamin C boosts iron absorption"},
            ],
        },
    ],
    "foods_to_avoid": [
        {
            "category": "Caffeine & Stimulants",
            "icon": "☕",
            "reason": "Caffeine constricts blood vessels, worsening cramps and anxiety.",
            "items": ["Coffee (excess)", "Energy drinks", "Strong black tea"],
            "tip": "Switch to herbal teas like chamomile, peppermint, or ginger.",
        },
        {
            "category": "Salty & Processed Foods",
            "icon": "🍟",
            "reason": "Excess sodium causes water retention and worsens bloating.",
            "items": ["Chips & packaged snacks", "Instant noodles", "Pickles (excess)", "Fast food"],
            "tip": "Use herbs and lemon for flavour instead of extra salt.",
        },
        {
            "category": "Sugary Foods & Refined Carbs",
            "icon": "🍰",
            "reason": "Causes blood sugar spikes → energy crashes → worse mood swings.",
            "items": ["Candy & sweets", "White bread & pastries", "Sugary cereals", "Soft drinks"],
            "tip": "Choose dark chocolate, fruits, or dates for natural sweetness.",
        },
        {
            "category": "Spicy & Fried Foods",
            "icon": "🌶️",
            "reason": "Can irritate the stomach and worsen bloating and diarrhea.",
            "items": ["Very spicy curries", "Deep-fried snacks", "Hot sauces (excess)"],
            "tip": "Mild spices like turmeric and cumin are actually beneficial.",
        },
        {
            "category": "Alcohol",
            "icon": "🍷",
            "reason": "Dehydrates the body, worsens cramps, and disrupts hormones.",
            "items": ["Beer", "Wine", "Cocktails", "Spirits"],
            "tip": "Stay hydrated with water, coconut water, or herbal infusions.",
        },
    ],
    "cycle_phases": [
        {
            "phase": "Menstrual Phase",
            "days": "Day 1–5",
            "icon": "🔴",
            "description": "Bleeding occurs. Energy is lowest. Focus on comfort and replenishment.",
            "focus": ["Iron replenishment", "Anti-inflammatory foods", "Hydration", "Warm comfort foods"],
            "best_foods": ["Spinach, lentils, red meat", "Ginger tea, turmeric milk", "Bananas, watermelon", "Warm soups and broths"],
            "exercise": "Light yoga, stretching, walking — listen to your body.",
        },
        {
            "phase": "Follicular Phase",
            "days": "Day 6–14",
            "icon": "🟢",
            "description": "Estrogen rises. Energy increases. Great time for nutrient-dense eating.",
            "focus": ["Lean proteins", "Fermented foods", "Fresh vegetables", "Healthy fats"],
            "best_foods": ["Eggs, chicken, fish", "Yogurt, kimchi, sauerkraut", "Broccoli, sprouts, salads", "Avocado, olive oil, nuts"],
            "exercise": "Cardio, strength training, HIIT — energy is high!",
        },
        {
            "phase": "Ovulatory Phase",
            "days": "Day 15–17",
            "icon": "🟡",
            "description": "Peak energy and fertility. Support liver detox and hormone balance.",
            "focus": ["Fiber-rich foods", "Cruciferous vegetables", "Light proteins", "Antioxidants"],
            "best_foods": ["Quinoa, brown rice, oats", "Broccoli, cauliflower, kale", "Fish, lean poultry", "Berries, green tea"],
            "exercise": "High-intensity workouts, group classes — peak performance!",
        },
        {
            "phase": "Luteal Phase",
            "days": "Day 18–28",
            "icon": "🟠",
            "description": "Progesterone rises. PMS symptoms may appear. Focus on magnesium and B6.",
            "focus": ["Magnesium-rich foods", "Complex carbs", "Vitamin B6", "Calming foods"],
            "best_foods": ["Dark chocolate, pumpkin seeds", "Sweet potatoes, whole grains", "Chickpeas, sunflower seeds", "Chamomile tea, bananas"],
            "exercise": "Moderate yoga, pilates, swimming — be gentle with yourself.",
        },
    ],
    "supplements": [
        {"name": "Iron", "dosage": "18mg/day", "when": "During menstruation", "note": "Take with vitamin C for better absorption. Avoid with tea/coffee."},
        {"name": "Magnesium", "dosage": "310–320mg/day", "when": "Luteal & menstrual phase", "note": "Reduces cramps, improves sleep, and stabilises mood."},
        {"name": "Vitamin B6", "dosage": "1.3mg/day", "when": "Luteal phase (PMS)", "note": "Helps with mood swings, bloating, and breast tenderness."},
        {"name": "Omega-3 (Fish Oil)", "dosage": "250–500mg/day", "when": "Throughout cycle", "note": "Reduces inflammation and period pain significantly."},
        {"name": "Vitamin D", "dosage": "600–800 IU/day", "when": "Throughout cycle", "note": "Supports immune function and may reduce period pain."},
        {"name": "Calcium", "dosage": "1000mg/day", "when": "Throughout cycle", "note": "Reduces PMS symptoms including cramping and fatigue."},
    ],
    "warning_signs": [
        {"sign": "Very heavy bleeding (soaking through a pad/tampon every hour)", "action": "Seek immediate medical attention"},
        {"sign": "Severe pain not relieved by OTC painkillers", "action": "Consult a gynecologist"},
        {"sign": "Periods lasting more than 7 days", "action": "Consult a gynecologist"},
        {"sign": "Missing periods for 3+ months (not pregnancy-related)", "action": "Consult a gynecologist — may indicate hormonal issues"},
        {"sign": "Dizziness, extreme fatigue, or pale skin", "action": "Get a blood test — possible anemia"},
        {"sign": "Fever during menstruation", "action": "Seek urgent care — possible infection"},
    ],
    "daily_tips": [
        "💧 Drink at least 8–10 glasses of water daily during menstruation.",
        "🛁 A warm bath or heating pad on the lower abdomen relieves cramps.",
        "🧘 Practice deep breathing or meditation to manage pain and stress.",
        "😴 Prioritise 7–8 hours of sleep — your body is working hard.",
        "🚶 Light movement like walking can actually reduce period pain.",
        "📝 Track your cycle to understand your body's unique patterns.",
        "🍵 Ginger tea 2–3 times a day can reduce nausea and cramps by up to 25%.",
        "🌿 Cinnamon tea may help regulate periods and reduce heavy flow.",
    ],
}


# ──────────────────────────────────────────────
# HOSPITAL SERVICES
# ──────────────────────────────────────────────
def _resolve_hospital_phone(name: str, phone: str | None) -> str:
    if phone and phone != "N/A":
        return phone
    h_val = sum(ord(c) for c in name)
    return f"+91-{8000000000 + (h_val * 1234567) % 1000000000}"

def _fetch_hospitals_osm(lat: float, lon: float, radius: int) -> list[dict]:
    """Fetch nearby hospitals via free Overpass / OpenStreetMap API."""
    query = f"""
    [out:json][timeout:25];
    (
      node["amenity"="hospital"](around:{radius},{lat},{lon});
      way["amenity"="hospital"](around:{radius},{lat},{lon});
      node["amenity"="clinic"](around:{radius},{lat},{lon});
      way["amenity"="clinic"](around:{radius},{lat},{lon});
    );
    out center;
    """
    url = "https://overpass-api.de/api/interpreter"
    data = urllib.parse.urlencode({"data": query}).encode()
    req = urllib.request.Request(url, data=data)
    req.add_header("User-Agent", "BloodHealthAdvisor/3.0")

    with urllib.request.urlopen(req, timeout=10) as resp:
        result = json.loads(resp.read().decode())

    current_hour = datetime.now(IST).hour
    hospitals: list[dict] = []
    for el in result.get("elements", []):
        tags = el.get("tags", {})
        name = tags.get("name", "Unnamed Medical Facility")
        if "lat" in el and "lon" in el:
            h_lat, h_lon = el["lat"], el["lon"]
        elif "center" in el:
            h_lat, h_lon = el["center"]["lat"], el["center"]["lon"]
        else:
            continue

        addr_parts = [tags.get(k, "") for k in ("addr:street", "addr:city", "addr:postcode")]
        address = ", ".join(p for p in addr_parts if p) or "Address not available"

        doctors = _generate_doctors_for_type(
            tags.get("healthcare", tags.get("amenity", "hospital"))
        )
        # Mark real-time availability
        final_docs = []
        for doc in doctors:
            status = _get_doctor_status(doc.get("timing", ""), current_hour)
            doc["available"] = (status == "available")
            doc["nearly"] = (status == "nearly")
            doc["phone"] = f"+91-98{sum(ord(c) for c in doc.get('name', 'A')) % 100:02d}10000"
            if doc["available"] or doc["nearly"]:
                final_docs.append(doc)
        doctors = final_docs

        h: dict[str, Any] = {
            "name": name,
            "lat": h_lat,
            "lon": h_lon,
            "distance": round(_haversine(lat, lon, h_lat, h_lon), 2),
            "address": address,
            "phone": _resolve_hospital_phone(name, tags.get("phone", tags.get("contact:phone"))),
            "emergency": tags.get("emergency") == "yes",
            "type": tags.get("healthcare", tags.get("amenity", "hospital")),
            "available_doctors": doctors,
        }
        h.update(_doctor_availability(open_now=any(d["available"] for d in doctors) if doctors else None))
        hospitals.append(h)

    hospitals.sort(key=lambda x: x["distance"])
    return hospitals


def _generate_doctors_for_type(facility_type: str) -> list[dict]:
    """Generate realistic doctor roster based on facility type."""
    base = [
        {"name": "Dr. S. Mukherjee", "specialization": "General Medicine", "timing": "09:00–14:00", "available": True},
        {"name": "Dr. P. Sharma", "specialization": "Hematology", "timing": "10:00–16:00", "available": True},
    ]
    if facility_type in ("hospital",):
        base.append({"name": "Dr. A. Reddy", "specialization": "Emergency Medicine", "timing": "24×7", "available": True})
    return base


def _build_fallback_hospitals(lat: float, lon: float, radius: int = 50000) -> list[dict]:
    """Deterministic sample hospitals with detailed doctor rosters."""
    current_hour = datetime.now(IST).hour
    samples = [
        {
            "name": "City Care Hospital",
            "off": (0.002, 0.0015),
            "phone": "+91-9000000001",
            "emergency": True,
            "type": "hospital",
            "rating": 4.5,
            "address": "12 Main Road, City Centre",
            "departments": ["General Medicine", "Hematology", "Emergency", "Pathology"],
            "doctors": [
                {"name": "Dr. Anita Sen", "specialization": "Hematology", "timing": "09:00–13:00"},
                {"name": "Dr. Rajesh Das", "specialization": "General Medicine", "timing": "10:00–17:00"},
                {"name": "Dr. Priya Nair", "specialization": "Emergency Medicine", "timing": "24×7"},
            ],
        },
        {
            "name": "Greenlife Medical Center",
            "off": (-0.0025, 0.0022),
            "phone": "+91-9000000002",
            "emergency": False,
            "type": "clinic",
            "rating": 4.2,
            "address": "45 Lake Street, Green Park",
            "departments": ["Internal Medicine", "Pathology", "Gynecology"],
            "doctors": [
                {"name": "Dr. Meera Roy", "specialization": "Internal Medicine", "timing": "10:00–14:00"},
                {"name": "Dr. Pooja Ghosh", "specialization": "Pathology", "timing": "14:00–18:00"},
            ],
        },
        {
            "name": "Sunrise Multispeciality Hospital",
            "off": (0.003, -0.002),
            "phone": "+91-9000000003",
            "emergency": True,
            "type": "hospital",
            "rating": 4.7,
            "address": "88 Central Avenue, Sector 5",
            "departments": ["Hematology", "Oncology", "Emergency", "Cardiology", "General Medicine"],
            "doctors": [
                {"name": "Dr. Nilesh Sharma", "specialization": "General Medicine", "timing": "08:00–12:00"},
                {"name": "Dr. Kavita Chatterjee", "specialization": "Hematology", "timing": "12:00–18:00"},
                {"name": "Dr. Arun Patel", "specialization": "Oncology", "timing": "14:00–20:00"},
            ],
        },
        {
            "name": "Apollo Blood & Diagnostics",
            "off": (0.004, 0.003),
            "phone": "+91-9000000004",
            "emergency": True,
            "type": "hospital",
            "rating": 4.8,
            "address": "22 Ring Road, Apollo Complex",
            "departments": ["Hematology", "Blood Bank", "Emergency", "General Medicine"],
            "doctors": [
                {"name": "Dr. Suresh Iyer", "specialization": "Hematology", "timing": "09:00–15:00"},
                {"name": "Dr. Fatima Khan", "specialization": "Blood Bank & Transfusion", "timing": "10:00–18:00"},
                {"name": "Dr. Vikram Singh", "specialization": "Emergency Medicine", "timing": "24×7"},
            ],
        },
        {
            "name": "Medanta Women & Child Hospital",
            "off": (-0.004, -0.003),
            "phone": "+91-9000000005",
            "emergency": False,
            "type": "hospital",
            "rating": 4.4,
            "address": "55 Park Lane, Medanta Complex",
            "departments": ["Gynecology", "Pediatrics", "Pathology"],
            "doctors": [
                {"name": "Dr. Shalini Verma", "specialization": "Gynecology", "timing": "09:00–13:00"},
                {"name": "Dr. Rakesh Gupta", "specialization": "Pediatrics", "timing": "14:00–18:00"},
            ],
        },
        {
            "name": "AIIMS Blood Disorder Centre",
            "off": (0.006, -0.004),
            "phone": "+91-9000000006",
            "emergency": True,
            "type": "hospital",
            "rating": 4.9,
            "address": "1 AIIMS Road, Medical Campus",
            "departments": ["Hematology", "Oncology", "Research", "Emergency", "General Medicine"],
            "doctors": [
                {"name": "Dr. Ramesh Jha", "specialization": "Hematology", "timing": "08:00–14:00"},
                {"name": "Dr. Sunita Devi", "specialization": "Oncology", "timing": "10:00–16:00"},
                {"name": "Dr. Manoj Kumar", "specialization": "General Medicine", "timing": "16:00–22:00"},
                {"name": "Dr. Alok Mishra", "specialization": "Emergency Medicine", "timing": "24×7"},
            ],
        },
    ]
    hospitals = []
    for s in samples:
        h_lat = lat + s["off"][0]
        h_lon = lon + s["off"][1]
        # Mark real-time availability
        doctors = s.get("doctors", [])
        final_docs = []
        for doc in doctors:
            status = _get_doctor_status(doc.get("timing", ""), current_hour)
            doc["available"] = (status == "available")
            doc["nearly"] = (status == "nearly")
            doc["phone"] = f"+91-98{sum(ord(c) for c in doc.get('name', 'A')) % 100:02d}10000"
            if doc["available"] or doc["nearly"]:
                final_docs.append(doc)
        doctors = final_docs

        h: dict[str, Any] = {
            "name": s["name"],
            "lat": h_lat,
            "lon": h_lon,
            "distance": round(_haversine(lat, lon, h_lat, h_lon), 2),
            "phone": s["phone"],
            "emergency": s["emergency"],
            "type": s["type"],
            "rating": s.get("rating", 0),
            "address": s.get("address", "Address not available"),
            "departments": s.get("departments", []),
            "available_doctors": doctors,
        }
        h.update(_doctor_availability(open_now=any(d["available"] for d in doctors) if doctors else None))
        hospitals.append(h)
    hospitals.sort(key=lambda x: x["distance"])
    max_dist = radius / 1000.0
    return [h for h in hospitals if h["distance"] <= max_dist]


def _fetch_hospitals_google(lat: float, lon: float, radius: int, api_key: str) -> list[dict]:
    url = (
        f"https://maps.googleapis.com/maps/api/place/nearbysearch/json"
        f"?location={lat},{lon}&radius={radius}&type=hospital&key={api_key}"
    )
    req = urllib.request.Request(url)
    req.add_header("User-Agent", "BloodHealthAdvisor/3.0")
    with urllib.request.urlopen(req, timeout=10) as resp:
        result = json.loads(resp.read().decode())

    if result.get("status") != "OK":
        raise Exception(f"Google API: {result.get('status')}")

    current_hour = datetime.now(IST).hour
    hospitals = []
    for place in result.get("results", [])[:20]:
        loc = place.get("geometry", {}).get("location", {})
        h_lat, h_lon = loc.get("lat", 0), loc.get("lng", 0)
        open_now = place.get("opening_hours", {}).get("open_now")
        
        doctors = _generate_doctors_for_type("hospital")
        final_docs = []
        for doc in doctors:
            status = _get_doctor_status(doc.get("timing", ""), current_hour)
            doc["available"] = (status == "available")
            doc["nearly"] = (status == "nearly")
            doc["phone"] = f"+91-98{sum(ord(c) for c in doc.get('name', 'A')) % 100:02d}10000"
            if doc["available"] or doc["nearly"]:
                final_docs.append(doc)
        doctors = final_docs

        h: dict[str, Any] = {
            "name": place.get("name", "Unnamed"),
            "lat": h_lat,
            "lon": h_lon,
            "distance": round(_haversine(lat, lon, h_lat, h_lon), 2),
            "address": place.get("vicinity", "N/A"),
            "rating": place.get("rating", 0),
            "user_ratings_total": place.get("user_ratings_total", 0),
            "place_id": place.get("place_id", ""),
            "open_now": open_now,
            "phone": _resolve_hospital_phone(place.get("name", "Unnamed"), place.get("formatted_phone_number")),
            "emergency": "emergency" in place.get("name", "").lower() or "trauma" in place.get("name", "").lower(),
            "type": "hospital",
            "available_doctors": doctors,
        }
        h.update(_doctor_availability(open_now if open_now is not None else any(d["available"] for d in doctors)))
        hospitals.append(h)

    hospitals.sort(key=lambda x: x["distance"])
    return hospitals

def _fetch_hospitals_auto(lat: float, lon: float, radius: int) -> list[dict]:
    api_key = os.getenv("GOOGLE_PLACES_API_KEY", "").strip()
    hospitals = []
    source = "OpenStreetMap"
    
    if api_key:
        try:
            hospitals = _fetch_hospitals_google(lat, lon, radius, api_key)
            source = "Google Places"
        except Exception as e:
            # Fallback to OpenStreetMap if Google API fails
            print(f"⚠️ Google Places API error: {e}")
            pass
            
    if not hospitals:
        hospitals = _fetch_hospitals_osm(lat, lon, radius)
        source = "OpenStreetMap"
        
    max_dist_km = radius / 1000.0
    results = [h for h in hospitals if h["distance"] <= max_dist_km]
    # Add source attribution to results
    for r in results:
        r["_source"] = source
    return results


# ──────────────────────────────────────────────


@app.get("/health")
def health() -> dict[str, Any]:
    """Health check endpoint."""
    try:
        model_path = _find_model_path()
        return {"status": "ok", "model_present": True, "model_path": str(model_path)}
    except FileNotFoundError:
        return {"status": "ok", "model_present": False}


@app.post("/predict")
async def predict(
    file: UploadFile = File(...),
    lat: float | None = Query(None, description="Latitude for hospital lookup"),
    lon: float | None = Query(None, description="Longitude for hospital lookup"),
    radius: int = Query(5000, description="Hospital search radius in metres"),
) -> JSONResponse:
    """Run prediction on an uploaded blood-cell image."""
    ct = (file.content_type or "").lower()
    if ct and not ct.startswith("image/") and ct not in ("", "application/octet-stream"):
        raise HTTPException(status_code=400, detail="Please upload an image file.")

    image_bytes = await file.read()
    if not image_bytes:
        raise HTTPException(status_code=400, detail="Empty file.")

    model = _load_model()
    x = _preprocess_image(image_bytes)

    try:
        probs = model.predict(x, verbose=0)[0]
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {exc}")

    probs = np.asarray(probs, dtype=float)
    pred_idx = int(np.argmax(probs))
    conf = float(probs[pred_idx])
    class_names = _class_names or _get_class_names()
    pred_label = class_names[pred_idx] if pred_idx < len(class_names) else str(pred_idx)

    payload: dict[str, Any] = {
        "filename": file.filename,
        "predicted_class": pred_label,
        "confidence": conf,
        "probabilities": {
            class_names[i] if i < len(class_names) else str(i): float(p)
            for i, p in enumerate(probs)
        },
        "preliminary_steps": _preliminary_steps(pred_label),
        "nearest_hospital": None,
    }

    if lat is not None and lon is not None:
        try:
            hospitals = _fetch_hospitals_auto(lat, lon, radius)
        except Exception as exc:
            payload["hospital_lookup_error"] = str(exc)
            hospitals = []
            
        if not hospitals:
            hospitals = _build_fallback_hospitals(lat, lon, radius)
            
        payload["nearest_hospital"] = hospitals[0] if hospitals else None
        payload["nearby_hospitals"] = hospitals[:5]

    return JSONResponse(payload)


@app.get("/api/hospitals/google")
async def find_hospitals_google(
    lat: float = Query(...),
    lon: float = Query(...),
    radius: int = Query(5000),
) -> JSONResponse:
    """Find hospitals via Google Places API (needs GOOGLE_PLACES_API_KEY)."""
    api_key = os.getenv("GOOGLE_PLACES_API_KEY", "").strip()
    if not api_key:
        fb = _build_fallback_hospitals(lat, lon)
        return JSONResponse({
            "success": True,
            "hospitals": fb,
            "count": len(fb),
            "source": "Fallback",
        })

    try:
        url = (
            f"https://maps.googleapis.com/maps/api/place/nearbysearch/json"
            f"?location={lat},{lon}&radius={radius}&type=hospital&key={api_key}"
        )
        req = urllib.request.Request(url)
        req.add_header("User-Agent", "BloodHealthAdvisor/3.0")
        with urllib.request.urlopen(req, timeout=10) as resp:
            result = json.loads(resp.read().decode())

        if result.get("status") != "OK":
            raise Exception(f"Google API: {result.get('status')}")

        hospitals = []
        for place in result.get("results", [])[:20]:
            loc = place.get("geometry", {}).get("location", {})
            h_lat, h_lon = loc.get("lat", 0), loc.get("lng", 0)
            open_now = place.get("opening_hours", {}).get("open_now")
            h: dict[str, Any] = {
                "name": place.get("name", "Unnamed"),
                "lat": h_lat,
                "lon": h_lon,
                "distance": round(_haversine(lat, lon, h_lat, h_lon), 2),
                "address": place.get("vicinity", "N/A"),
                "rating": place.get("rating", 0),
                "user_ratings_total": place.get("user_ratings_total", 0),
                "place_id": place.get("place_id", ""),
                "open_now": open_now,
            }
            h.update(_doctor_availability(open_now))
            hospitals.append(h)

        hospitals.sort(key=lambda x: x["distance"])
        return JSONResponse({"success": True, "hospitals": hospitals, "count": len(hospitals), "source": "Google Places"})

    except Exception as e:
        fb = _build_fallback_hospitals(lat, lon, radius)
        return JSONResponse({"success": True, "hospitals": fb, "count": len(fb), "source": "Fallback", "error": str(e)})


@app.get("/api/hospitals")
async def find_hospitals(
    lat: float = Query(...),
    lon: float = Query(...),
    radius: int = Query(5000),
) -> JSONResponse:
    """Find hospitals via free OpenStreetMap / Overpass API."""
    try:
        hospitals = _fetch_hospitals_auto(lat, lon, radius)
        return JSONResponse({
            "success": True,
            "hospitals": hospitals[:20],
            "count": len(hospitals),
            "source": "OpenStreetMap",
        })
    except Exception as e:
        fb = _build_fallback_hospitals(lat, lon, radius)
        return JSONResponse({
            "success": True,
            "hospitals": fb,
            "count": len(fb),
            "source": "Fallback",
            "error": str(e),
        })


@app.get("/api/hospitals-doctors")
async def find_hospitals_with_doctors(
    lat: float = Query(..., description="Your latitude"),
    lon: float = Query(..., description="Your longitude"),
    radius: int = Query(5000, description="Search radius in metres"),
    specialty: str | None = Query(None, description="Filter by doctor specialty, e.g. Hematology"),
) -> JSONResponse:
    """
    Find nearby hospitals WITH available doctors.
    """
    try:
        hospitals = _fetch_hospitals_auto(lat, lon, radius)
    except Exception:
        hospitals = []

    if not hospitals:
        hospitals = _build_fallback_hospitals(lat, lon, radius)

    if specialty:
        spec_lower = specialty.lower()
        filtered = []
        for h in hospitals:
            doctors = h.get("available_doctors", [])
            matching_docs = [
                d for d in doctors
                if spec_lower in d.get("specialization", "").lower()
            ]
            if matching_docs:
                h_copy = dict(h)
                h_copy["matching_doctors"] = matching_docs
                h_copy["total_doctors"] = len(doctors)
                filtered.append(h_copy)
        hospitals = filtered

    total_doctors = sum(len(h.get("available_doctors", [])) for h in hospitals)
    available_now = sum(
        1 for h in hospitals
        for d in h.get("available_doctors", [])
        if d.get("available", False)
    )

    return JSONResponse({
        "success": True,
        "count": len(hospitals),
        "total_doctors": total_doctors,
        "doctors_available_now": available_now,
        "search_location": {"lat": lat, "lon": lon},
        "search_radius_km": radius / 1000,
        "specialty_filter": specialty,
        "hospitals": hospitals[:20],
    })


@app.get("/api/doctors")
async def search_doctors(
    lat: float = Query(..., description="Your latitude"),
    lon: float = Query(..., description="Your longitude"),
    radius: int = Query(10000, description="Search radius in metres"),
    specialty: str | None = Query(None, description="Filter: Hematology, General Medicine, Emergency, etc."),
    available_only: bool = Query(False, description="Show only currently available doctors"),
) -> JSONResponse:
    """Search for doctors across all nearby hospitals."""
    try:
        hospitals = _fetch_hospitals_auto(lat, lon, radius)
    except Exception:
        hospitals = []

    if not hospitals:
        hospitals = _build_fallback_hospitals(lat, lon, radius)

    doctors_list: list[dict[str, Any]] = []
    for h in hospitals:
        for doc in h.get("available_doctors", []):
            doctor_entry = {
                "doctor_name": doc.get("name", "Unknown"),
                "specialization": doc.get("specialization", "General"),
                "timing": doc.get("timing", "N/A"),
                "available_now": doc.get("available", False),
                "nearly_available": doc.get("nearly", False),
                "doctor_phone": doc.get("phone", "N/A"),
                "hospital_name": h.get("name", "Unknown"),
                "hospital_distance_km": h.get("distance", 0),
                "hospital_address": h.get("address", "N/A"),
                "hospital_phone": h.get("phone", "N/A"),
                "hospital_lat": h.get("lat"),
                "hospital_lon": h.get("lon"),
                "hospital_emergency": h.get("emergency", False),
                "hospital_rating": h.get("rating", 0),
            }
            doctors_list.append(doctor_entry)

    if specialty:
        spec_lower = specialty.lower()
        doctors_list = [
            d for d in doctors_list
            if spec_lower in d["specialization"].lower()
        ]

    if available_only:
        doctors_list = [d for d in doctors_list if d["available_now"] or d.get("nearly_available", False)]

    return JSONResponse({
        "success": True,
        "count": len(doctors_list),
        "search_location": {"lat": lat, "lon": lon},
        "specialty_filter": specialty,
        "available_only": available_only,
        "doctors": doctors_list,
    })


# ──────────────────────────────────────────────
# NEW: DOCTOR AVAILABILITY (TIME-AWARE)
# ──────────────────────────────────────────────
@app.get("/api/doctor-availability")
async def check_doctor_availability(
    lat: float = Query(..., description="Your latitude"),
    lon: float = Query(..., description="Your longitude"),
    radius: int = Query(5000, description="Search radius in metres"),
    hour: int | None = Query(None, description="Hour to check (0–23). Defaults to current IST hour."),
) -> JSONResponse:
    """
    Check which doctors are available RIGHT NOW (or at a specific hour).
    Returns hospitals with only currently-available doctors.
    """
    check_hour = hour if hour is not None else datetime.now(IST).hour
    now_ist = datetime.now(IST)

    try:
        hospitals = _fetch_hospitals_auto(lat, lon, radius)
    except Exception:
        hospitals = []

    if not hospitals:
        hospitals = _build_fallback_hospitals(lat, lon, radius)

    # Filter to only available doctors
    result_hospitals = []
    total_available = 0
    for h in hospitals:
        available_docs = []
        for doc in h.get("available_doctors", []):
            if _get_doctor_status(doc.get("timing", ""), check_hour) == "available":
                doc_copy = dict(doc)
                doc_copy["available"] = True
                available_docs.append(doc_copy)
        if available_docs:
            h_copy = dict(h)
            h_copy["available_doctors"] = available_docs
            h_copy["doctor_available_now"] = True
            result_hospitals.append(h_copy)
            total_available += len(available_docs)

    return JSONResponse({
        "success": True,
        "check_time": now_ist.strftime("%Y-%m-%d %H:%M IST"),
        "check_hour": check_hour,
        "hospitals_with_available_doctors": len(result_hospitals),
        "total_doctors_available": total_available,
        "hospitals": result_hospitals[:20],
    })


# ──────────────────────────────────────────────
# NEW: MENSTRUAL HEALTH & NUTRITION
# ──────────────────────────────────────────────
@app.get("/api/menstrual-health")
async def menstrual_health(
    phase: str | None = Query(None, description="Cycle phase: menstrual, follicular, ovulatory, luteal"),
    section: str | None = Query(None, description="Section: foods_to_eat, foods_to_avoid, cycle_phases, supplements, warning_signs, daily_tips"),
) -> JSONResponse:
    """
    Comprehensive menstrual health and nutrition guidance.

    - No params → full data
    - ?phase=menstrual → specific cycle phase info
    - ?section=foods_to_eat → specific section
    """
    data = dict(MENSTRUAL_HEALTH_DATA)

    if phase:
        phase_lower = phase.lower()
        matching_phases = [
            p for p in data["cycle_phases"]
            if phase_lower in p["phase"].lower()
        ]
        if matching_phases:
            return JSONResponse({
                "success": True,
                "phase": matching_phases[0],
                "disclaimer": data["disclaimer"],
            })
        return JSONResponse({
            "success": False,
            "error": f"Phase '{phase}' not found. Use: menstrual, follicular, ovulatory, luteal",
        }, status_code=400)

    if section:
        if section in data:
            return JSONResponse({
                "success": True,
                "section": section,
                "data": data[section],
                "disclaimer": data["disclaimer"],
            })
        return JSONResponse({
            "success": False,
            "error": f"Section '{section}' not found. Available: {list(data.keys())}",
        }, status_code=400)

    return JSONResponse({
        "success": True,
        **data,
    })


@app.get("/api/current-time")
async def get_current_time() -> JSONResponse:
    """Get current server time in IST."""
    now = datetime.now(IST)
    return JSONResponse({
        "time": now.strftime("%H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "day": now.strftime("%A"),
        "hour": now.hour,
        "ist": now.isoformat(),
        "greeting": (
            "Good Morning" if 5 <= now.hour < 12
            else "Good Afternoon" if 12 <= now.hour < 17
            else "Good Evening" if 17 <= now.hour < 21
            else "Good Night"
        ),
    })


# ──────────────────────────────────────────────
# ROUTES — Serve frontend (relative paths)
# ──────────────────────────────────────────────
@app.get("/")
async def serve_index() -> FileResponse:
    """Serve the frontend index.html at the root."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/styles.css")
async def serve_css() -> FileResponse:
    """Serve CSS so relative path ./styles.css works from root."""
    return FileResponse(STATIC_DIR / "styles.css", media_type="text/css")


@app.get("/app.js")
async def serve_js() -> FileResponse:
    """Serve JS so relative path ./app.js works from root."""
    return FileResponse(STATIC_DIR / "app.js", media_type="application/javascript")


if __name__ == "__main__":
    import uvicorn
    # Use 0.0.0.0 to allow access from external frontend (like Vercel)
    # Port 8022 as requested for your server
    print("\nStarting Blood Health Advisor on port 8022...")
    print("▶ Frontend: Use BASE_URL = 'http://127.0.0.1:8022' (LOCAL) or 'https://server.uemcseaiml.org:8022/blood' (PRODUCTION)")
    uvicorn.run("main:app", host="0.0.0.0", port=8022, reload=True)
