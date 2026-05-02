"""
Student Stress Detector — Flask REST API
Run: python api/app.py
Then POST to http://localhost:5000/predict
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import joblib
import pandas as pd
import numpy as np
import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from train_model import StressPredictor, STRESS_LABELS

ROOT_DIR = os.path.dirname(os.path.dirname(__file__))
FRONTEND_DIST = os.path.join(ROOT_DIR, "frontend", "dist")

app = Flask(__name__, static_folder=FRONTEND_DIST, static_url_path="")
CORS(app)

MODEL_DIR = os.path.join(ROOT_DIR, "models")

# Load models at startup
predictor = StressPredictor(os.path.join(MODEL_DIR, "primary_model.pkl"))
with open(os.path.join(MODEL_DIR, "model_metadata.json")) as f:
    metadata = json.load(f)

SOLUTIONS_DB = {
    "sleep": {
        "title": "Improve your sleep schedule",
        "category": "Sleep hygiene",
        "body": "Poor sleep amplifies every other stress factor. A consistent 7-9 hour routine is one of the highest-ROI changes you can make.",
        "tips": ["Sleep & wake at the same time daily", "30-min screen-free wind-down ritual",
                 "No caffeine after 2 PM", "Keep room cool & dark (18–20°C)"]
    },
    "academics": {
        "title": "Reframe your academic approach",
        "category": "Academic strategy",
        "body": "High academic load often stems from poor planning. Structured time-blocking can cut perceived overwhelm significantly.",
        "tips": ["Pomodoro: 25 min on, 5 min off", "Weekly study plan every Sunday",
                 "Visit professor office hours early", "Form a 3-person study group"]
    },
    "exercise": {
        "title": "Add movement to your day",
        "category": "Physical wellness",
        "body": "Even 20–30 min of physical activity 3x/week reduces cortisol and dramatically improves mood and sleep quality.",
        "tips": ["Walk or cycle to class", "7-min HIIT workout at home",
                 "Join a campus sports club", "Stretch every 90 minutes of sitting"]
    },
    "mental": {
        "title": "Build mental health micro-routines",
        "category": "Mental health",
        "body": "Anxiety and low mood are common but not inevitable. Small daily practices compound into lasting resilience.",
        "tips": ["5-min morning journaling", "Box breathing: 4s in, 4s hold, 4s out, 4s hold",
                 "3 gratitude entries every night", "Try Headspace or Calm app"]
    },
    "social": {
        "title": "Strengthen your support network",
        "category": "Social connection",
        "body": "Social isolation is a major stress amplifier. One meaningful conversation a day measurably lowers cortisol.",
        "tips": ["Text one friend today", "Join a campus club or society",
                 "Set clear boundaries in stressful relationships",
                 "Attend one social event this week"]
    },
    "professional": {
        "title": "Reach out to a professional",
        "category": "Professional support",
        "body": "Your stress indicators suggest you'd benefit from talking to a counselor or therapist. This is a sign of strength.",
        "tips": ["Book a campus counseling session", "Try iCall (India) or BetterHelp",
                 "Tell a trusted adult how you're feeling",
                 "Helpline: iCall 9152987821 | Vandrevala 1860-2662-345"]
    },
    "quick_wins": {
        "title": "Immediate stress relief",
        "category": "Quick wins (5 min or less)",
        "body": "These techniques work within minutes and require no equipment. Use them whenever overwhelmed.",
        "tips": ["Cold water on face and wrists", "4-7-8 breathing technique",
                 "Step outside for a 10-minute walk", "Listen to one song you love"]
    },
    "music": {
        "title": "Listen to music on Spotify",
        "category": "Music & Relaxation",
        "body": "Music is scientifically proven to reduce stress and anxiety. Spotify has curated playlists for every mood.",
        "tips": [
            "🎵 Spotify: 'Peaceful Piano' - perfect for focusing and calming down",
            "🎵 Spotify: 'Good as Hell' - uplifting vibes and positive energy",
            "🎵 Spotify: 'Chill Hits' - modern tracks for relaxing and unwinding",
            "🎵 Spotify: 'Stress Relief' - scientifically designed for anxiety relief"
        ],
        "spotify_link": "https://open.spotify.com/search/stress%20relief"
    }
}


def generate_solutions(result: dict, inputs: dict) -> list:
    """Select relevant solutions based on predicted stress and input signals."""
    solutions = []
    sleep = inputs.get("sleep_hours", 7)
    anxiety = inputs.get("anxiety", 2)
    depression = inputs.get("depression_flag", 0)
    exercise = inputs.get("exercise", 2)
    social = inputs.get("social_isolation", 2)
    study = inputs.get("study_load", 3)
    pct = result["stress_pct"]

    if sleep < 6.5:
        solutions.append(SOLUTIONS_DB["sleep"])
    if study >= 4 or inputs.get("cgpa", 7.5) < 6:
        solutions.append(SOLUTIONS_DB["academics"])
    if exercise >= 3:
        solutions.append(SOLUTIONS_DB["exercise"])
    if anxiety >= 3 or depression >= 2:
        solutions.append(SOLUTIONS_DB["mental"])
    if social >= 3:
        solutions.append(SOLUTIONS_DB["social"])
    if pct >= 70:
        solutions.append(SOLUTIONS_DB["professional"])
    # Always include quick wins and music
    solutions.append(SOLUTIONS_DB["quick_wins"])
    solutions.append(SOLUTIONS_DB["music"])
    return solutions


# ── Routes ────────────────────────────────────────

@app.route("/api", methods=["GET"])
def api_index():
    return jsonify({
        "service": "Student Stress Detector API",
        "version": "1.0.0",
        "endpoints": {
            "POST /predict": "Predict stress level from survey inputs",
            "GET  /health":  "Service health check",
            "GET  /features": "List model features and importance",
            "GET  /metadata": "Model metadata and accuracy stats"
        }
    })


@app.route("/", defaults={"path": ""})
@app.route("/<path:path>")
def serve_frontend(path):
    if os.path.isdir(FRONTEND_DIST):
        file_path = os.path.join(FRONTEND_DIST, path)
        if path and os.path.isfile(file_path):
            return send_from_directory(FRONTEND_DIST, path)
        return send_from_directory(FRONTEND_DIST, "index.html")

    return jsonify({
        "message": "React frontend has not been built yet.",
        "next_steps": [
            "cd frontend",
            "npm install",
            "npm run build",
            "python ../api/app.py"
        ],
        "api": "/api"
    }), 200


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model_loaded": predictor.model is not None})


@app.route("/metadata", methods=["GET"])
def get_metadata():
    return jsonify(metadata)


@app.route("/features", methods=["GET"])
def get_features():
    importances = joblib.load(os.path.join(MODEL_DIR, "feature_importances.pkl"))
    return jsonify({
        "features": predictor.FEATURE_NAMES,
        "importances": importances.to_dict(),
        "top_5": importances.head(5).to_dict()
    })


@app.route("/predict", methods=["POST"])
def predict():
    """
    POST /predict
    Body (JSON):
    {
        "sleep_hours":       7.0,   // 2-10
        "cgpa":              7.5,   // 0-10
        "study_load":        3,     // 1-5
        "attendance":        2,     // 1-4 (1=always, 4=rarely)
        "screen_hours":      4.0,   // 0-12
        "social_isolation":  2,     // 1-4 (1=active, 4=isolated)
        "exercise":          2,     // 1-4 (1=daily, 4=never)
        "weight_change":     0,     // 0-3
        "anxiety":           2,     // 1-5
        "depression_flag":   1,     // 0-4
        "concentration":     1,     // 0-4
        "panic":             0,     // 0-4
        "peer_pressure":     2,     // 1-5
        "home_stress":       2,     // 1-4
        "relationship_stress": 1,  // 0-4
        "financial":         0      // 0-4
    }
    """
    try:
        data = request.get_json(force=True)
        if not data:
            return jsonify({"error": "No JSON body provided"}), 400

        # Validate types
        numeric_fields = [
            "sleep_hours", "cgpa", "study_load", "attendance", "screen_hours",
            "social_isolation", "exercise", "weight_change", "anxiety",
            "depression_flag", "concentration", "panic", "peer_pressure",
            "home_stress", "relationship_stress", "financial"
        ]
        inputs = {}
        for field in numeric_fields:
            if field in data:
                try:
                    inputs[field] = float(data[field])
                except (ValueError, TypeError):
                    return jsonify({"error": f"Field '{field}' must be numeric"}), 400

        # Run inference
        result = predictor.from_survey(inputs)

        # Add solutions
        solutions = generate_solutions(result, inputs)
        result["solutions"] = solutions

        # Compute factor scores for visualization
        result["factors"] = {
            "Sleep quality":        round(max(0, (8 - inputs.get("sleep_hours", 7)) / 6 * 100)),
            "Academic load":        round((inputs.get("study_load", 3) - 1) / 4 * 100),
            "Mental health":        round(((inputs.get("anxiety", 2) - 1) +
                                           inputs.get("depression_flag", 0) +
                                           inputs.get("panic", 0)) / 14 * 100),
            "Social environment":   round(((inputs.get("social_isolation", 2) - 1) +
                                           (inputs.get("peer_pressure", 2) - 1)) / 8 * 100),
            "Lifestyle":            round((inputs.get("screen_hours", 4) / 12 +
                                           (inputs.get("exercise", 2) - 1) / 3) / 2 * 100),
        }

        return jsonify(result)

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.route("/batch_predict", methods=["POST"])
def batch_predict():
    """Predict stress for multiple students at once."""
    try:
        data = request.get_json(force=True)
        if not isinstance(data, list):
            return jsonify({"error": "Body must be a JSON array of student objects"}), 400
        results = []
        for i, student in enumerate(data):
            result = predictor.from_survey(student)
            result["student_index"] = i
            results.append(result)
        summary = {
            "total": len(results),
            "low_count": sum(1 for r in results if r["stress_level"] == 0),
            "moderate_count": sum(1 for r in results if r["stress_level"] == 1),
            "high_count": sum(1 for r in results if r["stress_level"] == 2),
            "avg_stress_pct": round(np.mean([r["stress_pct"] for r in results]), 1),
        }
        return jsonify({"summary": summary, "results": results})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


if __name__ == "__main__":
    print("\n  Student Stress Detector API")
    print(f"  Primary model accuracy: {metadata['primary']['accuracy']*100:.1f}%")
    print("  Running on http://localhost:5000\n")
    app.run(debug=True, port=5000)
