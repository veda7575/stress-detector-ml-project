# Student Stress Detector — ML Project

A machine learning system that predicts student stress levels and recommends
personalised interventions, trained on 3 real Kaggle datasets (2,044 records).

---

## Datasets Used

| Dataset | Rows | Features | Source |
|---------|------|----------|--------|
| StressLevelDataset.csv | 1,100 | 21 | Kaggle — Student Stress Levels |
| Stress_Dataset.csv | 843 | 26 | Kaggle — Student Stress Monitoring |
| Student_Mental_health.csv | 101 | 11 | Kaggle — Student Mental Health |

---

## Model Performance

| Model | Algorithm | Test Accuracy | 5-Fold CV |
|-------|-----------|--------------|-----------|
| Primary (StressLevelDataset) | RF + GBT Ensemble | **88.2%** | 88.5% ± 2.0% |
| Secondary (Stress_Dataset) | Random Forest | **92.3%** | 92.9% ± 1.2% |

**Top 5 predictive features:**
1. blood_pressure (0.148)
2. sleep_quality (0.079)
3. depression (0.070)
4. academic_performance (0.068)
5. teacher_student_relationship (0.066)

---

## Project Structure

```
stress_detector/
├── data/
│   ├── StressLevelDataset.csv
│   ├── Stress_Dataset.csv
│   └── Student_Mental_health.csv
├── models/
│   ├── primary_model.pkl       ← Main ensemble model
│   ├── secondary_model.pkl     ← Survey-based model
│   ├── feature_importances.pkl
│   └── model_metadata.json
├── api/
│   └── app.py                  ← Flask REST API
├── notebooks/
│   ├── eda_analysis.py         ← EDA + 7 visualisation charts
│   └── charts/                 ← Generated PNG files
├── train_model.py              ← Full training pipeline
├── requirements.txt
└── README.md
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train the models
```bash
python train_model.py
```

### 3. Run EDA analysis
```bash
python notebooks/eda_analysis.py
# Charts saved to notebooks/charts/
```

### 4. Start the Flask API
```bash
python api/app.py
# Running on http://localhost:5000
```

### 5. Run the React frontend during development
```bash
cd frontend
npm install
npm run dev
# React dev server: http://localhost:5173
# API requests are proxied to Flask on http://localhost:5000
```

### 6. Serve React and Flask as one app
```bash
cd frontend
npm install
npm run build
cd ..
python api/app.py
# Open http://localhost:5000
```

The React app calls the Flask API using same-origin paths such as `/predict`.
In development, Vite proxies those paths to Flask. In production, Flask serves
the built React files from `frontend/dist`; the API info route is available at
`/api`.

---

## API Usage

### Predict stress for a student

```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "sleep_hours": 5.5,
    "cgpa": 7.0,
    "study_load": 4,
    "anxiety": 4,
    "depression_flag": 2,
    "social_isolation": 3,
    "exercise": 3,
    "peer_pressure": 4,
    "home_stress": 3,
    "financial": 2,
    "panic": 1,
    "concentration": 2,
    "screen_hours": 7,
    "attendance": 3,
    "weight_change": 1,
    "relationship_stress": 2
  }'
```

**Response:**
```json
{
  "stress_level": 2,
  "stress_label": "High",
  "stress_pct": 78.4,
  "confidence": 82.1,
  "probabilities": { "Low": 6.2, "Moderate": 11.7, "High": 82.1 },
  "factors": {
    "Sleep quality": 42,
    "Academic load": 75,
    "Mental health": 64,
    "Social environment": 68,
    "Lifestyle": 55
  },
  "solutions": [
    {
      "title": "Improve your sleep schedule",
      "category": "Sleep hygiene",
      "tips": ["Sleep & wake at same time daily", "..."]
    }
  ]
}
```

### Batch prediction
```bash
curl -X POST http://localhost:5000/batch_predict \
  -H "Content-Type: application/json" \
  -d '[{"sleep_hours": 8, "anxiety": 1, ...}, {"sleep_hours": 4, "anxiety": 5, ...}]'
```

### Other endpoints
- `GET /` — API info
- `GET /health` — Service health check
- `GET /features` — Feature list + importances
- `GET /metadata` — Model accuracy stats

---

## Input Field Reference

| Field | Range | Description |
|-------|-------|-------------|
| sleep_hours | 2–10 | Average hours of sleep per night |
| cgpa | 0-10 | Current CGPA (0=failing, 10=excellent) |
| study_load | 1–5 | Academic workload (1=light, 5=overwhelming) |
| attendance | 1–4 | Class attendance (1=always, 4=rarely) |
| screen_hours | 0–12 | Daily non-study screen time |
| social_isolation | 1–4 | Social activity (1=active, 4=isolated) |
| exercise | 1–4 | Exercise frequency (1=daily, 4=never) |
| weight_change | 0–3 | Weight change (0=none, 3=significant) |
| anxiety | 1–5 | Anxiety level (1=minimal, 5=severe) |
| depression_flag | 0–4 | Depression severity (0=none, 4=severe) |
| concentration | 0–4 | Concentration difficulty (0=none, 4=severe) |
| panic | 0–4 | Panic attack frequency (0=never, 4=frequent) |
| peer_pressure | 1–5 | Peer pressure impact (1=none, 5=severe) |
| home_stress | 1–4 | Home environment stress (1=supportive, 4=stressful) |
| relationship_stress | 0–4 | Relationship stress (0=none, 4=always) |
| financial | 0–4 | Financial pressure (0=none, 4=severe) |

---

## ML Architecture

```
Input Survey (16 features)
         │
         ▼
  Feature Mapping Layer
  (normalises to StressLevelDataset space → 20 features)
         │
         ▼
  Ensemble Model
  ┌──────────────────────────────────┐
  │  Random Forest (weight=2)        │
  │  • 300 trees, max_depth=14       │
  │  • balanced class weights        │
  │                                  │
  │  Gradient Boosting (weight=1)    │
  │  • 200 estimators, lr=0.08       │
  │  • subsample=0.85                │
  └──────────────────────────────────┘
         │ Soft voting
         ▼
  Stress Level: Low / Moderate / High
         │
         ▼
  Solution Engine
  (rule-based, triggered by input signals)
```

---

## Solution Engine

The solution engine selects from 7 intervention categories based on the
student's input signals, not just the stress prediction:

| Trigger condition | Solution category |
|------------------|-------------------|
| sleep_hours < 6.5 | Sleep hygiene |
| study_load ≥ 4 or cgpa < 6 | Academic strategy |
| exercise ≥ 3 (rarely/never) | Physical wellness |
| anxiety ≥ 3 or depression ≥ 2 | Mental health routines |
| social_isolation ≥ 3 | Social connection |
| stress_pct ≥ 70% | Professional support |
| always | Quick wins (< 5 min) |

---

## Future Improvements

- [ ] Add SHAP explainability (per-student feature contribution)
- [ ] Longitudinal tracking (predict stress trajectory over semester)
- [ ] NLP analysis of open-text responses
- [ ] Federated learning for privacy-preserving multi-institution training
- [ ] Mobile app integration via REST API
