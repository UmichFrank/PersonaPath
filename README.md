# PersonaPath — Adaptive Personality & Career Explorer

**SIADS 699 Capstone | Team Foundry | University of Michigan**

PersonaPath is a text-driven personality assessment tool built on the Big Five (OCEAN) framework. Users write a short self-description and answer up to 8 adaptive follow-up questions. A trained ML pipeline infers personality scores and matches the user to evidence-based career clusters from the O\*NET database.

---

## Features

- 8-step adaptive questionnaire (Q1 = free-text intro, Q2–Q8 = targeted follow-ups)
- Real-time Big Five inference via SBERT embeddings + Random Forest
- O\*NET career cluster alignment across 900+ occupations
- Pandora personality archetype classification (6 archetypes)
- One-click PDF export of the full results report
- Graceful mock/demo mode when ML model files are absent

---

## Project Structure

```
699 Capstone/
├── app.py                      # Main Streamlit application
├── requirements.txt            # Python dependencies
├── big5_predictors.pkl         # Trained Big Five Random Forest models
├── onet_career_artifacts.pkl   # O*NET occupation embeddings & cluster map
├── kmeans_pipeline_a.pkl       # Pandora archetype pipeline (Big Five space)
├── kmeans_pipeline_b.pkl       # Pandora archetype pipeline (SBERT space)
└── sbert_model/                # Local copy of all-MiniLM-L6-v2 (optional)
```

---

## Installation & Setup

### 1. Install dependencies

```bash
pip install -r requirements.txt
```

### 2. (Optional) Download SBERT model for offline use

If the environment has no internet access, save the model locally once from a connected machine:

```bash
python3 -c "
from sentence_transformers import SentenceTransformer
SentenceTransformer('all-MiniLM-L6-v2').save('sbert_model')
"
```

The app automatically detects `sbert_model/` and loads it without a network request.

### 3. Run the app

```bash
cd "699 Capstone"
streamlit run app.py
```

Opens at `http://localhost:8501` by default.

---

## Model Modes

The top bar displays a live status indicator for each component:

| Indicator | Component | Requires |
|-----------|-----------|----------|
| ● SBERT | Sentence embedding | `sentence-transformers` + `sbert_model/` |
| ● B5 RandomForest | Big Five prediction | `big5_predictors.pkl` + SBERT |
| ● Pandora Ensemble | Archetype (SBERT pipeline) | `kmeans_pipeline_b.pkl` + SBERT |
| ● O\*NET matching | Career cluster alignment | `onet_career_artifacts.pkl` |
| ● Archetypes | Pandora archetypes (Big Five) | `kmeans_pipeline_a.pkl` |

When ML artefacts are absent the app falls back to a rule-based mock mode so the full interface remains navigable.

---

## ML Pipeline

### Big Five Prediction
- Text is embedded using `all-MiniLM-L6-v2` (384-dim dense vectors)
- A separate Random Forest classifier predicts each OCEAN trait
- Adaptive questions target traits with the highest prediction uncertainty
- Scores update after each response; the final profile is an ensemble of all 8 inputs

### Career Cluster Alignment
- O\*NET occupation Big Five profiles are pre-embedded and grouped into 4 career clusters
- Cosine similarity between the user's OCEAN vector and each cluster centroid yields an alignment score

### Pandora Personality Archetypes
- KMeans (k=6) fitted on the Pandora Reddit dataset
- Two pipelines: Big Five space (`kmeans_pipeline_a`) and SBERT space (`kmeans_pipeline_b`)
- When SBERT is available, scores are blended for a richer archetype estimate

---

## Dependencies

| Package | Purpose | Min Version |
|---------|---------|-------------|
| `streamlit` | Web application framework | 1.35.0 |
| `plotly` | Interactive charts | 5.18.0 |
| `numpy` | Numerical computation | 1.26.0 |
| `scikit-learn` | Random Forest, KMeans | 1.4.0 |
| `sentence-transformers` | SBERT text embeddings | 2.7.0 |
| `reportlab` | PDF report generation | 4.1.0 |

---

## 🛡️ User Privacy & Responsible AI

At Team Foundry, we prioritize user trust and data ethics. PersonaPath is designed with the following principles:

* **Data Minimalism:** User text inputs are processed in real-time to generate personality insights and are **not stored** in any permanent database by default.
* **Non-Prescriptive Guidance:** Our career recommendations are framed as **exploratory tools** rather than definitive professional advice. We aim to spark self-reflection, not to limit a user's potential based on a single algorithm.
* **Transparency:** We provide a "Confidence Indicator" for our predictions, ensuring users understand that AI models have inherent uncertainties.