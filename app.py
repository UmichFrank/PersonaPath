"""
PersonaPath: Adaptive Personality Explorer
Team Foundry | SIADS 699 Capstone

Run: streamlit run app.py

Model mode: drops gracefully to mock mode if pkl artifacts are not found.
To use real models, copy these files into the same directory as app.py:
  - big5_predictors.pkl
  - onet_career_artifacts.pkl
  - question_pool.json   (optional override)
"""

import streamlit as st
import streamlit.components.v1 as stc
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os, pickle, json, re, io
from collections import defaultdict
from datetime import datetime

try:
    from streamlit_mic_recorder import speech_to_text
    HAS_MIC_RECORDER = True
except ImportError:
    speech_to_text = None
    HAS_MIC_RECORDER = False

# ═══════════════════════════════════════════════════════
# CONFIG
# ═══════════════════════════════════════════════════════
st.set_page_config(
    page_title="PersonaPath",
    page_icon=None,
    layout="wide",
    initial_sidebar_state="collapsed",
)

NAVY  = "#1E3A5F"
CYAN  = "#00B4D8"
WHITE = "#FFFFFF"
OFF   = "#F7FBFD"
GREY  = "#4A5568"
LGREY = "#E2E8F0"
GREEN = "#10B981"
AMBER = "#F59E0B"
RED   = "#E53E3E"
PURPLE = "#7C3AED"

BIG5_KEYS  = ["O", "C", "E", "A", "N"]
BIG5_NAMES = ["Openness", "Conscientiousness", "Extraversion", "Agreeableness", "Neuroticism"]
BIG5_DESC  = {
    "Openness":          "Curiosity, imagination, and openness to new experiences",
    "Conscientiousness": "Organisation, dependability, and goal-directed behaviour",
    "Extraversion":      "Sociability, assertiveness, and positive affect",
    "Agreeableness":     "Cooperation, empathy, and prosocial orientation",
    "Neuroticism":       "Emotional reactivity and tendency toward negative affect",
}

OCC_PREP_OPTIONS = [
    ("Open to all preparation levels", 1),
    ("High school / certificate and above", 2),
    ("Bachelor's / professional level and above", 3),
    ("Master's / advanced specialist level and above", 4),
    ("Doctoral / highly specialized roles", 5),
]
JOB_ZONE_LABELS = {
    1: "Job Zone 1 · Little preparation",
    2: "Job Zone 2 · Some preparation",
    3: "Job Zone 3 · Medium preparation",
    4: "Job Zone 4 · High preparation",
    5: "Job Zone 5 · Extensive preparation",
}
JOB_ZONE_SHORT = {
    1: "JZ1",
    2: "JZ2",
    3: "JZ3",
    4: "JZ4",
    5: "JZ5",
}
JOB_ZONE_COLS = [
    "Job Zone", "job_zone", "jobzone", "JobZone",
    "job_zone_num", "job_zone_level",
]
SOC_JOB_ZONE_BASE = {
    "11": 4, "13": 4, "15": 4, "17": 4, "19": 4, "21": 3, "23": 5,
    "25": 4, "27": 3, "29": 4, "31": 2, "33": 3, "35": 1, "37": 1,
    "39": 2, "41": 2, "43": 2, "45": 2, "47": 2, "49": 3, "51": 2, "53": 2,
}
JOB_ZONE_FLOOR_KEYWORDS = {
    5: ["physicist", "economist", "attorney", "lawyer", "surgeon", "anesthesiologist", "professor"],
    4: ["engineer", "developer", "scientist", "architect", "quant", "analyst", "manager",
        "director", "administrator", "specialist", "researcher", "statistician"],
}
JOB_ZONE_CAP_KEYWORDS = {
    1: ["dishwasher", "host", "cashier", "packer", "sorter"],
    2: ["operator", "assembler", "clerk", "driver", "helper", "attendant", "cashier", "server", "barista"],
    3: ["technician", "drafter", "inspector"],
}
JOB_ZONE_SENIORITY_KEYWORDS = ["chief", "head", "principal", "lead", "senior", "staff"]

# Reference occupations per trait × level — for illustrative purposes only
# Source: O*NET Big Five–occupation crosswalk (Sackett & Walmsley 2014)
# ═══════════════════════════════════════════════════════
# MBTI MAPPING  (Big Five → MBTI via McCrae & Costa, 1989)
# ═══════════════════════════════════════════════════════
# Multi-trait weighted mapping based on meta-analytic correlations
# (McCrae & Costa 1989; Furnham 1996; Wilt & Revelle 2015).
# Each dimension uses a primary trait plus secondary corrections.
# Weights: {trait: weight}.  Positive weight → pole_high.
MBTI_DIMS = [
    {"code": "EI", "pole_high": "E", "pole_low": "I",
     "weights": {"E": 0.80, "A": 0.15, "N": -0.05},
     "name_high": "Extraversion", "name_low": "Introversion",
     "desc_high": "Energised by social interaction; prefers action and breadth",
     "desc_low":  "Recharges through solitude; prefers depth and reflection"},
    {"code": "NS", "pole_high": "N", "pole_low": "S",
     "weights": {"O": 0.75, "C": -0.15, "E": 0.10},
     "name_high": "Intuition", "name_low": "Sensing",
     "desc_high": "Drawn to patterns, abstractions, and future possibilities",
     "desc_low":  "Trusts concrete facts, experience, and proven methods"},
    {"code": "TF", "pole_high": "F", "pole_low": "T",
     "weights": {"A": 0.75, "O": 0.10, "N": 0.15},
     "name_high": "Feeling", "name_low": "Thinking",
     "desc_high": "Prioritises empathy, harmony, and interpersonal values",
     "desc_low":  "Prioritises logic, consistency, and objective analysis"},
    {"code": "JP", "pole_high": "J", "pole_low": "P",
     "weights": {"C": 0.75, "N": -0.15, "O": -0.10},
     "name_high": "Judging", "name_low": "Perceiving",
     "desc_high": "Prefers structure, plans, and decisive closure",
     "desc_low":  "Prefers flexibility, spontaneity, and open options"},
]

# 16 MBTI type descriptions (short, non-clinical)
MBTI_TYPE_INFO = {
    "INTJ": ("Architect",       "Strategic, independent thinker with a vision for improvement"),
    "INTP": ("Logician",        "Analytical innovator driven by logic and curiosity"),
    "ENTJ": ("Commander",       "Bold, strategic leader who drives organisation and efficiency"),
    "ENTP": ("Debater",         "Quick-witted explorer who thrives on intellectual challenge"),
    "INFJ": ("Advocate",        "Idealistic, empathetic planner seeking meaningful impact"),
    "INFP": ("Mediator",        "Thoughtful idealist guided by values and inner vision"),
    "ENFJ": ("Protagonist",     "Charismatic leader focused on inspiring and helping others"),
    "ENFP": ("Campaigner",      "Enthusiastic, creative free spirit who sees possibilities everywhere"),
    "ISTJ": ("Logistician",     "Practical, reliable organiser who values tradition and duty"),
    "ISFJ": ("Defender",        "Warm, conscientious protector dedicated to helping others"),
    "ESTJ": ("Executive",       "Organised, decisive manager who upholds order and standards"),
    "ESFJ": ("Consul",          "Caring, sociable supporter who values harmony and cooperation"),
    "ISTP": ("Virtuoso",        "Bold, practical experimenter and hands-on problem solver"),
    "ISFP": ("Adventurer",      "Flexible, charming artist who lives in the moment"),
    "ESTP": ("Entrepreneur",    "Energetic, perceptive doer who thrives on action"),
    "ESFP": ("Entertainer",     "Spontaneous, energetic performer who loves engaging with people"),
}


def predict_mbti(profile: dict) -> dict:
    """Map a Big Five profile to MBTI dimensions.

    Uses multi-trait weighted mapping based on meta-analytic correlations
    (McCrae & Costa, 1989; Furnham, 1996; Wilt & Revelle, 2015).
    Each MBTI dimension is a weighted combination of multiple Big Five
    traits, producing more accurate results than single-trait lookup.

    Returns dict with keys: type (str), dimensions (list of dicts with
    pole, score, name, desc).
    """
    dims = []
    type_letters = ""
    for d in MBTI_DIMS:
        # Weighted sum of Big Five traits (centred around 0.5)
        raw = sum(w * profile[t] for t, w in d["weights"].items())
        # Normalise: weights sum may != 1, so clamp to [0, 1]
        raw = max(0.0, min(1.0, raw))
        if raw >= 0.50:
            pole = d["pole_high"]
            name = d["name_high"]
            desc = d["desc_high"]
            strength = raw
        else:
            pole = d["pole_low"]
            name = d["name_low"]
            desc = d["desc_low"]
            strength = 1.0 - raw
        type_letters += pole
        dims.append({
            "code": d["code"],
            "pole": pole,
            "score": strength,       # 0.5–1.0 — how clearly this pole
            "name": name,
            "desc": desc,
            "pole_high": d["pole_high"],
            "pole_low": d["pole_low"],
            "raw": raw,
        })
    nickname, tagline = MBTI_TYPE_INFO.get(type_letters, ("—", ""))
    return {"type": type_letters, "nickname": nickname, "tagline": tagline,
            "dimensions": dims}


BIG5_REFS = {
    "Openness": {
        "High":     ["Research Scientist", "UX Designer", "Architect", "Creative Director", "Journalist"],
        "Moderate": ["Marketing Analyst", "Product Manager", "Business Consultant"],
        "Low":      ["Quality Inspector", "Accountant", "Logistics Coordinator"],
    },
    "Conscientiousness": {
        "High":     ["Project Manager", "Surgeon", "Financial Analyst", "Auditor", "Civil Engineer"],
        "Moderate": ["Operations Coordinator", "HR Generalist", "Administrative Manager"],
        "Low":      ["Entrepreneur", "Artist", "Freelance Strategist"],
    },
    "Extraversion": {
        "High":     ["Sales Manager", "PR Specialist", "Teacher", "HR Manager", "Event Coordinator"],
        "Moderate": ["Business Analyst", "Team Lead", "Product Owner"],
        "Low":      ["Software Engineer", "Data Analyst", "Archivist", "Research Analyst"],
    },
    "Agreeableness": {
        "High":     ["Social Worker", "Nurse", "School Counselor", "Mediator", "Pediatrician"],
        "Moderate": ["Product Manager", "Operations Manager", "HR Business Partner"],
        "Low":      ["Lawyer", "Financial Trader", "Competitive Intelligence Analyst"],
    },
    "Neuroticism": {
        "High":     ["Creative Writer", "Therapist", "Musician", "Actor / Performer"],
        "Moderate": ["Graphic Designer", "Photographer", "Brand Strategist"],
        "Low":      ["Surgeon", "Airline Pilot", "Air Traffic Controller", "Crisis Negotiator"],
    },
}

# ═══════════════════════════════════════════════════════
# QUESTION POOL  (from notebook Section 6.1)
# ═══════════════════════════════════════════════════════
QUESTION_POOL = [
    # ── Openness ──────────────────────────────────────────────────────────────
    {"q": "Describe what you enjoy doing in your free time.",                          "primary": "O", "secondary": "E"},
    {"q": "Describe a recent creative project or idea you had.",                       "primary": "O", "secondary": None},
    {"q": "How do you approach learning something completely new?",                    "primary": "O", "secondary": "C"},
    {"q": "What kind of books, films, or art do you enjoy?",                           "primary": "O", "secondary": None},
    {"q": "What motivates you to try something you've never done before?",             "primary": "O", "secondary": "E"},
    {"q": "If you could live in any time period or place, where would you go and why?","primary": "O", "secondary": None},
    {"q": "How do you react when an idea or theory challenges your existing beliefs?", "primary": "O", "secondary": "A"},
    {"q": "Describe a moment when you changed your mind about something important.",   "primary": "O", "secondary": "A"},
    {"q": "Do you enjoy exploring topics that have no clear right or wrong answer?",   "primary": "O", "secondary": None},
    {"q": "What does an ideal vacation look like for you?",                            "primary": "O", "secondary": "E"},
    {"q": "How do you feel about experimenting with new foods, places, or hobbies?",   "primary": "O", "secondary": None},
    # ── Conscientiousness ─────────────────────────────────────────────────────
    {"q": "How do you typically organise your daily tasks or work?",                   "primary": "C", "secondary": None},
    {"q": "Describe your ideal work environment.",                                     "primary": "C", "secondary": "E"},
    {"q": "How do you prepare for an important presentation?",                         "primary": "C", "secondary": "N"},
    {"q": "How important is maintaining a routine for you?",                           "primary": "C", "secondary": None},
    {"q": "Walk me through how you plan a project from start to finish.",              "primary": "C", "secondary": None},
    {"q": "How do you prioritise when you have many tasks competing for your time?",   "primary": "C", "secondary": "N"},
    {"q": "What does 'done' look like to you when you finish a task?",                 "primary": "C", "secondary": None},
    {"q": "How do you handle it when a plan you made carefully falls apart?",          "primary": "C", "secondary": "N"},
    {"q": "How do you track promises or commitments you've made to others?",           "primary": "C", "secondary": "A"},
    {"q": "Describe a time you had to meet a high standard with very little time.",    "primary": "C", "secondary": "N"},
    # ── Extraversion ─────────────────────────────────────────────────────────
    {"q": "When you meet someone new, how do you usually feel?",                       "primary": "E", "secondary": "N"},
    {"q": "How do you decide whether to attend a social event?",                       "primary": "E", "secondary": None},
    {"q": "What role do you usually take in group projects?",                          "primary": "E", "secondary": "A"},
    {"q": "Do you prefer working alone or in a team? Why?",                            "primary": "E", "secondary": "A"},
    {"q": "How do you feel about taking on leadership roles?",                         "primary": "E", "secondary": "C"},
    {"q": "Where do you get your energy — from being around others or from solitude?", "primary": "E", "secondary": None},
    {"q": "How comfortable are you speaking up in a meeting or class?",                "primary": "E", "secondary": "N"},
    {"q": "How do you usually kick off a conversation with a stranger?",               "primary": "E", "secondary": "A"},
    {"q": "How do you recharge after a long or demanding week?",                       "primary": "E", "secondary": "N"},
    {"q": "Describe your social life — how often do you connect with others?",         "primary": "E", "secondary": None},
    {"q": "How do you feel the morning after a large party or networking event?",      "primary": "E", "secondary": "N"},
    # ── Agreeableness ────────────────────────────────────────────────────────
    {"q": "How do you usually handle disagreements with others?",                      "primary": "A", "secondary": "N"},
    {"q": "When someone asks for help, what's your typical reaction?",                 "primary": "A", "secondary": None},
    {"q": "How do you react when someone criticises your work?",                       "primary": "A", "secondary": "N"},
    {"q": "What do you do when you see someone struggling?",                           "primary": "A", "secondary": None},
    {"q": "How important is harmony in your relationships — at home or at work?",      "primary": "A", "secondary": None},
    {"q": "Describe a time you compromised your own preference to help someone else.", "primary": "A", "secondary": None},
    {"q": "When you sense a friend is upset, what do you typically do?",               "primary": "A", "secondary": "E"},
    {"q": "How do you feel when someone disagrees with you publicly?",                 "primary": "A", "secondary": "N"},
    {"q": "Do you find it easy or hard to say 'no' when people ask for favours?",      "primary": "A", "secondary": "N"},
    {"q": "How do you handle a colleague who is difficult or uncooperative?",          "primary": "A", "secondary": "C"},
    # ── Neuroticism ───────────────────────────────────────────────────────────
    {"q": "What happens when you face an unexpected deadline?",                        "primary": "N", "secondary": "C"},
    {"q": "How do you handle situations where things don't go as planned?",            "primary": "N", "secondary": None},
    {"q": "What do you do when you feel overwhelmed or stressed?",                     "primary": "N", "secondary": None},
    {"q": "How do you typically respond when you make a significant mistake?",         "primary": "N", "secondary": None},
    {"q": "How do you cope when you're waiting for important news or a big decision?", "primary": "N", "secondary": None},
    {"q": "Describe how you feel in high-pressure situations like exams or interviews.","primary": "N", "secondary": "E"},
    {"q": "How long does it take you to recover after a stressful event?",             "primary": "N", "secondary": None},
    {"q": "What's your inner monologue like when something goes wrong?",               "primary": "N", "secondary": None},
    {"q": "How often do you worry about things that haven't happened yet?",            "primary": "N", "secondary": None},
    {"q": "How do you manage your mood when life feels chaotic or unpredictable?",     "primary": "N", "secondary": "C"},
]

# ═══════════════════════════════════════════════════════
# IPIP-NEO LIKERT ITEMS (subset: 5 per trait = 25 total)
# Keyed +/- per IPIP convention; scored on 1-5 Likert scale
# Source: Johnson (2014) 120-item IPIP-NEO-PI-R
# ═══════════════════════════════════════════════════════
LIKERT_ITEMS = [
    # ── Openness ──
    {"text": "I have a vivid imagination.",                        "trait": "O", "keyed": "+"},
    {"text": "I enjoy hearing new ideas.",                         "trait": "O", "keyed": "+"},
    {"text": "I prefer variety to routine.",                       "trait": "O", "keyed": "+"},
    {"text": "I am not interested in abstract ideas.",             "trait": "O", "keyed": "-"},
    {"text": "I do not like art.",                                 "trait": "O", "keyed": "-"},
    # ── Conscientiousness ──
    {"text": "I am always prepared.",                              "trait": "C", "keyed": "+"},
    {"text": "I pay attention to details.",                        "trait": "C", "keyed": "+"},
    {"text": "I carry out my plans.",                              "trait": "C", "keyed": "+"},
    {"text": "I waste my time.",                                   "trait": "C", "keyed": "-"},
    {"text": "I find it difficult to get down to work.",           "trait": "C", "keyed": "-"},
    # ── Extraversion ──
    {"text": "I feel comfortable around people.",                  "trait": "E", "keyed": "+"},
    {"text": "I start conversations.",                             "trait": "E", "keyed": "+"},
    {"text": "I talk to a lot of different people at parties.",    "trait": "E", "keyed": "+"},
    {"text": "I keep in the background.",                          "trait": "E", "keyed": "-"},
    {"text": "I don't like to draw attention to myself.",          "trait": "E", "keyed": "-"},
    # ── Agreeableness ──
    {"text": "I am interested in people.",                         "trait": "A", "keyed": "+"},
    {"text": "I sympathize with others' feelings.",                "trait": "A", "keyed": "+"},
    {"text": "I take time out for others.",                        "trait": "A", "keyed": "+"},
    {"text": "I am not really interested in others.",              "trait": "A", "keyed": "-"},
    {"text": "I insult people.",                                   "trait": "A", "keyed": "-"},
    # ── Neuroticism ──
    {"text": "I get stressed out easily.",                         "trait": "N", "keyed": "+"},
    {"text": "I worry about things.",                              "trait": "N", "keyed": "+"},
    {"text": "I get upset easily.",                                "trait": "N", "keyed": "+"},
    {"text": "I am relaxed most of the time.",                     "trait": "N", "keyed": "-"},
    {"text": "I seldom feel blue.",                                "trait": "N", "keyed": "-"},
]

LIKERT_LABELS = [
    "Very Inaccurate", "Moderately Inaccurate",
    "Neither Accurate Nor Inaccurate",
    "Moderately Accurate", "Very Accurate",
]


def score_likert(responses: list[int]) -> dict:
    """Score Likert responses → Big Five in [0,1].

    responses: list of 25 ints (1-5), one per LIKERT_ITEMS.
    Returns dict {O, C, E, A, N} each in [0.0, 1.0].
    """
    from collections import defaultdict
    totals = defaultdict(list)
    for item, resp in zip(LIKERT_ITEMS, responses):
        if resp is None or resp < 1:
            continue
        # Reverse-score negative-keyed items
        val = resp if item["keyed"] == "+" else (6 - resp)
        totals[item["trait"]].append(val)
    scores = {}
    for trait in BIG5_KEYS:
        vals = totals.get(trait, [])
        if vals:
            # Map 1-5 average → 0-1 range
            scores[trait] = (sum(vals) / len(vals) - 1.0) / 4.0
        else:
            scores[trait] = 0.5
    return scores


# ═══════════════════════════════════════════════════════
# PANDORA PERSONALITY ARCHETYPES
# Derived empirically from KMeans clustering of Big Five scores
# fitted on the Pandora Reddit dataset (kmeans_pipeline_a).
# More grounded than MBTI — continuous traits, data-driven clusters.
# ═══════════════════════════════════════════════════════
PANDORA_ARCHETYPES = {
    0: {
        "name":   "Balanced Generalist",
        "short":  "BG",
        "color":  "#00B4D8",   # CYAN
        "desc":   (
            "Well-rounded and adaptable across social and technical domains. "
            "Performs consistently in varied environments without extreme peaks or valleys."
        ),
        "traits": "Moderate across all five dimensions",
        "centroid": {"O": 0.52, "C": 0.51, "E": 0.52, "A": 0.53, "N": 0.49},
    },
    1: {
        "name":   "Resilient Realist",
        "short":  "RR",
        "color":  "#64748B",
        "desc":   (
            "Emotionally stable and pragmatic. Works independently and cuts through "
            "ambiguity without being swayed by social pressure or abstract ideas."
        ),
        "traits": "High Emotional Stability · Low Openness · Low Agreeableness",
        "centroid": {"O": 0.19, "C": 0.42, "E": 0.40, "A": 0.19, "N": 0.37},
    },
    2: {
        "name":   "Anxious Achiever",
        "short":  "AA",
        "color":  "#10B981",   # GREEN
        "desc":   (
            "Highly organised, sociable, and cooperative — but prone to stress. "
            "Delivers strong results in structured teams while internalising pressure."
        ),
        "traits": "High Conscientiousness · High Agreeableness · High Extraversion · High Neuroticism",
        "centroid": {"O": 0.18, "C": 0.73, "E": 0.63, "A": 0.74, "N": 0.64},
    },
    3: {
        "name":   "Driven Visionary",
        "short":  "DV",
        "color":  "#7C3AED",   # PURPLE
        "desc":   (
            "Intellectually ambitious with high engagement across all dimensions. "
            "Channels stress into creative momentum; thrives on challenge and ideation."
        ),
        "traits": "High Openness · High Conscientiousness · High Extraversion · High Neuroticism",
        "centroid": {"O": 0.83, "C": 0.70, "E": 0.73, "A": 0.74, "N": 0.73},
    },
    4: {
        "name":   "Empathetic Helper",
        "short":  "EH",
        "color":  "#F59E0B",   # AMBER
        "desc":   (
            "Warm, emotionally stable, and deeply people-oriented. Prioritises human "
            "connection over rigid structure; a natural in support and community roles."
        ),
        "traits": "High Agreeableness · High Emotional Stability · Low Conscientiousness",
        "centroid": {"O": 0.56, "C": 0.25, "E": 0.37, "A": 0.83, "N": 0.36},
    },
    5: {
        "name":   "Independent Creator",
        "short":  "IC",
        "color":  "#E11D48",
        "desc":   (
            "Highly curious and original, with strong emotional independence. "
            "Prefers autonomous creative work over consensus-driven processes."
        ),
        "traits": "High Openness · High Emotional Stability · Low Agreeableness",
        "centroid": {"O": 0.83, "C": 0.40, "E": 0.45, "A": 0.19, "N": 0.39},
    },
}

# Pre-decoded B5 profiles for kmeans_pipeline_b SBERT clusters
# (computed by running RandomForest on each 384-dim cluster centre)
# Used for RF + embedding ensemble when SBERT is available locally.
SBERT_ARCHETYPE_B5 = [
    {"O": 0.70, "C": 0.39, "E": 0.46, "A": 0.42, "N": 0.49},  # B0
    {"O": 0.55, "C": 0.46, "E": 0.42, "A": 0.41, "N": 0.47},  # B1
    {"O": 0.15, "C": 0.61, "E": 0.62, "A": 0.57, "N": 0.68},  # B2
    {"O": 0.32, "C": 0.52, "E": 0.64, "A": 0.46, "N": 0.64},  # B3
    {"O": 0.33, "C": 0.60, "E": 0.53, "A": 0.57, "N": 0.60},  # B4
    {"O": 0.19, "C": 0.64, "E": 0.55, "A": 0.54, "N": 0.57},  # B5
]

# ═══════════════════════════════════════════════════════
# CAREER CLUSTERS  (from notebook Section 8 O*NET results)
# ═══════════════════════════════════════════════════════
# icon_label / icon_color used by CSS badges (no emoji)
CAREER_CLUSTERS = [
    {
        "id": 0,
        "name": "People-Centred Leadership",
        "icon_label": "PL",
        "icon_color": GREEN,
        "centroid": {"O": 0.770, "C": 0.755, "E": 0.774, "A": 0.802, "N": 0.224},
        "description": "Roles that require strong people skills, empathy, and the ability to inspire and manage teams.",
        "examples": ["Education & Training", "Healthcare Management", "Social Work", "Human Resources", "Counselling"],
        "n_occ": 229,
    },
    {
        "id": 1,
        "name": "Technical & Analytical",
        "icon_label": "TA",
        "icon_color": CYAN,
        "centroid": {"O": 0.675, "C": 0.581, "E": 0.447, "A": 0.398, "N": 0.579},
        "description": "Precision-focused roles requiring analytical thinking, technical expertise, and independent work.",
        "examples": ["Data Science", "Engineering", "Finance & Quant", "Quality Assurance", "Research"],
        "n_occ": 216,
    },
    {
        "id": 2,
        "name": "Creative & Collaborative",
        "icon_label": "CC",
        "icon_color": PURPLE,
        "centroid": {"O": 0.631, "C": 0.617, "E": 0.614, "A": 0.614, "N": 0.388},
        "description": "Roles that blend creativity with teamwork — producing, communicating, and coordinating.",
        "examples": ["Media & Communications", "Design", "Marketing", "Project Management", "Arts & Entertainment"],
        "n_occ": 282,
    },
    {
        "id": 3,
        "name": "Hands-On Operations",
        "icon_label": "HO",
        "icon_color": AMBER,
        "centroid": {"O": 0.401, "C": 0.362, "E": 0.421, "A": 0.387, "N": 0.646},
        "description": "Practical, task-oriented roles where physical skill, precision, and independence are valued.",
        "examples": ["Skilled Trades", "Manufacturing", "Construction", "Transportation", "Field Operations"],
        "n_occ": 167,
    },
]

# ═══════════════════════════════════════════════════════
# MODEL LAYER — real pkl if available, mock otherwise
# ═══════════════════════════════════════════════════════
APP_DIR = os.path.dirname(os.path.abspath(__file__))

@st.cache_resource(show_spinner=False)
def load_models():
    b5_path    = os.path.join(APP_DIR, "big5_predictors.pkl")
    onet_path  = os.path.join(APP_DIR, "onet_career_artifacts.pkl")
    km_a_path  = os.path.join(APP_DIR, "kmeans_pipeline_a.pkl")
    km_b_path  = os.path.join(APP_DIR, "kmeans_pipeline_b.pkl")

    sbert_model  = None
    b5_models    = None
    onet_arts    = None
    km_pipe_a    = None   # k=6 in Big Five space — no SBERT needed
    km_pipe_b    = None   # k=6 in SBERT embedding space — needs SBERT
    live_b5      = False
    live_onet    = False

    import warnings

    # ── O*NET + personality archetype pipelines (no SBERT needed) ──
    if os.path.exists(onet_path):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with open(onet_path, "rb") as f:
                    onet_arts = pickle.load(f)
            live_onet = True
        except Exception:
            onet_arts = None

    if os.path.exists(km_a_path):
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with open(km_a_path, "rb") as f:
                    km_pipe_a = pickle.load(f)
        except Exception:
            km_pipe_a = None

    # ── Big Five predictors + embedding archetype (needs SBERT) ──
    # Prefer a local copy at sbert_model/ so the app works without internet access.
    sbert_local = os.path.join(APP_DIR, "sbert_model")
    sbert_name  = sbert_local if os.path.isdir(sbert_local) else "all-MiniLM-L6-v2"
    if os.path.exists(b5_path):
        try:
            from sentence_transformers import SentenceTransformer
            sbert_model = SentenceTransformer(sbert_name)
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                with open(b5_path, "rb") as f:
                    b5_models = pickle.load(f)
            live_b5 = True
            # Also load pipeline_b for ensemble (only useful when SBERT works)
            if os.path.exists(km_b_path):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    with open(km_b_path, "rb") as f:
                        km_pipe_b = pickle.load(f)
        except Exception:
            sbert_model = None
            b5_models   = None
            km_pipe_b   = None

    return sbert_model, b5_models, onet_arts, km_pipe_a, km_pipe_b, live_b5, live_onet

sbert_model, b5_models, onet_arts, km_pipe_a, km_pipe_b, USE_REAL_B5, USE_REAL_ONET = load_models()
USE_REAL_MODELS = USE_REAL_B5  # for display badge


def predict_big5(texts: list[str]) -> dict:
    """Return Big Five scores {O,C,E,A,N} in [0,1].

    When SBERT + kmeans_pipeline_b are both available, blends the RandomForest
    prediction with the Pandora-derived SBERT cluster prior for smoother,
    more stable results (weighted ensemble: 75% RF + 25% cluster centroid).
    """
    combined = " ".join(texts)
    if USE_REAL_B5:
        emb = sbert_model.encode([combined[:2000]])          # (1, 384)
        emb_2d = emb.reshape(1, -1)

        # ── Base prediction: RandomForest trained on Pandora embeddings ──
        rf_pred = {t: float(b5_models[t].predict_proba(emb_2d)[0, 1])
                   for t in BIG5_KEYS}

        # ── Ensemble with Pandora SBERT cluster prior (pipeline_b) ──
        if km_pipe_b is not None:
            try:
                km_b   = km_pipe_b["kmeans"]
                sc_b   = km_pipe_b["scaler"]
                emb_sc = sc_b.transform(emb_2d)
                cid    = int(km_b.predict(emb_sc)[0])
                # Distance-based blend weight: closer to centroid → stronger prior
                dist   = float(np.linalg.norm(
                    emb_sc - km_b.cluster_centers_[cid]))
                # Sigmoid decay: weight ∈ [0.10, 0.25]
                w = 0.10 + 0.15 * np.exp(-dist / 2.0)
                cluster_b5 = SBERT_ARCHETYPE_B5[cid]
                return {t: float(np.clip(
                    (1 - w) * rf_pred[t] + w * cluster_b5[t], 0.05, 0.95))
                    for t in BIG5_KEYS}
            except Exception:
                pass  # fall back to plain RF

        return rf_pred
    # ── Mock: keyword heuristics ──────────────────────────────────
    text = combined.lower()
    def density(words):
        return min(1.0, sum(text.count(w) for w in words) / max(len(text.split()) / 25, 1))
    scores = {
        "O": 0.45 + 0.40 * density(["creative","imagine","curious","novel","idea","art","learn","explore","new","innovative","dream"]),
        "C": 0.45 + 0.40 * density(["organise","plan","schedule","deadline","goal","system","routine","prepare","careful","detail"]),
        "E": 0.45 + 0.40 * density(["people","social","team","talk","outgoing","meeting","friends","energy","lead","fun","party"]),
        "A": 0.45 + 0.40 * density(["help","cooperate","kind","empathy","support","trust","care","collaborate","agree","share"]),
        "N": 0.55 - 0.35 * density(["calm","stable","relax","confident","steady","comfortable","secure","fine","okay","manage"]),
    }
    # Add slight variation from text length
    rng = np.random.default_rng(abs(hash(combined[:50])) % (2**31))
    noise = rng.normal(0, 0.04, 5)
    for i, k in enumerate(BIG5_KEYS):
        scores[k] = float(np.clip(scores[k] + noise[i], 0.15, 0.95))
    return scores


def predict_archetype(profile: dict) -> tuple[int, float]:
    """Assign a Pandora personality archetype (0–5) from a Big Five profile.

    Uses kmeans_pipeline_a (fitted on Pandora Reddit Big Five distributions).
    Falls back to nearest-centroid distance if pipeline unavailable.

    Returns (archetype_id, confidence_score 0–1).
    """
    dims = ["O", "C", "E", "A", "ES"]
    # Convert N → ES for pipeline_a which uses ES dimension
    vec = np.array([
        1.0 - profile["N"] if d == "ES" else profile[d]
        for d in dims
    ])

    if km_pipe_a is not None:
        try:
            km_a  = km_pipe_a["kmeans"]
            sc_a  = km_pipe_a["scaler"]
            vec_s = sc_a.transform([vec])
            cid   = int(km_a.predict(vec_s)[0])
            # Confidence: 1 - normalised distance to assigned centre
            dists = np.linalg.norm(km_a.cluster_centers_ - vec_s, axis=1)
            conf  = float(1.0 - dists[cid] / (dists.sum() + 1e-9))
            conf  = float(np.clip(conf, 0.0, 1.0))
            return cid, conf
        except Exception:
            pass

    # Fallback: use hard-coded centroids from PANDORA_ARCHETYPES
    best_id, best_dist = 0, float("inf")
    for aid, arch in PANDORA_ARCHETYPES.items():
        c = arch["centroid"]
        c_vec = np.array([1.0 - c["N"] if d == "ES" else c[d] for d in dims])
        d = float(np.linalg.norm(c_vec - vec))
        if d < best_dist:
            best_dist, best_id = d, aid
    total = sum(
        1.0 / (np.linalg.norm(
            np.array([1.0 - a["centroid"]["N"] if d == "ES" else a["centroid"][d]
                      for d in dims]) - vec) + 1e-9)
        for a in PANDORA_ARCHETYPES.values()
    )
    conf = float(np.clip(
        (1.0 / (best_dist + 1e-9)) / (total + 1e-9), 0.0, 1.0))
    return best_id, conf


def cosine_sim(a: dict, b: dict) -> float:
    keys = BIG5_KEYS
    va = np.array([a[k] for k in keys])
    vb = np.array([b[k] for k in keys])
    denom = np.linalg.norm(va) * np.linalg.norm(vb)
    return float(np.dot(va, vb) / denom) if denom > 0 else 0.0


def rank_clusters(profile: dict) -> list:
    """Rank career clusters against a Big Five profile.

    When real O*NET artifacts are loaded, uses the trained KMeans model with
    N → ES conversion (Emotional Stability = 1 − Neuroticism) and the fitted
    MinMaxScaler.  Falls back to cosine similarity against hard-coded centroids
    in demo mode.
    """
    if USE_REAL_ONET and onet_arts is not None:
        try:
            km      = onet_arts["km_career"]
            dims    = onet_arts["BIG5_DIMS"]  # ['O','C','E','A','ES']

            # Build input vector in O*NET dimension order.
            # Our Big Five predictions are already in [0,1]; cluster centers are too.
            # Convert N (Neuroticism) → ES (Emotional Stability) = 1 - N.
            # The scaler maps from raw 1-5 Likert data — do NOT apply it here.
            onet_vec = np.array([
                1.0 - profile["N"] if d == "ES" else profile[d]
                for d in dims
            ])
            centers = km.cluster_centers_   # shape (n_clusters, 5), in [0,1] space

            # Distance-based alignment: score = 1 / (1 + euclidean_distance)
            dists  = np.linalg.norm(centers - onet_vec, axis=1)
            raw    = 1.0 / (1.0 + dists)
            # Normalise: max cluster → 95%, spread down to ~20% for real differentiation
            norm   = raw / raw.max()
            scores_onet = 0.20 + 0.75 * norm

            # Map cluster index → our display card
            result = []
            for cluster in CAREER_CLUSTERS:
                cluster_id  = cluster["id"]
                result.append((cluster, float(scores_onet[cluster_id])))
            return sorted(result, key=lambda x: -x[1])
        except Exception:
            pass  # fall through to cosine fallback

    # ── Fallback: cosine similarity (demo mode) ────────────────────
    def _cosine_b5(a, b_dict):
        vb = np.array([b_dict[k] for k in BIG5_KEYS])
        va = np.array([a[k] for k in BIG5_KEYS])
        denom = np.linalg.norm(va) * np.linalg.norm(vb)
        return float(np.dot(va, vb) / denom) if denom > 0 else 0.0

    scored = [(c, _cosine_b5(profile, c["centroid"])) for c in CAREER_CLUSTERS]
    return sorted(scored, key=lambda x: -x[1])


def _parse_job_zone(raw) -> int | None:
    if raw is None:
        return None
    if isinstance(raw, (int, float, np.integer, np.floating)):
        if np.isnan(raw):
            return None
        zone = int(round(float(raw)))
        return zone if 1 <= zone <= 5 else None
    text = str(raw).strip()
    if not text or text.lower() == "nan":
        return None
    match = re.search(r"[1-5]", text)
    if not match:
        return None
    return int(match.group(0))


def _estimate_job_zone(row) -> tuple[int, str]:
    for col in JOB_ZONE_COLS:
        if col in row.index:
            zone = _parse_job_zone(row[col])
            if zone is not None:
                return zone, "onet"

    occ_code = str(row.get("occ_code", ""))
    title = str(row.get("Title", "")).lower()
    zone = SOC_JOB_ZONE_BASE.get(occ_code[:2], 3)

    for floor, keywords in JOB_ZONE_FLOOR_KEYWORDS.items():
        if any(keyword in title for keyword in keywords):
            zone = max(zone, floor)
    for cap, keywords in JOB_ZONE_CAP_KEYWORDS.items():
        if any(keyword in title for keyword in keywords):
            zone = min(zone, cap)
    if any(keyword in title for keyword in JOB_ZONE_SENIORITY_KEYWORDS):
        zone = max(zone, 4)

    return int(np.clip(zone, 1, 5)), "proxy"


def _prepare_occ_explorer_df(df):
    if "job_zone" in df.columns and "job_zone_source" in df.columns and "job_zone_label" in df.columns:
        return df

    df = df.copy()
    zones = []
    sources = []
    labels = []
    for _, row in df.iterrows():
        zone, source = _estimate_job_zone(row)
        zones.append(zone)
        sources.append(source)
        labels.append(JOB_ZONE_LABELS.get(zone, f"Job Zone {zone}"))
    df["job_zone"] = zones
    df["job_zone_source"] = sources
    df["job_zone_label"] = labels
    return df


def rank_occupations(
    profile: dict,
    per_cluster: int = 3,
    min_job_zone: int = 1,
    complexity_weight: float = 0.0,
):
    """Fine-grained O*NET occupation matching — grouped by career cluster.

    For each of the 4 O*NET clusters, returns the `per_cluster` closest
    occupations by Euclidean distance in Big Five space.  Groups are ordered
    by their cluster-level alignment score (best cluster first), so the output
    naturally follows the same ranking as rank_clusters().

    Uses occ_b5_df from onet_career_artifacts.pkl — no SBERT needed.

    Returns (groups, meta), where groups is a list of cluster dicts and meta
    captures the preparation filter / reranking settings used.
    """
    if onet_arts is None:
        return [], {}
    try:
        df   = _prepare_occ_explorer_df(onet_arts["occ_b5_df"])
        dims = onet_arts["BIG5_DIMS"]          # ['O','C','E','A','ES']
        km   = onet_arts["km_career"]

        # User vector (N → ES conversion)
        user_vec = np.array([
            1.0 - profile["N"] if d == "ES" else profile[d]
            for d in dims
        ])

        filtered_df = df[df["job_zone"] >= int(min_job_zone)].copy()
        filter_relaxed = False
        if filtered_df.empty:
            filtered_df = df.copy()
            filter_relaxed = True

        occ_vecs = filtered_df[dims].values             # (n, 5)
        dists    = np.linalg.norm(occ_vecs - user_vec, axis=1)
        scores   = 1.0 / (1.0 + dists)        # raw similarity, no floor
        zone_boost = 1.0 + float(complexity_weight) * (
            (filtered_df["job_zone"].to_numpy() - 1.0) / 4.0
        )
        blended_scores = scores * zone_boost

        # Cluster-level alignment (same formula as rank_clusters)
        centers      = km.cluster_centers_
        c_dists      = np.linalg.norm(centers - user_vec, axis=1)
        c_raw        = 1.0 / (1.0 + c_dists)
        c_norm       = c_raw / c_raw.max()
        c_scores     = 0.20 + 0.75 * c_norm   # consistent with rank_clusters()

        # Build per-cluster groups, ordered by cluster score
        groups = []
        for cid in np.argsort(c_dists):        # best cluster first
            cid = int(cid)
            # Find display cluster name by id
            display_cluster = next((c for c in CAREER_CLUSTERS if c["id"] == cid), None)
            cl_name  = display_cluster["name"]  if display_cluster else f"Cluster {cid}"
            cl_color = display_cluster["icon_color"] if display_cluster else CYAN

            mask  = (filtered_df["cluster"] == cid).values
            idx_c = np.where(mask)[0]
            if len(idx_c) == 0:
                continue
            top_c = idx_c[np.argsort(-blended_scores[idx_c])[:per_cluster]]

            occs = []
            for idx in top_c:
                row = filtered_df.iloc[idx]
                occs.append({
                    "occ_code": row["occ_code"],
                    "title":    row["Title"],
                    "score":    float(blended_scores[idx]),
                    "base_score": float(scores[idx]),
                    "job_zone": int(row["job_zone"]),
                    "job_zone_label": row["job_zone_label"],
                    "job_zone_source": row["job_zone_source"],
                })
            groups.append({
                "cluster_id":    cid,
                "cluster_name":  cl_name,
                "cluster_color": cl_color,
                "cluster_score": float(c_scores[cid]),
                "occupations":   occs,
            })

        meta = {
            "min_job_zone": int(min_job_zone),
            "complexity_weight": float(complexity_weight),
            "total_occupations": int(len(df)),
            "filtered_occupations": int(len(filtered_df)),
            "filter_relaxed": filter_relaxed,
            "using_job_zone_proxy": bool((filtered_df["job_zone_source"] == "proxy").any()),
            "job_zone_source_label": (
                "SOC/title proxy"
                if bool((filtered_df["job_zone_source"] == "proxy").all())
                else "O*NET Job Zone with proxy fallback"
            ),
        }
        return groups, meta
    except Exception:
        return [], {}


def select_next_question(profile: dict, asked: set) -> dict | None:
    """Pick question targeting the trait with highest uncertainty."""
    uncertainties = {k: 0.5 - abs(profile[k] - 0.5) for k in BIG5_KEYS}
    available = [q for i, q in enumerate(QUESTION_POOL)
                 if i not in asked and q["primary"] in uncertainties]
    if not available:
        available = [q for i, q in enumerate(QUESTION_POOL) if i not in asked]
    if not available:
        return None
    most_uncertain = max(uncertainties, key=uncertainties.get)
    targeted = [q for q in available if q["primary"] == most_uncertain]
    pool = targeted if targeted else available
    return pool[0]


# ═══════════════════════════════════════════════════════
# CHARTS
# ═══════════════════════════════════════════════════════
def radar_chart(profile: dict, title: str = "") -> go.Figure:
    cats  = BIG5_NAMES + [BIG5_NAMES[0]]
    vals  = [profile[k] for k in BIG5_KEYS] + [profile[BIG5_KEYS[0]]]
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(
        r=vals, theta=cats, fill="toself",
        fillcolor=f"rgba(0,180,216,0.18)",
        line=dict(color=CYAN, width=2.5),
        name="Your Profile",
    ))
    fig.update_layout(
        polar=dict(
            bgcolor=OFF,
            radialaxis=dict(visible=True, range=[0, 1], tickfont=dict(size=10, color=GREY),
                            gridcolor=LGREY, linecolor=LGREY),
            angularaxis=dict(tickfont=dict(size=12, color=NAVY, family="Arial")),
        ),
        showlegend=False,
        title=dict(text=title, font=dict(color=NAVY, size=14)),
        margin=dict(l=90, r=90, t=50, b=60),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=360,
    )
    return fig


def trait_bars(profile: dict) -> go.Figure:
    names  = BIG5_NAMES
    scores = [profile[k] for k in BIG5_KEYS]
    colors = [CYAN if s >= 0.55 else AMBER if s >= 0.45 else "#CBD5E0" for s in scores]
    fig = go.Figure(go.Bar(
        x=scores, y=names, orientation="h",
        marker=dict(color=colors, line=dict(width=0)),
        text=[f"{s:.0%}" for s in scores],
        textposition="outside",
        textfont=dict(color=NAVY, size=12),
    ))
    fig.add_vline(x=0.5, line=dict(color=LGREY, width=1.5, dash="dot"))
    fig.update_layout(
        xaxis=dict(range=[0, 1.12], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(tickfont=dict(size=13, color=GREY), autorange="reversed"),
        margin=dict(l=10, r=60, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=230,
    )
    return fig


def match_bar(ranked: list) -> go.Figure:
    names  = [c["name"] for c, _ in ranked]
    scores = [s for _, s in ranked]
    colors = [CYAN, "#93C5FD", "#BAE6FD", "#E0F2FE"]
    # Add rank labels: #1 Best Match, #2, #3, #4 Least Match
    rank_labels = {0: " (Best Match)", len(names)-1: " (Least Match)"}
    fig = go.Figure(go.Bar(
        x=scores, y=names, orientation="h",
        marker=dict(color=colors[:len(names)], line=dict(width=0)),
        text=[f"#{i+1} {s:.0%}{rank_labels.get(i, '')}" for i, s in enumerate(scores)],
        textposition="outside",
        textfont=dict(color=NAVY, size=11),
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 1.25], showgrid=False, zeroline=False, showticklabels=False),
        yaxis=dict(tickfont=dict(size=12, color=GREY), autorange="reversed", automargin=True),
        margin=dict(l=0, r=100, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=200,
    )
    return fig


# ═══════════════════════════════════════════════════════
# PDF EXPORT
# ═══════════════════════════════════════════════════════
def generate_pdf(profile: dict, ranked: list, q_count: int, occ_settings: dict | None = None) -> bytes:
    """Build a styled single-page PDF report and return raw bytes."""
    from reportlab.lib.pagesizes import A4
    from reportlab.lib import colors
    from reportlab.lib.units import mm
    from reportlab.platypus import (
        SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable
    )
    from reportlab.lib.styles import ParagraphStyle
    from reportlab.lib.enums import TA_LEFT, TA_CENTER, TA_RIGHT
    from reportlab.graphics.shapes import Drawing, Rect, String, Line
    from reportlab.graphics import renderPDF

    W, H = A4
    buf = io.BytesIO()

    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=14*mm, bottomMargin=14*mm,
    )

    # ── Colour constants ───────────────────────────────
    C_NAVY  = colors.HexColor("#1E3A5F")
    C_CYAN  = colors.HexColor("#00B4D8")
    C_GREEN = colors.HexColor("#10B981")
    C_AMBER = colors.HexColor("#F59E0B")
    C_GREY  = colors.HexColor("#4A5568")
    C_LGREY = colors.HexColor("#E2E8F0")
    C_OFF   = colors.HexColor("#F7FBFD")
    C_WHITE = colors.white

    CLUSTER_COLORS = {
        "People-Centred Leadership": colors.HexColor("#10B981"),
        "Technical & Analytical":   colors.HexColor("#00B4D8"),
        "Creative & Collaborative": colors.HexColor("#7C3AED"),
        "Hands-On Operations":      colors.HexColor("#F59E0B"),
    }

    # ── Styles ─────────────────────────────────────────
    def sty(name, **kw):
        return ParagraphStyle(name, **kw)

    s_header   = sty("hdr",   fontSize=22, textColor=C_NAVY,  fontName="Helvetica-Bold",
                               spaceAfter=1*mm, leading=26)
    s_sub      = sty("sub",   fontSize=9,  textColor=C_GREY,  fontName="Helvetica",
                               spaceAfter=2*mm)
    s_section  = sty("sec",   fontSize=11, textColor=C_NAVY,  fontName="Helvetica-Bold",
                               spaceBefore=2*mm, spaceAfter=1.5*mm)
    s_body     = sty("body",  fontSize=9,  textColor=C_GREY,  fontName="Helvetica",
                               leading=13)
    s_small    = sty("sm",    fontSize=7.5,textColor=C_GREY,  fontName="Helvetica")
    s_right    = sty("rt",    fontSize=8,  textColor=C_GREY,  fontName="Helvetica",
                               alignment=TA_RIGHT)

    story = []

    # ── Header bar (simulated with a coloured table row) ──
    date_str = datetime.now().strftime("%B %d, %Y")
    hdr_data = [[
        Paragraph("<font color='#FFFFFF'><b>PersonaPath</b></font>", sty("ht",
            fontSize=16, textColor=C_WHITE, fontName="Helvetica-Bold")),
        Paragraph(f"<font color='#93C5FD'>SIADS 699 Capstone · Team Foundry · {date_str}</font>",
            sty("hts", fontSize=8, textColor=colors.HexColor("#93C5FD"),
                fontName="Helvetica", alignment=TA_RIGHT)),
    ]]
    hdr_tbl = Table(hdr_data, colWidths=[90*mm, None])
    hdr_tbl.setStyle(TableStyle([
        ("BACKGROUND", (0,0), (-1,-1), C_NAVY),
        ("VALIGN",     (0,0), (-1,-1), "MIDDLE"),
        ("TOPPADDING", (0,0), (-1,-1), 8),
        ("BOTTOMPADDING",(0,0),(-1,-1), 8),
        ("LEFTPADDING", (0,0),(0,-1), 6),
        ("RIGHTPADDING",(-1,0),(-1,-1), 6),
        ("ROUNDEDCORNERS", [6]),
    ]))
    story.append(hdr_tbl)
    story.append(Spacer(1, 2.5*mm))

    # ── Results title ──────────────────────────────────
    story.append(Paragraph("Your PersonaPath Results", s_header))
    story.append(Paragraph(
        f"Based on your written text and {q_count} adaptive follow-up question{'s' if q_count!=1 else ''}.",
        s_sub))

    # ── Best match banner ──────────────────────────────
    top_cluster, top_score = ranked[0]
    cl_color = CLUSTER_COLORS.get(top_cluster["name"], C_CYAN)
    banner_data = [[
        Paragraph("<font color='#10B981'><b>BEST CAREER MATCH</b></font>",
            sty("bml", fontSize=7, textColor=C_GREEN, fontName="Helvetica-Bold")),
        "",
    ],[
        Paragraph(f"<b>{top_cluster['name']}</b>",
            sty("bmn", fontSize=14, textColor=C_NAVY, fontName="Helvetica-Bold")),
        Paragraph(f"<font color='#00B4D8'><b>{top_score:.0%}</b></font> alignment",
            sty("bms", fontSize=14, textColor=C_GREY, fontName="Helvetica",
                alignment=TA_RIGHT)),
    ],[
        Paragraph(top_cluster["description"], s_body),
        Paragraph(f"<font color='#94A3B8'>{top_cluster['n_occ']} O*NET occupations</font>",
            sty("bmocc", fontSize=8, textColor=C_GREY, fontName="Helvetica",
                alignment=TA_RIGHT)),
    ]]
    banner_tbl = Table(banner_data, colWidths=[130*mm, 44*mm])  # 130+44=174mm = full usable A4 width
    banner_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), colors.HexColor("#F0FFF9")),
        ("LINEAFTER",    (0,0), (0,-1),  0, C_WHITE),
        ("BOX",          (0,0), (-1,-1), 1.2, C_GREEN),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 5),
        ("LEFTPADDING",  (0,0), (0,-1),  8),
        ("RIGHTPADDING", (-1,0),(-1,-1), 8),
        ("VALIGN",       (0,0), (-1,-1), "TOP"),
        ("SPAN",         (0,0), (-1,0)),
    ]))
    story.append(banner_tbl)
    story.append(Spacer(1, 2.5*mm))

    # ── Two-column layout: Big Five | Career Alignment ─
    # Left: Big Five bar chart (drawn manually)
    def big5_drawing(profile, w=95*mm):
        bar_h = 9
        gap   = 4
        n     = len(BIG5_KEYS)                       # 5
        # Exact height: n slots + small top/bottom buffer
        h = n * (bar_h + gap) + 10                   # 5×13 + 10 = 75 pts ≈ 26.5mm
        d = Drawing(w, h)
        label_w  = 48*mm                              # shifted right ~1cm (was 38*mm)
        bar_area = w - label_w - 12*mm                # 35mm bar area + 12mm trailing for pct text
        for i, (k, name) in enumerate(zip(BIG5_KEYS, BIG5_NAMES)):
            score = profile[k]
            # Draw from top: first bar near h, last bar near 0
            y = h - (i+1)*(bar_h + gap)
            # Label
            d.add(String(0, y+1, name, fontSize=8, fillColor=C_GREY,
                         fontName="Helvetica"))
            # Background track
            d.add(Rect(label_w, y, bar_area, bar_h,
                       fillColor=C_LGREY, strokeColor=None))
            # Score bar
            fill = C_GREEN if score>=0.60 else (C_AMBER if score>=0.42 else colors.HexColor("#CBD5E0"))
            d.add(Rect(label_w, y, bar_area*score, bar_h,
                       fillColor=fill, strokeColor=None))
            # Percentage label
            d.add(String(label_w + bar_area + 2, y+1, f"{score:.0%}",
                         fontSize=8, fillColor=C_NAVY, fontName="Helvetica-Bold"))
        return d

    # Right: Career alignment bars
    def career_drawing(ranked, w=75*mm):
        bar_h = 9
        gap   = 5
        n     = len(ranked)                           # 4
        # Exact height
        h = n * (bar_h + gap) + 10                   # 4×14 + 10 = 66 pts ≈ 23.3mm
        d = Drawing(w, h)
        label_w  = 40*mm
        bar_area = w - label_w - 12*mm
        cl_colors_list = [C_CYAN,
                          colors.HexColor("#93C5FD"),
                          colors.HexColor("#BAE6FD"),
                          colors.HexColor("#E0F2FE")]
        for i, (cluster, score) in enumerate(ranked):
            y = h - (i+1)*(bar_h + gap)
            short = cluster["name"][:22] + ("…" if len(cluster["name"])>22 else "")
            d.add(String(0, y+1, short, fontSize=7.5, fillColor=C_GREY,
                         fontName="Helvetica"))
            d.add(Rect(label_w, y, bar_area, bar_h,
                       fillColor=C_LGREY, strokeColor=None))
            d.add(Rect(label_w, y, bar_area*score, bar_h,
                       fillColor=cl_colors_list[i], strokeColor=None))
            d.add(String(label_w + bar_area + 2, y+1, f"{score:.0%}",
                         fontSize=8, fillColor=C_NAVY, fontName="Helvetica-Bold"))
        return d

    b5_d     = big5_drawing(profile)
    career_d = career_drawing(ranked)


    chart_data = [[
        Paragraph("<b>Big Five Personality Profile</b>", s_section),
        Paragraph("<b>Career Cluster Alignment</b>", s_section),
    ],[
        b5_d,
        career_d,
    ]]
    # A4 usable width = 210mm - 18mm*2 = 174mm; left col wider for shifted Big Five labels
    chart_tbl = Table(chart_data, colWidths=[97*mm, 77*mm])
    chart_tbl.setStyle(TableStyle([
        ("VALIGN",      (0,0), (-1,-1), "TOP"),
        ("TOPPADDING",  (0,0), (-1,-1), 4),
        ("BOTTOMPADDING",(0,0),(-1,-1), 4),
        ("LEFTPADDING", (0,0), (-1,-1), 6),
        ("RIGHTPADDING",(0,0), (-1,-1), 6),
        ("BOX",         (0,0), (-1,-1), 0.8, C_LGREY),   # outer border
        ("LINEBEFORE",  (1,0), (1,-1),  0.5, C_LGREY),   # divider between columns
    ]))
    story.append(chart_tbl)
    story.append(Spacer(1, 2*mm))

    # ── Career cluster detail cards ────────────────────
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_LGREY, spaceAfter=3*mm))
    story.append(Paragraph("<b>Career Cluster Details</b>", s_section))

    for rank_i, (cluster, score) in enumerate(ranked):
        cl_c = CLUSTER_COLORS.get(cluster["name"], C_CYAN)
        rank_labels = {0:"#1 Best Match", 1:"#2", 2:"#3", 3:"#4"}
        examples_str = " · ".join(cluster["examples"])
        card_data = [[
            Paragraph(
                f'<font color="#{cl_c.hexval()[2:]}">'
                f'<b>{rank_labels[rank_i]}  {cluster["name"]}</b></font>',
                sty(f"cn{rank_i}", fontSize=10, textColor=cl_c, fontName="Helvetica-Bold")),
            Paragraph(f"<b>{score:.0%}</b>",
                sty(f"cs{rank_i}", fontSize=12, textColor=C_CYAN,
                    fontName="Helvetica-Bold", alignment=TA_RIGHT)),
        ],[
            Paragraph(cluster["description"], s_body),
            Paragraph(f"{cluster['n_occ']} occupations", s_right),
        ],[
            Paragraph(f"<i>{examples_str}</i>",
                sty(f"ce{rank_i}", fontSize=7.5, textColor=C_GREY,
                    fontName="Helvetica-Oblique", leading=11)),
            "",
        ]]
        card_tbl = Table(card_data, colWidths=[146*mm, 28*mm])  # 146+28=174mm = full usable width
        bg = colors.HexColor("#F0FFF9") if rank_i == 0 else C_OFF
        border_c = cl_c if rank_i == 0 else C_LGREY
        card_tbl.setStyle(TableStyle([
            ("BACKGROUND",   (0,0), (-1,-1), bg),
            ("BOX",          (0,0), (-1,-1), 0.8, border_c),
            ("TOPPADDING",   (0,0), (-1,-1), 5),
            ("BOTTOMPADDING",(0,0), (-1,-1), 4),
            ("LEFTPADDING",  (0,0), (0,-1),  7),
            ("RIGHTPADDING", (-1,0),(-1,-1), 7),
            ("VALIGN",       (0,0), (-1,-1), "TOP"),
            ("SPAN",         (0,2), (-1,2)),
        ]))
        story.append(card_tbl)
        story.append(Spacer(1, 2*mm))

    # ── MBTI Interpretive Type (PDF) ──────────────────
    mbti_pdf = predict_mbti(profile)
    C_INDIGO = colors.HexColor("#6366F1")
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_LGREY, spaceAfter=2*mm))
    story.append(Paragraph("<b>MBTI Type (derived from Big Five)</b>", s_section))
    mbti_dim_strs = " · ".join(
        f"{d['pole']} ({d['name']}, {d['score']:.0%})" for d in mbti_pdf["dimensions"]
    )
    mbti_card_data = [[
        Paragraph(
            f'<font color="#6366F1"><b>{mbti_pdf["type"]}  —  The {mbti_pdf["nickname"]}</b></font>',
            sty("mbti_t", fontSize=12, textColor=C_INDIGO, fontName="Helvetica-Bold")),
    ],[
        Paragraph(mbti_pdf["tagline"], s_body),
    ],[
        Paragraph(f"<i>{mbti_dim_strs}</i>",
            sty("mbti_d", fontSize=7.5, textColor=C_GREY,
                fontName="Helvetica-Oblique", leading=11)),
    ]]
    mbti_tbl = Table(mbti_card_data, colWidths=[174*mm])
    mbti_tbl.setStyle(TableStyle([
        ("BACKGROUND",   (0,0), (-1,-1), colors.HexColor("#EEF2FF")),
        ("BOX",          (0,0), (-1,-1), 0.8, C_INDIGO),
        ("TOPPADDING",   (0,0), (-1,-1), 5),
        ("BOTTOMPADDING",(0,0), (-1,-1), 4),
        ("LEFTPADDING",  (0,0), (0,-1),  7),
        ("RIGHTPADDING", (-1,0),(-1,-1), 7),
        ("VALIGN",       (0,0), (-1,-1), "TOP"),
    ]))
    story.append(mbti_tbl)
    story.append(Spacer(1, 2*mm))

    # ── O*NET Occupation Explorer ──────────────────────
    occ_settings = occ_settings or {}
    occ_groups, occ_meta = rank_occupations(
        profile,
        per_cluster=3,
        min_job_zone=int(occ_settings.get("min_job_zone", 3)),
        complexity_weight=float(occ_settings.get("complexity_weight", 0.10)),
    )
    if occ_groups:
        story.append(HRFlowable(width="100%", thickness=0.5, color=C_LGREY, spaceAfter=2*mm))
        story.append(Paragraph("<b>O*NET Occupation Explorer</b>", s_section))
        source_note = "Preparation layer uses proxy estimates from SOC/title signals." if occ_meta.get("using_job_zone_proxy") else "Preparation layer uses O*NET Job Zone data."
        story.append(Paragraph(
            "Top 3 closest occupations per career cluster after filtering by minimum preparation level "
            f"({JOB_ZONE_LABELS.get(occ_meta.get('min_job_zone', 3), 'Job Zone 3')}). "
            f"Complexity boost: {occ_meta.get('complexity_weight', 0.0):.2f}. {source_note}",
            sty("occ_note", fontSize=7.5, textColor=C_GREY, fontName="Helvetica",
                spaceAfter=2*mm, leading=11)
        ))
        # Build flat table: cluster header rows + occupation rows
        CLUSTER_COL_MAP = {c["id"]: CLUSTER_COLORS.get(c["name"], C_CYAN) for c in CAREER_CLUSTERS}
        pdf_occ_rows = []
        for group in occ_groups:
            cl_col = CLUSTER_COL_MAP.get(group["cluster_id"], C_CYAN)
            # Cluster header row
            pdf_occ_rows.append([
                Paragraph(
                    f'<font color="#{cl_col.hexval()[2:]}"><b>{group["cluster_name"]}</b></font>',
                    sty(f'ch{group["cluster_id"]}', fontSize=8.5, fontName="Helvetica-Bold",
                        textColor=cl_col)),
                Paragraph(
                    f'<font color="#{cl_col.hexval()[2:]}"><b>{group["cluster_score"]:.0%} cluster fit</b></font>',
                    sty(f'cs{group["cluster_id"]}', fontSize=8, fontName="Helvetica-Bold",
                        textColor=cl_col, alignment=2)),
            ])
            for j, occ in enumerate(group["occupations"]):
                prep_text = f"{JOB_ZONE_SHORT.get(occ['job_zone'], 'JZ?')} · {occ['job_zone_label'].split('·', 1)[-1].strip()}"
                if occ["job_zone_source"] == "proxy":
                    prep_text += " (proxy)"
                pdf_occ_rows.append([
                    Paragraph(f"{j+1}.  {occ['title']}<br/><font color='#94A3B8'>{prep_text}</font>",
                        sty(f"ot{group['cluster_id']}{j}", fontSize=7.5, fontName="Helvetica",
                            textColor=C_NAVY, leftIndent=8, leading=11)),
                    Paragraph(f"{occ['score']:.0%}",
                        sty(f"os{group['cluster_id']}{j}", fontSize=7.5,
                            fontName="Helvetica-Bold", textColor=C_GREY, alignment=2)),
                ])

        occ_tbl = Table(pdf_occ_rows, colWidths=[145*mm, 22*mm])
        row_styles = [
            ("TOPPADDING",    (0, 0), (-1, -1), 3),
            ("BOTTOMPADDING", (0, 0), (-1, -1), 2),
            ("LEFTPADDING",   (0, 0), (-1, -1), 5),
            ("RIGHTPADDING",  (0, 0), (-1, -1), 5),
            ("VALIGN",        (0, 0), (-1, -1), "MIDDLE"),
            ("BOX",           (0, 0), (-1, -1), 0.5, C_LGREY),
        ]
        # Shade cluster header rows (every 4th row starting at 0)
        for gi in range(len(occ_groups)):
            hrow = gi * 4
            row_styles.append(("BACKGROUND", (0, hrow), (-1, hrow), C_OFF))
            row_styles.append(("LINEABOVE",  (0, hrow), (-1, hrow), 0.5, C_LGREY))
        occ_tbl.setStyle(TableStyle(row_styles))
        story.append(occ_tbl)
        story.append(Spacer(1, 2*mm))

    # ── Footer ─────────────────────────────────────────
    story.append(Spacer(1, 2*mm))
    story.append(HRFlowable(width="100%", thickness=0.5, color=C_LGREY))
    story.append(Paragraph(
        "Results generated by ML models trained on Essays Big Five, Pandora Reddit, and MBTI Reddit datasets. "
        "Treat as a reflective tool — not a clinical or professional assessment. "
        "Team Foundry · SIADS 699 Capstone · University of Michigan.",
        sty("ft", fontSize=7, textColor=colors.HexColor("#94A3B8"),
            fontName="Helvetica", leading=10, spaceBefore=2*mm)
    ))

    doc.build(story)
    return buf.getvalue()


# ═══════════════════════════════════════════════════════
# CSS
# ═══════════════════════════════════════════════════════
def inject_css():
    st.markdown(f"""
    <style>
    /* ── Fonts: Inter (UI) · Fira Code / JetBrains Mono (stats) ── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&family=Fira+Code:wght@400;500;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    html, body, [class*="css"] {{ font-family: 'Inter', Arial, sans-serif; }}

    /* ── Top bar ───────────────────────────────────── */
    .topbar {{
        background: {NAVY};
        color: {WHITE};
        padding: 0.7rem 2rem;
        border-radius: 0 0 10px 10px;
        display: flex;
        align-items: center;
        gap: 1rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 3px 14px rgba(30,58,95,0.18);
        border-bottom: 2px solid {CYAN};
    }}
    .topbar-logo {{
        width: 42px; height: 42px;
        border-radius: 8px;
        background: {CYAN};
        display: flex; align-items: center; justify-content: center;
        font-size: 0.85rem; font-weight: 800; color: {WHITE};
        letter-spacing: 0.02em; flex-shrink: 0;
    }}
    .topbar-title {{ font-size: 1.4rem; font-weight: 800; letter-spacing: -0.02em; }}
    .topbar-sub   {{ font-size: 0.82rem; color: #93C5FD; margin-top: 2px; }}

    /* ── Mode badge ────────────────────────────────── */
    .mode-badge-live {{
        background: {GREEN}; color: #fff;
        font-size: 0.68rem; font-weight: 700;
        padding: 3px 10px; border-radius: 20px; margin-left: 10px;
        vertical-align: middle; letter-spacing: 0.05em; text-transform: uppercase;
    }}
    .mode-badge-demo {{
        background: {AMBER}; color: #fff;
        font-size: 0.68rem; font-weight: 700;
        padding: 3px 10px; border-radius: 20px; margin-left: 10px;
        vertical-align: middle; letter-spacing: 0.05em; text-transform: uppercase;
    }}

    /* ── Cards (Swiss Modernism flat card) ─────────── */
    .card {{
        background: {WHITE};
        border: 1px solid {LGREY};
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        box-shadow: 0 2px 4px rgba(0,0,0,0.05);
        margin-bottom: 1rem;
        transition: transform 0.18s ease, box-shadow 0.18s ease;
    }}
    .card:hover {{
        transform: translateY(-2px) scale(1.005);
        box-shadow: 0 6px 18px rgba(30,58,95,0.09);
    }}
    .card-accent  {{ border-left: 4px solid {CYAN};  }}
    .card-green   {{ border-left: 4px solid {GREEN}; background: #F0FFF9; }}
    .card-amber   {{ border-left: 4px solid {AMBER}; background: #FFFBEB; }}

    /* ── Step pill + heading ───────────────────────── */
    .step-pill {{
        display: inline-block;
        background: {CYAN};
        color: {WHITE};
        font-size: 0.7rem; font-weight: 700;
        padding: 0.22rem 0.8rem;
        border-radius: 4px;
        margin-bottom: 0.6rem;
        letter-spacing: 0.06em; text-transform: uppercase;
    }}
    .step-title {{
        font-size: 1.4rem; font-weight: 800;
        color: {NAVY}; margin-bottom: 0.3rem; letter-spacing: -0.02em;
    }}
    .step-sub {{
        font-size: 0.95rem; color: {GREY}; margin-bottom: 1.2rem; line-height: 1.55;
    }}

    /* ── Trait badge ───────────────────────────────── */
    .trait-badge {{
        display: inline-block;
        background: {OFF}; border: 1px solid {LGREY};
        color: {NAVY}; font-size: 0.78rem; font-weight: 600;
        padding: 0.2rem 0.6rem; border-radius: 6px; margin-right: 0.4rem;
    }}

    /* ── Cluster icon badge ────────────────────────── */
    .cluster-icon {{
        display: inline-flex; align-items: center; justify-content: center;
        width: 36px; height: 36px; border-radius: 9px;
        font-size: 0.72rem; font-weight: 800; color: {WHITE};
        letter-spacing: 0.03em; flex-shrink: 0; margin-right: 0.6rem;
    }}

    /* ── Feature icon badge ────────────────────────── */
    .feat-icon {{
        display: inline-flex; align-items: center; justify-content: center;
        width: 32px; height: 32px; border-radius: 8px;
        font-size: 0.68rem; font-weight: 800; color: {WHITE};
        flex-shrink: 0;
    }}

    /* ── Rank badge ────────────────────────────────── */
    .rank-badge {{
        display: inline-flex; align-items: center; justify-content: center;
        width: 26px; height: 26px; border-radius: 6px;
        font-size: 0.78rem; font-weight: 800; color: {WHITE};
        margin-right: 0.45rem; flex-shrink: 0;
    }}
    .r1 {{ background: {NAVY}; }}
    .r2 {{ background: {GREY}; }}
    .r3 {{ background: #94A3B8; }}
    .r4 {{ background: {LGREY}; color: {GREY}; }}

    /* ── Career card ───────────────────────────────── */
    .career-card {{
        background: {WHITE};
        border: 1.5px solid {LGREY};
        border-radius: 10px;
        padding: 1.1rem 1.3rem;
        margin-bottom: 0.8rem;
        transition: transform 0.18s ease, box-shadow 0.18s ease, border-color 0.18s ease;
    }}
    .career-card:hover {{
        transform: translateY(-2px) scale(1.01);
        box-shadow: 0 6px 20px rgba(30,58,95,0.08);
        border-color: #BAE6FD;
    }}
    .career-card.top {{
        border-color: {CYAN};
        border-left: 4px solid {CYAN};
        box-shadow: 0 2px 8px rgba(0,180,216,0.10);
    }}
    .career-card.top:hover {{
        box-shadow: 0 6px 20px rgba(0,180,216,0.16);
    }}
    .career-name {{ font-size: 0.98rem; font-weight: 700; color: {NAVY}; }}
    /* Fira Code / JetBrains Mono for stat numbers — crisp and data-forward */
    .career-pct  {{
        font-size: 1.35rem; font-weight: 700; color: {CYAN};
        font-family: 'Fira Code', 'JetBrains Mono', monospace; letter-spacing: -0.03em;
    }}
    .career-desc {{ font-size: 0.86rem; color: {GREY}; margin: 0.35rem 0; line-height: 1.45; }}
    .career-chip {{
        display: inline-block;
        background: {OFF}; border: 1px solid {LGREY};
        color: {GREY}; font-size: 0.74rem;
        padding: 0.15rem 0.5rem; border-radius: 20px;
        margin: 0.15rem 0.1rem 0 0;
        transition: background 0.15s;
    }}
    .career-chip:hover {{ background: #EEF6FF; }}

    /* ── Progress bar ──────────────────────────────── */
    .progress-steps {{ display: flex; align-items: center; gap: 0.3rem; margin-bottom: 1.5rem; }}
    .ps {{ width: 28px; height: 6px; border-radius: 3px; }}
    .ps.done   {{ background: {CYAN}; }}
    .ps.active {{ background: {NAVY}; }}
    .ps.todo   {{ background: {LGREY}; }}

    /* ── Info strip (replaces warning emoji) ───────── */
    .info-strip {{
        border-left: 4px solid #CBD5E0;
        background: #F8FAFC;
        padding: 0.65rem 1rem;
        border-radius: 0 8px 8px 0;
        font-size: 0.8rem;
        color: #64748B;
        margin-top: 1rem;
        line-height: 1.5;
    }}

    /* ── Disclaimer ────────────────────────────────── */
    .disclaimer {{
        font-size: 0.78rem; color: #94A3B8;
        border-top: 1px solid {LGREY};
        padding-top: 0.8rem; margin-top: 1rem; line-height: 1.5;
    }}

    /* ── Voice button (streamlit-mic-recorder) ────── */
    [data-testid="stVerticalBlock"] iframe[title*="mic_recorder"] {{
        height: 42px !important;
    }}
    /* Style the mic recorder container to align right */
    div:has(> iframe[title*="mic_recorder"]) {{
        display: flex;
        justify-content: flex-end;
        margin-top: -12px;
        margin-bottom: 4px;
    }}

    /* ── Hide Streamlit chrome ─────────────────────── */
    #MainMenu, footer, header {{ visibility: hidden; }}
    .stDeployButton {{ display: none; }}
    .block-container {{ padding-top: 1rem; max-width: 1100px; margin-left: auto !important; margin-right: auto !important; }}

    /* ── Button polish ─────────────────────────────── */
    .stButton > button[kind="primary"] {{
        background: {NAVY};
        border: none; border-radius: 6px; font-weight: 600;
        letter-spacing: 0.01em; transition: opacity 0.15s;
    }}
    .stButton > button[kind="primary"]:hover {{ opacity: 0.85; }}
    </style>
    """, unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════
# STATE INIT
# ═══════════════════════════════════════════════════════
STEPS_TOTAL = 7  # welcome + text + 3 Q&A + processing + results

def init_state():
    defaults = {
        "step":      0,
        "texts":     [],
        "asked":     set(),
        "profile":   None,
        "ranked":    None,
        "q_count":   0,
        "mode":      "open",        # "open" | "likert" | "hybrid"
        "likert_responses": [None] * len(LIKERT_ITEMS),
        "likert_page": 0,           # which batch of 5 Likert items
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()

def reset():
    for k in list(st.session_state.keys()):
        del st.session_state[k]
    st.rerun()

def go_step(n):
    st.session_state.step = n
    st.rerun()


# ═══════════════════════════════════════════════════════
# COMPONENTS
# ═══════════════════════════════════════════════════════
def topbar():
    # ── Primary mode badge ──────────────────────────────────────────
    if USE_REAL_B5 and USE_REAL_ONET:
        mode_badge = '<span class="mode-badge-live">Live models</span>'
    elif USE_REAL_ONET:
        mode_badge = '<span class="mode-badge-live">O*NET live</span> <span class="mode-badge-demo">B5 demo</span>'
    else:
        mode_badge = '<span class="mode-badge-demo">Demo mode</span>'

    # ── Secondary capability dots ───────────────────────────────────
    # Each dot = one feature; green = active, grey = unavailable
    def dot(label, active):
        col = GREEN if active else "#64748B"
        icon = "●" if active else "○"
        return (f'<span style="font-size:0.65rem;color:{col};'
                f'margin-right:0.55rem;white-space:nowrap;">'
                f'{icon} {label}</span>')

    caps = (
        dot("SBERT",            USE_REAL_B5)
      + dot("B5 RandomForest",  USE_REAL_B5)
      + dot("Pandora Ensemble", USE_REAL_B5 and km_pipe_b is not None)
      + dot("O*NET matching",   USE_REAL_ONET)
      + dot("Archetypes",       km_pipe_a is not None)
    )

    st.markdown(f"""
    <div class="topbar">
      <div class="topbar-logo">PP</div>
      <div style="flex:1;">
        <div class="topbar-title">PersonaPath {mode_badge}</div>
        <div class="topbar-sub" style="margin-top:3px;">{caps}</div>
      </div>
    </div>""", unsafe_allow_html=True)


def progress_indicator(current_step: int):
    labels = ["Welcome", "Q 1", "Q 2", "Q 3", "Q 4", "Q 5", "Q 6", "Q 7", "Q 8", "Results"]
    n = len(labels)
    bars = ""
    for i in range(n):
        cls = "done" if i < current_step else ("active" if i == current_step else "todo")
        bars += f'<div class="ps {cls}"></div>'
    # Q pages (steps 1-8): show "Q X of 8"; Welcome/Results: just label
    if 1 <= current_step <= 8:
        step_text = f'Q {current_step} of 8 &nbsp;—&nbsp; <b>{labels[current_step]}</b>'
    else:
        step_text = f'<b>{labels[min(current_step, n-1)]}</b>'
    label_html = f'<span style="font-size:0.82rem;color:{GREY};">{step_text}</span>'
    st.markdown(f'<div class="progress-steps">{bars}</div>{label_html}', unsafe_allow_html=True)


def cluster_icon_html(label: str, color: str, size: int = 36) -> str:
    """Return an inline HTML badge with 2-letter label and solid background color."""
    return (
        f'<span style="display:inline-flex;align-items:center;justify-content:center;'
        f'width:{size}px;height:{size}px;border-radius:{size//4}px;'
        f'background:{color};color:#fff;font-size:{size*0.3:.0f}px;'
        f'font-weight:800;letter-spacing:0.03em;flex-shrink:0;">'
        f'{label}</span>'
    )


def voice_input_widget(textarea_key: str, lang: str = "en") -> None:
    """Render a compact mic button. Recognised text is injected into the textarea via session_state."""
    if not HAS_MIC_RECORDER:
        st.caption("Voice input unavailable: install `streamlit-mic-recorder` to enable microphone capture.")
        return

    stt_key = f"stt_{textarea_key}"

    text = speech_to_text(
        language=lang,
        start_prompt="Voice",
        stop_prompt="Stop",
        just_once=False,
        use_container_width=False,
        key=stt_key,
    )

    if text:
        # Append recognised text into the textarea's session_state key
        prev = st.session_state.get(textarea_key, "")
        st.session_state[textarea_key] = (prev + " " + text).strip() if prev else text


def rank_badge_html(rank: int) -> str:
    colors = {1: NAVY, 2: GREY, 3: "#94A3B8", 4: "#CBD5E0"}
    text_colors = {1: WHITE, 2: WHITE, 3: WHITE, 4: GREY}
    bg = colors.get(rank, "#CBD5E0")
    tc = text_colors.get(rank, GREY)
    return (
        f'<span style="display:inline-flex;align-items:center;justify-content:center;'
        f'width:24px;height:24px;border-radius:6px;background:{bg};'
        f'color:{tc};font-size:0.75rem;font-weight:800;margin-right:0.45rem;'
        f'flex-shrink:0;vertical-align:middle;">{rank}</span>'
    )


# ═══════════════════════════════════════════════════════
# PAGES
# ═══════════════════════════════════════════════════════

# ── Page 0: Welcome ────────────────────────────────────
def page_welcome():
    col_main, col_side = st.columns([3, 2], gap="large")
    with col_main:
        st.markdown(f"""
        <div style="margin-top:1rem;margin-bottom:1.6rem;">
          <div style="font-size:0.78rem;font-weight:700;color:{CYAN};letter-spacing:0.12em;
                      text-transform:uppercase;margin-bottom:0.6rem;">
            SIADS 699 Capstone &nbsp;·&nbsp; Team Foundry
          </div>
          <div style="font-size:2.6rem;font-weight:800;color:{NAVY};line-height:1.15;
                      letter-spacing:-0.03em;margin-bottom:0.9rem;">
            Discover careers that<br>fit your personality.
          </div>
          <div style="font-size:1.02rem;color:{GREY};max-width:500px;line-height:1.65;">
            Write freely about yourself, answer a few adaptive questions, and PersonaPath
            infers your Big Five personality profile and matches it to real career clusters
            from the O*NET database.
          </div>
        </div>""", unsafe_allow_html=True)

        # Bento-style feature grid (3 cards in a row)
        features = [
            ("B5", NAVY,   "Big Five Model",
             "SBERT + LR trained on 8,000+ samples",
             "8,000+", "samples"),
            ("ON", CYAN,   "O*NET Matching",
             "894 occupations, Sackett & Walmsley crosswalk",
             "894", "occupations"),
            ("QA", PURPLE, "Adaptive Q&A",
             "Entropy-guided, targets most uncertain traits",
             "3", "questions"),
        ]
        feat_html = '<div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.7rem;margin-bottom:1.5rem;">'
        for label, color, title, desc, stat, stat_label in features:
            icon = cluster_icon_html(label, color, size=32)
            feat_html += f"""
            <div style="background:{WHITE};border:1px solid {LGREY};border-radius:20px;
                        padding:1.1rem 1.1rem 0.9rem;
                        box-shadow:0 4px 6px rgba(0,0,0,0.04);
                        transition:transform 0.18s,box-shadow 0.18s;"
                 onmouseover="this.style.transform='translateY(-2px)';this.style.boxShadow='0 8px 20px rgba(30,58,95,0.10)'"
                 onmouseout="this.style.transform='';this.style.boxShadow='0 4px 6px rgba(0,0,0,0.04)'">
              {icon}
              <div style="font-size:0.82rem;font-weight:700;color:{NAVY};margin-top:0.6rem;">{title}</div>
              <div style="font-size:0.75rem;color:{GREY};line-height:1.4;margin-top:0.15rem;">{desc}</div>
              <div style="margin-top:0.6rem;border-top:1px solid {LGREY};padding-top:0.5rem;">
                <span style="font-family:'JetBrains Mono',monospace;font-size:1.15rem;
                             font-weight:700;color:{color};letter-spacing:-0.02em;">{stat}</span>
                <span style="font-size:0.72rem;color:{GREY};margin-left:0.3rem;">{stat_label}</span>
              </div>
            </div>"""
        feat_html += '</div>'
        st.markdown(feat_html, unsafe_allow_html=True)

        st.markdown(f"""
        <div style="margin-bottom:0.8rem;">
          <div style="font-size:0.82rem;font-weight:700;color:{NAVY};margin-bottom:0.4rem;">
            Choose your assessment style
          </div>
        </div>""", unsafe_allow_html=True)

        mode = st.radio(
            "Assessment mode",
            options=["open", "likert", "hybrid"],
            format_func=lambda m: {
                "open":   "Open-Text (write freely + adaptive Q&A)",
                "likert": "Standard Scale (25 IPIP-NEO Likert items, like bigfive-test.com)",
                "hybrid": "Hybrid (Likert scale first, then open-text for fine-tuning)",
            }[m],
            index=0,
            label_visibility="collapsed",
        )

        if st.button("Get Started  →", type="primary", use_container_width=False):
            st.session_state.mode = mode
            if mode == "likert":
                go_step(100)   # Likert flow
            elif mode == "hybrid":
                go_step(100)   # Likert first, then open-text
            else:
                go_step(1)     # Original open-text flow

        st.markdown(f"""
        <div class="info-strip" style="margin-top:1.2rem;">
          PersonaPath is an academic research prototype. Results are for exploration only
          and should not be used for professional, clinical, or hiring decisions.
        </div>""", unsafe_allow_html=True)

    with col_side:
        demo_profile = {"O": 0.72, "C": 0.58, "E": 0.65, "A": 0.80, "N": 0.38}
        st.plotly_chart(radar_chart(demo_profile, "Sample Profile"), use_container_width=True)

        # Mini stats row below radar
        st.markdown(f"""
        <div style="display:grid;grid-template-columns:repeat(3,1fr);gap:0.5rem;margin-top:-0.3rem;">
          <div style="background:{OFF};border:1px solid {LGREY};border-radius:12px;
                      padding:0.6rem;text-align:center;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                        font-weight:700;color:{NAVY};">4</div>
            <div style="font-size:0.68rem;color:{GREY};">clusters</div>
          </div>
          <div style="background:{OFF};border:1px solid {LGREY};border-radius:12px;
                      padding:0.6rem;text-align:center;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                        font-weight:700;color:{CYAN};">5</div>
            <div style="font-size:0.68rem;color:{GREY};">traits</div>
          </div>
          <div style="background:{OFF};border:1px solid {LGREY};border-radius:12px;
                      padding:0.6rem;text-align:center;">
            <div style="font-family:'JetBrains Mono',monospace;font-size:1.1rem;
                        font-weight:700;color:{GREEN};">~5 min</div>
            <div style="font-size:0.68rem;color:{GREY};">total</div>
          </div>
        </div>
        <div style="text-align:center;font-size:0.78rem;color:{GREY};margin-top:0.7rem;">
          Example output — your results will vary
        </div>""", unsafe_allow_html=True)


# ── Page 1: Initial Text Input ──────────────────────────
def page_text():
    progress_indicator(1)
    st.markdown(f"""
    <div class="step-pill">Q 1 of 8</div>
    <div class="step-title">Tell us about yourself</div>
    <div class="step-sub">Write freely — there are no right or wrong answers.
    Describe your interests, how you work, what energises you, or anything else that feels relevant.</div>
    """, unsafe_allow_html=True)

    # Voice → sets session_state["ta_main"] before textarea renders
    voice_input_widget("ta_main")

    user_text = st.text_area(
        label="Your text",
        placeholder="Example: I love solving puzzles and building things from scratch. I prefer working independently "
                    "on deep technical problems, though I enjoy explaining my findings to others once I'm done...",
        height=220,
        max_chars=3000,
        key="ta_main",
        label_visibility="collapsed",
    )
    char_count = len(user_text)
    col_hint, col_btn = st.columns([3, 1])
    with col_hint:
        color = GREEN if char_count >= 100 else AMBER if char_count >= 30 else GREY
        st.markdown(
            f'<span style="font-size:0.82rem;color:{color};">'
            f'{char_count}/3000 characters'
            + (" — great, ready to analyse!" if char_count >= 50
               else " — a bit more detail will improve accuracy" if char_count >= 8
               else " — write something to get started")
            + '</span>', unsafe_allow_html=True
        )
    with col_btn:
        disabled = char_count < 8
        if st.button("Analyse  →", type="primary", disabled=disabled, use_container_width=True):
            st.session_state.texts = [user_text]
            sbert_profile = predict_big5(st.session_state.texts)
            # In hybrid mode, blend SBERT + Likert scores (50/50)
            if st.session_state.mode == "hybrid" and "likert_profile" in st.session_state:
                lp = st.session_state["likert_profile"]
                blended = {t: 0.5 * sbert_profile[t] + 0.5 * lp[t] for t in BIG5_KEYS}
                st.session_state.profile = blended
            else:
                st.session_state.profile = sbert_profile
            st.session_state.ranked  = rank_clusters(st.session_state.profile)
            go_step(2)


# ── Pages 2-4: Adaptive Q&A ─────────────────────────────
def page_question(q_num: int):
    progress_indicator(q_num)

    q = select_next_question(st.session_state.profile, st.session_state.asked)
    if q is None:
        go_step(9)
        return

    q_idx = next(i for i, qp in enumerate(QUESTION_POOL) if qp is q)

    trait_full = dict(zip(BIG5_KEYS, BIG5_NAMES))[q["primary"]]
    follow_num = q_num        # q_num 2→Q2, 3→Q3, … 8→Q8
    st.markdown(f"""
    <div class="step-pill">Q {follow_num} of 8 — Adaptive</div>
    <div class="step-title">Refining your profile</div>
    <div class="step-sub">
      This question helps clarify your <b>{trait_full}</b> score.
    </div>""", unsafe_allow_html=True)



    with st.expander("View current profile snapshot", expanded=False):
        col_r, col_b = st.columns(2)
        with col_r:
            st.plotly_chart(radar_chart(st.session_state.profile), use_container_width=True)
        with col_b:
            st.plotly_chart(trait_bars(st.session_state.profile), use_container_width=True)

    st.markdown(f"""
    <div class="card card-accent" style="margin:1rem 0;">
      <div style="font-size:1.05rem;font-weight:600;color:{NAVY};">
        {q['q']}
      </div>
    </div>""", unsafe_allow_html=True)

    # Voice → sets session_state key before textarea renders
    qa_key = f"qa_{q_num}_{q_idx}"
    voice_input_widget(qa_key)

    answer = st.text_area(
        "Your answer",
        placeholder="Write as much or as little as feels natural...",
        height=140,
        key=qa_key,
        label_visibility="collapsed",
    )

    col_skip, col_next = st.columns([1, 1])
    with col_skip:
        if st.button("Skip this question", use_container_width=True):
            st.session_state.asked.add(q_idx)
            st.session_state.q_count += 1
            next_step = q_num + 1 if q_num < 8 else 9
            go_step(next_step)
    with col_next:
        if st.button("Next  →", type="primary", disabled=len(answer.strip()) < 5,
                     use_container_width=True):
            st.session_state[f"_q_text_{follow_num}"] = q["q"]
            st.session_state.texts.append(answer)
            st.session_state.asked.add(q_idx)
            st.session_state.q_count += 1
            sbert_p = predict_big5(st.session_state.texts)
            if st.session_state.mode == "hybrid" and "likert_profile" in st.session_state:
                lp = st.session_state["likert_profile"]
                sbert_p = {t: 0.5 * sbert_p[t] + 0.5 * lp[t] for t in BIG5_KEYS}
            st.session_state.profile = sbert_p
            st.session_state.ranked  = rank_clusters(st.session_state.profile)
            next_step = q_num + 1 if q_num < 8 else 9
            go_step(next_step)


# ── Page 9+: Results ────────────────────────────────────
def page_results():
    progress_indicator(9)
    profile = st.session_state.profile
    ranked  = st.session_state.ranked

    # Detect if profile is "balanced" (all traits within ±0.15 of 0.5)
    trait_spread = max(profile.values()) - min(profile.values())
    is_balanced  = trait_spread < 0.30

    balance_note = ""
    if is_balanced:
        balance_note = (
            f'<div style="display:inline-block;background:#EFF6FF;border:1px solid #BFDBFE;'
            f'border-radius:10px;padding:0.35rem 0.8rem;font-size:0.82rem;color:#1E40AF;'
            f'margin-top:0.4rem;">'
            f'Your traits are broadly balanced — you have meaningful fit across multiple career directions.'
            f'</div>'
        )

    st.markdown(f"""
    <div class="step-title" style="font-size:1.6rem;">Your PersonaPath Results</div>
    <div class="step-sub">{'Based on 25 IPIP-NEO Likert-scale items.' if st.session_state.mode == 'likert'
        else f'Based on your text and {st.session_state.q_count} follow-up question{"s" if st.session_state.q_count != 1 else ""}.'
        + (' (Hybrid: Likert + open-text blended)' if st.session_state.mode == 'hybrid' else '')}</div>
    {balance_note}
    """, unsafe_allow_html=True)

    # ── Top match banner (Bento hero card) ────────────
    top_cluster, top_score = ranked[0]
    icon_html = cluster_icon_html(top_cluster["icon_label"], top_cluster["icon_color"], size=48)
    chips_top = " ".join(
        f'<span style="display:inline-block;background:{OFF};border:1px solid {LGREY};'
        f'color:{GREY};font-size:0.73rem;padding:0.12rem 0.48rem;border-radius:20px;'
        f'margin:0.1rem 0.1rem 0 0;">{ex}</span>'
        for ex in top_cluster["examples"][:3]
    )
    st.markdown(f"""
    <div style="background:linear-gradient(135deg,#F0FFF9 0%,#E6F7FF 100%);
         border:1.5px solid {GREEN};border-radius:24px;
         padding:1.4rem 1.8rem;margin-bottom:1.5rem;
         box-shadow:0 4px 16px rgba(16,185,129,0.10);
         display:flex;align-items:flex-start;gap:1.2rem;flex-wrap:wrap;">
      {icon_html}
      <div style="flex:1;min-width:200px;">
        <div style="font-size:0.72rem;font-weight:700;color:{GREEN};text-transform:uppercase;
                    letter-spacing:0.1em;margin-bottom:4px;">Best Career Match</div>
        <div style="display:flex;align-items:baseline;gap:0.8rem;flex-wrap:wrap;">
          <span style="font-size:1.3rem;font-weight:800;color:{NAVY};letter-spacing:-0.02em;">
            {top_cluster['name']}
          </span>
          <span style="font-family:'JetBrains Mono',monospace;font-size:1.6rem;
                       font-weight:700;color:{CYAN};letter-spacing:-0.04em;">
            {top_score:.0%}
          </span>
          <span style="font-size:0.82rem;color:{GREY};">alignment</span>
        </div>
        <div style="color:{GREY};font-size:0.9rem;margin-top:0.4rem;line-height:1.45;">
          {top_cluster['description']}
        </div>
        <div style="margin-top:0.5rem;">{chips_top}</div>
      </div>
      <div style="text-align:right;flex-shrink:0;">
        <div style="font-size:0.68rem;color:{GREY};margin-bottom:2px;text-transform:uppercase;
                    letter-spacing:0.08em;">O*NET occupations</div>
        <div style="font-family:'JetBrains Mono',monospace;font-size:1.8rem;
                    font-weight:700;color:{NAVY};">{top_cluster['n_occ']}</div>
      </div>
    </div>""", unsafe_allow_html=True)

    # ── Pandora Personality Archetype ──────────────────
    archetype_id, arch_conf = predict_archetype(profile)
    arch = PANDORA_ARCHETYPES[archetype_id]
    arch_col = arch["color"]
    # Radar comparison: user profile vs archetype centroid
    radar_labels = BIG5_NAMES + [BIG5_NAMES[0]]
    user_vals    = [profile[k] for k in BIG5_KEYS] + [profile[BIG5_KEYS[0]]]
    arch_c       = arch["centroid"]
    arch_vals    = [arch_c[k] for k in BIG5_KEYS] + [arch_c[BIG5_KEYS[0]]]

    arch_radar = go.Figure()
    arch_radar.add_trace(go.Scatterpolar(
        r=arch_vals, theta=radar_labels, fill="toself",
        fillcolor=f"rgba(100,116,139,0.12)",
        line=dict(color="#94A3B8", width=1.5, dash="dot"),
        name="Archetype avg",
    ))
    arch_radar.add_trace(go.Scatterpolar(
        r=user_vals, theta=radar_labels, fill="toself",
        fillcolor=f"rgba(0,180,216,0.18)",
        line=dict(color=CYAN, width=2.5),
        name="Your profile",
    ))
    arch_radar.update_layout(
        polar=dict(
            bgcolor=OFF,
            radialaxis=dict(visible=True, range=[0,1], tickfont=dict(size=9,color=GREY),
                            gridcolor=LGREY, linecolor=LGREY),
            angularaxis=dict(tickfont=dict(size=11,color=NAVY,family="Arial")),
        ),
        showlegend=True,
        legend=dict(font=dict(size=10,color=GREY), orientation="h",
                    yanchor="bottom", y=-0.18, xanchor="center", x=0.5),
        margin=dict(l=45,r=45,t=30,b=40),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        height=300,
    )

    col_arch_l, col_arch_r = st.columns([1, 1], gap="large")
    with col_arch_l:
        st.markdown(f"""
        <div style="border-left:3px solid {arch_col};padding-left:0.75rem;margin-bottom:0.7rem;">
          <div style="font-size:0.68rem;font-weight:700;color:{arch_col};text-transform:uppercase;
                      letter-spacing:0.1em;margin-bottom:3px;">Pandora Personality Archetype</div>
          <div style="font-size:0.72rem;color:{GREY};">
            Empirically derived from Pandora Reddit dataset · More stable than MBTI type assignment
          </div>
        </div>
        <div style="background:{WHITE};border:1.5px solid {arch_col}33;
                    border-left:4px solid {arch_col};border-radius:10px;
                    padding:1rem 1.2rem;margin-bottom:0.6rem;">
          <div style="display:flex;align-items:center;gap:0.7rem;margin-bottom:0.5rem;">
            <div style="background:{arch_col};color:#fff;font-size:0.78rem;font-weight:800;
                        padding:0.28rem 0.65rem;border-radius:6px;letter-spacing:0.04em;
                        flex-shrink:0;">{arch['short']}</div>
            <div style="font-size:1.05rem;font-weight:800;color:{NAVY};">{arch['name']}</div>
          </div>
          <div style="font-size:0.86rem;color:{GREY};line-height:1.55;margin-bottom:0.5rem;">
            {arch['desc']}
          </div>
          <div style="font-size:0.72rem;color:{arch_col};font-weight:600;
                      background:{arch_col}15;border-radius:4px;padding:0.25rem 0.55rem;
                      display:inline-block;">{arch['traits']}</div>
        </div>
        <div style="font-size:0.7rem;color:#94A3B8;line-height:1.5;margin-bottom:0.4rem;">
          <b>Archetypes vs MBTI:</b> Pandora archetypes use continuous Big Five scores
          fitted on real behavioural data, providing finer granularity than 16 binary types.
          PersonaPath derives your MBTI type from the same Big Five profile
          (McCrae &amp; Costa, 1989), offering both the nuance of continuous traits
          and the familiarity of MBTI language.
        </div>""", unsafe_allow_html=True)

        # All 6 archetypes mini comparison
        mini_html = '<div style="display:flex;flex-wrap:wrap;gap:0.3rem;margin-top:0.5rem;">'
        for aid, a in PANDORA_ARCHETYPES.items():
            is_me = (aid == archetype_id)
            bg    = a["color"] if is_me else OFF
            fc    = "#fff"     if is_me else GREY
            bdr   = f"2px solid {a['color']}" if is_me else f"1px solid {LGREY}"
            mini_html += (
                f'<div style="background:{bg};color:{fc};border:{bdr};'
                f'border-radius:5px;padding:3px 8px;font-size:0.68rem;font-weight:700;">'
                f'{a["short"]} {a["name"]}</div>'
            )
        mini_html += "</div>"
        st.markdown(mini_html, unsafe_allow_html=True)

    with col_arch_r:
        st.plotly_chart(arch_radar, use_container_width=True)

    st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)

    # ── MBTI Interpretive Layer (derived from Big Five) ────────────────────
    mbti = predict_mbti(profile)
    MBTI_COL = "#6366F1"   # indigo-500

    col_mbti_l, col_mbti_r = st.columns([1, 1], gap="large")

    with col_mbti_l:
        st.markdown(f"""
        <div style="border-left:3px solid {MBTI_COL};padding-left:0.75rem;margin-bottom:0.7rem;">
          <div style="font-size:0.68rem;font-weight:700;color:{MBTI_COL};text-transform:uppercase;
                      letter-spacing:0.1em;margin-bottom:3px;">MBTI Personality Type</div>
          <div style="font-size:0.72rem;color:{GREY};">
            Derived from Big Five profile via McCrae &amp; Costa (1989) mapping
          </div>
        </div>
        <div style="background:{WHITE};border:1.5px solid {MBTI_COL}33;
                    border-left:4px solid {MBTI_COL};border-radius:10px;
                    padding:1rem 1.2rem;margin-bottom:0.6rem;">
          <div style="display:flex;align-items:center;gap:0.8rem;margin-bottom:0.6rem;flex-wrap:wrap;">
            <div style="background:{MBTI_COL};color:#fff;font-size:1.3rem;font-weight:800;
                        padding:0.35rem 0.9rem;border-radius:8px;letter-spacing:0.12em;
                        flex-shrink:0;">{mbti['type']}</div>
            <div>
              <div style="font-size:1.05rem;font-weight:800;color:{NAVY};">The {mbti['nickname']}</div>
              <div style="font-size:0.82rem;color:{GREY};margin-top:1px;">{mbti['tagline']}</div>
            </div>
          </div>""", unsafe_allow_html=True)

        # Dimension detail bars
        dim_html = ""
        for d in mbti["dimensions"]:
            pct = d["score"]
            bar_w = int(pct * 100)
            # Gradient position: raw score 0→left, 1→right
            dot_pos = int(d["raw"] * 100)
            dim_html += f"""
            <div style="margin-bottom:0.55rem;">
              <div style="display:flex;justify-content:space-between;align-items:baseline;margin-bottom:3px;">
                <span style="font-size:0.78rem;color:{GREY};">
                  <b style="color:{NAVY};">{d['pole']}</b> · {d['name']}
                </span>
                <span style="font-family:'Fira Code','JetBrains Mono',monospace;
                             font-size:0.82rem;font-weight:600;color:{MBTI_COL};">{pct:.0%}</span>
              </div>
              <div style="position:relative;height:8px;background:linear-gradient(to right,
                          #E0E7FF 0%, {MBTI_COL} 100%);border-radius:4px;">
                <div style="position:absolute;top:-2px;left:{dot_pos}%;
                            width:12px;height:12px;background:{WHITE};border:2.5px solid {MBTI_COL};
                            border-radius:50%;transform:translateX(-50%);
                            box-shadow:0 1px 4px rgba(99,102,241,0.3);"></div>
              </div>
              <div style="display:flex;justify-content:space-between;margin-top:2px;">
                <span style="font-size:0.65rem;color:#94A3B8;">{d['pole_low']}</span>
                <span style="font-size:0.65rem;color:#94A3B8;">{d['pole_high']}</span>
              </div>
            </div>"""
        st.markdown(dim_html + "</div>", unsafe_allow_html=True)

        st.markdown(f"""
        <div style="font-size:0.7rem;color:#94A3B8;line-height:1.5;margin-top:0.4rem;">
          <b>How this works:</b> PersonaPath derives your MBTI type from your Big Five profile
          using multi-trait weighted mapping based on meta-analytic correlations
          (McCrae &amp; Costa, 1989; Furnham, 1996; Wilt &amp; Revelle, 2015).
          Each MBTI dimension draws on multiple Big Five traits for higher accuracy.
        </div>""", unsafe_allow_html=True)

    with col_mbti_r:
        # MBTI radar chart (4 dimensions)
        mbti_labels = [d["code"] for d in mbti["dimensions"]]
        mbti_raw    = [d["raw"] for d in mbti["dimensions"]]
        # Close the polygon
        mbti_labels_c = mbti_labels + [mbti_labels[0]]
        mbti_raw_c    = mbti_raw + [mbti_raw[0]]

        mbti_radar = go.Figure()
        mbti_radar.add_trace(go.Scatterpolar(
            r=[0.5]*len(mbti_labels_c), theta=mbti_labels_c,
            fill=None, mode="lines",
            line=dict(color="#E2E8F0", width=1, dash="dot"),
            showlegend=False,
        ))
        mbti_radar.add_trace(go.Scatterpolar(
            r=mbti_raw_c, theta=mbti_labels_c, fill="toself",
            fillcolor="rgba(99,102,241,0.15)",
            line=dict(color=MBTI_COL, width=2.5),
            name="Your MBTI profile",
        ))
        mbti_radar.update_layout(
            polar=dict(
                bgcolor=OFF,
                radialaxis=dict(visible=True, range=[0, 1],
                                tickfont=dict(size=9, color=GREY),
                                gridcolor=LGREY, linecolor=LGREY),
                angularaxis=dict(tickfont=dict(size=12, color=NAVY, family="Arial")),
            ),
            showlegend=False,
            margin=dict(l=50, r=50, t=30, b=40),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            height=300,
        )
        st.plotly_chart(mbti_radar, use_container_width=True)

        # Dimension labels below radar
        type_chips = " ".join(
            f'<span style="display:inline-block;background:{MBTI_COL}15;border:1px solid {MBTI_COL}33;'
            f'color:{MBTI_COL};font-size:0.72rem;font-weight:700;'
            f'padding:3px 10px;border-radius:6px;margin:2px;">'
            f'{d["pole"]} — {d["name"]}</span>'
            for d in mbti["dimensions"]
        )
        st.markdown(f'<div style="text-align:center;margin-top:-0.3rem;">{type_chips}</div>',
                    unsafe_allow_html=True)

    st.markdown("<div style='margin-bottom:0.5rem;'></div>", unsafe_allow_html=True)

    # ── Two columns: Personality | Careers ────────────────────────────────────
    # Spacer ratio 0.35 ≈ 13% each side → visually aligns with card content indentation
    _gl, col_left, col_right, _gr = st.columns([0.25, 1, 1, 0.25], gap="large")

    with col_left:
        st.markdown(f"""
        <div style="border-left:3px solid {CYAN};padding-left:0.75rem;margin-bottom:0.6rem;">
          <div style="font-weight:700;color:{NAVY};font-size:1.05rem;">Big Five Profile</div>
          <div style="font-size:0.76rem;color:{GREY};margin-top:1px;">Personality trait scores inferred from your text</div>
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(trait_bars(profile), use_container_width=True)

        # Disclaimer shown once above all trait cards
        st.markdown(f"""
        <div style="font-size:0.72rem;color:#94A3B8;background:#F8FAFC;
                    border:1px solid {LGREY};border-radius:6px;
                    padding:0.35rem 0.7rem;margin-bottom:0.6rem;line-height:1.5;">
          &#9432;&nbsp; Reference roles are illustrative only, drawn from O*NET Big Five–occupation
          research (Sackett &amp; Walmsley, 2014). Not career advice or professional assessment.
        </div>""", unsafe_allow_html=True)

        for k, name in zip(BIG5_KEYS, BIG5_NAMES):
            score = profile[k]
            level = "High" if score >= 0.60 else ("Moderate" if score >= 0.42 else "Low")
            color = GREEN  if score >= 0.60 else (AMBER       if score >= 0.42 else "#94A3B8")
            bar_w = int(score * 100)
            ref_jobs = BIG5_REFS.get(name, {}).get(level, [])
            ref_chips = "".join(
                f'<span style="display:inline-block;background:#F1F5F9;'
                f'border:1px solid {LGREY};color:{GREY};font-size:0.68rem;'
                f'padding:1px 6px;border-radius:4px;margin:2px 2px 0 0;">{j}</span>'
                for j in ref_jobs
            )
            st.markdown(f"""
            <div style="display:flex;align-items:flex-start;margin-bottom:0.7rem;
                        background:{WHITE};border:1px solid {LGREY};border-radius:10px;
                        padding:0.65rem 0.9rem;gap:0.7rem;">
              <span style="background:{color};color:#fff;font-size:0.68rem;font-weight:700;
                           padding:3px 8px;border-radius:6px;margin-top:2px;
                           white-space:nowrap;flex-shrink:0;">{level}</span>
              <div style="flex:1;min-width:0;">
                <div style="display:flex;align-items:baseline;gap:0.4rem;">
                  <span style="font-weight:700;color:{NAVY};font-size:0.92rem;">{name}</span>
                  <span style="font-family:'Fira Code','JetBrains Mono',monospace;font-size:0.9rem;
                               font-weight:600;color:{color};">{score:.0%}</span>
                </div>
                <div style="height:4px;background:{LGREY};border-radius:2px;margin:4px 0 3px;">
                  <div style="height:4px;width:{bar_w}%;background:{color};border-radius:2px;"></div>
                </div>
                <div style="font-size:0.78rem;color:{GREY};margin-bottom:4px;">{BIG5_DESC[name]}</div>
                {f'<div style="margin-top:3px;"><span style="font-size:0.67rem;color:#94A3B8;font-weight:600;letter-spacing:0.04em;text-transform:uppercase;">Related roles · </span>{ref_chips}</div>' if ref_chips else ''}
              </div>
            </div>""", unsafe_allow_html=True)


    with col_right:
        st.markdown(f"""
        <div style="border-left:3px solid {GREEN};padding-left:0.75rem;margin-bottom:0.6rem;">
          <div style="font-weight:700;color:{NAVY};font-size:1.05rem;">Career Cluster Alignment</div>
          <div style="font-size:0.76rem;color:{GREY};margin-top:1px;">Matched to O*NET clusters via Big Five profile</div>
        </div>""", unsafe_allow_html=True)
        st.plotly_chart(match_bar(ranked), use_container_width=True)

        rank_suffix = {0: "Best Match", 1: "2nd", 2: "3rd", 3: "Least Match"}
        for i, (cluster, score) in enumerate(ranked):
            card_class = "career-card top" if i == 0 else "career-card"
            rank_html  = rank_badge_html(i + 1)
            icon_sm    = cluster_icon_html(cluster["icon_label"], cluster["icon_color"], size=30)
            chips = " ".join(
                f'<span class="career-chip">{ex}</span>' for ex in cluster["examples"]
            )
            rlabel = rank_suffix.get(i, "")
            st.markdown(f"""
            <div class="{card_class}">
              <div style="display:flex;justify-content:space-between;align-items:center;
                          margin-bottom:0.25rem;">
                <div style="display:flex;align-items:center;gap:0.4rem;">
                  {rank_html}{icon_sm}
                  <span class="career-name">{cluster['name']}</span>
                </div>
                <div style="text-align:right;">
                  <span class="career-pct">{score:.0%}</span>
                  <div style="font-size:0.65rem;color:{GREY};font-weight:600;">{rlabel}</div>
                </div>
              </div>
              <div class="career-desc">{cluster['description']}</div>
              <div style="margin-top:0.4rem;">{chips}</div>
              <div style="font-size:0.76rem;color:#94A3B8;margin-top:0.4rem;">
                {cluster['n_occ']} O*NET occupations in this cluster
              </div>
            </div>""", unsafe_allow_html=True)

    # ── Fine-grained O*NET Occupation Matches ──────────
    st.markdown(f"""
    <div style="margin-top:1.8rem;">
      <div style="border-left:3px solid {CYAN};padding-left:0.75rem;margin-bottom:0.8rem;">
        <div style="font-weight:700;color:{NAVY};font-size:1.05rem;">
          O*NET Occupation Explorer
        </div>
        <div style="font-size:0.76rem;color:{GREY};margin-top:1px;">
          Add a preparation filter, then rerank by Big Five fit plus optional complexity emphasis
          · <span style="color:#94A3B8;">Exploratory only — not a career recommendation</span>
        </div>
      </div>
    </div>""", unsafe_allow_html=True)

    prep_labels = [label for label, _ in OCC_PREP_OPTIONS]
    prep_to_zone = {label: zone for label, zone in OCC_PREP_OPTIONS}
    ctl_l, ctl_r = st.columns([1.2, 1], gap="medium")
    with ctl_l:
        selected_prep = st.selectbox(
            "Minimum education / preparation level",
            prep_labels,
            index=2,
            key="occ_min_prep_label",
        )
        min_job_zone = prep_to_zone[selected_prep]
        st.caption("Job Zone 3 is a good default for many technical and professional roles; Job Zone 4 better fits advanced specialist paths.")
    with ctl_r:
        complexity_weight = st.slider(
            "Complexity emphasis",
            min_value=0.0,
            max_value=0.30,
            value=0.10,
            step=0.05,
            key="occ_complexity_weight",
            help="0.00 keeps the current personality-only ranking inside the filtered set. Higher values slightly favor more complex roles.",
        )
        st.caption("Use this to nudge the ranking toward more cognitively demanding occupations after the preparation filter is applied.")

    occ_settings = {
        "min_job_zone": min_job_zone,
        "complexity_weight": complexity_weight,
    }
    occ_groups, occ_meta = rank_occupations(
        profile,
        per_cluster=3,
        min_job_zone=min_job_zone,
        complexity_weight=complexity_weight,
    )
    if occ_groups:
        source_note = (
            "Current artifact does not include O*NET Job Zone, so this demo estimates preparation level from SOC group + title keywords."
            if occ_meta.get("using_job_zone_proxy")
            else "Using O*NET Job Zone data from the occupation artifact."
        )
        if occ_meta.get("filter_relaxed"):
            source_note += " No occupations met the selected minimum, so the explorer fell back to the full occupation set."
        st.markdown(
            f"""
            <div style="font-size:0.74rem;color:{GREY};background:#F8FAFC;
                        border:1px solid {LGREY};border-radius:8px;padding:0.55rem 0.8rem;
                        margin:0.2rem 0 0.9rem 0;line-height:1.55;">
              Showing occupations at <b>{JOB_ZONE_LABELS[min_job_zone]}</b> and above across
              <b>{occ_meta.get('filtered_occupations', 0)}</b> of <b>{occ_meta.get('total_occupations', 0)}</b> roles.
              Complexity boost = <b>{complexity_weight:.2f}</b>. {source_note}
            </div>
            """,
            unsafe_allow_html=True,
        )

        import streamlit.components.v1 as components

        _go_l, _go_r = st.columns(2, gap="medium")
        occ_cols = [_go_l, _go_r]
        for gi, group in enumerate(occ_groups):
            cl_col   = group["cluster_color"]
            cl_score = group["cluster_score"]
            icon_label = next(
                (c["icon_label"] for c in CAREER_CLUSTERS if c["id"] == group["cluster_id"]), "??")

            # Build occupation rows HTML (plain string concat — no f-string nesting)
            rows = ""
            for j, occ in enumerate(group["occupations"]):
                pct      = occ["score"]
                bar_w    = int(min(pct, 1.0) * 100)
                onet_url = "https://www.onetonline.org/link/summary/" + occ["occ_code"]
                title    = occ["title"]
                pct_str  = f"{pct:.0%}"
                prep_txt = f"{JOB_ZONE_SHORT.get(occ['job_zone'], 'JZ?')} · {occ['job_zone_label'].split('·', 1)[-1].strip()}"
                if occ["job_zone_source"] == "proxy":
                    prep_txt += " (proxy)"
                rows += (
                    '<div style="display:flex;align-items:center;gap:8px;'
                    'padding:6px 0;border-bottom:1px solid #E2E8F0;">'
                    '<span style="font-size:11px;font-weight:700;color:#4A5568;'
                    'width:16px;text-align:center;flex-shrink:0;">' + str(j+1) + '</span>'
                    '<div style="flex:1;min-width:0;">'
                    '<a href="' + onet_url + '" target="_blank" '
                    'style="font-size:13px;font-weight:600;color:#1E3A5F;'
                    'text-decoration:none;display:block;white-space:nowrap;'
                    'overflow:hidden;text-overflow:ellipsis;">' + title + '</a>'
                    '<div style="font-size:10px;color:#64748B;margin-top:2px;">' + prep_txt + '</div>'
                    '<div style="height:3px;background:#E2E8F0;border-radius:2px;margin-top:4px;">'
                    '<div style="height:3px;width:' + str(bar_w) + '%;background:' + cl_col + ';'
                    'border-radius:2px;opacity:0.75;"></div></div>'
                    '</div>'
                    '<span style="font-family:monospace;font-size:12px;font-weight:700;'
                    'color:' + cl_col + ';flex-shrink:0;">' + pct_str + '</span>'
                    '</div>'
                )

            card_html = (
                '<html><head><meta charset="utf-8"></head><body style="margin:0;padding:0;">'
                '<div style="background:#FFFFFF;border:1px solid #E2E8F0;border-radius:10px;'
                'padding:12px 14px;font-family:Inter,Arial,sans-serif;">'
                '<div style="display:flex;align-items:center;justify-content:space-between;'
                'margin-bottom:8px;">'
                '<div style="display:flex;align-items:center;gap:7px;">'
                '<div style="background:' + cl_col + ';color:#fff;font-size:11px;font-weight:800;'
                'padding:4px 8px;border-radius:6px;letter-spacing:.04em;">' + icon_label + '</div>'
                '<span style="font-weight:700;color:#1E3A5F;font-size:14px;">'
                + group["cluster_name"] + '</span></div>'
                '<span style="font-family:monospace;font-size:13px;font-weight:700;color:'
                + cl_col + ';">' + f"{cl_score:.0%}" + ' cluster fit</span>'
                '</div>'
                + rows +
                '<div style="font-size:10px;color:#94A3B8;margin-top:6px;">'
                'Click any title to view full O*NET profile</div>'
                '</div></body></html>'
            )

            with occ_cols[gi % 2]:
                components.html(card_html, height=205, scrolling=False)

    # ── Footer actions ─────────────────────────────────
    st.markdown("<div style='margin-top:1.5rem;'></div>", unsafe_allow_html=True)
    col_a, col_b, _ = st.columns([1, 1, 2])
    with col_a:
        if st.button("Start Over", use_container_width=True):
            reset()
    with col_b:
        pdf_bytes = generate_pdf(profile, ranked, st.session_state.q_count, occ_settings)
        fname = f"personapath_{datetime.now().strftime('%Y%m%d_%H%M')}.pdf"
        st.download_button(
            "Export PDF Report",
            data=pdf_bytes,
            file_name=fname,
            mime="application/pdf",
            use_container_width=True,
        )

    st.markdown(f"""
    <div class="disclaimer">
      Results are generated by ML models trained on self-report datasets (Essays Big Five, Pandora Reddit, MBTI Reddit).
      Accuracy is limited — treat this as a reflective tool, not a definitive assessment.
      Team Foundry &nbsp;·&nbsp; SIADS 699 Capstone &nbsp;·&nbsp; University of Michigan.
    </div>""", unsafe_allow_html=True)


# ── Likert Scale Pages ─────────────────────────────────
def page_likert():
    """Render Likert items in batches of 5 (one trait per page)."""
    page = st.session_state.likert_page  # 0-4
    batch_start = page * 5
    batch_end   = batch_start + 5
    batch_items = LIKERT_ITEMS[batch_start:batch_end]
    trait_name  = dict(zip(BIG5_KEYS, BIG5_NAMES))[batch_items[0]["trait"]]
    total_pages = 5

    # Progress bar
    pct = (page + 1) / total_pages
    st.markdown(f"""
    <div style="display:flex;align-items:center;gap:0.6rem;margin-bottom:1rem;">
      <div style="flex:1;height:6px;background:{LGREY};border-radius:3px;">
        <div style="height:6px;width:{pct*100:.0f}%;background:{CYAN};border-radius:3px;
                    transition:width 0.3s;"></div>
      </div>
      <span style="font-size:0.8rem;color:{GREY};white-space:nowrap;">
        Part {page+1} of {total_pages}
      </span>
    </div>""", unsafe_allow_html=True)

    st.markdown(f"""
    <div class="step-pill">Standard Scale — {trait_name}</div>
    <div class="step-title">Rate how accurately each statement describes you</div>
    <div class="step-sub">
      Use the scale from <b>Very Inaccurate</b> to <b>Very Accurate</b>.
      There are no right or wrong answers.
    </div>""", unsafe_allow_html=True)

    responses = st.session_state.likert_responses

    for i, item in enumerate(batch_items):
        idx = batch_start + i
        st.markdown(f"""
        <div style="background:{WHITE};border:1px solid {LGREY};border-radius:10px;
                    padding:0.8rem 1rem;margin-bottom:0.6rem;">
          <div style="font-size:0.95rem;font-weight:600;color:{NAVY};margin-bottom:0.4rem;">
            {idx+1}. {item['text']}
          </div>
        </div>""", unsafe_allow_html=True)

        current_val = responses[idx] if responses[idx] is not None else 0
        val = st.select_slider(
            f"Item {idx+1}",
            options=[1, 2, 3, 4, 5],
            value=current_val if current_val >= 1 else 3,
            format_func=lambda v: LIKERT_LABELS[v - 1],
            key=f"likert_{idx}",
            label_visibility="collapsed",
        )
        responses[idx] = val

    st.session_state.likert_responses = responses

    # Navigation
    col_back, col_next = st.columns(2)
    with col_back:
        if page > 0:
            if st.button("← Back", use_container_width=True):
                st.session_state.likert_page = page - 1
                st.rerun()
    with col_next:
        if page < total_pages - 1:
            if st.button("Next →", type="primary", use_container_width=True):
                st.session_state.likert_page = page + 1
                st.rerun()
        else:
            if st.button("See Results →", type="primary", use_container_width=True):
                likert_profile = score_likert(responses)
                mode = st.session_state.mode
                if mode == "hybrid":
                    # Store Likert scores and go to open-text for blending
                    st.session_state["likert_profile"] = likert_profile
                    go_step(1)  # open-text page
                else:
                    # Pure Likert mode — use scores directly
                    st.session_state.profile = likert_profile
                    st.session_state.ranked  = rank_clusters(likert_profile)
                    st.session_state.q_count = 25
                    go_step(9)


# ═══════════════════════════════════════════════════════
# ROUTER
# ═══════════════════════════════════════════════════════
inject_css()
topbar()

step = st.session_state.step

if   step == 0:   page_welcome()
elif step == 100: page_likert()
elif step == 1:   page_text()
elif step == 2:   page_question(2)
elif step == 3:   page_question(3)
elif step == 4:   page_question(4)
elif step == 5:   page_question(5)
elif step == 6:   page_question(6)
elif step == 7:   page_question(7)
elif step == 8:   page_question(8)
else:             page_results()
