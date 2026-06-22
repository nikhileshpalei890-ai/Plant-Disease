import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# ==========================================
# 1. PAGE CONFIGURATION & THEME STYLING
# ==========================================
st.set_page_config(
    page_title="Plant Disease Classifier & Remedy Guide",
    page_icon="🌿",
    layout="centered"
)

# Custom Theme CSS Styling Embedded directly so it works locally and on Cloud
st.markdown("""
    <style>
    .stApp {
        background-color: #E8F8F5;
    }
    .custom-title {
        text-align: center;
        color: #1E8449;
        font-weight: bold;
        margin-bottom: 5px;
    }
    .custom-subtitle {
        text-align: center;
        color: #27AE60;
        font-size: 16px;
        margin-bottom: 25px;
    }
    .custom-footer {
        text-align: center;
        color: #7F8C8D;
        font-size: 13px;
        margin-top: 60px;
        border-top: 1px solid #BDC3C7;
        padding-top: 15px;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. COMPREHENSIVE DISEASE LOGIC DICTIONARY
# ==========================================
DISEASE_GUIDE = {
    'Apple___Apple_scab': {
        'name': 'Apple Scab (Fungal)',
        'solution': 'Rake and destroy fallen leaves in autumn to reduce fungal spores. Apply preventative fungicides early in the spring.'
    },
    'Apple___Black_rot': {
        'name': 'Apple Black Rot (Fungal)',
        'solution': 'Prune out dead wood, mummified fruits, and cankers during the dormant season.'
    },
    'Apple___Cedar_apple_rust': {
        'name': 'Cedar Apple Rust (Fungal)',
        'solution': 'Remove nearby galls on eastern red cedars if possible. Apply protective fungicides.'
    },
    'Apple___healthy': {
        'name': 'Healthy Apple Leaf',
        'solution': 'No treatment needed! Maintain balanced watering and ensure proper sunlight.'
    },
    'Blueberry___healthy': {
        'name': 'Healthy Blueberry Leaf',
        'solution': 'Looks great! Maintain acidic soil conditions (pH 4.5–5.5).'
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'name': 'Cherry Powdery Mildew (Fungal)',
        'solution': 'Prune to improve air circulation inside the tree canopy. Apply neem oil or potassium bicarbonate.'
    },
    'Cherry_(including_sour)___healthy': {
        'name': 'Healthy Cherry Leaf',
        'solution': 'Excellent condition. Continue monitoring for pests and ensure deep, infrequent watering.'
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'name': 'Corn Gray Leaf Spot (Fungal)',
        'solution': 'Practice crop rotation with non-host plants next season. Clean plant debris after harvest.'
    },
    'Corn_(maize)___Common_rust_': {
        'name': 'Corn Common Rust (Fungal)',
        'solution': 'Typically doesn\'t cause severe loss; use resistant hybrids in the future.'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'name': 'Northern Corn Leaf Blight
