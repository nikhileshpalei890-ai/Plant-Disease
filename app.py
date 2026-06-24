import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import json
import time

# ─────────────────────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Plant Disease Detector",
    page_icon="🌿",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────────────────────

st.markdown("""
<style>
    /* Main background */
    .stApp {
        background: linear-gradient(135deg, #f0fdf4 0%, #dcfce7 100%);
    }

    /* Title */
    .main-title {
        text-align: center;
        font-size: 2.5rem;
        font-weight: 800;
        color: #166534;
        margin-bottom: 0.2rem;
    }

    .sub-title {
        text-align: center;
        font-size: 1rem;
        color: #4b7c59;
        margin-bottom: 2rem;
    }

    /* Upload box */
    .upload-box {
        background: white;
        border-radius: 16px;
        padding: 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        margin-bottom: 1.5rem;
    }

    /* Result card */
    .result-card {
        background: white;
        border-radius: 16px;
        padding: 1.5rem 2rem;
        box-shadow: 0 4px 20px rgba(0,0,0,0.08);
        border-left: 6px solid #16a34a;
        margin-top: 1.5rem;
    }

    .result-disease {
        font-size: 1.6rem;
        font-weight: 700;
        color: #166534;
    }

    .result-confidence {
        font-size: 1rem;
        color: #4b7c59;
        margin-top: 0.3rem;
    }

    /* Healthy badge */
    .badge-healthy {
        background: #dcfce7;
        color: #166534;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.5rem;
    }

    .badge-disease {
        background: #fee2e2;
        color: #991b1b;
        padding: 4px 14px;
        border-radius: 999px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 0.5rem;
    }

    /* Top 3 */
    .top3-title {
        font-size: 1rem;
        font-weight: 600;
        color: #374151;
        margin-top: 1.2rem;
        margin-bottom: 0.5rem;
    }

    /* Footer */
    .footer {
        text-align: center;
        font-size: 0.8rem;
        color: #9ca3af;
        margin-top: 3rem;
        padding-bottom: 1rem;
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# CLASS NAMES
# These must match the exact folder names in your training data
# ─────────────────────────────────────────────────────────────

# IMPORTANT: Replace this list with your actual class names
# You can get them from: list(train_data.class_indices.keys())
# in your Notebook 3
CLASS_NAMES = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 'Grape___healthy', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus', 'Tomato___healthy']

DISEASE_SOLUTIONS = {
    'Apple___Apple_scab': {
        'cause': 'Fungal infection by Venturia inaequalis. Spreads in cool, wet weather.',
        'solution': 'Apply fungicides like Captan or Mancozeb at bud break. Remove and destroy infected leaves. Ensure good air circulation by pruning.'
    },
    'Apple___Black_rot': {
        'cause': 'Fungal infection by Botryosphaeria obtusa. Enters through wounds or dead bark.',
        'solution': 'Prune infected branches 15cm below visible infection. Apply copper-based fungicides. Remove mummified fruits from tree and ground.'
    },
    'Apple___Cedar_apple_rust': {
        'cause': 'Fungal infection by Gymnosporangium juniperi-virginianae. Requires both apple and juniper trees to complete lifecycle.',
        'solution': 'Apply fungicides in spring when orange masses appear on junipers. Remove nearby juniper trees if possible. Use resistant apple varieties.'
    },
    'Apple___healthy': {
        'cause': 'No disease detected.',
        'solution': 'Plant is healthy. Maintain regular watering, fertilization, and annual pruning. Monitor regularly for early signs of disease.'
    },
    'Blueberry___healthy': {
        'cause': 'No disease detected.',
        'solution': 'Plant is healthy. Maintain acidic soil pH (4.5-5.5). Mulch around plants and ensure consistent moisture.'
    },
    'Cherry_(including_sour)___Powdery_mildew': {
        'cause': 'Fungal infection by Podosphaera clandestina. Thrives in warm dry days and cool nights.',
        'solution': 'Apply sulfur-based or potassium bicarbonate fungicides. Improve air circulation through pruning. Avoid overhead irrigation.'
    },
    'Cherry_(including_sour)___healthy': {
        'cause': 'No disease detected.',
        'solution': 'Plant is healthy. Maintain proper pruning for air circulation. Water at base of plant to keep foliage dry.'
    },
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': {
        'cause': 'Fungal infection by Cercospora zeae-maydis. Favored by warm humid conditions.',
        'solution': 'Use resistant hybrids. Apply foliar fungicides like Azoxystrobin. Practice crop rotation. Till crop residue after harvest.'
    },
    'Corn_(maize)___Common_rust_': {
        'cause': 'Fungal infection by Puccinia sorghi. Spreads rapidly through wind-borne spores.',
        'solution': 'Apply foliar fungicides early. Plant resistant varieties. Avoid late planting. Monitor fields regularly during humid weather.'
    },
    'Corn_(maize)___Northern_Leaf_Blight': {
        'cause': 'Fungal infection by Exserohilum turcicum. Favored by moderate temperatures and wet weather.',
        'solution': 'Use resistant hybrids. Apply fungicides at early tasseling stage. Practice crop rotation. Remove infected crop debris.'
    },
    'Corn_(maize)___healthy': {
        'cause': 'No disease detected.',
        'solution': 'Plant is healthy. Maintain proper spacing, fertilization, and irrigation. Scout regularly for pests and disease.'
    },
    'Grape___Black_rot': {
        'cause': 'Fungal infection by Guignardia bidwellii. Spreads during warm wet weather.',
        'solution': 'Apply Mancozeb or Captan starting at bud break. Remove mummified berries and infected leaves. Prune for good air circulation.'
    },
    'Grape___healthy': {
        'cause': 'No disease detected.',
        'solution': 'Plant is healthy. Maintain proper pruning, trellising, and canopy management. Monitor regularly.'
    },
    'Peach___Bacterial_spot': {
        'cause': 'Bacterial infection by Xanthomonas arboricola. Spreads in warm, wet, and windy conditions.',
        'solution': 'Apply copper-based bactericides in early spring. Avoid overhead irrigation. Plant resistant varieties. Remove infected plant material.'
    },
    'Peach___healthy': {
        'cause': 'No disease detected.',
        'solution': 'Plant is healthy. Maintain proper pruning for air circulation. Apply dormant sprays to prevent fungal and bacterial issues.'
    },
    'Pepper,_bell___Bacterial_spot': {
        'cause': 'Bacterial infection by Xanthomonas campestris. Spreads through contaminated seeds, water, and tools.',
        'solution': 'Use disease-free seeds. Apply copper-based bactericides. Avoid working with wet plants. Practice crop rotation for 2-3 years.'
    },
    'Pepper,_bell___healthy': {
        'cause': 'No disease detected.',
        'solution': 'Plant is healthy. Maintain proper spacing, consistent watering at base, and regular fertilization.'
    },
    'Potato___Early_blight': {
        'cause': 'Fungal infection by Alternaria solani. Affects older leaves first during warm humid weather.',
        'solution': 'Apply Mancozeb or Chlorothalonil fungicide. Remove infected lower leaves. Avoid overhead watering. Ensure adequate potassium nutrition.'
    },
    'Potato___Late_blight': {
        'cause': 'Caused by Phytophthora infestans. Extremely destructive — spreads rapidly in cool wet weather.',
        'solution': 'Apply metalaxyl or cymoxanil fungicides immediately. Destroy infected plants. Improve field drainage. Do not compost infected material.'
    },
    'Potato___healthy': {
        'cause': 'No disease detected.',
        'solution': 'Plant is healthy. Maintain proper hilling, irrigation, and fertilization. Use certified disease-free seed potatoes.'
    },
    'Raspberry___healthy': {
        'cause': 'No disease detected.',
        'solution': 'Plant is healthy. Prune old canes after harvest. Maintain good air circulation and avoid waterlogged soil.'
    },
    'Soybean___healthy': {
        'cause': 'No disease detected.',
        'solution': 'Plant is healthy. Maintain proper row spacing and monitor for aphids and other pests regularly.'
    },
    'Squash___Powdery_mildew': {
        'cause': 'Fungal infection by Podosphaera xanthii. Thrives in warm dry conditions with high humidity at night.',
        'solution': 'Apply potassium bicarbonate or neem oil spray. Remove heavily infected leaves. Improve air circulation. Avoid excess nitrogen fertilization.'
    },
    'Strawberry___Leaf_scorch': {
        'cause': 'Fungal infection by Diplocarpon earlianum. Spreads in wet conditions.',
        'solution': 'Remove infected leaves immediately. Apply Captan fungicide. Avoid overhead irrigation. Renovate beds after harvest.'
    },
    'Strawberry___healthy': {
        'cause': 'No disease detected.',
        'solution': 'Plant is healthy. Maintain proper bed renovation, mulching, and drip irrigation to keep foliage dry.'
    },
    'Tomato___Bacterial_spot': {
        'cause': 'Bacterial infection by Xanthomonas species. Spreads through rain splash and contaminated tools.',
        'solution': 'Apply copper-based bactericides. Use disease-free transplants. Avoid working with wet plants. Rotate crops annually.'
    },
    'Tomato___Early_blight': {
        'cause': 'Fungal infection by Alternaria solani. Affects lower older leaves first.',
        'solution': 'Remove infected lower leaves. Apply Mancozeb or copper fungicide every 7-10 days. Mulch around base. Avoid wetting foliage.'
    },
    'Tomato___Late_blight': {
        'cause': 'Caused by Phytophthora infestans. Extremely aggressive in cool wet weather.',
        'solution': 'Apply systemic fungicides immediately. Remove and destroy infected plants. Do not compost. Improve air circulation.'
    },
    'Tomato___Leaf_Mold': {
        'cause': 'Fungal infection by Passalora fulva. Common in greenhouse conditions with high humidity.',
        'solution': 'Reduce humidity below 85%. Improve ventilation. Apply Chlorothalonil fungicide. Remove infected leaves promptly.'
    },
    'Tomato___Septoria_leaf_spot': {
        'cause': 'Fungal infection by Septoria lycopersici. Starts on lower leaves and moves upward.',
        'solution': 'Remove infected leaves. Apply Mancozeb or Copper fungicide. Avoid overhead irrigation. Stake plants for better air flow.'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'cause': 'Viral disease spread by silverleaf whiteflies (Bemisia tabaci).',
        'solution': 'Control whiteflies with imidacloprid insecticide. Use yellow sticky traps. Use reflective mulches. Remove and destroy infected plants immediately.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'cause': 'Tobacco Mosaic Virus (TMV). Spreads through contaminated hands, tools, and infected seeds.',
        'solution': 'No chemical cure. Remove and destroy infected plants. Wash hands with soap before handling. Disinfect tools. Use TMV-resistant varieties.'
    },
    'Tomato___healthy': {
        'cause': 'No disease detected.',
        'solution': 'Plant is healthy. Maintain consistent watering, proper staking, and regular monitoring for pests.'
    },
}

DEFAULT_SOLUTION = {
    'cause': 'Specific cause varies by region and conditions.',
    'solution': 'Consult a local agricultural expert or Krishi Vigyan Kendra (KVK) for region-specific treatment advice.'
}

# Clean display names (removes underscores for UI)
def clean_name(raw_name):
    parts = raw_name.split('___')
    plant   = parts[0].replace('_', ' ')
    disease = parts[1].replace('_', ' ') if len(parts) > 1 else ''
    return plant, disease


# ─────────────────────────────────────────────────────────────
# LOAD MODEL
# @st.cache_resource loads the model once and reuses it
# Without this it would reload on every user interaction
# ─────────────────────────────────────────────────────────────

@st.cache_resource
def load_model():
    # Model file must be in same folder as app.py
    # File name: phase3_best.keras
    try:
        model = tf.keras.models.load_model('phase2_best.keras')
        return model
    except Exception as e:
        st.error(f"❌ Model file not found: {e}")
        st.stop()


# ─────────────────────────────────────────────────────────────
# PREDICTION FUNCTION
# ─────────────────────────────────────────────────────────────

def predict(image, model):
    # Resize to MobileNet input size
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0

    # Handle RGBA images (4 channels → 3 channels)
    if img_array.shape[-1] == 4:
        img_array = img_array[:, :, :3]

    img_array = np.expand_dims(img_array, axis=0)

    # Get predictions
    preds    = model.predict(img_array, verbose=0)
    top3_idx = np.argsort(preds[0])[::-1][:3]

    results = []
    for idx in top3_idx:
        results.append({
            'class_index': int(idx),
            'class_name':  CLASS_NAMES[idx],
            'confidence':  float(preds[0][idx]) * 100
        })

    return results


# ─────────────────────────────────────────────────────────────
# MAIN UI
# ─────────────────────────────────────────────────────────────

# Header
st.markdown('<p class="main-title">🌿 Plant Disease Detector</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Upload a leaf image to identify the disease using Deep Learning</p>', unsafe_allow_html=True)

# Load model
model = load_model()

# Upload section
st.markdown('<div class="upload-box">', unsafe_allow_html=True)
uploaded_file = st.file_uploader(
    "Upload a plant leaf image",
    type=['jpg', 'jpeg', 'png'],
    help="Supported formats: JPG, JPEG, PNG"
)
st.markdown('</div>', unsafe_allow_html=True)


if uploaded_file is not None:
    # Show image
    image = Image.open(uploaded_file)

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.image(image, caption="Uploaded Image", use_column_width=True)

    # Predict button
    if st.button("🔍 Detect Disease", use_container_width=True, type="primary"):

        with st.spinner("Analyzing leaf..."):
            time.sleep(0.5)   # small delay for UX
            results = predict(image, model)

        top         = results[0]
        plant, disease = clean_name(top['class_name'])
        confidence  = top['confidence']
        is_healthy  = 'healthy' in top['class_name'].lower()

        # Result card
        st.markdown(f"""
        <div class="result-card">
            <div style="font-size:0.9rem; color:#6b7280; margin-bottom:4px;">
                Plant Species
            </div>
            <div style="font-size:1.2rem; font-weight:600; color:#111827;">
                🌱 {plant}
            </div>
            <div style="font-size:0.9rem; color:#6b7280; margin-top:1rem; margin-bottom:4px;">
                Diagnosis
            </div>
            <div class="result-disease">{disease}</div>
            <div class="result-confidence">Confidence: {confidence:.1f}%</div>
            <div class="{'badge-healthy' if is_healthy else 'badge-disease'}">
                {'✅ Healthy' if is_healthy else '⚠️ Disease Detected'}
            </div>
        </div>
        """, unsafe_allow_html=True)

        solution = DISEASE_SOLUTIONS.get(top['class_name'], DEFAULT_SOLUTION)

        st.markdown(f"""
        <div style="margin-top:1rem; border-top:1px solid #e5e7eb; padding-top:1rem;">
            <div style="font-size:0.85rem; color:#6b7280;">🦠 Cause</div>
            <div style="font-size:0.95rem; color:#374151; margin-bottom:0.8rem;">
                {solution['cause']}
            </div>
            <div style="font-size:0.85rem; color:#6b7280;">💊 Treatment</div>
            <div style="font-size:0.95rem; color:#166534;">
                {solution['solution']}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # Confidence bar
        st.markdown("<br>", unsafe_allow_html=True)
        st.progress(int(confidence))

        # Top 3 predictions
        st.markdown('<div class="top3-title">Top 3 Predictions</div>', unsafe_allow_html=True)

        for i, result in enumerate(results):
            p, d = clean_name(result['class_name'])
            label = f"{p} — {d}"
            conf  = result['confidence']

            col_label, col_bar, col_pct = st.columns([3, 5, 1])
            with col_label:
                st.markdown(f"<small>{label}</small>", unsafe_allow_html=True)
            with col_bar:
                st.progress(int(conf))
            with col_pct:
                st.markdown(f"<small>{conf:.1f}%</small>", unsafe_allow_html=True)

        # Warning for low confidence
        if confidence < 60:
            st.warning(
                "⚠️ Low confidence prediction. "
                "Try uploading a clearer image with the leaf centered."
            )

else:
    # Placeholder when no image uploaded
    st.markdown("""
    <div style="text-align:center; padding: 2rem; color: #9ca3af;">
        <div style="font-size: 3rem;">🍃</div>
        <div style="margin-top: 0.5rem;">
            Upload a leaf image to get started
        </div>
        <div style="font-size:0.85rem; margin-top:0.5rem;">
            Works best with clear, well-lit images of a single leaf
        </div>
    </div>
    """, unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────
# SIDEBAR — About
# ─────────────────────────────────────────────────────────────

with st.sidebar:
    st.markdown("### 📊 About This Model")
    st.markdown("""
    **Model:** MobileNetV2  
    **Technique:** Transfer Learning + Fine Tuning
                
    **Training Data:**
    - PlantVillage Dataset (54,306 images)
                  
    **Testing Data:**
    - Collected during Field Visit

    **Classes:** 33 plant-disease combinations  
    **PlantVillage Accuracy:** ~95%

    ---
    **Developed by:**  
    **NIKHILESH PALEI**
    --Examination Roll No:
    5503U24034

    **CHINMAYEE ROUTRAY**
    --Examination Roll No:
    5503U24021
                      
    B.Sc. Computer Science  
    Prananath College (Autonomous),Khordha
              
    Utkal University  

    *Field Work Project — 4th Semester*
    """)

    st.markdown("### 🌿 Supported Plants")
    plants = sorted(set(n.split('___')[0].replace('_', ' ') for n in CLASS_NAMES))
    for p in plants:
        st.markdown(f"- {p}")


# Footer
st.markdown("""
<div class="footer">
    Plant Disease Detection and Classification &nbsp;|&nbsp;
    MobileNetV2 + Transfer Learning &nbsp;|&nbsp;
    Utkal University Field Work Project
</div>
""", unsafe_allow_html=True)
