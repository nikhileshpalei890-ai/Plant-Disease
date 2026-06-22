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
        'name': 'Northern Corn Leaf Blight(Fungal)',
        'solution': 'Manage residue with tillage and rotate away from corn for at least 1 year.'
    },
    'Corn_(maize)___healthy': {
        'name': 'Healthy Corn Plant',
        'solution': 'Healthy growth! Ensure consistent nitrogen fertilization and keep fields free of weeds.'
    },
    'Grape___Black_rot': {
        'name': 'Grape Black Rot (Fungal)',
        'solution': 'Strictly prune and remove all mummified berries from the vines.'
    },
    'Grape___healthy': {
        'name': 'Healthy Grape Leaf',
        'solution': 'Healthy vines. Keep up with canopy management to maximize airflow.'
    },
    'Peach___Bacterial_spot': {
        'name': 'Peach Bacterial Spot (Bacterial)',
        'solution': 'Avoid overhead irrigation. Apply copper-based bactericides during late dormancy.'
    },
    'Peach___healthy': {
        'name': 'Healthy Peach Leaf',
        'solution': 'Vibrant and healthy. Thin fruits properly in early summer to optimize nutrient distribution.'
    },
    'Pepper,_bell___Bacterial_spot': {
        'name': 'Bell Pepper Bacterial Spot (Bacterial)',
        'solution': 'Remove infected plants immediately to prevent spreading. Spray copper fungicides weekly.'
    },
    'Pepper,_bell___healthy': {
        'name': 'Healthy Bell Pepper Leaf',
        'solution': 'Perfectly healthy. Provide well-draining soil and a consistent watering regimen.'
    },
    'Potato___Early_blight': {
        'name': 'Potato Early Blight (Fungal)',
        'solution': 'Maintain high plant vigor with proper nitrogen fertilization. Apply preventative fungicides.'
    },
    'Potato___Late_blight': {
        'name': 'Potato Late Blight (Oomycete - Highly Destructive)',
        'solution': 'Destroy infected plants immediately (do not compost). Apply aggressive protectant fungicides.'
    },
    'Potato___healthy': {
        'name': 'Healthy Potato Leaf',
        'solution': 'Keep it up! Mound soil around the base of the plants to protect developing tubers.'
    },
    'Raspberry___healthy': {
        'name': 'Healthy Raspberry Leaf',
        'solution': 'Looking strong. Keep canes tied neatly to support trellises.'
    },
    'Soybean___healthy': {
        'name': 'Healthy Soybean Leaf',
        'solution': 'Great canopy density! Monitor regularly for insects like aphids as pods begin to fill.'
    },
    'Squash___Powdery_mildew': {
        'name': 'Squash Powdery Mildew (Fungal)',
        'solution': 'Plant in full sun and optimize spacing. Use organic sprays containing baking soda.'
    },
    'Strawberry___Leaf_scorch': {
        'name': 'Strawberry Leaf Scorch (Fungal)',
        'solution': 'Remove older, heavily blotched leaves. Avoid overhead watering late in the day.'
    },
    'Strawberry___healthy': {
        'name': 'Healthy Strawberry Leaf',
        'solution': 'Healthy patch! Use straw mulch underneath the plants to keep berries off bare soil.'
    },
    'Tomato___Bacterial_spot': {
        'name': 'Tomato Bacterial Spot (Bacterial)',
        'solution': 'Avoid working among wet plants. Apply a tank-mix of copper fungicide and Mancozeb.'
    },
    'Tomato___Early_blight': {
        'name': 'Tomato Early Blight (Fungal)',
        'solution': 'Prune lower leaves touching the soil. Mulch around the base of the plant to prevent soil splash.'
    },
    'Tomato___Late_blight': {
        'name': 'Tomato Late Blight (Oomycete)',
        'solution': 'Requires urgent attention. Pull out and bag highly infected plants to stop spores.'
    },
    'Tomato___Leaf_Mold': {
        'name': 'Tomato Leaf Mold (Fungal)',
        'solution': 'Common in greenhouses. Significantly reduce relative humidity and increase ventilation.'
    },
    'Tomato___Septoria_leaf_spot': {
        'name': 'Tomato Septoria Leaf Spot (Fungal)',
        'solution': 'Eliminate nightshade weeds nearby. Clean up all garden debris in the fall.'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'name': 'Tomato Yellow Leaf Curl Virus (Vectored by Whiteflies)',
        'solution': 'Viruses cannot be cured. Focus on controlling the vector: use yellow sticky traps.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'name': 'Tomato Mosaic Virus (Viral)',
        'solution': 'No chemical cure. Remove and burn infected plants. Sterilize tools thoroughly.'
    },
    'Tomato___healthy': {
        'name': 'Healthy Tomato Leaf',
        'solution': 'Superb condition! Keep staking the plant for structure and prune suckers.'
    }
}

CLASS_NAMES = [
    'Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy', 
    'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew', 'Cherry_(including_sour)___healthy', 
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 'Corn_(maize)___Common_rust_', 
    'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy', 'Grape___Black_rot', 
    'Grape___healthy', 'Peach___Bacterial_spot', 'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 
    'Pepper,_bell___healthy', 'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy', 
    'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew', 'Strawberry___Leaf_scorch', 
    'Strawberry___healthy', 'Tomato___Bacterial_spot', 'Tomato___Early_blight', 'Tomato___Late_blight', 
    'Tomato___Leaf_Mold', 'Tomato___Septoria_leaf_spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
    'Tomato___Tomato_mosaic_virus', 'Tomato___healthy'
]

# ==========================================
# 3. MODEL LOADING WITH INTERFERENCE STABILITY
# ==========================================
@st.cache_resource
def load_my_model():
    # compile=False ensures cross-platform Keras deserialization stability
    return tf.keras.models.load_model("phase2_best.keras", compile=False)

with st.spinner("Loading Deep Learning Model..."):
    model = load_my_model()

# ==========================================
# 4. USER INTERFACE GENERATION
# ==========================================
st.markdown("<h1 class='custom-title'>🌿 Plant Disease Detector</h1>", unsafe_allow_html=True)
st.markdown("<p class='custom-subtitle'>A deep learning web app that classifies plant leaf diseases from images.</p>", unsafe_allow_html=True)

# Interactive Model Info Card
with st.expander("📊 Project Documentation & Model Architecture Specs", expanded=False):
    st.markdown("""
    * **Architecture:** MobileNetV2 (Transfer Learning)
    * **Training:** PlantVillage Dataset + PlantDoc Domain Adaptation
    * **Classes:** 38 plant-disease combinations
    * **Accuracy:** ~95% on PlantVillage test set
    """)

uploaded_file = st.file_uploader("Upload a plant leaf image", type=["jpg", "jpeg", "png"])

if uploaded_file is None:
    st.markdown("""
    <br><center>
        <h3 style='color: #27AE60;'>How to Use:</h3>
        <ol style='text-align: left; display: inline-block; color: #555;'>
            <li>Upload a clear photo of a plant leaf</li>
            <li>System automatically handles detection</li>
            <li>Get the disease name, confidence score, and remedies</li>
        </ol>
        <br><br>
        <small style='color:gray;'>Works best with clear, well-lit images of a single leaf</small>
    </center>
    """, unsafe_allow_html=True)

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Target Crop Specimen", use_container_width=True)
    st.write("🔄 Running Diagnostic Inference...")
    
    # Preprocessing Input Tensor
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    # Classification Logic
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    
    raw_class_name = CLASS_NAMES[predicted_class_idx]
    diagnostic_info = DISEASE_GUIDE[raw_class_name]
    
    # Output Layout Generation
    st.write("---")
    col1, col2 = st.columns([2, 1])
    
    with col1:
        if "healthy" in raw_class_name.lower():
            st.success(f"### Diagnostic Result:\n**{diagnostic_info['name']}**")
        else:
            st.error(f"### Diagnostic Result:\n**{diagnostic_info['name']}**")
            
    with col2:
        st.metric(label="Model Confidence", value=f"{confidence:.2f}%")
    
    st.write("") 
    with st.container(border=True):
        st.markdown("### 📋 Recommended Treatment Actions")
        st.info(diagnostic_info['solution'])

# Global Application Academic Footer
st.markdown("""
    <div class='custom-footer'>
        <strong>Tech Stack:</strong> TensorFlow / Keras | Streamlit | GitHub Deployment <br>
        Field Work Project | 4th Semester | Utkal University
    </div>
""", unsafe_allow_html=True)
