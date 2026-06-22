import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# 1. Page Configuration
st.set_page_config(
    page_title="Plant Disease Classifier & Remedy Guide",
    page_icon="🌿",
    layout="centered"
)

# 2. Comprehensive Dictionary of Classes, Proper Names, and Solutions
DISEASE_GUIDE = {
    'Apple___Apple_scab': {
        'name': 'Apple Scab (Fungal)',
        'solution': 'Rake and destroy fallen leaves in autumn to reduce fungal spores. Apply preventative fungicides (e.g., copper-based or sulfur) early in the spring as buds break.'
    },
    'Apple___Black_rot': {
        'name': 'Apple Black Rot (Fungal)',
        'solution': 'Prune out dead wood, mummified fruits, and cankers during the dormant season. Apply labeled fungicides from silver tip stage through harvest.'
    },
    'Apple___Cedar_apple_rust': {
        'name': 'Cedar Apple Rust (Fungal)',
        'solution': 'Remove nearby structural galls on eastern red cedars if possible. Apply protective fungicides containing Myclobutanil when apple flower buds show pink.'
    },
    'Apple___healthy': {
        'name': 'Healthy Apple Leaf',
        'solution': 'No treatment needed! Maintain balanced watering, ensure proper sunlight, and continue regular pruning schedules.'
    },
    'Blueberry___healthy': {
        'name': 'Healthy Blueberry Leaf',
        'solution': 'Looks great! Maintain acidic soil conditions (pH 4.5–5.5) and keep a thick layer of organic mulch around the roots.'
    },
    'Cherry___Powdery_mildew': {
        'name': 'Cherry Powdery Mildew (Fungal)',
        'solution': 'Prune to improve air circulation inside the tree canopy. Apply neem oil, potassium bicarbonate, or horticultural oils at the first sign of symptoms.'
    },
    'Cherry___healthy': {
        'name': 'Healthy Cherry Leaf',
        'solution': 'Excellent condition. Continue monitoring for pests and ensure deep, infrequent watering during dry spells.'
    },
    'Corn___Cercospora_leaf_spot Gray_leaf_spot': {
        'name': 'Corn Gray Leaf Spot (Fungal)',
        'solution': 'Practice crop rotation with non-host plants (like soybeans) next season. Clean plant debris after harvest or practice deep tillage. Apply foliar fungicides if severe.'
    },
    'Corn___Common_rust_': {
        'name': 'Corn Common Rust (Fungal)',
        'solution': 'Typically doesn\'t cause severe loss; use resistant hybrids in the future. In severe cases or seed corn production, apply preventative strobilurin or triazole fungicides.'
    },
    'Corn___Northern_Leaf_Blight': {
        'name': 'Northern Corn Leaf Blight (Fungal)',
        'solution': 'Manage residue with tillage and rotate away from corn for at least 1 year. Apply labeled fungicides if lesions appear on upper leaves before silking.'
    },
    'Corn___healthy': {
        'name': 'Healthy Corn Plant',
        'solution': 'Healthy growth! Ensure consistent nitrogen fertilization and keep fields free of competing weeds.'
    },
    'Grape___Black_rot': {
        'name': 'Grape Black Rot (Fungal)',
        'solution': 'Strictly prune and remove all mummified berries from the vines. Apply preventative fungicides starting at bud break until 4 weeks post-bloom.'
    },
    'Grape___healthy': {
        'name': 'Healthy Grape Leaf',
        'solution': 'Healthy vines. Keep up with canopy management to maximize airflow and sunlight penetration.'
    },
    'Peach___Bacterial_spot': {
        'name': 'Peach Bacterial Spot (Bacterial)',
        'solution': 'Avoid overhead irrigation. Apply copper-based bactericides during late dormancy/bud split, or use oxytetracycline treatments during the growing season.'
    },
    'Peach___healthy': {
        'name': 'Healthy Peach Leaf',
        'solution': 'Vibrant and healthy. Thin fruits properly in early summer to prevent branch breakage and optimize nutrient distribution.'
    },
    'Pepper,_bell___Bacterial_spot': {
        'name': 'Bell Pepper Bacterial Spot (Bacterial)',
        'solution': 'Remove infected plants immediately to prevent spreading. Spray copper fungicides mixed with Mancozeb weekly during humid, warm periods.'
    },
    'Pepper,_bell___healthy': {
        'name': 'Healthy Bell Pepper Leaf',
        'solution': 'Perfectly healthy. Provide well-draining soil and a consistent watering regimen to avoid blossom end rot.'
    },
    'Potato___Early_blight': {
        'name': 'Potato Early Blight (Fungal)',
        'solution': 'Maintain high plant vigor with proper nitrogen fertilization. Apply preventative fungicides (Chlorothalonil or copper soap) every 7–10 days during rainy weather.'
    },
    'Potato___Late_blight': {
        'name': 'Potato Late Blight (Oomycete - Highly Destructive)',
        'solution': 'Destroy infected plants immediately (do not compost). Apply aggressive protectant fungicides like Mancozeb or specialized systemic options if conditions remain cool and wet.'
    },
    'Potato___healthy': {
        'name': 'Healthy Potato Leaf',
        'solution': 'Keep it up! Mound soil around the base of the plants to protect developing tubers from sunlight.'
    },
    'Raspberry___healthy': {
        'name': 'Healthy Raspberry Leaf',
        'solution': 'Looking strong. Keep canes tied neatly to support trellises and prune old floricanes right after harvest.'
    },
    'Soybean___healthy': {
        'name': 'Healthy Soybean Leaf',
        'solution': 'Great canopy density! Monitor regularly for insects like aphids or stink bugs as pods begin to fill.'
    },
    'Squash___Powdery_mildew': {
        'name': 'Squash Powdery Mildew (Fungal)',
        'solution': 'Plant in full sun and optimize spacing. Use organic sprays containing baking soda (potassium bicarbonate) or neem oil thoroughly across all leaf surfaces.'
    },
    'Strawberry___Leaf_scorch': {
        'name': 'Strawberry Leaf Scorch (Fungal)',
        'solution': 'Remove older, heavily blotched leaves. Avoid overhead watering late in the day. Apply copper or capture-based fungicides before blossoms open.'
    },
    'Strawberry___healthy': {
        'name': 'Healthy Strawberry Leaf',
        'solution': 'Healthy patch! Use straw mulch underneath the plants to keep berries off bare soil and prevent rot.'
    },
    'Tomato___Bacterial_spot': {
        'name': 'Tomato Bacterial Spot (Bacterial)',
        'solution': 'Avoid working among wet plants. Apply a tank-mix of copper fungicide and Mancozeb to protect healthy foliage from spreading bacteria.'
    },
    'Tomato___Early_blight': {
        'name': 'Tomato Early Blight (Fungal)',
        'solution': 'Prune lower leaves touching the soil. Mulch around the base of the plant to prevent soil splash. Treat with copper fungicides or Chlorothalonil.'
    },
    'Tomato___Late_blight': {
        'name': 'Tomato Late Blight (Oomycete)',
        'solution': 'Requires urgent attention. Pull out and bag highly infected plants to stop spores traveling on the wind. Spray remaining healthy plants with preventative fungicides.'
    },
    'Tomato___Leaf_Mold': {
        'name': 'Tomato Leaf Mold (Fungal)',
        'solution': 'Common in greenhouses. Significantly reduce relative humidity, increase ventilation, and apply preventative bio-fungicides or copper sprays.'
    },
    'Tomato___Septoria_leaf_spot': {
        'name': 'Tomato Septoria Leaf Spot (Fungal)',
        'solution': 'Eliminate nightshade weeds nearby. Clean up all garden debris in the fall. Apply organic copper sprays or chemical fungicides on a regular cadence during wet summer weeks.'
    },
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': {
        'name': 'Tomato Yellow Leaf Curl Virus (Vectored by Whiteflies)',
        'solution': 'Viruses cannot be cured. Focus on controlling the vector: use yellow sticky traps, introduce natural predators, or use insecticidal soaps to kill Whiteflies.'
    },
    'Tomato___Tomato_mosaic_virus': {
        'name': 'Tomato Mosaic Virus (Viral)',
        'solution': 'No chemical cure. Remove and burn infected plants. Wash hands and sterilize tools thoroughly with a TSP or bleach solution before handling healthy plants.'
    },
    'Tomato___healthy': {
        'name': 'Healthy Tomato Leaf',
        'solution': 'Superb condition! Keep staking the plant for structure, prune suckers for optimal airflow, and enjoy your harvest.'
    }
}

# Extract ordered class list matching model indices
CLASS_NAMES = ['Apple___Apple_scab', 
               'Apple___Black_rot', 
               'Apple___Cedar_apple_rust', 
               'Apple___healthy', 
               'Blueberry___healthy', 
               'Cherry_(including_sour)___Powdery_mildew', 
               'Cherry_(including_sour)___healthy', 
               'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot', 
               'Corn_(maize)___Common_rust_', 
               'Corn_(maize)___Northern_Leaf_Blight', 
               'Corn_(maize)___healthy', 
               'Grape___Black_rot', 
               'Grape___healthy', 
               'Peach___Bacterial_spot', 
               'Peach___healthy', 
               'Pepper,_bell___Bacterial_spot', 
               'Pepper,_bell___healthy', 
               'Potato___Early_blight', 
               'Potato___Late_blight', 
               'Potato___healthy', 
               'Raspberry___healthy', 
               'Soybean___healthy',
               'Squash___Powdery_mildew', 
               'Strawberry___Leaf_scorch', 
               'Strawberry___healthy', 
               'Tomato___Bacterial_spot', 
               'Tomato___Early_blight', 
               'Tomato___Late_blight', 
               'Tomato___Leaf_Mold', 
               'Tomato___Septoria_leaf_spot', 
               'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 
               'Tomato___Tomato_mosaic_virus', 
               'Tomato___healthy']

# 3. Load Model with Caching
@st.cache_resource
def load_my_model():
    return tf.keras.models.load_model("phase2_best.keras")

with st.spinner("Loading Deep Learning Model..."):
    model = load_my_model()

# 4. User Interface Setup
st.title("🌿 Plant Disease Classifier & Remedy Guide")
st.write("Upload a crop leaf photograph to immediately detect issues and view treatment workflows.")

uploaded_file = st.file_uploader("Select a leaf image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    image = Image.open(uploaded_file)
    st.image(image, caption="Target Crop Specimen", use_container_width=True)
    
    st.write("🔄 Running Diagnostic Inference...")
    
    # 5. Mirroring Training Preprocessing
    img = image.resize((224, 224))
    img_array = tf.keras.preprocessing.image.img_to_array(img)
    img_array = img_array / 255.0  # Rescale matching 1./255 ImageDataGenerator
    img_array = np.expand_dims(img_array, axis=0)
    
    # 6. Prediction Logic
    predictions = model.predict(img_array)
    predicted_class_idx = np.argmax(predictions)
    confidence = np.max(predictions) * 100
    
    raw_class_name = CLASS_NAMES[predicted_class_idx]
    diagnostic_info = DISEASE_GUIDE[raw_class_name]
    
    # 7. Rendering Dynamic UI Components Based on Results
    st.write("---")
    
    if "healthy" in raw_class_name.lower():
        st.success(f"### Diagnostic Result: **{diagnostic_info['name']}**")
    else:
        st.error(f"### Diagnostic Result: **{diagnostic_info['name']}**")
        
    st.metric(label="Model Confidence Score", value=f"{confidence:.2f}%")
    
    # 8. Dynamic Treatment Card Presentation
    st.subheader("📋 Recommended Treatment Actions")
    st.info(diagnostic_info['solution'])