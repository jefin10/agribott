import os
import pickle
import numpy as np
import cv2
from flask import Flask, request, jsonify
from flask_cors import CORS
from skimage.feature import hog

# Load model
MODEL_PATH = os.path.join(os.path.dirname(__file__), "plant_disease_model_compact.pkl")
with open(MODEL_PATH, 'rb') as f:
    model_data = pickle.load(f)
clf = model_data['classifier']
pca = model_data['pca']
classes = model_data['classes']

# Treatment recommendations (add more as needed)
TREATMENTS = {
    'Apple___Apple_scab': 'Apply fungicide sprays during spring when leaves start to appear. Remove and destroy fallen leaves to reduce fungal spores.',
    'Apple___Black_rot': 'Prune out dead or diseased wood. Apply fungicides during growing season. Remove nearby wild hosts.',
    'Apple___Cedar_apple_rust': 'Apply fungicides in spring. Keep apple trees away from cedar and juniper trees. Remove galls from cedar trees.',
    'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot': 'Rotate crops with non-host crops. Apply fungicides during growing season. Consider resistant varieties.',
    'Corn_(maize)___Common_rust_': 'Apply appropriate fungicides. Plant resistant corn varieties. Practice proper field sanitation.',
    'Corn_(maize)___Northern_Leaf_Blight': 'Use resistant hybrids. Apply foliar fungicides. Practice crop rotation with non-host crops.',
    'Potato___Early_blight': 'Apply fungicides at first sign of disease. Rotate crops. Maintain adequate plant nutrition.',
    'Potato___Late_blight': 'Apply preventative fungicides. Eliminate cull piles and volunteer plants. Harvest during dry weather.',
    'Tomato___Bacterial_spot': 'Apply copper-based sprays. Rotate crops. Avoid overhead irrigation.',
    'Tomato___Early_blight': 'Remove lower infected leaves. Apply appropriate fungicides. Mulch around plants.',
    'Tomato___Late_blight': 'Apply preventative fungicides. Remove and destroy infected plants. Provide good airflow around plants.',
    'Tomato___Leaf_Mold': 'Improve air circulation. Reduce humidity. Apply fungicides.',
    'Tomato___Septoria_leaf_spot': 'Remove infected leaves. Apply fungicides. Rotate crops.',
    'Tomato___Spider_mites Two-spotted_spider_mite': 'Apply miticides or insecticidal soap. Increase humidity. Introduce predatory mites.',
    'Tomato___Target_Spot': 'Apply appropriate fungicides. Improve airflow. Remove infected plant material.',
    'Tomato___Tomato_Yellow_Leaf_Curl_Virus': 'Control whitefly vectors with appropriate insecticides. Use reflective mulches. Remove infected plants.',
    'Tomato___Tomato_mosaic_virus': 'Remove and destroy infected plants. Control aphids. Disinfect tools between plants.'
}
for c in classes:
    if 'healthy' in c and c not in TREATMENTS:
        TREATMENTS[c] = 'No treatment needed. Plant appears healthy.'

def extract_features(img, img_size=96):
    """
    Extract HOG and color statistics features from image.
    Make identical to training version.
    """
    # First resize, then handle color spaces (match training pipeline)
    img = cv2.resize(img, (img_size, img_size))
    
    # Handle grayscale images like in training
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_RGBA2RGB)
        
    # Rest of processing remains the same
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    hog_features = hog(gray, orientations=8, pixels_per_cell=(12, 12),
                      cells_per_block=(2, 2), visualize=False)
    hsv_img = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
    h_mean, h_std = np.mean(hsv_img[:,:,0]), np.std(hsv_img[:,:,0])
    s_mean, s_std = np.mean(hsv_img[:,:,1]), np.std(hsv_img[:,:,1])
    v_mean, v_std = np.mean(hsv_img[:,:,2]), np.std(hsv_img[:,:,2])
    r_mean, r_std = np.mean(img[:,:,0]), np.std(img[:,:,0])
    g_mean, g_std = np.mean(img[:,:,1]), np.std(img[:,:,1])
    b_mean, b_std = np.mean(img[:,:,2]), np.std(img[:,:,2])
    color_features = np.array([h_mean, h_std, s_mean, s_std, v_mean, v_std,
                              r_mean, r_std, g_mean, g_std, b_mean, b_std])
    combined_features = np.concatenate((hog_features, color_features))
    return combined_features
app = Flask(__name__)
CORS(app)
@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No image uploaded'})
    
    file = request.files['image']
    in_memory = np.frombuffer(file.read(), np.uint8)
    img = cv2.imdecode(in_memory, cv2.IMREAD_COLOR)
    
    if img is None:
        return jsonify({'success': False, 'error': 'Invalid image'})
    
    # REMOVE THIS LINE - Don't convert to RGB to match training pipeline
    # img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    features = extract_features(img)
    features_pca = pca.transform([features])
    proba = clf.predict_proba(features_pca)[0]
    pred = np.argmax(proba)
    disease = classes[pred]
    confidence = float(proba[pred])
    treatment = TREATMENTS.get(disease, "No specific treatment information available.")
    
    return jsonify({
        'success': True,
        'disease': disease.replace('___', ' - ').replace('_', ' '),
        'confidence': round(confidence * 100, 2),
        'treatment': treatment
    })
if __name__ == '__main__':
    app.run(debug=True)