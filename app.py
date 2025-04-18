import base64
import difflib
from io import BytesIO

import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from flask import Flask, render_template, request, jsonify
from sklearn.preprocessing import LabelEncoder

app = Flask(__name__)

# Configure modern matplotlib style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Load models and data
def load_models():
    """Load ML models and encoders"""
    model = joblib.load('assets/optimized_disease_predictor.pkl')
    label_encoder = joblib.load('assets/label_encoder.pkl')
    selected_features = joblib.load('assets/selected_features.pkl')
    return model, label_encoder, selected_features

model, label_encoder, selected_features = load_models()

# Complete list of symptoms
ALL_SYMPTOMS = [
    'itching', 'skin_rash', 'nodal_skin_eruptions', 'continuous_sneezing', 'shivering',
    'chills', 'joint_pain', 'stomach_pain', 'acidity', 'ulcers_on_tongue',
    'muscle_wasting', 'vomiting', 'burning_micturition', 'spotting_ urination', 'fatigue',
    'weight_gain', 'anxiety', 'cold_hands_and_feets', 'mood_swings', 'weight_loss',
    'restlessness', 'lethargy', 'patches_in_throat', 'irregular_sugar_level', 'cough',
    'high_fever', 'sunken_eyes', 'breathlessness', 'sweating', 'dehydration',
    'indigestion', 'headache', 'yellowish_skin', 'dark_urine', 'nausea',
    'loss_of_appetite', 'pain_behind_the_eyes', 'back_pain', 'constipation', 'abdominal_pain',
    'diarrhoea', 'mild_fever', 'yellow_urine', 'yellowing_of_eyes', 'acute_liver_failure',
    'fluid_overload', 'swelling_of_stomach', 'swelled_lymph_nodes', 'malaise', 'blurred_and_distorted_vision',
    'phlegm', 'throat_irritation', 'redness_of_eyes', 'sinus_pressure', 'runny_nose',
    'congestion', 'chest_pain', 'weakness_in_limbs', 'fast_heart_rate', 'pain_during_bowel_movements',
    'pain_in_anal_region', 'bloody_stool', 'irritation_in_anus', 'neck_pain', 'dizziness',
    'cramps', 'bruising', 'obesity', 'swollen_legs', 'swollen_blood_vessels',
    'puffy_face_and_eyes', 'enlarged_thyroid', 'brittle_nails', 'swollen_extremeties', 'excessive_hunger',
    'extra_marital_contacts', 'drying_and_tingling_lips', 'slurred_speech', 'knee_pain', 'hip_joint_pain',
    'muscle_weakness', 'stiff_neck', 'swelling_joints', 'movement_stiffness', 'spinning_movements',
    'loss_of_balance', 'unsteadiness', 'weakness_of_one_body_side', 'loss_of_smell', 'bladder_discomfort',
    'foul_smell_of urine', 'continuous_feel_of_urine', 'passage_of_gases', 'internal_itching', 'toxic_look_(typhos)',
    'depression', 'irritability', 'muscle_pain', 'altered_sensorium', 'red_spots_over_body',
    'belly_pain', 'abnormal_menstruation', 'dischromic _patches', 'watering_from_eyes', 'increased_appetite',
    'polyuria', 'family_history', 'mucoid_sputum', 'rusty_sputum', 'lack_of_concentration',
    'visual_disturbances', 'receiving_blood_transfusion', 'receiving_unsterile_injections', 'coma', 'stomach_bleeding',
    'distention_of_abdomen', 'history_of_alcohol_consumption', 'fluid_overload', 'blood_in_sputum', 'prominent_veins_on_calf',
    'palpitations', 'painful_walking', 'pus_filled_pimples', 'blackheads', 'scurring',
    'skin_peeling', 'silver_like_dusting', 'small_dents_in_nails', 'inflammatory_nails', 'blister',
    'red_sore_around_nose', 'yellow_crust_ooze'
]

@app.route('/')
def home():
    """Render homepage with symptom selection"""
    return render_template('index.html', symptoms=ALL_SYMPTOMS)

@app.route('/predict', methods=['POST'])
def predict():
    """Handle prediction requests"""
    try:
        # Process form data
        symptoms_input = request.form.getlist('symptoms')
        custom_symptoms = request.form.get('custom_symptoms', '').split(',')
        
        # Validate and process symptoms
        user_symptoms = process_symptoms(symptoms_input, custom_symptoms)
        if not user_symptoms:
            return render_template('index.html', 
                                symptoms=ALL_SYMPTOMS,
                                error="Please select or enter at least one valid symptom")

        # Make prediction
        prediction = get_prediction(user_symptoms)
        
        # Generate visualizations
        graph_image = create_prediction_graph(prediction)
        symptoms_chart = create_symptoms_chart(user_symptoms)
        
        return render_template('results.html',
                            symptoms_reported=list(user_symptoms.keys()),
                            prediction=prediction,
                            graph_image=graph_image,
                            symptoms_chart=symptoms_chart)

    except Exception as e:
        app.logger.error(f"Prediction error: {str(e)}")
        return render_template('index.html', 
                            symptoms=ALL_SYMPTOMS,
                            error="An error occurred during prediction. Please try again.")

@app.route('/api/predict', methods=['POST'])
def api_predict():
    """API endpoint for predictions"""
    try:
        data = request.get_json()
        if not data or 'symptoms' not in data:
            return jsonify({'error': 'Please provide symptoms list'}), 400
        
        # Process symptoms
        user_symptoms = {}
        for symptom in data['symptoms']:
            matched_symptom = get_closest_symptom(symptom)
            if matched_symptom:
                user_symptoms[matched_symptom] = 1
        
        if not user_symptoms:
            return jsonify({'error': 'No valid symptoms provided'}), 400
        
        # Make prediction
        prediction = get_prediction(user_symptoms)
        
        return jsonify({
            'symptoms_reported': list(user_symptoms.keys()),
            'prediction': prediction
        })

    except Exception as e:
        return jsonify({'error': str(e)}), 500

def process_symptoms(selected_symptoms, custom_symptoms):
    """Process and validate symptoms"""
    user_symptoms = {}
    
    # Add selected symptoms
    for symptom in selected_symptoms:
        if symptom in ALL_SYMPTOMS:
            user_symptoms[symptom] = 1
    
    # Add custom symptoms with fuzzy matching
    for symptom in custom_symptoms:
        symptom = symptom.strip()
        if symptom:
            matched_symptom = get_closest_symptom(symptom)
            if matched_symptom:
                user_symptoms[matched_symptom] = 1
    
    return user_symptoms

def get_closest_symptom(input_symptom):
    """Fuzzy match symptoms"""
    input_symptom = input_symptom.replace(' ', '_').lower()
    matches = difflib.get_close_matches(input_symptom, ALL_SYMPTOMS, n=1, cutoff=0.6)
    return matches[0] if matches else None

def get_prediction(symptoms):
    """Get prediction from model"""
    input_data = pd.DataFrame(0, index=[0], columns=selected_features)
    for symptom, value in symptoms.items():
        if symptom in input_data.columns:
            input_data[symptom] = value
    
    probabilities = model.predict_proba(input_data)[0]
    top3_indices = probabilities.argsort()[-3:][::-1]
    top3_diseases = label_encoder.inverse_transform(top3_indices)
    top3_probs = probabilities[top3_indices]
    
    return {
        'primary_prediction': {
            'disease': top3_diseases[0],
            'confidence': float(top3_probs[0])
        },
        'alternative_predictions': [
            {'disease': disease, 'confidence': float(prob)}
            for disease, prob in zip(top3_diseases[1:], top3_probs[1:])
        ]
    }

def create_prediction_graph(prediction):
    """Create prediction probability graph"""
    diseases = [prediction['primary_prediction']['disease']]
    probabilities = [prediction['primary_prediction']['confidence']]
    
    for alt in prediction['alternative_predictions']:
        diseases.append(alt['disease'])
        probabilities.append(alt['confidence'])
    
    plt.figure(figsize=(10, 6))
    ax = sns.barplot(x=diseases, y=probabilities, palette="viridis")
    
    # Add value labels
    for i, p in enumerate(ax.patches):
        ax.annotate(f'{probabilities[i]:.1%}', 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='center', 
                   xytext=(0, 10), 
                   textcoords='offset points')
    
    plt.title('Disease Prediction Confidence', fontsize=16, pad=20)
    plt.xlabel('')
    plt.ylabel('Confidence Score')
    plt.xticks(rotation=45, ha='right')
    plt.ylim(0, 1)
    plt.tight_layout()
    
    return save_plot_to_base64()

def create_symptoms_chart(symptoms):
    """Create symptoms importance chart"""
    symptoms_list = list(symptoms.keys())
    
    plt.figure(figsize=(8, 6))
    sns.barplot(x=[1]*len(symptoms_list), y=symptoms_list, orient='h', palette="rocket")
    plt.title('Reported Symptoms', fontsize=14)
    plt.xlabel('')
    plt.ylabel('')
    plt.xticks([])
    plt.tight_layout()
    
    return save_plot_to_base64()

def save_plot_to_base64():
    """Save matplotlib plot to base64 string"""
    buffer = BytesIO()
    plt.savefig(buffer, format='png', dpi=100, bbox_inches='tight')
    plt.close()
    buffer.seek(0)
    return base64.b64encode(buffer.getvalue()).decode('utf-8')

if __name__ == '__main__':
    app.run(debug=True)